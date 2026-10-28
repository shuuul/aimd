//! DSpark draft model for MiniCPM-OCR (`openbmb/MiniCPM-OCR-1.0-DSpark`,
//! `Qwen3DSparkModel`), following vLLM `qwen3_dspark.py` /
//! `qwen3_dflash.py` and the DSpark speculator
//! (`v1/worker/gpu/spec_decode/dspark/speculator.py`):
//!
//! - Context: the target's outputs of layers `target_layer_ids`
//!   (`[3, 7, 11, 15, 19]`) concatenated, `fc` -> `hidden_norm`. Every
//!   draft layer projects them to K/V (`k_norm`, RoPE at their absolute
//!   positions) and appends them to its cache; the context is never run
//!   through the draft blocks.
//! - Block: `sample_from_anchor` -> the anchor (last emitted token) and
//!   `N - 1` `mask_token_id`s at the positions after the context, embedded
//!   with the target's `embed_tokens`. Five Qwen3 blocks (pre-RMSNorm,
//!   `q_norm`/`k_norm`, full NeoX RoPE over 256 dims, theta 1e7, GQA 8/2)
//!   with non-causal attention over `[cached context, block]`; the block
//!   K/V are not cached.
//! - Proposals: `norm` -> the draft `lm_head` (reduced 114688-token
//!   vocabulary) gives base logits for all `N` block positions (position
//!   `i` predicts token `i + 1`). Sampling is sequential: step `i` adds the
//!   Markov bias `markov_w2(markov_w1[prev])` and takes the argmax, mapped
//!   to a target id by `d2t` (`id + d2t[id]`); `prev` starts at the anchor
//!   and is the previous mapped proposal.
//!
//! The confidence head only drives vLLM's adaptive verification, which the
//! official serving config disables, so it is not loaded.

use std::path::Path;

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask};
use aimd_weights::{WeightsError, load_checkpoint};
use serde::Deserialize;

use super::decoder::{MiniCpmDecoder, validity_mask};
use crate::models::kv_cache::KvCache;
use crate::models::paddleocr_vl::weights::{Emb, Lin, Loader};
use crate::models::unlimited_ocr::mlx_weights::LoadError;

/// The official draft checkpoint for MiniCPM-OCR 1.0.
pub const DSPARK_REPO: &str = "openbmb/MiniCPM-OCR-1.0-DSpark";

/// Drafter failures.
#[derive(Debug, thiserror::Error)]
pub enum DSparkError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("draft config: {0}")]
    Config(String),
}

#[derive(Debug, Clone, Deserialize)]
struct DflashConfig {
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    markov_rank: usize,
    #[serde(default)]
    markov_head_type: Option<String>,
    #[serde(default)]
    projector_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    rope_theta: f64,
}

/// The fields of the draft `config.json` this port reads.
#[derive(Debug, Clone, Deserialize)]
pub struct DSparkConfig {
    architectures: Vec<String>,
    /// Proposals per draft forward (`num_speculative_tokens`).
    pub block_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    vocab_size: usize,
    draft_vocab_size: usize,
    num_target_layers: usize,
    rope_parameters: RopeParams,
    #[serde(default)]
    layer_types: Vec<String>,
    dflash_config: DflashConfig,
}

impl DSparkConfig {
    pub fn target_layer_ids(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }
}

struct DraftLayer {
    in_norm: MlxArray,
    post_norm: MlxArray,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: MlxArray,
    k_norm: MlxArray,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// The DSpark drafter and its per-request context caches.
pub struct DSparkDrafter {
    cfg: DSparkConfig,
    fc: Lin,
    hidden_norm: MlxArray,
    layers: Vec<DraftLayer>,
    norm: MlxArray,
    lm_head: Lin,
    markov_w1: Emb,
    markov_w2: Lin,
    /// Draft id -> target id, `[draft_vocab]` u32.
    d2t: MlxArray,
}

/// `d2t[i] + i` from the checkpoint's int64 delta table.
fn absolute_ids(bytes: &[u8], n: usize, target_vocab: usize) -> Result<Vec<u32>, DSparkError> {
    if bytes.len() != n * 8 {
        return Err(DSparkError::Config(format!(
            "d2t holds {} bytes",
            bytes.len()
        )));
    }
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let delta = i64::from_le_bytes(*b);
            let id = i as i64 + delta;
            u32::try_from(id)
                .ok()
                .filter(|&t| (t as usize) < target_vocab)
                .ok_or_else(|| DSparkError::Config(format!("d2t[{i}] maps to {id}")))
        })
        .collect()
}

impl DSparkDrafter {
    /// Loads the draft snapshot for `target` (whose embeddings it shares).
    pub fn load(dir: &Path, target: &MiniCpmDecoder) -> Result<Self, DSparkError> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| DSparkError::Config(format!("read config.json: {e}")))?;
        let cfg: DSparkConfig =
            serde_json::from_str(&text).map_err(|e| DSparkError::Config(e.to_string()))?;
        let t = target.config();
        let dc = &cfg.dflash_config;
        let checks = [
            (
                cfg.architectures.iter().any(|a| a == "Qwen3DSparkModel"),
                "architecture",
            ),
            (
                dc.projector_type.as_deref().is_none_or(|p| p == "dspark"),
                "projector_type",
            ),
            (
                dc.markov_head_type
                    .as_deref()
                    .is_none_or(|p| p == "vanilla"),
                "markov_head_type",
            ),
            (
                cfg.layer_types.iter().all(|l| l == "full_attention"),
                "layer_types",
            ),
            (cfg.vocab_size == t.vocab_size, "vocab_size"),
            (
                cfg.num_target_layers == t.num_hidden_layers,
                "num_target_layers",
            ),
            (cfg.hidden_size == t.hidden_size, "hidden_size"),
            (cfg.draft_vocab_size <= cfg.vocab_size, "draft_vocab_size"),
            (
                (dc.mask_token_id as usize) < cfg.vocab_size,
                "mask_token_id",
            ),
            (cfg.block_size >= 1, "block_size"),
            (
                !dc.target_layer_ids.is_empty()
                    && dc.target_layer_ids.windows(2).all(|w| w[0] < w[1])
                    && dc.target_layer_ids.iter().all(|&l| l < t.num_hidden_layers),
                "target_layer_ids",
            ),
            (
                cfg.num_attention_heads
                    .is_multiple_of(cfg.num_key_value_heads),
                "attention heads",
            ),
        ];
        if let Some((_, what)) = checks.iter().find(|(ok, _)| !ok) {
            return Err(DSparkError::Config(format!("unsupported draft {what}")));
        }

        let store = load_checkpoint(dir)?;
        let ld = Loader::new(&store, None)?;
        let (d, hd, ff) = (cfg.hidden_size, cfg.head_dim, cfg.intermediate_size);
        let (nq, nkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("layers.{l}");
            layers.push(DraftLayer {
                in_norm: ld.tensor(&format!("{p}.input_layernorm.weight"), Some(&[d]))?,
                post_norm: ld
                    .tensor(&format!("{p}.post_attention_layernorm.weight"), Some(&[d]))?,
                q: ld.linear(&format!("{p}.self_attn.q_proj"), false, &[nq * hd, d])?,
                k: ld.linear(&format!("{p}.self_attn.k_proj"), false, &[nkv * hd, d])?,
                v: ld.linear(&format!("{p}.self_attn.v_proj"), false, &[nkv * hd, d])?,
                o: ld.linear(&format!("{p}.self_attn.o_proj"), false, &[d, nq * hd])?,
                q_norm: ld.tensor(&format!("{p}.self_attn.q_norm.weight"), Some(&[hd]))?,
                k_norm: ld.tensor(&format!("{p}.self_attn.k_norm.weight"), Some(&[hd]))?,
                gate: ld.linear(&format!("{p}.mlp.gate_proj"), false, &[ff, d])?,
                up: ld.linear(&format!("{p}.mlp.up_proj"), false, &[ff, d])?,
                down: ld.linear(&format!("{p}.mlp.down_proj"), false, &[d, ff])?,
            });
        }
        let nt = dc.target_layer_ids.len();
        let r = dc.markov_rank;
        let dv = cfg.draft_vocab_size;
        let d2t_info = store
            .tensor_info("d2t")
            .ok_or_else(|| DSparkError::Config("missing d2t".into()))?;
        if format!("{:?}", d2t_info.dtype) != "I64" || d2t_info.shape != [dv] {
            return Err(DSparkError::Config(format!(
                "d2t {:?} {:?}",
                d2t_info.dtype, d2t_info.shape
            )));
        }
        let ids = absolute_ids(store.tensor_bytes("d2t")?, dv, cfg.vocab_size)?;
        let d2t = MlxArray::from_u32(&[dv], &ids)?;
        let drafter = Self {
            fc: ld.linear("fc", false, &[d, nt * d])?,
            hidden_norm: ld.tensor("hidden_norm.weight", Some(&[d]))?,
            layers,
            norm: ld.tensor("norm.weight", Some(&[d]))?,
            lm_head: ld.linear("lm_head", false, &[dv, d])?,
            markov_w1: Emb::Dense(
                ld.tensor("markov_head.markov_w1.weight", Some(&[cfg.vocab_size, r]))?,
            ),
            markov_w2: ld.linear("markov_head.markov_w2", false, &[dv, r])?,
            d2t,
            cfg,
        };
        drop(store);
        Ok(drafter)
    }

    pub fn config(&self) -> &DSparkConfig {
        &self.cfg
    }

    /// Fresh per-layer context caches for one request.
    pub(crate) fn make_cache(&self) -> Vec<KvCache> {
        self.layers.iter().map(|_| KvCache::new(None)).collect()
    }

    /// The position of the next anchor: context tokens consumed so far.
    pub(crate) fn context_len(cache: &[KvCache]) -> usize {
        cache.first().map_or(0, KvCache::offset)
    }

    fn heads(&self, x: &MlxArray, lin: &Lin, heads: usize) -> Result<MlxArray, MlxError> {
        let (b, l) = (x.dim(0)?, x.dim(1)?);
        lin.forward(x)?
            .reshape(&[b, l, heads, self.cfg.head_dim])?
            .transpose(&[0, 2, 1, 3])
    }

    fn rope(&self, x: &MlxArray, offset: usize) -> Result<MlxArray, MlxError> {
        x.rope(
            self.cfg.head_dim,
            self.cfg.rope_parameters.rope_theta,
            1.0,
            offset as i32,
        )
    }

    /// [`Self::rope`] with row `b` of `x` (`[B, H, L, D]`) starting at
    /// `offsets[b]`.
    fn rope_rows(&self, x: &MlxArray, offsets: &[usize]) -> Result<MlxArray, MlxError> {
        if offsets.iter().all(|&o| o == offsets[0]) {
            return self.rope(x, offsets[0]);
        }
        let s = x.shape()?;
        let rows: Vec<MlxArray> = offsets
            .iter()
            .enumerate()
            .map(|(b, &o)| {
                let row = x.slice(&[b, 0, 0, 0], &[b + 1, s[1], s[2], s[3]], &[1, 1, 1, 1])?;
                self.rope(&row, o)
            })
            .collect::<Result<_, _>>()?;
        let refs: Vec<&MlxArray> = rows.iter().collect();
        MlxArray::concatenate(&refs, 0)
    }

    /// Batched [`Self::draft`]: row `b` drafts after `anchors[b]`, and the
    /// first `fresh[b]` columns of its `context` row (`[B, S, features]`)
    /// are new context (the rest are holes, masked from then on).
    pub(crate) fn draft_batch(
        &self,
        target: &MiniCpmDecoder,
        anchors: &[u32],
        context: &MlxArray,
        fresh: &[usize],
        state: &mut DraftBatch,
    ) -> Result<MlxArray, MlxError> {
        let cfg = &self.cfg;
        let (nq, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let eps = cfg.rms_norm_eps;
        let n = cfg.block_size;
        let b = anchors.len();
        let s = context.dim(1)?;
        let starts = state.lens.clone();
        for ((len, valid), &f) in state.lens.iter_mut().zip(&mut state.valid).zip(fresh) {
            *len += f;
            valid.extend((0..s).map(|c| c < f));
        }
        let mask = validity_mask(&state.valid, n)?;

        let h_ctx = self
            .fc
            .forward(context)?
            .rms_norm(Some(&self.hidden_norm), eps)?;
        let mut ids = vec![cfg.dflash_config.mask_token_id; b * n];
        for (i, &a) in anchors.iter().enumerate() {
            ids[i * n] = a;
        }
        let mut x = target.embed_ids(&MlxArray::from_u32(&[b, n], &ids)?)?;
        let scale = (hd as f64).powf(-0.5) as f32;
        for (layer, c) in self.layers.iter().zip(state.caches.iter_mut()) {
            let ck = self
                .heads(&h_ctx, &layer.k, nkv)?
                .rms_norm(Some(&layer.k_norm), eps)?;
            let cv = self.heads(&h_ctx, &layer.v, nkv)?;
            let (keys, values) = c.update_and_fetch(&self.rope_rows(&ck, &starts)?, &cv)?;

            let xn = x.rms_norm(Some(&layer.in_norm), eps)?;
            let q = self
                .heads(&xn, &layer.q, nq)?
                .rms_norm(Some(&layer.q_norm), eps)?;
            let bk = self
                .heads(&xn, &layer.k, nkv)?
                .rms_norm(Some(&layer.k_norm), eps)?;
            let bv = self.heads(&xn, &layer.v, nkv)?;
            let q = self.rope_rows(&q, &state.lens)?;
            let keys = MlxArray::concatenate(&[&keys, &self.rope_rows(&bk, &state.lens)?], 2)?;
            let values = MlxArray::concatenate(&[&values, &bv], 2)?;
            let o = q
                .scaled_dot_product_attention(&keys, &values, scale, SdpaMask::Array(&mask))?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[b, n, nq * hd])?;
            let h = x.add(&layer.o.forward(&o)?)?;
            let hn = h.rms_norm(Some(&layer.post_norm), eps)?;
            x = h.add(
                &layer
                    .down
                    .forward(&layer.gate.forward(&hn)?.swiglu(&layer.up.forward(&hn)?)?)?,
            )?;
        }
        let base = self.lm_head.forward(&x.rms_norm(Some(&self.norm), eps)?)?;
        let dv = cfg.draft_vocab_size;
        let base = base.reshape(&[b, n, dv])?;

        let mut prev = MlxArray::from_u32(&[b], anchors)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bias = self.markov_w2.forward(&self.markov_w1.embed(&prev)?)?;
            let logits = base
                .slice(&[0, i, 0], &[b, i + 1, dv], &[1, 1, 1])?
                .reshape(&[b, dv])?
                .add(&bias)?;
            let draft = logits.argmax(-1, false)?;
            prev = self.d2t.take(&draft, 0)?;
            out.push(prev.reshape(&[b, 1])?);
        }
        let refs: Vec<&MlxArray> = out.iter().collect();
        MlxArray::concatenate(&refs, 1)?.astype(MlxDtype::U32)
    }

    /// Drafts `block_size` target ids after `anchor`. `context` holds the
    /// captured target features (`[1, S, n_layers * hidden]`) of the
    /// tokens since the previous call; they are appended to `cache` first.
    pub(crate) fn draft(
        &self,
        target: &MiniCpmDecoder,
        anchor: u32,
        context: &MlxArray,
        cache: &mut [KvCache],
    ) -> Result<MlxArray, MlxError> {
        let cfg = &self.cfg;
        let (nq, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let eps = cfg.rms_norm_eps;
        let n = cfg.block_size;
        let s = context.dim(1)?;
        let start = Self::context_len(cache);
        let block_pos = start + s;

        let h_ctx = self
            .fc
            .forward(context)?
            .rms_norm(Some(&self.hidden_norm), eps)?;
        let mut ids = vec![cfg.dflash_config.mask_token_id; n];
        ids[0] = anchor;
        let mut x = target.embed_ids(&MlxArray::from_u32(&[1, n], &ids)?)?;
        let scale = (hd as f64).powf(-0.5) as f32;
        for (layer, c) in self.layers.iter().zip(cache.iter_mut()) {
            let ck = self
                .heads(&h_ctx, &layer.k, nkv)?
                .rms_norm(Some(&layer.k_norm), eps)?;
            let cv = self.heads(&h_ctx, &layer.v, nkv)?;
            let (keys, values) = c.update_and_fetch(&self.rope(&ck, start)?, &cv)?;

            let xn = x.rms_norm(Some(&layer.in_norm), eps)?;
            let q = self
                .heads(&xn, &layer.q, nq)?
                .rms_norm(Some(&layer.q_norm), eps)?;
            let bk = self
                .heads(&xn, &layer.k, nkv)?
                .rms_norm(Some(&layer.k_norm), eps)?;
            let bv = self.heads(&xn, &layer.v, nkv)?;
            let q = self.rope(&q, block_pos)?;
            let keys = MlxArray::concatenate(&[&keys, &self.rope(&bk, block_pos)?], 2)?;
            let values = MlxArray::concatenate(&[&values, &bv], 2)?;
            let o = q
                .scaled_dot_product_attention(&keys, &values, scale, SdpaMask::None)?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[1, n, nq * hd])?;
            let h = x.add(&layer.o.forward(&o)?)?;
            let hn = h.rms_norm(Some(&layer.post_norm), eps)?;
            x = h.add(
                &layer
                    .down
                    .forward(&layer.gate.forward(&hn)?.swiglu(&layer.up.forward(&hn)?)?)?,
            )?;
        }
        let base = self.lm_head.forward(&x.rms_norm(Some(&self.norm), eps)?)?;
        let dv = cfg.draft_vocab_size;
        let base = base.reshape(&[n, dv])?;

        // Sequential Markov sampling (vLLM `_sample_sequential`, greedy).
        let mut prev = MlxArray::from_u32(&[1], &[anchor])?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bias = self.markov_w2.forward(&self.markov_w1.embed(&prev)?)?;
            let logits = base.slice(&[i, 0], &[i + 1, dv], &[1, 1])?.add(&bias)?;
            let draft = logits.argmax(-1, false)?;
            prev = self.d2t.take(&draft, 0)?;
            out.push(prev.try_clone()?);
        }
        let refs: Vec<&MlxArray> = out.iter().collect();
        MlxArray::concatenate(&refs, 0)?.astype(MlxDtype::U32)
    }
}

/// Drafter state of a batch: the context caches (one row per sequence)
/// and, per row, the context positions consumed and each cache column's
/// validity (holes: rejected draft positions).
pub(crate) struct DraftBatch {
    pub(crate) caches: Vec<KvCache>,
    pub(crate) lens: Vec<usize>,
    pub(crate) valid: Vec<Vec<bool>>,
}

impl DraftBatch {
    pub(crate) fn new(drafter: &DSparkDrafter, rows: usize) -> Self {
        Self {
            caches: drafter.make_cache(),
            lens: vec![0; rows],
            valid: vec![Vec::new(); rows],
        }
    }

    /// Keeps rows `keep` (batch indices, ascending; `idx` the same as u32).
    pub(crate) fn select_rows(&mut self, keep: &[usize], idx: &MlxArray) -> Result<(), MlxError> {
        for c in &mut self.caches {
            c.select_rows(idx)?;
        }
        self.lens = keep.iter().map(|&j| self.lens[j]).collect();
        self.valid = keep
            .iter()
            .map(|&j| std::mem::take(&mut self.valid[j]))
            .collect();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpm_dspark_d2t_is_delta_encoded() {
        let deltas: [i64; 3] = [0, 5, 7];
        let bytes: Vec<u8> = deltas.iter().flat_map(|d| d.to_le_bytes()).collect();
        assert_eq!(absolute_ids(&bytes, 3, 16).unwrap(), vec![0, 6, 9]);
        assert!(absolute_ids(&bytes, 3, 9).is_err());
        assert!(absolute_ids(&bytes[..16], 3, 16).is_err());
    }
}
