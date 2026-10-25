//! Unlimited-OCR DeepSeek-V2 MoE decoder on MLX, plus the Python-exact
//! KV caches (`KVCache` step-256 growth and `RingSlidingKVCache`).
//!
//! # Op-for-op port of the mlx_vlm graph
//!
//! Source of truth: mlx_vlm 0.7.0 `models/deepseekocr/language.py`
//! (`LlamaAttention`, `MoEGate`, `DeepseekV2MoE`, `DeepseekV2DecoderLayer`),
//! `models/switch_layers.py` (`SwitchGLU` + `_gather_sort`), `models/mlp.py`
//! (`DeepseekMLP`), `models/cache.py` (`KVCache`), and
//! `models/unlimited_ocr/language.py` (`RingSlidingKVCache`,
//! `DeepseekV2RingSlidingModel`):
//!
//! - 12 pre-RMSNorm layers (eps 1e-6). Attention is plain MHA (10 heads x
//!   128, 4-bit q/k/v/o), NeoX RoPE (base 1e4) at `offset = cache.offset`,
//!   SDPA scale `128^-0.5` with the `"causal"` mask for `N > 1` and no mask
//!   for single-token steps.
//! - Layer 0 is a dense SwiGLU MLP (intermediate 6848); layers 1..11 are
//!   MoE: bf16 gate logits `x @ W_g.T`, `softmax(precise=True)`, greedy
//!   top-6 via `argpartition` + `take_along_axis` (routed scaling 1.0),
//!   `SwitchGLU` over 4-bit experts through `gather_qmm` (sorted path when
//!   `indices.size >= 64`), weighted sum over the selected experts, plus
//!   the 2-expert shared MLP (intermediate 1792).
//! - Final RMSNorm and the untied 4-bit `lm_head`.
//! - Embeddings are `nn.QuantizedEmbedding` (row gather + dequantize).
//!
//! The caches live in [`crate::models::kv_cache`].

use std::path::Path;

use aimd_mlx::{MlxArray, MlxError, SdpaMask};
use aimd_weights::{WeightStore, WeightsError, load_checkpoint};

use crate::models::kv_cache::KvCache;

use super::config::{LanguageConfig, UnlimitedOcrConfig, parse_config};
use super::mlx_weights::{LoadError, QWeights, dense};

/// Errors surfaced by the decoder.
#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
    #[error("layout: {0}")]
    Layout(String),
}

struct Attention {
    q: QWeights,
    k: QWeights,
    v: QWeights,
    o: QWeights,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_base: f64,
}

impl Attention {
    fn forward(&self, x: &MlxArray, cache: &mut KvCache) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let hd = self.head_dim;
        let q = self
            .q
            .linear(x)?
            .reshape(&[b, l, self.heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let k = self
            .k
            .linear(x)?
            .reshape(&[b, l, self.kv_heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let v = self
            .v
            .linear(x)?
            .reshape(&[b, l, self.kv_heads, hd])?
            .transpose(&[0, 2, 1, 3])?;
        let offset = cache.offset() as i32;
        let q = q.rope(hd, self.rope_base, 1.0, offset)?;
        let k = k.rope(hd, self.rope_base, 1.0, offset)?;
        let (keys, values) = cache.update_and_fetch(&k, &v)?;
        let mask = if l > 1 {
            SdpaMask::Causal
        } else {
            SdpaMask::None
        };
        let scale = (hd as f32).powf(-0.5);
        let o = q
            .scaled_dot_product_attention(&keys, &values, scale, mask)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[b, l, self.heads * hd])?;
        self.o.linear(&o)
    }
}

/// `DeepseekMLP`: `down(swiglu(gate(x), up(x)))`.
struct DenseMlp {
    gate: QWeights,
    up: QWeights,
    down: QWeights,
}

impl DenseMlp {
    fn load(store: &WeightStore, p: &str, gs: u32, bits: u32) -> Result<Self, LoadError> {
        Ok(Self {
            gate: QWeights::load(store, &format!("{p}.gate_proj"), gs, bits)?,
            up: QWeights::load(store, &format!("{p}.up_proj"), gs, bits)?,
            down: QWeights::load(store, &format!("{p}.down_proj"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let g = self.gate.linear(x)?;
        let u = self.up.linear(x)?;
        self.down.linear(&g.swiglu(&u)?)
    }
}

struct Moe {
    /// `gate.weight.T` (bf16 `[hidden, n_experts]` view).
    gate_wt: MlxArray,
    n_experts: usize,
    top_k: usize,
    sw_gate: QWeights,
    sw_up: QWeights,
    sw_down: QWeights,
    shared: DenseMlp,
}

impl Moe {
    /// `MoEGate` (softmax, greedy): `(inds [B, L, k] u32, scores [B, L, k])`.
    fn route(&self, x: &MlxArray) -> Result<(MlxArray, MlxArray), MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let gates = x.matmul(&self.gate_wt, None)?;
        let scores = gates.softmax_precise(-1, true)?;
        let kth = (self.n_experts - self.top_k) as i32;
        let inds = scores.argpartition(kth, -1)?.slice(
            &[0, 0, self.n_experts - self.top_k],
            &[b, l, self.n_experts],
            &[1, 1, 1],
        )?;
        let sel = scores.take_along_axis(&inds, -1)?;
        Ok((inds, sel))
    }

    /// `SwitchGLU.__call__(x, indices)` -> `[B, L, k, D]`.
    fn switch_glu(&self, x: &MlxArray, inds: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l, d) = (s[0], s[1], s[2]);
        let k = self.top_k;
        let n = b * l * k;
        let x5 = x.reshape(&[b, l, 1, 1, d])?;
        let act = |xin: &MlxArray, idx: &MlxArray, sorted: bool| -> Result<MlxArray, MlxError> {
            let up = self.sw_up.gather(xin, idx, sorted)?;
            let gate = self.sw_gate.gather(xin, idx, sorted)?;
            self.sw_down.gather(&gate.swiglu(&up)?, idx, sorted)
        };
        let y = if n >= 64 {
            // _gather_sort / _scatter_unsort.
            let flat = inds.reshape(&[n])?;
            let order = flat.argsort(0)?;
            let inv = order.argsort(0)?;
            let rows = order.floor_divide(&MlxArray::scalar_u32(k as u32)?)?;
            let xs = x5.reshape(&[b * l, 1, d])?.take(&rows, 0)?;
            let idx = flat.take(&order, 0)?;
            act(&xs, &idx, true)?
                .take(&inv, 0)?
                .reshape(&[b, l, k, 1, d])?
        } else {
            act(&x5, inds, false)?
        };
        y.reshape(&[b, l, k, d])
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, l) = (s[0], s[1]);
        let (inds, scores) = self.route(x)?;
        let y = self.switch_glu(x, &inds)?;
        let y = y
            .multiply(&scores.reshape(&[b, l, self.top_k, 1])?)?
            .sum_axis(-2, false)?;
        y.add(&self.shared.forward(x)?)
    }
}

enum Mlp {
    Dense(DenseMlp),
    Moe(Box<Moe>),
}

struct Layer {
    input_ln: MlxArray,
    post_ln: MlxArray,
    attn: Attention,
    mlp: Mlp,
}

/// The Unlimited-OCR language model on MLX.
pub struct OcrDecoder {
    cfg: LanguageConfig,
    embed: QWeights,
    layers: Vec<Layer>,
    norm: MlxArray,
    lm_head: QWeights,
    eps: f32,
}

impl std::fmt::Debug for OcrDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrDecoder")
            .field("layers", &self.layers.len())
            .field("hidden", &self.cfg.hidden_size)
            .finish_non_exhaustive()
    }
}

impl OcrDecoder {
    pub fn load(checkpoint_dir: &Path) -> Result<Self, DecoderError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| DecoderError::Io(format!("reading config.json: {e}")))?;
        let cfg = parse_config(&raw).map_err(|e| DecoderError::Config(e.to_string()))?;
        Self::from_store(&store, &cfg)
    }

    pub fn from_store(
        store: &WeightStore,
        full: &UnlimitedOcrConfig,
    ) -> Result<Self, DecoderError> {
        let cfg = full.language_config.clone();
        let q = full
            .quantization
            .as_ref()
            .ok_or_else(|| DecoderError::Config("checkpoint is not quantized".into()))?;
        let (gs, bits) = (q.group_size, q.bits);
        if cfg.topk_method != "greedy" {
            return Err(DecoderError::Config(format!(
                "unsupported topk_method {}",
                cfg.topk_method
            )));
        }
        if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        {
            return Err(DecoderError::Config(
                "hidden_size not divisible by heads".into(),
            ));
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let p = "language_model.model";
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l = format!("{p}.layers.{i}");
            let attn = Attention {
                q: QWeights::load(store, &format!("{l}.self_attn.q_proj"), gs, bits)?,
                k: QWeights::load(store, &format!("{l}.self_attn.k_proj"), gs, bits)?,
                v: QWeights::load(store, &format!("{l}.self_attn.v_proj"), gs, bits)?,
                o: QWeights::load(store, &format!("{l}.self_attn.o_proj"), gs, bits)?,
                heads: cfg.num_attention_heads,
                kv_heads: cfg.num_key_value_heads,
                head_dim,
                rope_base: cfg.rope_theta,
            };
            let mlp = if i >= cfg.first_k_dense_replace && cfg.n_routed_experts > 0 {
                let gate_w = dense(
                    store,
                    &format!("{l}.mlp.gate.weight"),
                    Some(&[cfg.n_routed_experts, cfg.hidden_size]),
                )?;
                Mlp::Moe(Box::new(Moe {
                    gate_wt: gate_w.transpose(&[1, 0])?,
                    n_experts: cfg.n_routed_experts,
                    top_k: cfg.num_experts_per_tok,
                    sw_gate: QWeights::load(
                        store,
                        &format!("{l}.mlp.switch_mlp.gate_proj"),
                        gs,
                        bits,
                    )?,
                    sw_up: QWeights::load(store, &format!("{l}.mlp.switch_mlp.up_proj"), gs, bits)?,
                    sw_down: QWeights::load(
                        store,
                        &format!("{l}.mlp.switch_mlp.down_proj"),
                        gs,
                        bits,
                    )?,
                    shared: DenseMlp::load(store, &format!("{l}.mlp.shared_experts"), gs, bits)?,
                }))
            } else {
                Mlp::Dense(DenseMlp::load(store, &format!("{l}.mlp"), gs, bits)?)
            };
            layers.push(Layer {
                input_ln: dense(
                    store,
                    &format!("{l}.input_layernorm.weight"),
                    Some(&[cfg.hidden_size]),
                )?,
                post_ln: dense(
                    store,
                    &format!("{l}.post_attention_layernorm.weight"),
                    Some(&[cfg.hidden_size]),
                )?,
                attn,
                mlp,
            });
        }
        let embed = QWeights::load(store, &format!("{p}.embed_tokens"), gs, bits)?;
        let lm_head = QWeights::load(store, "language_model.lm_head", gs, bits)?;
        if embed.shape != [cfg.vocab_size, cfg.hidden_size]
            || lm_head.shape != [cfg.vocab_size, cfg.hidden_size]
        {
            return Err(DecoderError::Layout(format!(
                "embed {:?} / lm_head {:?} vs vocab {} x hidden {}",
                embed.shape, lm_head.shape, cfg.vocab_size, cfg.hidden_size
            )));
        }
        Ok(Self {
            eps: cfg.rms_norm_eps,
            norm: dense(store, &format!("{p}.norm.weight"), Some(&[cfg.hidden_size]))?,
            embed,
            layers,
            lm_head,
            cfg,
        })
    }

    pub fn config(&self) -> &LanguageConfig {
        &self.cfg
    }

    /// `make_cache()`: ring caches when the config (or `window`) sets a
    /// sliding window.
    pub(crate) fn make_cache(&self, window: Option<usize>) -> Vec<KvCache> {
        let w = window.or(self.cfg.window());
        (0..self.layers.len()).map(|_| KvCache::new(w)).collect()
    }

    /// `embed_tokens(ids)` for `[1, n]` ids -> `[1, n, hidden]` bf16.
    pub(crate) fn embed_ids(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        self.embed.embed(ids)
    }

    /// Validated host ids -> `[1, n]` u32 array.
    pub(crate) fn ids_array(&self, ids: &[u32]) -> Result<MlxArray, DecoderError> {
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab_size) {
            return Err(DecoderError::Layout(format!(
                "token id {bad} outside vocab {}",
                self.cfg.vocab_size
            )));
        }
        Ok(MlxArray::from_u32(&[1, ids.len()], ids)?)
    }

    /// `DeepseekV2RingSlidingModel.__call__(inputs_embeds=h, cache)`:
    /// layers + final norm -> `[B, L, hidden]`.
    pub(crate) fn forward(
        &self,
        h: &MlxArray,
        caches: &mut [KvCache],
    ) -> Result<MlxArray, MlxError> {
        let mut h = h.try_clone()?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let r = layer
                .attn
                .forward(&h.rms_norm(Some(&layer.input_ln), self.eps)?, cache)?;
            let hh = h.add(&r)?;
            let xn = hh.rms_norm(Some(&layer.post_ln), self.eps)?;
            let r = match &layer.mlp {
                Mlp::Dense(m) => m.forward(&xn)?,
                Mlp::Moe(m) => m.forward(&xn)?,
            };
            h = hh.add(&r)?;
        }
        h.rms_norm(Some(&self.norm), self.eps)
    }

    /// Untied quantized `lm_head`.
    pub(crate) fn logits(&self, hidden: &MlxArray) -> Result<MlxArray, MlxError> {
        self.lm_head.linear(hidden)
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::unlimited_ocr::generate::{
        OCR_IMAGE_TOKEN_ID, OcrGenerationSettings, greedy_ocr_token_stream, splice_image_features,
    };
    use crate::models::unlimited_ocr::test_support::{
        Parity, compare_rows, mlx_rows_f32, ocr_decoder_golden as golden, shared_pipeline,
    };
    use std::collections::BTreeSet;

    /// Logits rows vs the bf16 reference. Prefill is bit-exact; decode
    /// steps drift by bf16 ulps because brew mlx 0.32.4's small-M
    /// `quantized_matmul` kernel rounds differently from the reference's
    /// mlx 0.32.2 (measured: ring logits rel rms 1.3e-2, max 1.1% of max
    /// |ref|; greedy tokens still identical).
    const LOGITS: Parity = Parity {
        rel_rms: 0.03,
        max_frac: 0.05,
    };
    /// Layer-0 key rows (one RMSNorm + q-proj + RoPE from the embeddings).
    const KEYS: Parity = Parity {
        rel_rms: 0.01,
        max_frac: 0.01,
    };

    fn argmax(row: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = i;
            }
        }
        best as u32
    }

    struct Golden {
        m: serde_json::Value,
        prompt: Vec<u32>,
        embeds: MlxArray,
    }

    fn golden_inputs(dec: &OcrDecoder) -> Golden {
        let m = golden::manifest();
        let prompt = golden::u32s(&m["prompt_ids"]);
        let feats = golden::image_features(&m);
        let embeds = splice_image_features(dec, &prompt, &feats, OCR_IMAGE_TOKEN_ID).unwrap();
        Golden { m, prompt, embeds }
    }

    /// Full-prompt prefill (unchunked, like the exporter) over ring caches.
    fn prefill(dec: &OcrDecoder, g: &Golden, window: usize) -> (Vec<KvCache>, Vec<f32>, usize) {
        let mut caches = dec.make_cache(Some(window));
        let h = dec.forward(&g.embeds, &mut caches).unwrap();
        let (logits, vocab) = mlx_rows_f32(&dec.logits(&h).unwrap());
        (caches, logits, vocab)
    }

    fn step(dec: &OcrDecoder, caches: &mut [KvCache], tok: u32) -> Vec<f32> {
        let emb = dec.embed_ids(&dec.ids_array(&[tok]).unwrap()).unwrap();
        let h = dec.forward(&emb, caches).unwrap();
        mlx_rows_f32(&dec.logits(&h).unwrap()).0
    }

    /// Prefill logits rows + the window-128 forced greedy stream (16
    /// tokens, EOS at step 15) against mlx-vlm.
    #[test]
    fn decoder_prefill_and_forced_stream_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let (mut caches, logits, vocab) = prefill(dec, &g, 128);
        let kept = golden::usizes(&g.m["logits_kept_positions"]);
        let want = golden::read_bf16_f32("logits.bf16");
        let mut fails = Vec::new();
        compare_rows(
            "prefill logits",
            &logits,
            vocab,
            &kept,
            &want,
            LOGITS,
            &mut fails,
        );
        assert!(fails.is_empty(), "{}", fails.join("\n"));

        let want_tokens = golden::u32s(&g.m["greedy_tokens"]);
        let p_len = g.prompt.len();
        let mut got = vec![argmax(&logits[(p_len - 1) * vocab..p_len * vocab])];
        for &tok in &want_tokens[..want_tokens.len() - 1] {
            got.push(argmax(&step(dec, &mut caches, tok)));
        }
        assert_eq!(got, want_tokens, "forced greedy stream");
    }

    /// Window-4 ring run: forced stream, logits rows of the first ring
    /// writes, and layer-0 key rows after decode step 4.
    #[test]
    fn decoder_ring_window_golden() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let ring = &g.m["ring"];
        let window = ring["window"].as_u64().unwrap() as usize;
        let forced = golden::u32s(&ring["forced_tokens"]);
        let kept = golden::usizes(&ring["kept_steps"]);
        let dump = &ring["cache_dump"];
        let (after, from, to) = (
            dump["after_step"].as_u64().unwrap() as usize,
            dump["from_row"].as_u64().unwrap() as usize,
            dump["to_row"].as_u64().unwrap() as usize,
        );
        let (mut caches, logits, vocab) = prefill(dec, &g, window);
        let p_len = g.prompt.len();
        let mut got = vec![argmax(&logits[(p_len - 1) * vocab..p_len * vocab])];
        let mut rows = Vec::new();
        let mut fails = Vec::new();
        for (k, &tok) in forced.iter().enumerate().take(forced.len() - 1) {
            let row = step(dec, &mut caches, tok);
            got.push(argmax(&row));
            if kept.contains(&k) {
                rows.extend(row);
            }
            if k == after {
                assert_eq!(caches[0].prefill_length(), Some(p_len));
                let keys = caches[0].keys_buffer().unwrap();
                let s = keys.shape().unwrap();
                let sl = keys
                    .slice(&[0, 0, from, 0], &[1, s[1], to, s[3]], &[1, 1, 1, 1])
                    .unwrap()
                    .transpose(&[0, 2, 1, 3])
                    .unwrap();
                let (k_rows, _) = mlx_rows_f32(&sl);
                let cols = s[1] * s[3];
                let want = golden::read_f32("ring_cache_keys.f32");
                let idx: Vec<usize> = (0..to - from).collect();
                compare_rows(
                    "ring layer-0 keys",
                    &k_rows,
                    cols,
                    &idx,
                    &want,
                    KEYS,
                    &mut fails,
                );
            }
        }
        let idx: Vec<usize> = (0..kept.len()).collect();
        let want = golden::read_bf16_f32("ring_logits.bf16");
        compare_rows("ring logits", &rows, vocab, &idx, &want, LOGITS, &mut fails);
        assert!(fails.is_empty(), "{}", fails.join("\n"));
        assert_eq!(got, forced, "ring forced stream");
    }

    /// The production loop (chunked prefill, guard, pipelined decode)
    /// stops on EOS at the golden step and drops it.
    #[test]
    fn decoder_token_stream_stops_on_eos() {
        let Some(p) = shared_pipeline() else {
            return;
        };
        let dec = p.decoder();
        let g = golden_inputs(dec);
        let want = golden::u32s(&g.m["greedy_tokens"]);
        let eos_step = g.m["first_eos_step"].as_u64().unwrap() as usize;
        let eos: BTreeSet<u32> = golden::u32s(&g.m["eos_ids"]).into_iter().collect();
        let settings = OcrGenerationSettings {
            max_new_tokens: 64,
            ..OcrGenerationSettings::default()
        };
        let (stream, _) =
            greedy_ocr_token_stream(dec, &g.prompt, &g.embeds, &eos, settings).unwrap();
        assert_eq!(
            stream.finish_reason,
            crate::models::unlimited_ocr::generate::FinishReason::Stop
        );
        assert_eq!(stream.tokens, want[..eos_step]);
        // max_new_tokens caps sampled tokens (Length).
        let settings = OcrGenerationSettings {
            max_new_tokens: 3,
            ..settings
        };
        let (stream, _) =
            greedy_ocr_token_stream(dec, &g.prompt, &g.embeds, &eos, settings).unwrap();
        assert_eq!(
            stream.finish_reason,
            crate::models::unlimited_ocr::generate::FinishReason::Length
        );
        assert_eq!(stream.tokens, want[..3]);
    }
}
