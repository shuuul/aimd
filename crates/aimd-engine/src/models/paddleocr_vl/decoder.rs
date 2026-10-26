//! ERNIE-4.5-0.3B decoder of PaddleOCR-VL (mlx-vlm
//! `paddleocr_vl/language.py`): 18 pre-RMSNorm blocks, GQA 16/2 heads of
//! 128, sectioned 3-D mRoPE, SwiGLU MLP, untied `lm_head`.
//!
//! mRoPE follows the Metal path of mlx-vlm (`rope_utils`
//! `_maybe_fast_precomputed_rotary`, style `sectioned_half_split`):
//! `freqs = pos.astype(f32)[..., None] * inv_freq` per axis (`inv_freq =
//! 1 / theta^(arange(0, 128, 2) / 128)`), `cos`/`sin` cast to bf16, the
//! frequency index `f` taking its angle from axis `t` (`f < 16`), `h`
//! (`16 <= f < 40`) or `w` (`f >= 40`) for both rotated halves, and the
//! rotation `x * cos + rotate_half(x) * sin` evaluated in f32 then cast to
//! bf16. (mlx-vlm's non-Metal fallback computes it in bf16 instead; this
//! port always follows the Metal kernel.)
//!
//! The rotation runs as one fused kernel ([`MlxArray::rope_half_split`],
//! value-identical to the unfused graph), and single-token steps slice
//! their tables from one [`PaddleDecoder::rope_sequential`] table built per
//! generation (positions are equal on all three axes there).
//!
//! Prefill chunks use mlx-vlm's boolean causal mask array
//! (`create_causal_mask(N, offset)`: key `j` visible to query `i` iff
//! `j <= offset + i`); single-token steps use no mask.

use aimd_mlx::{BinaryOp, MlxArray, MlxDtype, MlxError, SdpaMask, UnaryOp};

use super::config::{PaddleOcrVlConfig, TextConfig};
use super::weights::{Emb, Lin, Loader};
use crate::models::kv_cache::KvCache;
use crate::models::unlimited_ocr::mlx_weights::LoadError;

/// Decoder failures.
#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error("decoder layout: {0}")]
    Layout(String),
}

struct DecoderLayer {
    in_norm: MlxArray,
    post_norm: MlxArray,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// Per-forward rotary tables: f32 `[1, 1, L, head_dim]`, `cos = [c, c]`
/// and `sin_signed = [-s, s]` (the `rotate_half` sign folded in).
pub struct Rope {
    cos: MlxArray,
    sin_signed: MlxArray,
}

impl Rope {
    /// Rows `i..i + 1` (one decode step) as views of a sequential table.
    pub fn row(&self, i: usize) -> Result<Self, MlxError> {
        let s = self.cos.shape()?;
        let row = |t: &MlxArray| t.slice(&[0, 0, i, 0], &[1, 1, i + 1, s[3]], &[1, 1, 1, 1]);
        Ok(Self {
            cos: row(&self.cos)?,
            sin_signed: row(&self.sin_signed)?,
        })
    }
}

/// ERNIE decoder + `lm_head`.
pub struct PaddleDecoder {
    cfg: TextConfig,
    mrope_section: [usize; 3],
    inv_freq: MlxArray,
    embed: Emb,
    layers: Vec<DecoderLayer>,
    norm: MlxArray,
    lm_head: Lin,
    /// Single-token steps use [`MlxArray::sdpa_decode`] (custom Metal
    /// kernel, bit-identical to mlx sdpa): GPU device only.
    sdpa_decode: bool,
}

impl PaddleDecoder {
    pub(crate) fn load(ld: &Loader<'_>, cfg: &PaddleOcrVlConfig) -> Result<Self, DecoderError> {
        let t = &cfg.text;
        let (d, hd) = (t.hidden_size, t.head_dim);
        let (nq, nkv, ff) = (
            t.num_attention_heads,
            t.num_key_value_heads,
            t.intermediate_size,
        );
        if t.use_bias || t.tie_word_embeddings {
            return Err(DecoderError::Layout(
                "biased projections / tied embeddings are not part of PaddleOCR-VL-1.6".into(),
            ));
        }
        let sec = cfg.mrope_section();
        if sec.len() != 3 || sec.iter().sum::<usize>() * 2 != hd {
            return Err(DecoderError::Layout(format!(
                "mrope_section {sec:?} does not cover head_dim {hd}"
            )));
        }
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for l in 0..t.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(DecoderLayer {
                in_norm: ld.tensor(&format!("{p}.input_layernorm.weight"), Some(&[d]))?,
                post_norm: ld
                    .tensor(&format!("{p}.post_attention_layernorm.weight"), Some(&[d]))?,
                q: ld.linear(&format!("{p}.self_attn.q_proj"), false, &[nq * hd, d])?,
                k: ld.linear(&format!("{p}.self_attn.k_proj"), false, &[nkv * hd, d])?,
                v: ld.linear(&format!("{p}.self_attn.v_proj"), false, &[nkv * hd, d])?,
                o: ld.linear(&format!("{p}.self_attn.o_proj"), false, &[d, nq * hd])?,
                gate: ld.linear(&format!("{p}.mlp.gate_proj"), false, &[ff, d])?,
                up: ld.linear(&format!("{p}.mlp.up_proj"), false, &[ff, d])?,
                down: ld.linear(&format!("{p}.mlp.down_proj"), false, &[d, ff])?,
            });
        }
        // compute_inv_freq(head_dim, rope_theta), f32 on device.
        let half = hd / 2;
        let steps: Vec<f32> = (0..half).map(|i| (2 * i) as f32).collect();
        let inv_freq = MlxArray::scalar_f32(1.0)?.divide(
            &MlxArray::scalar_f32(t.rope_theta as f32)?.power(
                &MlxArray::from_f32(&[half], &steps)?.divide(&MlxArray::scalar_f32(hd as f32)?)?,
            )?,
        )?;
        inv_freq.eval()?;
        Ok(Self {
            cfg: t.clone(),
            mrope_section: [sec[0], sec[1], sec[2]],
            inv_freq,
            embed: ld.embedding("model.embed_tokens", &[t.vocab_size, d])?,
            layers,
            norm: ld.tensor("model.norm.weight", Some(&[d]))?,
            lm_head: ld.linear("lm_head", false, &[t.vocab_size, d])?,
            sdpa_decode: aimd_mlx::metal_kernels_available().unwrap_or(false),
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    #[cfg(test)]
    pub(crate) fn embedding(&self) -> &Emb {
        &self.embed
    }

    #[cfg(test)]
    pub(crate) fn lm_head(&self) -> &Lin {
        &self.lm_head
    }

    #[cfg(test)]
    pub(crate) fn layer_linear(&self, layer: usize, name: &str) -> &Lin {
        let l = &self.layers[layer];
        match name {
            "q_proj" => &l.q,
            "k_proj" => &l.k,
            "v_proj" => &l.v,
            "o_proj" => &l.o,
            "gate_proj" => &l.gate,
            "up_proj" => &l.up,
            "down_proj" => &l.down,
            other => panic!("unknown decoder linear {other}"),
        }
    }

    /// Fresh per-layer `KVCache`s.
    pub(crate) fn make_cache(&self) -> Vec<KvCache> {
        (0..self.layers.len()).map(|_| KvCache::new(None)).collect()
    }

    /// `embed_tokens(ids)` for `[1, L]` u32 ids -> `[1, L, hidden]`.
    pub fn embed_ids(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        self.embed.embed(ids)
    }

    /// Rotary tables for explicit `[3][L]` (t, h, w) positions.
    pub fn rope(&self, positions: [&[i64]; 3]) -> Result<Rope, DecoderError> {
        let l = positions[0].len();
        if positions.iter().any(|p| p.len() != l) || l == 0 {
            return Err(DecoderError::Layout(
                "ragged / empty mRoPE positions".into(),
            ));
        }
        let half = self.cfg.head_dim / 2;
        let flat: Vec<f32> = positions
            .iter()
            .flat_map(|p| p.iter().map(|&v| v as f32))
            .collect();
        let freqs = MlxArray::from_f32(&[3, l, 1], &flat)?.multiply(&self.inv_freq)?;
        let section = |a: MlxArray, negate_first: bool| -> Result<MlxArray, MlxError> {
            // `cos(emb) * 1.0` cast to bf16, then the chunked axis selector.
            let a = a.astype(MlxDtype::Bf16)?;
            let [s0, s1, _] = self.mrope_section;
            let bounds = [(0, 0, s0), (1, s0, s0 + s1), (2, s0 + s1, half)];
            let parts = bounds
                .iter()
                .map(|&(axis, lo, hi)| a.slice(&[axis, 0, lo], &[axis + 1, l, hi], &[1, 1, 1]))
                .collect::<Result<Vec<_>, _>>()?;
            let sel = MlxArray::concatenate(&parts.iter().collect::<Vec<_>>(), 2)?;
            let first = if negate_first {
                sel.unary(UnaryOp::Negative)?
            } else {
                sel.try_clone()?
            };
            MlxArray::concatenate(&[&first, &sel], 2)?
                .astype(MlxDtype::F32)?
                .reshape(&[1, 1, l, 2 * half])
        };
        Ok(Rope {
            cos: section(freqs.unary(UnaryOp::Cos)?, false)?,
            sin_signed: section(freqs.unary(UnaryOp::Sin)?, true)?,
        })
    }

    /// Text positions `start..start + len` on all three axes.
    pub fn rope_sequential(&self, start: i64, len: usize) -> Result<Rope, DecoderError> {
        let p: Vec<i64> = (0..len as i64).map(|i| start + i).collect();
        self.rope([&p, &p, &p])
    }

    /// `x * cos + rotate_half(x) * sin` in f32, cast to bf16 (`x` is bf16).
    fn apply_rope(&self, x: &MlxArray, rope: &Rope) -> Result<MlxArray, MlxError> {
        x.rope_half_split(&rope.cos, &rope.sin_signed)
    }

    /// Boolean causal mask `[L, offset + L]` (`create_causal_mask`).
    fn causal_mask(offset: usize, l: usize) -> Result<MlxArray, MlxError> {
        let kv = offset + l;
        let cols: Vec<u32> = (0..kv as u32).collect();
        let rows: Vec<u32> = (0..l as u32).map(|i| offset as u32 + i + 1).collect();
        MlxArray::from_u32(&[1, kv], &cols)?
            .binary(&MlxArray::from_u32(&[l, 1], &rows)?, BinaryOp::Less)
    }

    /// Runs the blocks over `h` (`[1, L, hidden]`) with the given rotary
    /// tables, appending to `caches`; returns the final-norm hidden state.
    pub(crate) fn forward(
        &self,
        h: &MlxArray,
        rope: &Rope,
        caches: &mut [KvCache],
    ) -> Result<MlxArray, DecoderError> {
        let s = h.shape()?;
        if s.len() != 3 || s[0] != 1 || s[2] != self.cfg.hidden_size {
            return Err(DecoderError::Layout(format!("hidden {s:?}")));
        }
        if caches.len() != self.layers.len() {
            return Err(DecoderError::Layout("cache count != layers".into()));
        }
        let l = s[1];
        let mask = if l > 1 {
            Some(Self::causal_mask(caches[0].offset(), l)?)
        } else {
            None
        };
        let (nq, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let eps = self.cfg.rms_norm_eps;
        let scale = (hd as f64).powf(-0.5) as f32;
        let mut x = h.try_clone()?;
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let xn = x.rms_norm(Some(&layer.in_norm), eps)?;
            let q = layer
                .q
                .forward(&xn)?
                .reshape(&[1, l, nq, hd])?
                .transpose(&[0, 2, 1, 3])?;
            let k = layer
                .k
                .forward(&xn)?
                .reshape(&[1, l, nkv, hd])?
                .transpose(&[0, 2, 1, 3])?;
            let v = layer
                .v
                .forward(&xn)?
                .reshape(&[1, l, nkv, hd])?
                .transpose(&[0, 2, 1, 3])?;
            let q = self.apply_rope(&q, rope)?;
            let k = self.apply_rope(&k, rope)?;
            let (k, v) = cache.update_and_fetch(&k, &v)?;
            let o = match &mask {
                None if self.sdpa_decode => q.sdpa_decode(&k, &v, scale)?,
                Some(m) => {
                    let kv_len = k.dim(2)?;
                    let m = m.slice(&[0, 0], &[l, kv_len], &[1, 1])?;
                    q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::Array(&m))?
                }
                None => q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?,
            };
            let o = o.transpose(&[0, 2, 1, 3])?.reshape(&[1, l, nq * hd])?;
            let hh = x.add(&layer.o.forward(&o)?)?;
            let hn = hh.rms_norm(Some(&layer.post_norm), eps)?;
            let mlp = layer
                .down
                .forward(&layer.gate.forward(&hn)?.swiglu(&layer.up.forward(&hn)?)?)?;
            x = hh.add(&mlp)?;
        }
        Ok(x.rms_norm(Some(&self.norm), eps)?)
    }

    /// `lm_head(hidden)`.
    pub fn logits(&self, h: &MlxArray) -> Result<MlxArray, MlxError> {
        self.lm_head.forward(h)
    }
}
