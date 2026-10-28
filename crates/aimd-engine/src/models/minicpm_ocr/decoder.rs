//! Qwen3.5-0.8B hybrid decoder of MiniCPM-OCR (transformers
//! `qwen3_5/modeling_qwen3_5.py`): 24 pre-RMSNorm blocks, every 4th
//! (3, 7, ..., 23) full attention, the rest Gated DeltaNet, each followed by
//! a SwiGLU MLP; tied `lm_head`. bf16 elementwise ops follow torch's
//! rounding (f32 opmath, one rounding per op).
//!
//! - RMSNorms (`input`/`post_attention`/final/`q_norm`/`k_norm`) compute
//!   `norm(x) * (1 + w)` in f32 and round once; the gated DeltaNet norm
//!   keeps its raw weight.
//! - Full attention: `q_proj` -> `[8, 512]` split into query and output
//!   gate halves, `q_norm` / `k_norm`, partial RoPE over the first 64 of
//!   256 dims (theta 1e7, positions plain 1-D: all mRoPE axes are equal for
//!   MiniCPM; f32 `cos`/`sin` cast to bf16, bf16 rotation), GQA 8/2 SDPA
//!   (scale `256^-0.5`, causal prefill), then `o_proj(out * sigmoid(gate))`.
//! - Gated DeltaNet: `in_proj_qkv` through a depthwise causal conv (kernel
//!   4, 3-row conv state; single-token steps use an f32 multiply-sum),
//!   `silu`, q/k `l2norm` and the query's `128^-0.5` scale in f32, the
//!   gate `g` in f32, the gated delta recurrence on an f32
//!   `[B, 16, 128, 128]` state ([`MlxArray::gated_delta`]),
//!   `rms_norm(y, w) * silu(z)` in f32, `out_proj`.
//!
//! Quantization (`Some(4 | 6 | 8)`) touches only the full-attention
//! projections and the MLPs (mlx-vlm's convert output for this checkpoint);
//! embeddings, `lm_head` and every DeltaNet projection stay dense.

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask, UnaryOp};

use super::config::{MiniCpmOcrConfig, TextConfig};
use crate::models::kv_cache::KvCache;
use crate::models::paddleocr_vl::weights::{Emb, Lin, Loader};
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

struct FullAttention {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: MlxArray,
    k_norm: MlxArray,
}

struct DeltaNet {
    in_qkv: Lin,
    in_z: Lin,
    in_b: Lin,
    in_a: Lin,
    /// Depthwise conv weight `[C, K, 1]` (prefill).
    conv_w: MlxArray,
    /// The same weight as f32 `[K, C]` (single-token steps).
    conv_w_step: MlxArray,
    a_log: MlxArray,
    dt_bias: MlxArray,
    norm: MlxArray,
    out: Lin,
}

enum Mixer {
    Full(FullAttention),
    Linear(DeltaNet),
}

struct DecoderLayer {
    in_norm: MlxArray,
    post_norm: MlxArray,
    mixer: Mixer,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// Per-layer generation state.
pub(crate) enum LayerCache {
    Kv(KvCache),
    Delta {
        /// Last `K - 1` conv inputs `[B, K - 1, C]`.
        conv: Option<MlxArray>,
        /// Recurrent state `[B, Hv, Dv, Dk]` f32.
        state: Option<MlxArray>,
    },
}

impl LayerCache {
    /// Keeps batch rows `rows` (u32 indices).
    pub(crate) fn select_rows(&mut self, rows: &MlxArray) -> Result<(), MlxError> {
        match self {
            LayerCache::Kv(c) => c.select_rows(rows),
            LayerCache::Delta { conv, state } => {
                for a in [conv, state] {
                    if let Some(x) = a.take() {
                        *a = Some(x.take(rows, 0)?);
                    }
                }
                Ok(())
            }
        }
    }

    /// Arrays to evaluate after a prefill chunk.
    pub(crate) fn state(&self) -> Result<Vec<MlxArray>, MlxError> {
        match self {
            LayerCache::Kv(c) => c.state(),
            LayerCache::Delta { conv, state } => conv
                .iter()
                .chain(state.iter())
                .map(MlxArray::try_clone)
                .collect(),
        }
    }
}

/// Partial-rotary tables, bf16 `[B, 1, L, rotary_dim]` (`B = 1` for a
/// shared range): `cos = [c, c]`, `sin_signed = [-s, s]` (transformers'
/// `cos.to(x.dtype)`).
pub struct Rope {
    cos: MlxArray,
    sin_signed: MlxArray,
}

impl Rope {
    /// Position `i` (one decode step) of a sequential range.
    pub fn row(&self, i: usize) -> Result<Self, MlxError> {
        let row = |t: &MlxArray| {
            let s = t.shape()?;
            t.slice(&[0, 0, i, 0], &[s[0], 1, i + 1, s[3]], &[1, 1, 1, 1])
        };
        Ok(Self {
            cos: row(&self.cos)?,
            sin_signed: row(&self.sin_signed)?,
        })
    }

    /// Keeps batch rows `rows` (u32 indices) of per-row tables.
    pub(crate) fn select_rows(&self, rows: &MlxArray) -> Result<Self, MlxError> {
        Ok(Self {
            cos: self.cos.take(rows, 0)?,
            sin_signed: self.sin_signed.take(rows, 0)?,
        })
    }
}

/// Additive bf16 mask `[B, 1, 1, L + extra]` from per-row column validity
/// (`valid[b]` holds `L` flags); the `extra` trailing columns are valid.
pub(crate) fn validity_mask(valid: &[Vec<bool>], extra: usize) -> Result<MlxArray, MlxError> {
    let l = valid.first().map_or(0, Vec::len);
    let mut bits = Vec::with_capacity(valid.len() * (l + extra));
    for row in valid {
        debug_assert_eq!(row.len(), l);
        bits.extend(row.iter().map(|&v| !v));
        bits.extend(std::iter::repeat_n(false, extra));
    }
    MlxArray::from_bool(&[valid.len(), 1, 1, l + extra], &bits)?.where_(
        &MlxArray::scalar_bf16(f32::NEG_INFINITY)?,
        &MlxArray::scalar_bf16(0.0)?,
    )
}

/// Captured hidden states (`stage`, `[1, L, hidden]`) for parity tests.
pub type Taps = Vec<(String, MlxArray)>;

/// The inputs of one DeltaNet layer's recurrence over a speculative
/// verify block, enough to rebuild its caches for any accepted prefix.
pub(crate) struct DeltaRecord {
    layer: usize,
    /// Conv window `[B, K - 1 + L, C]` (previous state + the block).
    conv_in: MlxArray,
    /// Recurrent state before the block.
    state: MlxArray,
    q: MlxArray,
    k: MlxArray,
    v: MlxArray,
    g: MlxArray,
    beta: MlxArray,
}

/// What a forward pass records besides the caches: the outputs of
/// `layers` (DSpark target features) and, with `delta`, the DeltaNet
/// recurrence inputs for [`MiniCpmDecoder::rollback`].
pub(crate) struct Capture<'a> {
    pub(crate) layers: &'a [usize],
    pub(crate) hidden: Vec<MlxArray>,
    pub(crate) delta: Option<Vec<DeltaRecord>>,
}

impl<'a> Capture<'a> {
    pub(crate) fn new(layers: &'a [usize], record_delta: bool) -> Self {
        Self {
            layers,
            hidden: Vec::with_capacity(layers.len()),
            delta: record_delta.then(Vec::new),
        }
    }
}

/// Qwen3.5 decoder + tied `lm_head`.
pub struct MiniCpmDecoder {
    cfg: TextConfig,
    inv_freq: MlxArray,
    embed: Emb,
    lm_head: Lin,
    layers: Vec<DecoderLayer>,
    norm: MlxArray,
    sdpa_decode: bool,
}

/// `w + 1` in f32: mlx-vlm's sanitize shift of the Qwen3.5 RMSNorm
/// weights. The shifted weights stay f32; `nn.RMSNorm` returns the
/// input dtype, so every use casts the norm back to bf16.
fn shifted_norm(ld: &Loader<'_>, name: &str, dim: usize) -> Result<MlxArray, LoadError> {
    let w = ld.tensor(name, Some(&[dim]))?;
    let w = w.astype(MlxDtype::F32)?.add(&MlxArray::scalar_f32(1.0)?)?;
    w.eval()?;
    Ok(w)
}

/// `F.silu` on a bf16 tensor: f32 opmath, one rounding.
pub(crate) fn silu_bf16(x: &MlxArray) -> Result<MlxArray, MlxError> {
    x.astype(MlxDtype::F32)?.silu()?.astype(MlxDtype::Bf16)
}

/// `down(act(gate(x)) * up(x))` with torch's roundings: the SiLU once,
/// then the bf16 product.
pub(crate) fn swiglu_bf16(gate: &MlxArray, up: &MlxArray) -> Result<MlxArray, MlxError> {
    silu_bf16(gate)?.multiply(up)
}

impl MiniCpmDecoder {
    /// `dense` reads embeddings and DeltaNet modules; `quant` (the same
    /// store, with the requested bits) reads attention and MLP projections.
    pub(crate) fn load(
        dense: &Loader<'_>,
        quant: &Loader<'_>,
        cfg: &MiniCpmOcrConfig,
    ) -> Result<Self, DecoderError> {
        let t = &cfg.text_config;
        let (d, hd, ff) = (t.hidden_size, t.head_dim, t.intermediate_size);
        let (nq, nkv) = (t.num_attention_heads, t.num_key_value_heads);
        let (kd, vd) = (
            t.linear_num_key_heads * t.linear_key_head_dim,
            t.linear_num_value_heads * t.linear_value_head_dim,
        );
        let (hv, kk) = (t.linear_num_value_heads, t.linear_conv_kernel_dim);
        let conv_dim = 2 * kd + vd;
        if !t.attn_output_gate || t.attention_bias || !cfg.tie_word_embeddings {
            return Err(DecoderError::Layout(
                "expected gated, bias-free attention and tied embeddings".into(),
            ));
        }
        if t.linear_num_key_heads != hv || !t.rotary_dim().is_multiple_of(2) || t.rotary_dim() > hd
        {
            return Err(DecoderError::Layout(format!(
                "unsupported DeltaNet heads / rotary dim ({} k, {hv} v, rotary {})",
                t.linear_num_key_heads,
                t.rotary_dim()
            )));
        }
        let root = "model.language_model";
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for l in 0..t.num_hidden_layers {
            let p = format!("{root}.layers.{l}");
            let mixer = if t.is_linear(l) {
                let a = format!("{p}.linear_attn");
                let conv = dense.tensor(&format!("{a}.conv1d.weight"), Some(&[conv_dim, 1, kk]))?;
                let conv_w = conv.transpose(&[0, 2, 1])?;
                let conv_w_step = conv
                    .reshape(&[conv_dim, kk])?
                    .transpose(&[1, 0])?
                    .astype(MlxDtype::F32)?;
                conv_w_step.eval()?;
                Mixer::Linear(DeltaNet {
                    in_qkv: dense.linear(&format!("{a}.in_proj_qkv"), false, &[conv_dim, d])?,
                    in_z: dense.linear(&format!("{a}.in_proj_z"), false, &[vd, d])?,
                    in_b: dense.linear(&format!("{a}.in_proj_b"), false, &[hv, d])?,
                    in_a: dense.linear(&format!("{a}.in_proj_a"), false, &[hv, d])?,
                    conv_w,
                    conv_w_step,
                    a_log: dense.tensor(&format!("{a}.A_log"), Some(&[hv]))?,
                    dt_bias: dense.tensor(&format!("{a}.dt_bias"), Some(&[hv]))?,
                    norm: dense.tensor(
                        &format!("{a}.norm.weight"),
                        Some(&[t.linear_value_head_dim]),
                    )?,
                    out: dense.linear(&format!("{a}.out_proj"), false, &[d, vd])?,
                })
            } else {
                let a = format!("{p}.self_attn");
                Mixer::Full(FullAttention {
                    q: quant.linear(&format!("{a}.q_proj"), false, &[2 * nq * hd, d])?,
                    k: quant.linear(&format!("{a}.k_proj"), false, &[nkv * hd, d])?,
                    v: quant.linear(&format!("{a}.v_proj"), false, &[nkv * hd, d])?,
                    o: quant.linear(&format!("{a}.o_proj"), false, &[d, nq * hd])?,
                    q_norm: shifted_norm(dense, &format!("{a}.q_norm.weight"), hd)?,
                    k_norm: shifted_norm(dense, &format!("{a}.k_norm.weight"), hd)?,
                })
            };
            layers.push(DecoderLayer {
                in_norm: shifted_norm(dense, &format!("{p}.input_layernorm.weight"), d)?,
                post_norm: shifted_norm(dense, &format!("{p}.post_attention_layernorm.weight"), d)?,
                mixer,
                gate: quant.linear(&format!("{p}.mlp.gate_proj"), false, &[ff, d])?,
                up: quant.linear(&format!("{p}.mlp.up_proj"), false, &[ff, d])?,
                down: quant.linear(&format!("{p}.mlp.down_proj"), false, &[d, ff])?,
            });
        }
        let embed = dense.tensor(
            &format!("{root}.embed_tokens.weight"),
            Some(&[t.vocab_size, d]),
        )?;
        let lm_head = Lin::Dense {
            wt: embed.transpose(&[1, 0])?,
            bias: None,
        };
        // compute_inv_freq(rotary_dim, theta): 1 / theta^(arange(0, r, 2) / r).
        let r = t.rotary_dim();
        let steps: Vec<f32> = (0..r / 2).map(|i| (2 * i) as f32).collect();
        let inv_freq = MlxArray::scalar_f32(1.0)?.divide(
            &MlxArray::scalar_f32(t.rope_parameters.rope_theta as f32)?.power(
                &MlxArray::from_f32(&[r / 2], &steps)?.divide(&MlxArray::scalar_f32(r as f32)?)?,
            )?,
        )?;
        inv_freq.eval()?;
        Ok(Self {
            cfg: t.clone(),
            inv_freq,
            embed: Emb::Dense(embed),
            lm_head,
            layers,
            norm: shifted_norm(dense, &format!("{root}.norm.weight"), d)?,
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
    pub(crate) fn layer_linear(&self, layer: usize, name: &str) -> &Lin {
        let l = &self.layers[layer];
        match (name, &l.mixer) {
            ("gate_proj", _) => &l.gate,
            ("up_proj", _) => &l.up,
            ("down_proj", _) => &l.down,
            ("q_proj", Mixer::Full(a)) => &a.q,
            ("k_proj", Mixer::Full(a)) => &a.k,
            ("v_proj", Mixer::Full(a)) => &a.v,
            ("o_proj", Mixer::Full(a)) => &a.o,
            ("in_proj_qkv", Mixer::Linear(a)) => &a.in_qkv,
            ("out_proj", Mixer::Linear(a)) => &a.out,
            (other, _) => panic!("no decoder linear {other} in layer {layer}"),
        }
    }

    /// Fresh per-layer caches.
    pub(crate) fn make_cache(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Full(_) => LayerCache::Kv(KvCache::new(None)),
                Mixer::Linear(_) => LayerCache::Delta {
                    conv: None,
                    state: None,
                },
            })
            .collect()
    }

    /// `embed_tokens(ids)` for `[1, L]` u32 ids -> `[1, L, hidden]`.
    pub fn embed_ids(&self, ids: &MlxArray) -> Result<MlxArray, MlxError> {
        self.embed.embed(ids)
    }

    /// Rotary positions `start..start + len`.
    pub fn rope(&self, start: usize, len: usize) -> Result<Rope, DecoderError> {
        self.rope_rows(&[start as i64], len)
    }

    /// Per-row rotary positions `starts[b]..starts[b] + len` (`[B, 1, len,
    /// r]`); negative positions belong to left padding.
    pub(crate) fn rope_rows(&self, starts: &[i64], len: usize) -> Result<Rope, DecoderError> {
        if len == 0 || starts.is_empty() {
            return Err(DecoderError::Layout("empty rope range".into()));
        }
        // freqs = positions.float() * inv_freq (f32), accurate cos/sin,
        // then cast to bf16 like transformers' rotary embedding.
        let r = self.cfg.rotary_dim();
        let b = starts.len();
        let pos: Vec<f32> = starts
            .iter()
            .flat_map(|&s| (s..s + len as i64).map(|p| p as f32))
            .collect();
        let freqs = MlxArray::from_f32(&[b * len, 1], &pos)?
            .multiply(&self.inv_freq.reshape(&[1, r / 2])?)?;
        let table = |half: MlxArray, signed: bool| -> Result<MlxArray, MlxError> {
            let half = half.astype(MlxDtype::Bf16)?;
            let first = if signed {
                half.unary(UnaryOp::Negative)?
            } else {
                half.try_clone()?
            };
            MlxArray::concatenate(&[&first, &half], 1)?.reshape(&[b, 1, len, r])
        };
        Ok(Rope {
            cos: table(freqs.unary(UnaryOp::Cos)?, false)?,
            sin_signed: table(freqs.unary(UnaryOp::Sin)?, true)?,
        })
    }

    /// Rotates the first `rotary_dim` dims of `x` (`[B, H, L, head_dim]`):
    /// `x * cos + rotate_half(x) * sin` with every product and the sum
    /// rounded to bf16, as transformers' `apply_rotary_pos_emb`.
    fn apply_rope(&self, x: &MlxArray, rope: &Rope) -> Result<MlxArray, MlxError> {
        let s = x.shape()?;
        let (b, h, l, d) = (s[0], s[1], s[2], s[3]);
        let r = self.cfg.rotary_dim();
        let rot = x.slice(&[0, 0, 0, 0], &[b, h, l, r], &[1, 1, 1, 1])?;
        let pass = x.slice(&[0, 0, 0, r], &[b, h, l, d], &[1, 1, 1, 1])?;
        let rot = rot.rope_half_split(&rope.cos, &rope.sin_signed)?;
        MlxArray::concatenate(&[&rot, &pass], 3)
    }

    fn full_attention(
        &self,
        a: &FullAttention,
        x: &MlxArray,
        rope: &Rope,
        mask: Option<&MlxArray>,
        cache: &mut KvCache,
    ) -> Result<MlxArray, MlxError> {
        let (b, l) = (x.dim(0)?, x.dim(1)?);
        let (nq, nkv, hd) = (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        );
        let eps = self.cfg.rms_norm_eps;
        let qg = a.q.forward(x)?.reshape(&[b, l, nq, 2 * hd])?;
        let q = qg.slice(&[0, 0, 0, 0], &[b, l, nq, hd], &[1, 1, 1, 1])?;
        let gate = qg
            .slice(&[0, 0, 0, hd], &[b, l, nq, 2 * hd], &[1, 1, 1, 1])?
            .reshape(&[b, l, nq * hd])?;
        let q = q
            .rms_norm(Some(&a.q_norm), eps)?
            .astype(MlxDtype::Bf16)?
            .transpose(&[0, 2, 1, 3])?;
        let k =
            a.k.forward(x)?
                .reshape(&[b, l, nkv, hd])?
                .rms_norm(Some(&a.k_norm), eps)?
                .astype(MlxDtype::Bf16)?
                .transpose(&[0, 2, 1, 3])?;
        let v =
            a.v.forward(x)?
                .reshape(&[b, l, nkv, hd])?
                .transpose(&[0, 2, 1, 3])?;
        let q = self.apply_rope(&q, rope)?;
        let k = self.apply_rope(&k, rope)?;
        let (k, v) = cache.update_and_fetch(&k, &v)?;
        let scale = (hd as f64).powf(-0.5) as f32;
        let o = if let Some(m) = mask {
            q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::Array(m))?
        } else if l > 1 {
            q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::Causal)?
        } else if self.sdpa_decode {
            q.sdpa_decode(&k, &v, scale)?
        } else {
            q.scaled_dot_product_attention(&k, &v, scale, SdpaMask::None)?
        };
        let o = o.transpose(&[0, 2, 1, 3])?.reshape(&[b, l, nq * hd])?;
        a.o.forward(&o.multiply(&gate.sigmoid()?)?)
    }

    fn delta_net(
        &self,
        n: &DeltaNet,
        x: &MlxArray,
        conv_state: &mut Option<MlxArray>,
        rec_state: &mut Option<MlxArray>,
        record: Option<(usize, &mut Vec<DeltaRecord>)>,
    ) -> Result<MlxArray, MlxError> {
        let t = &self.cfg;
        let (bs, l) = (x.dim(0)?, x.dim(1)?);
        let (hk, dk) = (t.linear_num_key_heads, t.linear_key_head_dim);
        let (hv, dv) = (t.linear_num_value_heads, t.linear_value_head_dim);
        let (kd, vd, kk) = (hk * dk, hv * dv, t.linear_conv_kernel_dim);
        let c = 2 * kd + vd;
        let mixed = n.in_qkv.forward(x)?;
        let z = n.in_z.forward(x)?.reshape(&[bs, l, hv, dv])?;
        let b = n.in_b.forward(x)?;
        let a = n.in_a.forward(x)?;

        let prev = match conv_state.take() {
            Some(s) => s,
            None => MlxArray::zeros(&[bs, kk - 1, c], mixed.dtype()?)?,
        };
        let conv_in = MlxArray::concatenate(&[&prev, &mixed], 1)?;
        let total = l + kk - 1;
        *conv_state = Some(conv_in.slice(&[0, l, 0], &[bs, total, c], &[1, 1, 1])?);
        let conv_out = if l == 1 {
            conv_in
                .astype(MlxDtype::F32)?
                .multiply(&n.conv_w_step)?
                .sum_axis(1, false)?
                .astype(conv_in.dtype()?)?
                .reshape(&[bs, 1, c])?
        } else {
            conv_in.conv1d(&n.conv_w, 1, 0, 1, c)?
        };
        // F.silu on bf16: computed in f32, rounded once.
        let conv_out = silu_bf16(&conv_out)?;

        let part = |lo: usize, hi: usize, heads: usize, dim: usize| {
            conv_out
                .slice(&[0, 0, lo], &[bs, l, hi], &[1, 1, 1])?
                .reshape(&[bs, l, heads, dim])
        };
        // transformers' `l2norm(x, eps=1e-6)` in f32 (the epsilon on the
        // sum of squares, as an RMSNorm whose epsilon is divided by the head
        // width), and the query's `1 / sqrt(dk)` scale, also in f32.
        let inv = (dk as f64).powf(-0.5);
        let qk_eps = 1e-6 / dk as f32;
        let q = part(0, kd, hk, dk)?
            .astype(MlxDtype::F32)?
            .rms_norm(None, qk_eps)?
            .multiply(&MlxArray::scalar_f32((inv * inv) as f32)?)?;
        let k = part(kd, 2 * kd, hk, dk)?
            .astype(MlxDtype::F32)?
            .rms_norm(None, qk_eps)?
            .multiply(&MlxArray::scalar_f32(inv as f32)?)?;
        let v = part(2 * kd, c, hv, dv)?;

        // g = -exp(A_log) * softplus(a.float() + dt_bias), all in f32.
        let (g, beta) = a
            .astype(MlxDtype::F32)?
            .gated_delta_gates(&b, &n.a_log, &n.dt_bias)?;
        let state = match rec_state.take() {
            Some(s) => s,
            None => MlxArray::zeros(&[bs, hv, dv, dk], MlxDtype::F32)?,
        };
        let (y, new_state) = q.gated_delta(&k, &v, &g, &beta, &state)?;
        let y = y.astype(MlxDtype::Bf16)?;
        *rec_state = Some(new_state);
        if let Some((layer, out)) = record {
            out.push(DeltaRecord {
                layer,
                conv_in,
                state,
                q,
                k,
                v,
                g,
                beta,
            });
        }

        let dtype = y.dtype()?;
        let normed = y
            .rms_norm(Some(&n.norm), t.rms_norm_eps)?
            .astype(MlxDtype::F32)?;
        let out = z
            .astype(MlxDtype::F32)?
            .silu()?
            .multiply(&normed)?
            .astype(dtype)?
            .reshape(&[bs, l, vd])?;
        n.out.forward(&out)
    }

    /// Runs the blocks over `h` (`[B, L, hidden]`) at the `rope` positions,
    /// updating `caches`; returns the final-norm hidden state. With `taps`,
    /// records `layer_{i}` and `final_norm`.
    pub(crate) fn forward(
        &self,
        h: &MlxArray,
        rope: &Rope,
        caches: &mut [LayerCache],
        taps: Option<&mut Taps>,
    ) -> Result<MlxArray, DecoderError> {
        self.forward_capture(h, rope, None, caches, taps, None)
    }

    /// [`Self::forward`] that also fills `capture`. `mask` (additive,
    /// broadcast against `[B, heads, L, kv]`) replaces the full-attention
    /// causal / unmasked modes, for left-padded batches.
    pub(crate) fn forward_capture(
        &self,
        h: &MlxArray,
        rope: &Rope,
        mask: Option<&MlxArray>,
        caches: &mut [LayerCache],
        mut taps: Option<&mut Taps>,
        mut capture: Option<&mut Capture<'_>>,
    ) -> Result<MlxArray, DecoderError> {
        let s = h.shape()?;
        if s.len() != 3 || s[2] != self.cfg.hidden_size {
            return Err(DecoderError::Layout(format!("hidden {s:?}")));
        }
        if caches.len() != self.layers.len() {
            return Err(DecoderError::Layout("cache count != layers".into()));
        }
        let eps = self.cfg.rms_norm_eps;
        let mut x = h.try_clone()?;
        for (i, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let xn = x
                .rms_norm(Some(&layer.in_norm), eps)?
                .astype(MlxDtype::Bf16)?;
            let r = match (&layer.mixer, cache) {
                (Mixer::Full(a), LayerCache::Kv(c)) => {
                    self.full_attention(a, &xn, rope, mask, c)?
                }
                (Mixer::Linear(n), LayerCache::Delta { conv, state }) => {
                    let record = capture
                        .as_deref_mut()
                        .and_then(|c| c.delta.as_mut())
                        .map(|d| (i, d));
                    self.delta_net(n, &xn, conv, state, record)?
                }
                _ => return Err(DecoderError::Layout(format!("cache kind of layer {i}"))),
            };
            let hh = x.add(&r)?;
            let hn = hh
                .rms_norm(Some(&layer.post_norm), eps)?
                .astype(MlxDtype::Bf16)?;
            let mlp = layer.down.forward(&swiglu_bf16(
                &layer.gate.forward(&hn)?,
                &layer.up.forward(&hn)?,
            )?)?;
            x = hh.add(&mlp)?;
            if let Some(c) = capture.as_deref_mut()
                && c.layers.contains(&i)
            {
                c.hidden.push(x.try_clone()?);
            }
            if let Some(t) = taps.as_deref_mut() {
                t.push((format!("layer_{i}"), x.try_clone()?));
            }
        }
        let out = x.rms_norm(Some(&self.norm), eps)?.astype(MlxDtype::Bf16)?;
        if let Some(t) = taps {
            t.push(("final_norm".into(), out.try_clone()?));
        }
        Ok(out)
    }

    /// Keeps the first `keep` of the `total` positions a recorded verify
    /// forward appended: KV caches drop the rest, and each DeltaNet layer
    /// replays its recurrence over the kept prefix from the saved state
    /// (the step kernel computes every step the same way at any length, so
    /// this is the state the full block passed through).
    pub(crate) fn rollback(
        &self,
        caches: &mut [LayerCache],
        records: Vec<DeltaRecord>,
        keep: usize,
        total: usize,
    ) -> Result<(), DecoderError> {
        if keep == 0 || keep > total {
            return Err(DecoderError::Layout(format!(
                "rollback keep {keep} of {total}"
            )));
        }
        if keep == total {
            return Ok(());
        }
        for cache in caches.iter_mut() {
            if let LayerCache::Kv(c) = cache {
                c.trim(total - keep);
            }
        }
        let kk = self.cfg.linear_conv_kernel_dim;
        for r in records {
            let LayerCache::Delta { conv, state } = &mut caches[r.layer] else {
                return Err(DecoderError::Layout(format!(
                    "record for layer {}",
                    r.layer
                )));
            };
            let prefix = |x: &MlxArray| -> Result<MlxArray, MlxError> {
                let s = x.shape()?;
                let mut stop = s.clone();
                stop[1] = keep;
                x.slice(&vec![0; s.len()], &stop, &vec![1; s.len()])
            };
            let (_, st) = prefix(&r.q)?.gated_delta(
                &prefix(&r.k)?,
                &prefix(&r.v)?,
                &prefix(&r.g)?,
                &prefix(&r.beta)?,
                &r.state,
            )?;
            *state = Some(st);
            let s = r.conv_in.shape()?;
            *conv = Some(r.conv_in.slice(
                &[0, keep, 0],
                &[s[0], keep + kk - 1, s[2]],
                &[1, 1, 1],
            )?);
        }
        Ok(())
    }

    /// Batched [`Self::rollback`]: row `b` keeps the first `kept[b]` of the
    /// `total` positions. KV caches keep the first `max(kept)` (a shorter
    /// row's extra columns become holes its caller masks); each DeltaNet
    /// layer replays its recurrence over that prefix with every step past a
    /// row's own `kept` neutral (`g = 1`, `beta = 0`, which leave the state
    /// unchanged) and takes each row's conv window at its own end.
    pub(crate) fn rollback_rows(
        &self,
        caches: &mut [LayerCache],
        records: Vec<DeltaRecord>,
        kept: &[usize],
        total: usize,
    ) -> Result<(), DecoderError> {
        let keep = kept.iter().copied().max().unwrap_or(0);
        if kept.contains(&0) || keep > total {
            return Err(DecoderError::Layout(format!(
                "rollback keep {kept:?} of {total}"
            )));
        }
        if kept.iter().all(|&k| k == keep) {
            return self.rollback(caches, records, keep, total);
        }
        for cache in caches.iter_mut() {
            if let LayerCache::Kv(c) = cache {
                c.trim(total - keep);
            }
        }
        let b = kept.len();
        let kk = self.cfg.linear_conv_kernel_dim;
        let live: Vec<bool> = kept
            .iter()
            .flat_map(|&k| (0..keep).map(move |t| t < k))
            .collect();
        let live = MlxArray::from_bool(&[b, keep, 1], &live)?;
        let ends: Vec<u32> = kept
            .iter()
            .flat_map(|&k| (0..kk - 1).map(move |j| (k + j) as u32))
            .collect();
        let ends = MlxArray::from_u32(&[b, kk - 1, 1], &ends)?;
        for r in records {
            let LayerCache::Delta { conv, state } = &mut caches[r.layer] else {
                return Err(DecoderError::Layout(format!(
                    "record for layer {}",
                    r.layer
                )));
            };
            let prefix = |x: &MlxArray| -> Result<MlxArray, MlxError> {
                let s = x.shape()?;
                let mut stop = s.clone();
                stop[1] = keep;
                x.slice(&vec![0; s.len()], &stop, &vec![1; s.len()])
            };
            let scalar = |v: f32, like: &MlxArray| -> Result<MlxArray, MlxError> {
                MlxArray::scalar_f32(v)?.astype(like.dtype()?)
            };
            let g = prefix(&r.g)?;
            let beta = prefix(&r.beta)?;
            let g = live.where_(&g, &scalar(1.0, &g)?)?;
            let beta = live.where_(&beta, &scalar(0.0, &beta)?)?;
            let (_, st) =
                prefix(&r.q)?.gated_delta(&prefix(&r.k)?, &prefix(&r.v)?, &g, &beta, &r.state)?;
            *state = Some(st);
            let c = r.conv_in.dim(2)?;
            *conv = Some(
                r.conv_in
                    .take_along_axis(&ends.broadcast_to(&[b, kk - 1, c])?, 1)?,
            );
        }
        Ok(())
    }

    /// `lm_head(hidden)` (tied to `embed_tokens`).
    pub fn logits(&self, h: &MlxArray) -> Result<MlxArray, MlxError> {
        self.lm_head.forward(h)
    }
}
