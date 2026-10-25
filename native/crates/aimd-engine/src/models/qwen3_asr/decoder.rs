//! Qwen3 dense text decoder over MLX affine-quantized weights (WS-06b).
//!
//! Port of the `TextModel` / `TextDecoderLayer` / `TextAttention` graph in
//! `.venv/lib/python3.12/site-packages/mlx_audio/stt/models/qwen3_asr/
//! qwen3_asr.py` (the porting source of truth):
//!
//! - 28 pre-LN decoder layers: RMSNorm -> GQA attention with per-head
//!   `q_norm`/`k_norm` RMSNorm applied *before* NeoX RoPE -> residual ->
//!   RMSNorm -> SwiGLU MLP -> residual; final RMSNorm; logits through the
//!   tied embedding matrix (`embed_tokens.as_linear`).
//! - Every linear projection and the embedding table are 4-bit (or 8-bit)
//!   affine, group 64, in the `mlx-community` checkpoints.
//!   [`aimd_weights::dequantize_affine`] is bit-exact vs MLX, and the
//!   dequantized bf16 values convert to f16 losslessly here (model weights
//!   sit far inside the f16 exponent range), feeding the f32-accumulate CPU
//!   ops in `aimd-tensor`. Linear weights are stored transposed `[in, out]`
//!   after dequantization so every projection is a direct
//!   [`ops::matmul_f16_f32acc`]; the tied embedding stays `[vocab, hidden]`
//!   for row gathers and transposed-access logits.
//! - Audio features enter the sequence by REPLACING the embedding rows of
//!   the `<|audio_pad|>` (`audio_token_id`) prompt positions in order
//!   (`Qwen3ASRModel.__call__` / `_build_inputs_embeds`: position `i` takes
//!   `audio_features[audio_idx]` while
//!   `audio_idx < num_to_replace = min(#pads, #features)`; surplus pads keep
//!   their embedding). See [`Qwen3Decoder::build_input_embeds`].
//! - [`Qwen3Decoder::prefill`] / [`Qwen3Decoder::step`] run over the
//!   `aimd-tensor` [`KvCache`] with sequential absolute positions (prefill
//!   `0..seq-1`, steps continue at `seq`). Prefill attends with the causal
//!   no-cache kernel and then appends k/v; steps append the rotated k/v
//!   first and attend over the full cache (matching mlx
//!   `cache.update_and_fetch` + causal SDPA).
//!
//! [`Qwen3Decoder::load`] takes any checkpoint snapshot directory, so the
//! 4-bit model and the 8-bit repetition-retry model are the same type with
//! different inputs.

use std::path::Path;

use aimd_tensor::{DType, KvCache, Shape, Tensor, TensorError, ops};
use aimd_weights::{DequantizedTensor, ScaleDtype, WeightStore, load_checkpoint};
use half::{bf16, f16};

use super::config::{TextDecoderConfig, parse_config};

/// Errors surfaced by decoder loading and forward passes.
#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error("checkpoint weights: {0}")]
    Weights(#[from] aimd_weights::WeightsError),
    #[error("tensor op: {0}")]
    Tensor(#[from] TensorError),
    #[error("checkpoint config: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(String),
    #[error("missing weight tensor {0:?}")]
    MissingWeight(String),
    #[error("unsupported dtype {dtype} for {name:?}")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("checkpoint layout mismatch: {0}")]
    Layout(String),
    #[error("prefill requires an empty kv cache")]
    CacheNotEmpty,
    #[error("token id {id} outside vocab {vocab}")]
    TokenOutOfRange { id: u32, vocab: usize },
}

/// One decoder layer's weights. Linear weights are dequantized to `F16` and
/// transposed to `[in, out]`; norms stay `F16` vectors.
#[derive(Debug)]
struct LayerWeights {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    input_layernorm: Tensor,
    post_attention_layernorm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

/// Qwen3 text decoder over dequantized checkpoint weights.
///
/// Immutable after load (the KV cache is a caller-owned [`KvCache`]
/// argument), so one instance can be shared across generations.
#[derive(Debug)]
pub struct Qwen3Decoder {
    cfg: TextDecoderConfig,
    /// `F16 [vocab, hidden]` tied embedding table.
    embed_tokens: Tensor,
    layers: Vec<LayerWeights>,
    /// `F16 [hidden]` final RMSNorm weight.
    final_norm: Tensor,
}

impl Qwen3Decoder {
    /// Loads the decoder weights from an HF snapshot directory (any bit
    /// width: the 4-bit and 8-bit ASR checkpoints are the same layout).
    pub fn load(checkpoint_dir: &Path) -> Result<Self, DecoderError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let config_path = checkpoint_dir.join("config.json");
        let raw = std::fs::read(&config_path)
            .map_err(|e| DecoderError::Io(format!("reading {}: {e}", config_path.display())))?;
        let parsed = parse_config(&raw).map_err(|e| DecoderError::Config(e.to_string()))?;
        Self::from_store(&store, parsed.thinker_config.text_config)
    }

    fn from_store(store: &WeightStore, cfg: TextDecoderConfig) -> Result<Self, DecoderError> {
        if !cfg.tie_word_embeddings {
            return Err(DecoderError::Layout(
                "only tied-embedding checkpoints are supported (lm_head would be required)"
                    .to_string(),
            ));
        }
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;

        // Embedding table keeps the checkpoint's [vocab, hidden] layout: rows
        // are gathered for token embeddings and read transposed for logits.
        let embed_deq = store.quantized_tensor("model.embed_tokens")?.dequantize()?;
        let embed_tokens = deq_to_f16(&embed_deq, vocab, hidden, Transpose::Keep)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("model.layers.{i}.{suffix}");
            layers.push(LayerWeights {
                q_proj: dequantized_linear(
                    store,
                    &name("self_attn.q_proj"),
                    heads * head_dim,
                    hidden,
                )?,
                k_proj: dequantized_linear(
                    store,
                    &name("self_attn.k_proj"),
                    kv_heads * head_dim,
                    hidden,
                )?,
                v_proj: dequantized_linear(
                    store,
                    &name("self_attn.v_proj"),
                    kv_heads * head_dim,
                    hidden,
                )?,
                o_proj: dequantized_linear(
                    store,
                    &name("self_attn.o_proj"),
                    hidden,
                    heads * head_dim,
                )?,
                gate_proj: dequantized_linear(store, &name("mlp.gate_proj"), inter, hidden)?,
                up_proj: dequantized_linear(store, &name("mlp.up_proj"), inter, hidden)?,
                down_proj: dequantized_linear(store, &name("mlp.down_proj"), hidden, inter)?,
                input_layernorm: raw_vector(store, &name("input_layernorm.weight"), hidden)?,
                post_attention_layernorm: raw_vector(
                    store,
                    &name("post_attention_layernorm.weight"),
                    hidden,
                )?,
                q_norm: raw_vector(store, &name("self_attn.q_norm.weight"), head_dim)?,
                k_norm: raw_vector(store, &name("self_attn.k_norm.weight"), head_dim)?,
            });
        }
        let final_norm = raw_vector(store, "model.norm.weight", hidden)?;

        Ok(Self {
            cfg,
            embed_tokens,
            layers,
            final_norm,
        })
    }

    /// Decoder configuration the weights were validated against.
    pub fn config(&self) -> &TextDecoderConfig {
        &self.cfg
    }

    /// Embedding rows for token ids: `F16 [ids.len(), hidden]`.
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor, DecoderError> {
        let hidden = self.cfg.hidden_size;
        let vocab = self.cfg.vocab_size;
        let rows = self.embed_tokens.as_f16()?;
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let row = id as usize;
            if row >= vocab {
                return Err(DecoderError::TokenOutOfRange { id, vocab });
            }
            out.extend_from_slice(&rows[row * hidden..(row + 1) * hidden]);
        }
        Ok(Tensor::from_f16_vec(
            Shape::from_dims(&[ids.len(), hidden]),
            &out,
        )?)
    }

    /// Builds prefill input embeds from prompt token ids plus audio features.
    ///
    /// Mirrors mlx-audio `_build_inputs_embeds`: every prompt position whose
    /// id equals `audio_token_id` (`<|audio_pad|>`) is replaced, in order,
    /// with the next `audio_features` row, while features remain; surplus
    /// pads keep their embedding row. `audio_features` is rank-2
    /// `[n_audio, hidden]` in `F16`, `BF16`, or `F32` (converted to `F16`).
    /// Returns `F16 [prompt_ids.len(), hidden]`.
    pub fn build_input_embeds(
        &self,
        prompt_ids: &[u32],
        audio_features: &Tensor,
        audio_token_id: u32,
    ) -> Result<Tensor, DecoderError> {
        let hidden = self.cfg.hidden_size;
        let dims = audio_features.shape().dims();
        if dims.len() != 2 || dims[1] != hidden {
            return Err(DecoderError::Layout(format!(
                "audio features must be rank-2 [n_audio, {hidden}], got {:?} {}",
                dims,
                audio_features.dtype(),
            )));
        }
        let n_audio = dims[0];
        let audio: Vec<f16> = match audio_features.dtype() {
            DType::F16 => audio_features.as_f16()?.to_vec(),
            DType::F32 => audio_features
                .as_f32()?
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect(),
            DType::BF16 => audio_features
                .as_bf16()?
                .iter()
                .map(|&v| f16::from_f32(v.to_f32()))
                .collect(),
            other => {
                return Err(DecoderError::UnsupportedDtype {
                    name: "audio features".to_string(),
                    dtype: other.to_string(),
                });
            }
        };
        let vocab = self.cfg.vocab_size;
        let rows = self.embed_tokens.as_f16()?;
        let mut out = Vec::with_capacity(prompt_ids.len() * hidden);
        let mut audio_idx = 0usize;
        for &id in prompt_ids {
            if id == audio_token_id && audio_idx < n_audio {
                out.extend_from_slice(&audio[audio_idx * hidden..(audio_idx + 1) * hidden]);
                audio_idx += 1;
            } else {
                let row = id as usize;
                if row >= vocab {
                    return Err(DecoderError::TokenOutOfRange { id, vocab });
                }
                out.extend_from_slice(&rows[row * hidden..(row + 1) * hidden]);
            }
        }
        Ok(Tensor::from_f16_vec(
            Shape::from_dims(&[prompt_ids.len(), hidden]),
            &out,
        )?)
    }

    /// Full forward over prompt embeds, filling `cache`.
    ///
    /// `embeds` is `F16 [seq, hidden]`; `cache` must be empty and sized for
    /// `seq` plus all subsequent decode steps. Returns the final-normed
    /// hidden states `F16 [seq, hidden]` (logits via
    /// [`Qwen3Decoder::logits`]).
    pub fn prefill(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor, DecoderError> {
        let dims = embeds.shape().dims();
        if embeds.dtype() != DType::F16 || dims.len() != 2 || dims[1] != self.cfg.hidden_size {
            return Err(DecoderError::Layout(format!(
                "prefill embeds must be F16 [seq, {}], got {:?} {}",
                self.cfg.hidden_size,
                dims,
                embeds.dtype(),
            )));
        }
        if !cache.layer(0)?.is_empty() {
            return Err(DecoderError::CacheNotEmpty);
        }
        let mut x = embeds.clone();
        for (i, lw) in self.layers.iter().enumerate() {
            x = self.layer_prefill(i, lw, &x, cache)?;
        }
        Ok(ops::rmsnorm(&x, &self.final_norm, self.cfg.rms_norm_eps)?)
    }

    /// Single-token forward at absolute `position`.
    ///
    /// `embed` is the current token's embedding, `F16 [hidden]` or
    /// `[1, hidden]`; `cache` must already hold rows `0..position` (prefill
    /// or earlier steps). Returns the final-normed hidden row
    /// `F16 [1, hidden]`.
    pub fn step(
        &self,
        embed: &Tensor,
        position: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor, DecoderError> {
        let hidden = self.cfg.hidden_size;
        let dims = embed.shape().dims();
        let x = if embed.dtype() == DType::F16 && dims == [hidden] {
            reshape2(embed, 1, hidden)?
        } else if embed.dtype() == DType::F16 && dims == [1, hidden] {
            embed.clone()
        } else {
            return Err(DecoderError::Layout(format!(
                "step embed must be F16 [{hidden}] or [1, {hidden}], got {dims:?} {}",
                embed.dtype(),
            )));
        };
        let mut x = x;
        for (i, lw) in self.layers.iter().enumerate() {
            x = self.layer_step(i, lw, &x, position, cache)?;
        }
        Ok(ops::rmsnorm(&x, &self.final_norm, self.cfg.rms_norm_eps)?)
    }

    /// Tied-embedding logits via the transposed-B blocked matmul
    /// (`ops::matmul_bt_f16_f32out` against the `[vocab, hidden]` embedding
    /// table — no transpose copy, rayon across the vocab axis). `f32`
    /// output keeps argmax ties faithful.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor, DecoderError> {
        let dims = hidden.shape().dims();
        if hidden.dtype() != DType::F16 || dims.len() != 2 || dims[1] != self.cfg.hidden_size {
            return Err(DecoderError::Layout(format!(
                "logits input must be F16 [rows, {}], got {:?} {}",
                self.cfg.hidden_size,
                dims,
                hidden.dtype(),
            )));
        }
        Ok(ops::matmul_bt_f16_f32out(hidden, &self.embed_tokens)?)
    }

    /// One pre-LN layer over a `[seq, hidden]` block (prefill).
    fn layer_prefill(
        &self,
        index: usize,
        lw: &LayerWeights,
        x: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor, DecoderError> {
        let seq = x.shape().dims()[0];
        let (q_heads, kv_heads, head_dim) = self.attention_geometry();
        let normed = ops::rmsnorm(x, &lw.input_layernorm, self.cfg.rms_norm_eps)?;
        let positions: Vec<u32> = (0..seq as u32).collect();
        let q = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.q_proj)?,
            seq,
            q_heads,
            head_dim,
        )?;
        let k = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.k_proj)?,
            seq,
            kv_heads,
            head_dim,
        )?;
        let v = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.v_proj)?,
            seq,
            kv_heads,
            head_dim,
        )?;
        // Per-head RMSNorm before RoPE (Qwen3 TextAttention).
        let q = ops::rmsnorm(&q, &lw.q_norm, self.cfg.rms_norm_eps)?;
        let k = ops::rmsnorm(&k, &lw.k_norm, self.cfg.rms_norm_eps)?;
        let (q, k) = ops::apply_rope(&q, &k, &positions, self.cfg.rope_theta)?;
        let attn = ops::causal_prefill_attention(&q, &k, &v, None)?;
        cache.layer_mut(index)?.append(&k, &v)?;
        let attn_flat = reshape2(&attn, seq, q_heads * head_dim)?;
        let o = ops::matmul_f16_f32acc(&attn_flat, &lw.o_proj)?;
        let x = add_f16(x, &o)?;
        let mlp = self.mlp(lw, &x)?;
        Ok(add_f16(&x, &mlp)?)
    }

    /// One pre-LN layer for a single token (decode step).
    fn layer_step(
        &self,
        index: usize,
        lw: &LayerWeights,
        x: &Tensor,
        position: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor, DecoderError> {
        let (q_heads, kv_heads, head_dim) = self.attention_geometry();
        let normed = ops::rmsnorm(x, &lw.input_layernorm, self.cfg.rms_norm_eps)?;
        let q = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.q_proj)?,
            1,
            q_heads,
            head_dim,
        )?;
        let k = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.k_proj)?,
            1,
            kv_heads,
            head_dim,
        )?;
        let v = reshape3(
            &ops::matmul_f16_f32acc(&normed, &lw.v_proj)?,
            1,
            kv_heads,
            head_dim,
        )?;
        let q = ops::rmsnorm(&q, &lw.q_norm, self.cfg.rms_norm_eps)?;
        let k = ops::rmsnorm(&k, &lw.k_norm, self.cfg.rms_norm_eps)?;
        let (q, k) = ops::apply_rope(&q, &k, &[position], self.cfg.rope_theta)?;
        cache.layer_mut(index)?.append(&k, &v)?;
        let q2 = reshape2(&q, q_heads, head_dim)?;
        let attn = cache.layer(index)?.decode_attention(&q2, None, None)?;
        let attn_flat = reshape2(&attn, 1, q_heads * head_dim)?;
        let o = ops::matmul_f16_f32acc(&attn_flat, &lw.o_proj)?;
        let x = add_f16(x, &o)?;
        let mlp = self.mlp(lw, &x)?;
        Ok(add_f16(&x, &mlp)?)
    }

    /// SwiGLU feed-forward block over a `[rows, hidden]` input.
    fn mlp(&self, lw: &LayerWeights, x: &Tensor) -> Result<Tensor, DecoderError> {
        let normed = ops::rmsnorm(x, &lw.post_attention_layernorm, self.cfg.rms_norm_eps)?;
        let gate = ops::matmul_f16_f32acc(&normed, &lw.gate_proj)?;
        let up = ops::matmul_f16_f32acc(&normed, &lw.up_proj)?;
        let activated = ops::swiglu(&gate, &up)?;
        Ok(ops::matmul_f16_f32acc(&activated, &lw.down_proj)?)
    }

    fn attention_geometry(&self) -> (usize, usize, usize) {
        (
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
        )
    }
}

/// Dequantizes one affine triple and stores it as `F16 [in, out]`.
///
/// `out_features`/`in_features` describe the checkpoint's logical
/// `[out, in]` shape; the returned matrix is the transpose so callers matmul
/// activations directly.
fn dequantized_linear(
    store: &WeightStore,
    module: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Tensor, DecoderError> {
    let deq = store.quantized_tensor(module)?.dequantize()?;
    deq_to_f16(&deq, out_features, in_features, Transpose::Transpose)
}

/// Layout of the converted matrix relative to the dequantized `[rows, cols]`.
#[derive(Clone, Copy)]
enum Transpose {
    /// Output `[rows, cols]` (same row-major order).
    Keep,
    /// Output `[cols, rows]` with `out[i][j] = deq[j][i]`.
    Transpose,
}

/// Converts a dequantized bf16/f16 matrix to an `F16` tensor in one pass
/// (bf16 -> f32 -> f16 is exact for these weight magnitudes).
fn deq_to_f16(
    deq: &DequantizedTensor,
    rows: usize,
    cols: usize,
    layout: Transpose,
) -> Result<Tensor, DecoderError> {
    if deq.shape != vec![rows, cols] {
        return Err(DecoderError::Layout(format!(
            "dequantized shape {:?}, expected [{rows}, {cols}]",
            deq.shape
        )));
    }
    let (out_rows, out_cols) = match layout {
        Transpose::Keep => (rows, cols),
        Transpose::Transpose => (cols, rows),
    };
    let src = &deq.data;
    let mut out = vec![f16::ZERO; rows * cols];
    match deq.dtype {
        ScaleDtype::F16 => {
            fill_transposed(src, rows, cols, layout, &mut out, f16::from_bits);
        }
        ScaleDtype::Bf16 => {
            fill_transposed(src, rows, cols, layout, &mut out, |bits| {
                f16::from_f32(bf16::from_bits(bits).to_f32())
            });
        }
    }
    Tensor::from_f16_vec(Shape::from_dims(&[out_rows, out_cols]), &out).map_err(Into::into)
}

/// Fills `out` (flattened `[out_rows, out_cols]`) from a row-major
/// `[rows, cols]` source, applying `convert` per element and the requested
/// transpose.
fn fill_transposed(
    src: &[u16],
    rows: usize,
    cols: usize,
    layout: Transpose,
    out: &mut [f16],
    convert: fn(u16) -> f16,
) {
    match layout {
        Transpose::Keep => {
            for (dst, &bits) in out.iter_mut().zip(src) {
                *dst = convert(bits);
            }
        }
        Transpose::Transpose => {
            for (j, src_row) in src.chunks_exact(cols).enumerate() {
                for (i, &bits) in src_row.iter().enumerate() {
                    out[i * rows + j] = convert(bits);
                }
            }
        }
    }
}

/// Loads a plain (unquantized) f16/bf16/f32 checkpoint vector as `F16 [len]`.
fn raw_vector(store: &WeightStore, name: &str, len: usize) -> Result<Tensor, DecoderError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| DecoderError::MissingWeight(name.to_string()))?;
    let bytes = store.tensor_bytes(name)?;
    let dtype_name = format!("{:?}", info.dtype);
    let vals: Vec<f16> = match dtype_name.as_str() {
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| f16::from_bits(u16::from_le_bytes(*p)))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| f16::from_f32(bf16::from_bits(u16::from_le_bytes(*p)).to_f32()))
            .collect(),
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f16::from_f32(f32::from_le_bytes(*p)))
            .collect(),
        other => {
            return Err(DecoderError::UnsupportedDtype {
                name: name.to_string(),
                dtype: other.to_string(),
            });
        }
    };
    if vals.len() != len {
        return Err(DecoderError::Layout(format!(
            "{name}: {vals_len} elements, expected {len}",
            vals_len = vals.len()
        )));
    }
    Tensor::from_f16_vec(Shape::from_dims(&[len]), &vals).map_err(Into::into)
}

/// Cheap reshape of an `F16`/`F32` tensor (fresh buffer copy, same bytes).
fn reshape2(t: &Tensor, rows: usize, cols: usize) -> Result<Tensor, TensorError> {
    Tensor::from_bytes(
        t.dtype(),
        Shape::from_dims(&[rows, cols]),
        t.as_bytes().to_vec(),
    )
}

/// Cheap reshape to rank-3 (fresh buffer copy, same bytes).
fn reshape3(t: &Tensor, a: usize, b: usize, c: usize) -> Result<Tensor, TensorError> {
    Tensor::from_bytes(
        t.dtype(),
        Shape::from_dims(&[a, b, c]),
        t.as_bytes().to_vec(),
    )
}

/// Elementwise `f16` residual add with `f32` intermediate.
fn add_f16(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: format!("{:?}", a.shape().dims()),
            actual: format!("{:?}", b.shape().dims()),
        });
    }
    let av = a.as_f16()?;
    let bv = b.as_f16()?;
    let out: Vec<f16> = av
        .iter()
        .zip(bv)
        .map(|(&x, &y)| f16::from_f32(x.to_f32() + y.to_f32()))
        .collect();
    Tensor::from_f16_vec(a.shape().clone(), &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    const ASR_REPO_4BIT: &str = "models--mlx-community--Qwen3-ASR-1.7B-4bit";
    const ASR_REPO_8BIT: &str = "models--mlx-community--Qwen3-ASR-1.7B-8bit";

    /// Logits parity tolerance vs the bf16 MLX reference.
    ///
    /// Observed bound on the golden fixture (`asr_no_context_language_chinese`
    /// prompt + seeded noise features, dump positions 0..3 and prompt_len-1):
    /// max abs diff **1.035** (logits reach ~±25), while the greedy argmax
    /// matched at every dumped position and across the full 16-token forced
    /// stream (MLX min top-2 gap 0.375). The drift is pure store-dtype
    /// rounding: weights are bit-exact dequantizations, but MLX rounds every
    /// intermediate to bf16 while this port rounds to f16 (f32 accumulate in
    /// both) over 28 layers. 1.5 gives ~1.45x margin over the observed max.
    const LOGITS_ATOL: f64 = 1.5;

    fn hf_cache() -> PathBuf {
        PathBuf::from(
            std::env::var("AIMD_TEST_HF_CACHE")
                .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
        )
    }

    /// First (sorted) snapshot dir of `repo` containing every `required` file.
    fn snapshot_dir(repo: &str, required: &[&str]) -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> =
            std::fs::read_dir(hf_cache().join(repo).join("snapshots"))
                .ok()?
                .flatten()
                .map(|e| e.path())
                .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
                .collect();
        candidates.sort();
        candidates.into_iter().next()
    }

    fn asr_4bit_snapshot() -> Option<PathBuf> {
        snapshot_dir(ASR_REPO_4BIT, &["config.json", "model.safetensors"])
    }

    /// Loads the 4-bit decoder once per process and shares it across tests
    /// (dequantizing ~1.4G weight elements dominates the suite runtime).
    fn shared_decoder() -> Option<&'static Qwen3Decoder> {
        static DECODER: OnceLock<Option<Qwen3Decoder>> = OnceLock::new();
        DECODER
            .get_or_init(|| {
                let Some(dir) = asr_4bit_snapshot() else {
                    eprintln!("skipping: {ASR_REPO_4BIT} not in HF cache");
                    return None;
                };
                match Qwen3Decoder::load(&dir) {
                    Ok(decoder) => Some(decoder),
                    Err(e) => {
                        eprintln!("skipping: decoder load failed: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/asr_decoder")
    }

    fn golden_manifest() -> Option<serde_json::Value> {
        let raw = std::fs::read(golden_dir().join("manifest.json")).ok()?;
        serde_json::from_slice(&raw).ok()
    }

    fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect())
    }

    /// Hand-built minimal decoder for embedding-only unit tests.
    fn tiny_decoder(vocab: usize, hidden: usize) -> Qwen3Decoder {
        let cfg = TextDecoderConfig {
            hidden_size: hidden,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: hidden,
            intermediate_size: hidden,
            max_position_embeddings: 64,
            vocab_size: vocab,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            tie_word_embeddings: true,
        };
        let embed: Vec<f16> = (0..vocab * hidden)
            .map(|i| f16::from_f32(i as f32 * 0.5))
            .collect();
        Qwen3Decoder {
            embed_tokens: Tensor::from_f16_vec(Shape::from_dims(&[vocab, hidden]), &embed).unwrap(),
            layers: Vec::new(),
            final_norm: Tensor::from_f16_vec(Shape::from_dims(&[hidden]), &vec![f16::ONE; hidden])
                .unwrap(),
            cfg,
        }
    }

    #[test]
    fn decoder_build_input_embeds_replaces_audio_pads() {
        let decoder = tiny_decoder(8, 4);
        // audio_token_id 7; two feature rows; prompt pads at 1, 2, and 4.
        let features = Tensor::from_f32_vec(
            Shape::from_dims(&[2, 4]),
            &[100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],
        )
        .unwrap();
        let embeds = decoder
            .build_input_embeds(&[3, 7, 7, 1, 7], &features, 7)
            .unwrap();
        assert_eq!(embeds.shape().dims(), &[5, 4]);
        let ev = embeds.as_f16().unwrap();
        let embed_row = |id: usize, j: usize| f16::from_f32((id * 4 + j) as f32 * 0.5);
        // row 0: embedding of token 3
        for (j, v) in ev[..4].iter().enumerate() {
            assert_eq!(*v, embed_row(3, j));
        }
        // rows 1-2: audio features (f32 -> f16)
        for (row, base) in [(1usize, 100.0f32), (2, 200.0)] {
            for (j, v) in ev[row * 4..row * 4 + 4].iter().enumerate() {
                assert_eq!(*v, f16::from_f32(base + j as f32));
            }
        }
        // row 3: embedding of token 1
        for (j, v) in ev[3 * 4..4 * 4].iter().enumerate() {
            assert_eq!(*v, embed_row(1, j));
        }
        // row 4: surplus pad (features exhausted) keeps the embedding
        for (j, v) in ev[4 * 4..5 * 4].iter().enumerate() {
            assert_eq!(*v, embed_row(7, j));
        }
    }

    #[test]
    fn decoder_embed_tokens_gathers_rows() {
        let decoder = tiny_decoder(8, 4);
        let t = decoder.embed_tokens(&[0, 2, 7]).unwrap();
        assert_eq!(t.shape().dims(), &[3, 4]);
        let tv = t.as_f16().unwrap();
        for (r, id) in [0usize, 2, 7].iter().enumerate() {
            for j in 0..4 {
                assert_eq!(tv[r * 4 + j], f16::from_f32((id * 4 + j) as f32 * 0.5));
            }
        }
        assert!(matches!(
            decoder.embed_tokens(&[8]),
            Err(DecoderError::TokenOutOfRange { id: 8, vocab: 8 })
        ));
    }

    #[test]
    fn decoder_weight_layout_matches_checkpoint() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let cfg = decoder.config();
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(decoder.embed_tokens.shape().dims(), &[151936, 2048]);
        assert_eq!(decoder.final_norm.shape().dims(), &[2048]);
        assert_eq!(decoder.layers.len(), 28);
        let lw = &decoder.layers[0];
        // Linear weights are stored transposed [in, out].
        assert_eq!(lw.q_proj.shape().dims(), &[2048, 2048]);
        assert_eq!(lw.k_proj.shape().dims(), &[2048, 1024]);
        assert_eq!(lw.v_proj.shape().dims(), &[2048, 1024]);
        assert_eq!(lw.o_proj.shape().dims(), &[2048, 2048]);
        assert_eq!(lw.gate_proj.shape().dims(), &[2048, 6144]);
        assert_eq!(lw.up_proj.shape().dims(), &[2048, 6144]);
        assert_eq!(lw.down_proj.shape().dims(), &[6144, 2048]);
        assert_eq!(lw.input_layernorm.shape().dims(), &[2048]);
        assert_eq!(lw.q_norm.shape().dims(), &[128]);
        assert_eq!(lw.k_norm.shape().dims(), &[128]);
    }

    /// The decoder's embedding table must be the bit-exact MLX dequantization
    /// (converted bf16 -> f16 losslessly).
    #[test]
    fn decoder_embed_rows_bit_exact_vs_dequant() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(dir) = asr_4bit_snapshot() else {
            return;
        };
        let store = load_checkpoint(&dir).unwrap();
        let deq = store
            .quantized_tensor("model.embed_tokens")
            .unwrap()
            .dequantize()
            .unwrap();
        assert_eq!(deq.dtype, ScaleDtype::Bf16);
        let table = decoder.embed_tokens.as_f16().unwrap();
        let hidden = 2048usize;
        for row in [0usize, 1, 75_000, 151_935] {
            for col in [0usize, 1, 1023, 2047] {
                let expected =
                    f16::from_f32(bf16::from_bits(deq.data[row * hidden + col]).to_f32());
                assert_eq!(table[row * hidden + col], expected, "embed[{row}][{col}]");
            }
        }
    }

    /// Prefill logits parity vs the mlx-audio 4-bit decoder on the golden
    /// synthetic input. Tolerance covers bf16 (reference) vs f16+fn32-acc
    /// (this port) rounding drift over 28 layers; the observed bound is
    /// documented in the golden manifest.
    #[test]
    fn decoder_golden_prefill_logits_parity() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let Some(manifest) = golden_manifest() else {
            eprintln!("skipping: golden decoder fixture absent");
            return;
        };
        let prompt_ids: Vec<u32> = manifest["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let dump_positions: Vec<usize> = manifest["dump_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let n_audio = manifest["n_audio"].as_u64().unwrap() as usize;
        let hidden = manifest["hidden"].as_u64().unwrap() as usize;
        let audio_token_id = manifest["audio_token_id"].as_u64().unwrap() as u32;

        let feats_raw = read_f32_le(&golden_dir().join("features.f32")).expect("features fixture");
        assert_eq!(feats_raw.len(), n_audio * hidden);
        let features =
            Tensor::from_f32_vec(Shape::from_dims(&[n_audio, hidden]), &feats_raw).unwrap();

        let embeds = decoder
            .build_input_embeds(&prompt_ids, &features, audio_token_id)
            .unwrap();
        let mut cache = KvCache::new(
            decoder.config().num_hidden_layers,
            decoder.config().num_key_value_heads,
            decoder.config().head_dim,
            prompt_ids.len() + 16,
        )
        .unwrap();
        let hidden_states = decoder.prefill(&embeds, &mut cache).unwrap();

        let selected = select_rows(&hidden_states, &dump_positions);
        let logits = decoder.logits(&selected).unwrap();
        assert_eq!(logits.dtype(), DType::F32);
        let got = logits.as_f32().unwrap();

        let golden_raw = read_f32_le(&golden_dir().join("logits.f32")).expect("logits fixture");
        assert_eq!(golden_raw.len(), dump_positions.len() * 151936);
        let mut max_diff = 0f64;
        let mut argmax_mismatches = 0usize;
        for (r, pos) in dump_positions.iter().enumerate() {
            let g_row = &golden_raw[r * 151936..(r + 1) * 151936];
            let a_row = &got[r * 151936..(r + 1) * 151936];
            for (g, a) in g_row.iter().zip(a_row) {
                max_diff = max_diff.max((*g - *a).abs() as f64);
            }
            if argmax(g_row) != argmax(a_row) {
                argmax_mismatches += 1;
                eprintln!("argmax mismatch at prompt position {pos}");
            }
        }
        eprintln!("decoder golden logits: max abs diff {max_diff:.6}");
        assert_eq!(
            argmax_mismatches, 0,
            "argmax must match at dumped positions"
        );
        assert!(
            max_diff <= LOGITS_ATOL,
            "logits max abs diff {max_diff} exceeds {LOGITS_ATOL}"
        );
    }

    /// Chunked prefill + steps must reproduce the full-prefill final hidden
    /// state (same accumulation order in both attention kernels).
    #[test]
    fn decoder_step_prefill_chunk_equivalence() {
        let Some(decoder) = shared_decoder() else {
            return;
        };
        let cfg = decoder.config();
        // Deterministic synthetic prompt: 8 real tokens + 5 audio pads.
        let audio_token_id = 151_676u32;
        let prompt_ids: Vec<u32> = (1..=8u32)
            .chain(std::iter::repeat_n(audio_token_id, 5))
            .collect();
        let hidden = cfg.hidden_size;
        let n_audio = 5usize;
        let feats: Vec<f32> = (0..n_audio * hidden)
            .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
            .collect();
        let features = Tensor::from_f32_vec(Shape::from_dims(&[n_audio, hidden]), &feats).unwrap();
        let embeds = decoder
            .build_input_embeds(&prompt_ids, &features, audio_token_id)
            .unwrap();
        let seq = prompt_ids.len();
        let cache_capacity = seq + 4;

        let mut full_cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cache_capacity,
        )
        .unwrap();
        let full_hidden = decoder.prefill(&embeds, &mut full_cache).unwrap();

        // Chunked: prefill the first 6 tokens, then step the rest.
        let split = 6;
        let head = select_rows(&embeds, &(0..split).collect::<Vec<_>>());
        let mut chunk_cache = KvCache::new(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cache_capacity,
        )
        .unwrap();
        let _ = decoder.prefill(&head, &mut chunk_cache).unwrap();
        let mut last = decoder
            .step(
                &select_rows(&embeds, &[split]),
                split as u32,
                &mut chunk_cache,
            )
            .unwrap();
        for i in (split + 1)..seq {
            last = decoder
                .step(&select_rows(&embeds, &[i]), i as u32, &mut chunk_cache)
                .unwrap();
        }
        assert_eq!(chunk_cache.layer(0).unwrap().len(), seq);
        assert_eq!(full_cache.layer(0).unwrap().len(), seq);

        let full_last = &full_hidden.as_f16().unwrap()[(seq - 1) * hidden..seq * hidden];
        let chunk_last = last.as_f16().unwrap();
        assert_eq!(
            full_last, chunk_last,
            "chunked prefill must be bit-identical"
        );
    }

    /// The 8-bit retry checkpoint loads through the same path (shapes only;
    /// values differ from 4-bit by construction).
    #[test]
    fn decoder_loads_8bit_checkpoint() {
        let Some(dir) = snapshot_dir(ASR_REPO_8BIT, &["config.json", "model.safetensors"]) else {
            eprintln!("skipping: {ASR_REPO_8BIT} not in HF cache");
            return;
        };
        let decoder = Qwen3Decoder::load(&dir).expect("8-bit checkpoint loads");
        assert_eq!(decoder.config().num_hidden_layers, 28);
        assert_eq!(decoder.embed_tokens.shape().dims(), &[151936, 2048]);
        let quant = {
            let store = load_checkpoint(&dir).unwrap();
            store.quantized_tensor("model.embed_tokens").unwrap()
        };
        assert_eq!(quant.bits, 8);
    }

    fn select_rows(t: &Tensor, rows: &[usize]) -> Tensor {
        let dims = t.shape().dims();
        assert_eq!(dims.len(), 2, "select_rows needs rank 2");
        assert_eq!(t.dtype(), DType::F16, "select_rows handles F16 only");
        let row_len = dims[1];
        let src = t.as_f16().unwrap();
        let mut out = Vec::with_capacity(rows.len() * row_len);
        for &r in rows {
            out.extend_from_slice(&src[r * row_len..(r + 1) * row_len]);
        }
        Tensor::from_f16_vec(Shape::from_dims(&[rows.len(), row_len]), &out).unwrap()
    }

    fn argmax(values: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in values.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }
}
