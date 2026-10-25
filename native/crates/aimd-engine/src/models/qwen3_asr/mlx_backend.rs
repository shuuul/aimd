//! MLX/Metal Qwen3-ASR pipeline (WS-11b): the full graph — audio tower AND
//! text decoder — running on MLX Metal kernels through the [`aimd_mlx`] shim.
//!
//! # Op-for-op port of the mlx-audio graph
//!
//! The porting source of truth is
//! `.venv/lib/python3.12/site-packages/mlx_audio/stt/models/qwen3_asr/
//! qwen3_asr.py`; every stage here mirrors one of its calls with the same
//! dtypes, so Metal sees the same kernel sequence as mlx-audio:
//!
//! - Tower (`AudioEncoder`): the mel chunks are zero-padded to the longest
//!   chunk and convolved as one NHWC batch ([`MlxArray::conv2d`] 3x3 s2 p1 +
//!   exact-erf [`MlxArray::gelu`]), flattened time-major, projected by the
//!   bias-free `conv_out`, given the (CPU-computed, bit-identical) f32
//!   sinusoidal positional embedding, trimmed per chunk, concatenated into
//!   the encoder stream, and run through 24 pre-norm layers with
//!   block-diagonal attention ([`MlxArray::layer_norm`] +
//!   [`MlxArray::scaled_dot_product_attention`] with an additive -1e9 array
//!   mask). Activations are f32 (the mel is f32 and every bf16 weight is
//!   promoted per op, exactly like mlx-audio); tower weights are the
//!   UNQUANTIZED checkpoint bf16 tensors imported verbatim.
//! - Decoder (`TextModel`): 28 pre-LN layers with per-head q/k RMSNorm
//!   before NeoX RoPE ([`MlxArray::rope`], the primitive behind `nn.RoPE`),
//!   GQA causal attention ([`SdpaMask::Causal`] at prefill, no mask for
//!   single-token steps), SwiGLU MLPs, final RMSNorm, and tied-embedding
//!   logits. The 4-bit/8-bit affine triples (u32 payload + bf16
//!   scales/biases) are fed DIRECTLY to [`MlxArray::quantized_matmul`] with
//!   `transpose = true` — no eager dequantization; token embeddings gather
//!   payload rows with [`MlxArray::take`] and dequantize per gather,
//!   mirroring `nn.QuantizedEmbedding.__call__`. Decoder activations are
//!   bf16: the embedding output takes the bf16 scales' dtype and every
//!   quantized matmul promotes bf16 x bf16 -> bf16, matching mlx-audio's
//!   quantized-checkpoint dtype flow.
//! - Audio features enter the sequence by replacing `<|audio_pad|>`
//!   embedding rows in order (`_build_inputs_embeds`); the merge runs on
//!   the host over a readback of the dequantized prompt rows (a few hundred
//!   KB) and re-imports one bf16 array.
//!
//! # Weight import
//!
//! Every tensor is imported through the shim's bit-exact byte constructor
//! from [`aimd_weights::WeightStore`] views: tower bf16 verbatim, decoder
//! triples via [`MlxArray::quantized_weight`] (checkpoint quantization is
//! the contract).
//!
//! # Lazy evaluation and readback discipline
//!
//! mlx is lazy: the graph is driven by explicit readbacks only. Each decode
//! step's logits readback evaluates that whole step (the per-step eval);
//! tower output and prompt embeds are read back once at their boundary.
//! All byte readbacks materialize non-contiguous views inside the shim
//! (`contiguous` + `eval`, the WS-11a strided-view SIGSEGV lesson).
//!
//! # Greedy loop and retry
//!
//! The loop lives in Rust and mirrors `generate.rs`/mlx-audio
//! `generate_step`: the first token is the argmax at the last prompt
//! position, each next token is the argmax of the previous step's logits,
//! EOS terminates without entering the output, and `max_new_tokens` (4096)
//! is the hard budget. Argmax runs on the host over a bf16->f32 logits
//! readback (607 KB/step, negligible next to per-step compute) instead of
//! mlx `argmax`: first-index tie-breaking is pinned to the numpy/`np.argmax`
//! semantics the golden fixtures were exported with, independent of mlx's
//! reduction order. Repetition-loop detection, output parsing, and EOS
//! resolution are reused from [`crate::models::qwen3_asr::generate`], so the
//! 4-bit -> 8-bit retry composes exactly like the CPU
//! [`crate::models::qwen3_asr::pipeline::transcribe_with_retry`].
//!
//! The KV cache is [`MlxArray`] state grown per step with
//! [`MlxArray::concatenate`] along the sequence axis (the same
//! `cache.update_and_fetch` semantics as mlx `KVCache`).

use std::collections::BTreeSet;
use std::path::Path;

use aimd_mlx::{MlxArray, MlxDtype, MlxError, SdpaMask};
use aimd_weights::{ScaleDtype, WeightStore, WeightsError, load_checkpoint};
use half::bf16;

use crate::audio::{AudioError, MelConfig, load_wav, mel_spectrogram};
use crate::models::qwen3_asr::config::{
    AudioTowerConfig, Qwen3AsrCheckpointConfig, TextDecoderConfig, parse_config,
};
use crate::models::qwen3_asr::encoder::frames_to_audio_pads;
use crate::models::qwen3_asr::generate::{
    FinishReason, GenerationOutcome, GenerationSettings, TokenStream, detect_repetition_loop,
    parse_qwen_output, resolve_eos_ids,
};
use crate::tokenizer::{Tokenizer, TokenizerError, build_asr_prompt, resolve_language};

/// LayerNorm epsilon of mlx `nn.LayerNorm` (the tower default; mirrors the
/// CPU encoder port).
const LAYER_NORM_EPS: f32 = 1e-5;

/// Failures of the MLX pipeline. Variants mirror
/// [`crate::models::qwen3_asr::pipeline::PipelineError`] where the stage is
/// shared (audio, weights, config, tokenizer, language, empty output); MLX
/// and checkpoint-layout failures get their own variants.
#[derive(Debug, thiserror::Error)]
pub enum MlxPipelineError {
    #[error("audio: {0}")]
    Audio(#[from] AudioError),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("config mismatch: {0}")]
    ConfigMismatch(String),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("checkpoint layout mismatch: {0}")]
    Layout(String),
    #[error("unsupported language {0:?}")]
    UnsupportedLanguage(String),
    #[error("prompt token ids are empty")]
    EmptyPrompt,
    #[error("transcription was empty")]
    EmptyTranscription,
}

/// Resolved language for the prompt builder.
fn resolved_language(input: &str) -> Result<String, MlxPipelineError> {
    resolve_language(input).ok_or_else(|| MlxPipelineError::UnsupportedLanguage(input.to_string()))
}

// ---------------------------------------------------------------------------
// MLX audio tower (mlx-audio `AudioEncoder`)
// ---------------------------------------------------------------------------

/// One tower layer's weights: bf16 checkpoint tensors verbatim. Linear
/// weights are kept as transposed views `[in, out]` so every projection is
/// a direct matmul; biases and norm vectors keep the checkpoint layout.
struct MlxTowerLayer {
    q_wt: MlxArray,
    q_b: MlxArray,
    k_wt: MlxArray,
    k_b: MlxArray,
    v_wt: MlxArray,
    v_b: MlxArray,
    o_wt: MlxArray,
    o_b: MlxArray,
    attn_ln_w: MlxArray,
    attn_ln_b: MlxArray,
    fc1_wt: MlxArray,
    fc1_b: MlxArray,
    fc2_wt: MlxArray,
    fc2_b: MlxArray,
    fin_ln_w: MlxArray,
    fin_ln_b: MlxArray,
}

/// The audio tower: conv front-end + 24 windowed encoder layers + head,
/// all on MLX.
struct MlxTower {
    cfg: AudioTowerConfig,
    conv2d1_w: MlxArray,
    conv2d1_b: MlxArray,
    conv2d2_w: MlxArray,
    conv2d2_b: MlxArray,
    conv2d3_w: MlxArray,
    conv2d3_b: MlxArray,
    /// `conv_out.weight` `[d_model, flat_in]` as a transposed view.
    conv_out_wt: MlxArray,
    layers: Vec<MlxTowerLayer>,
    ln_post_w: MlxArray,
    ln_post_b: MlxArray,
    proj1_wt: MlxArray,
    proj1_b: MlxArray,
    proj2_wt: MlxArray,
    proj2_b: MlxArray,
    /// `head_dim^-0.5` scalar applied to projected queries (mirrors
    /// `AudioAttention.scaling` pre-scaling with `sdpa scale=1.0`).
    q_scale: MlxArray,
    /// `downsample_hidden_size` (conv channel count).
    channels: usize,
}

impl MlxTower {
    fn from_store(cfg: &AudioTowerConfig, store: &WeightStore) -> Result<Self, MlxPipelineError> {
        if cfg.activation_function != "gelu" {
            return Err(MlxPipelineError::ConfigMismatch(format!(
                "activation_function {:?}, only exact-erf \"gelu\" is implemented",
                cfg.activation_function
            )));
        }
        let head_dim = cfg.d_model / cfg.encoder_attention_heads;

        // Conv weights are MLX NHWC `[o, kh, kw, c]`; biases `[o]`.
        let conv2d1_w = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d1.weight",
            &[cfg.downsample_hidden_size, 3, 3, 1],
        )?;
        let conv2d1_b = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d1.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let conv2d2_w = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d2.weight",
            &[cfg.downsample_hidden_size, 3, 3, cfg.downsample_hidden_size],
        )?;
        let conv2d2_b = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d2.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let conv2d3_w = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d3.weight",
            &[cfg.downsample_hidden_size, 3, 3, cfg.downsample_hidden_size],
        )?;
        let conv2d3_b = mlx_bf16_tensor(
            store,
            "audio_tower.conv2d3.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let flat_in = cfg.downsample_hidden_size * conv_freq_after_front_end(cfg.num_mel_bins);
        let conv_out_wt = mlx_bf16_tensor(
            store,
            "audio_tower.conv_out.weight",
            &[cfg.d_model, flat_in],
        )?
        .transpose(&[1, 0])?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for index in 0..cfg.encoder_layers {
            let prefix = format!("audio_tower.layers.{index}.");
            let linear_wt =
                |name: &str, out: usize, inp: usize| -> Result<MlxArray, MlxPipelineError> {
                    Ok(
                        mlx_bf16_tensor(store, &format!("{prefix}{name}.weight"), &[out, inp])?
                            .transpose(&[1, 0])?,
                    )
                };
            let vector = |name: &str, len: usize| -> Result<MlxArray, MlxPipelineError> {
                mlx_bf16_tensor(store, &format!("{prefix}{name}"), &[len])
            };
            layers.push(MlxTowerLayer {
                q_wt: linear_wt("self_attn.q_proj", cfg.d_model, cfg.d_model)?,
                q_b: vector("self_attn.q_proj.bias", cfg.d_model)?,
                k_wt: linear_wt("self_attn.k_proj", cfg.d_model, cfg.d_model)?,
                k_b: vector("self_attn.k_proj.bias", cfg.d_model)?,
                v_wt: linear_wt("self_attn.v_proj", cfg.d_model, cfg.d_model)?,
                v_b: vector("self_attn.v_proj.bias", cfg.d_model)?,
                o_wt: linear_wt("self_attn.out_proj", cfg.d_model, cfg.d_model)?,
                o_b: vector("self_attn.out_proj.bias", cfg.d_model)?,
                attn_ln_w: vector("self_attn_layer_norm.weight", cfg.d_model)?,
                attn_ln_b: vector("self_attn_layer_norm.bias", cfg.d_model)?,
                fc1_wt: linear_wt("fc1", cfg.encoder_ffn_dim, cfg.d_model)?,
                fc1_b: vector("fc1.bias", cfg.encoder_ffn_dim)?,
                fc2_wt: linear_wt("fc2", cfg.d_model, cfg.encoder_ffn_dim)?,
                fc2_b: vector("fc2.bias", cfg.d_model)?,
                fin_ln_w: vector("final_layer_norm.weight", cfg.d_model)?,
                fin_ln_b: vector("final_layer_norm.bias", cfg.d_model)?,
            });
        }

        Ok(Self {
            cfg: cfg.clone(),
            conv2d1_w,
            conv2d1_b,
            conv2d2_w,
            conv2d2_b,
            conv2d3_w,
            conv2d3_b,
            conv_out_wt,
            layers,
            ln_post_w: mlx_bf16_tensor(store, "audio_tower.ln_post.weight", &[cfg.d_model])?,
            ln_post_b: mlx_bf16_tensor(store, "audio_tower.ln_post.bias", &[cfg.d_model])?,
            proj1_wt: mlx_bf16_tensor(
                store,
                "audio_tower.proj1.weight",
                &[cfg.d_model, cfg.d_model],
            )?
            .transpose(&[1, 0])?,
            proj1_b: mlx_bf16_tensor(store, "audio_tower.proj1.bias", &[cfg.d_model])?,
            proj2_wt: mlx_bf16_tensor(
                store,
                "audio_tower.proj2.weight",
                &[cfg.output_dim, cfg.d_model],
            )?
            .transpose(&[1, 0])?,
            proj2_b: mlx_bf16_tensor(store, "audio_tower.proj2.bias", &[cfg.output_dim])?,
            q_scale: MlxArray::scalar_f32((head_dim as f64).powf(-0.5) as f32)?,
            channels: cfg.downsample_hidden_size,
        })
    }

    /// Runs the tower over a frequency-major mel (`[n_mels, frames]`,
    /// exactly the layout `mel_spectrogram` emits) and returns the
    /// row-major `[pads, output_dim]` f32 features plus the pad count.
    fn forward(
        &self,
        mel_fm: &[f32],
        frames: usize,
    ) -> Result<(Vec<f32>, usize), MlxPipelineError> {
        let cfg = &self.cfg;
        let n_mels = cfg.num_mel_bins;
        let d = cfg.d_model;
        if frames == 0 {
            return Err(MlxPipelineError::Layout("frames must be >= 1".to_string()));
        }
        if mel_fm.len() != frames * n_mels {
            return Err(MlxPipelineError::Layout(format!(
                "mel holds {} values, expected frames ({frames}) * num_mel_bins ({n_mels})",
                mel_fm.len()
            )));
        }

        // 1) Padded chunk batch [chunks, n_mels, max_len, 1] f32 — the
        //    reference zero-pads every chunk to the longest chunk and
        //    convolves the batch, then trims each row to its true length.
        let chunk_size = cfg.n_window * 2;
        let chunk_lengths = mel_chunk_lengths(frames, chunk_size);
        let chunks = chunk_lengths.len();
        let max_chunk_len = chunk_lengths.iter().copied().max().unwrap_or(0);
        let mut batch = vec![0f32; chunks * n_mels * max_chunk_len];
        let mut start = 0;
        for (ci, &clen) in chunk_lengths.iter().enumerate() {
            for f in 0..n_mels {
                let src = &mel_fm[f * frames + start..f * frames + start + clen];
                let dst = (ci * n_mels + f) * max_chunk_len;
                batch[dst..dst + clen].copy_from_slice(src);
            }
            start += clen;
        }
        let mut x = MlxArray::from_f32(&[chunks, n_mels, max_chunk_len, 1], &batch)?;

        // 2) Conv front-end: three 3x3 s2 p1 convolutions, each + GELU
        //    (biases applied separately, like `nn.Conv2d`).
        x = x.conv2d(&self.conv2d1_w, [2, 2], [1, 1])?;
        x = x.add(&self.conv2d1_b)?;
        x = x.gelu()?;
        let h1 = conv_time_out(n_mels);
        let w1 = conv_time_out(max_chunk_len);
        x = x.conv2d(&self.conv2d2_w, [2, 2], [1, 1])?;
        x = x.add(&self.conv2d2_b)?;
        x = x.gelu()?;
        let h2 = conv_time_out(h1);
        let w2 = conv_time_out(w1);
        x = x.conv2d(&self.conv2d3_w, [2, 2], [1, 1])?;
        x = x.add(&self.conv2d3_b)?;
        x = x.gelu()?;
        let h3 = conv_time_out(h2);
        let w3 = conv_time_out(w2);

        // 3) Flatten time-major with index `c * freq' + f`
        //    (transpose(0, 2, 3, 1).reshape(b, t, c * f)), conv_out
        //    (bias-free), positional embedding (restarts at 0 per chunk —
        //    added over the padded length, identical for the kept prefix).
        let x = x
            .transpose(&[0, 2, 3, 1])?
            .reshape(&[chunks, w3, self.channels * h3])?;
        let x = x.matmul(&self.conv_out_wt, None)?;
        let pos = MlxArray::from_f32(&[w3, d], &sinusoidal_position_embedding(w3, d))?;
        let x = x.add(&pos)?;

        // 4) Per-chunk trim + concatenate into the encoder stream.
        let mut parts: Vec<MlxArray> = Vec::with_capacity(chunks);
        for (ci, &clen) in chunk_lengths.iter().enumerate() {
            let valid = chunk_time_after_front_end(clen);
            parts.push(x.slice(&[ci, 0, 0], &[ci + 1, valid, d], &[1, 1, 1])?);
        }
        let part_refs: Vec<&MlxArray> = parts.iter().collect();
        let mut hidden = MlxArray::concatenate(&part_refs, 1)?; // [1, total, d]
        let total = hidden.dim(1)?;

        // 5) Block-diagonal attention mask (CPU-built, imported) over the
        //    stream; masked scores underflow to zero in f32, so the additive
        //    -1e9 mask is equivalent to block-wise attention.
        let window = self.attention_window(&chunk_lengths);
        let blocks = attention_block_ranges(total, window);
        let mask = MlxArray::from_f32(&[1, 1, total, total], &block_mask_f32(total, &blocks))?;

        // 6) Encoder layers.
        for layer in &self.layers {
            hidden = self.layer_forward(layer, hidden, &mask)?;
        }

        // 7) Head: ln_post -> proj1 -> GELU -> proj2.
        let mut out =
            hidden.layer_norm(Some(&self.ln_post_w), Some(&self.ln_post_b), LAYER_NORM_EPS)?;
        out = out.matmul(&self.proj1_wt, None)?.add(&self.proj1_b)?;
        out = out.gelu()?;
        out = out.matmul(&self.proj2_wt, None)?.add(&self.proj2_b)?;
        let pads = out.dim(1)?;
        let out = out.reshape(&[pads, cfg.output_dim])?;
        let features = out.to_f32_vec()?;
        Ok((features, pads))
    }

    /// Per-forward attention window: `max chunk-after-cnn length *
    /// (n_window_infer / chunk_size)`, mirroring the reference.
    fn attention_window(&self, chunk_lengths: &[usize]) -> usize {
        let max_after_cnn = chunk_lengths
            .iter()
            .map(|&len| chunk_time_after_front_end(len))
            .max()
            .unwrap_or(0);
        max_after_cnn * (self.cfg.infer_window() / (self.cfg.n_window * 2))
    }

    /// One pre-norm encoder layer with block-diagonal attention.
    fn layer_forward(
        &self,
        lw: &MlxTowerLayer,
        x: MlxArray,
        mask: &MlxArray,
    ) -> Result<MlxArray, MlxError> {
        let d = self.cfg.d_model;
        let heads = self.cfg.encoder_attention_heads;
        let head_dim = d / heads;
        let total = x.dim(1)?;

        let normed = x.layer_norm(Some(&lw.attn_ln_w), Some(&lw.attn_ln_b), LAYER_NORM_EPS)?;
        let q = normed
            .matmul(&lw.q_wt, None)?
            .add(&lw.q_b)?
            .multiply(&self.q_scale)?;
        let k = normed.matmul(&lw.k_wt, None)?.add(&lw.k_b)?;
        let v = normed.matmul(&lw.v_wt, None)?.add(&lw.v_b)?;
        let q = q
            .reshape(&[1, total, heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let k = k
            .reshape(&[1, total, heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let v = v
            .reshape(&[1, total, heads, head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let attn = q.scaled_dot_product_attention(&k, &v, 1.0, SdpaMask::Array(mask))?;
        let attn = attn.transpose(&[0, 2, 1, 3])?.reshape(&[1, total, d])?;
        let o = attn.matmul(&lw.o_wt, None)?.add(&lw.o_b)?;
        let x = x.add(&o)?;

        let normed = x.layer_norm(Some(&lw.fin_ln_w), Some(&lw.fin_ln_b), LAYER_NORM_EPS)?;
        let h = normed.matmul(&lw.fc1_wt, None)?.add(&lw.fc1_b)?;
        let h = h.gelu()?;
        let h = h.matmul(&lw.fc2_wt, None)?.add(&lw.fc2_b)?;
        x.add(&h)
    }
}

// ---------------------------------------------------------------------------
// MLX text decoder (mlx-audio `TextModel` over quantized weights)
// ---------------------------------------------------------------------------

/// One affine-quantized module: the packed u32 payload plus bf16 (or f16)
/// scales/biases, fed DIRECTLY to `quantized_matmul` (no eager dequant).
struct MlxQuantWeights {
    w: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    group_size: i32,
    bits: i32,
}

impl MlxQuantWeights {
    /// Assembles the module's `{base}.weight` / `.scales` / `.biases` triple
    /// from the checkpoint store. Logical shape `[out, in]`; payload
    /// `[out, in*bits/32]` u32; scales/biases `[out, in/group_size]`.
    fn from_store(store: &WeightStore, module: &str) -> Result<Self, MlxPipelineError> {
        let quant = store.quantized_tensor(module)?;
        let cols = *quant
            .shape
            .last()
            .ok_or_else(|| MlxPipelineError::Layout(format!("{module}: empty quantized shape")))?;
        let rows: usize = quant.shape[..quant.shape.len() - 1].iter().product();
        let words = cols * quant.bits as usize / 32;
        let groups = cols / quant.group_size as usize;
        let scale_dtype = match quant.scale_dtype {
            ScaleDtype::F16 => MlxDtype::F16,
            ScaleDtype::Bf16 => MlxDtype::Bf16,
        };
        let (w, scales, biases) = MlxArray::quantized_weight(
            &quant.payload,
            &[rows, words],
            &quant.scales,
            &quant.biases,
            &[rows, groups],
            scale_dtype,
        )?;
        Ok(Self {
            w,
            scales,
            biases,
            group_size: quant.group_size as i32,
            bits: quant.bits as i32,
        })
    }

    /// `x @ deq(w).T` with `x` rank-2 `[rows, in]` -> `[rows, out]`.
    fn matmul(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        x.quantized_matmul(
            &self.w,
            &self.scales,
            Some(&self.biases),
            true,
            Some(self.group_size),
            Some(self.bits),
            None,
        )
    }
}

/// One decoder layer's weights: quantized projections + bf16 norm vectors.
struct MlxDecoderLayer {
    q: MlxQuantWeights,
    k: MlxQuantWeights,
    v: MlxQuantWeights,
    o: MlxQuantWeights,
    gate: MlxQuantWeights,
    up: MlxQuantWeights,
    down: MlxQuantWeights,
    input_ln: MlxArray,
    post_ln: MlxArray,
    q_norm: MlxArray,
    k_norm: MlxArray,
}

/// Which forward pass a layer call is: full-sequence prefill (causal mask,
/// positions 0..L-1) or a single decode step at an absolute position.
#[derive(Clone, Copy)]
enum Pass {
    Prefill,
    Step { position: u32 },
}

impl Pass {
    fn seq_len(self, x: &MlxArray) -> Result<usize, MlxError> {
        match self {
            Pass::Prefill => x.dim(1),
            Pass::Step { .. } => Ok(1),
        }
    }
    fn rope_offset(self) -> i32 {
        match self {
            Pass::Prefill => 0,
            Pass::Step { position } => position as i32,
        }
    }
    fn mask<'a>(self) -> SdpaMask<'a> {
        match self {
            Pass::Prefill => SdpaMask::Causal,
            Pass::Step { .. } => SdpaMask::None,
        }
    }
}

/// Qwen3 dense text decoder over MLX quantized weights. Immutable after
/// load; the KV cache is a caller-owned [`MlxKvCache`].
struct MlxDecoder {
    cfg: TextDecoderConfig,
    embed: MlxQuantWeights,
    layers: Vec<MlxDecoderLayer>,
    final_norm: MlxArray,
}

impl MlxDecoder {
    fn from_store(cfg: &TextDecoderConfig, store: &WeightStore) -> Result<Self, MlxPipelineError> {
        if !cfg.tie_word_embeddings {
            return Err(MlxPipelineError::Layout(
                "only tied-embedding checkpoints are supported (lm_head would be required)"
                    .to_string(),
            ));
        }
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;

        let embed = MlxQuantWeights::from_store(store, "model.embed_tokens")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("model.layers.{i}.{suffix}");
            layers.push(MlxDecoderLayer {
                q: MlxQuantWeights::from_store(store, &name("self_attn.q_proj"))?,
                k: MlxQuantWeights::from_store(store, &name("self_attn.k_proj"))?,
                v: MlxQuantWeights::from_store(store, &name("self_attn.v_proj"))?,
                o: MlxQuantWeights::from_store(store, &name("self_attn.o_proj"))?,
                gate: MlxQuantWeights::from_store(store, &name("mlp.gate_proj"))?,
                up: MlxQuantWeights::from_store(store, &name("mlp.up_proj"))?,
                down: MlxQuantWeights::from_store(store, &name("mlp.down_proj"))?,
                input_ln: mlx_raw_vector(store, &name("input_layernorm.weight"), hidden)?,
                post_ln: mlx_raw_vector(store, &name("post_attention_layernorm.weight"), hidden)?,
                q_norm: mlx_raw_vector(store, &name("self_attn.q_norm.weight"), head_dim)?,
                k_norm: mlx_raw_vector(store, &name("self_attn.k_norm.weight"), head_dim)?,
            });
        }
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            layers,
            final_norm: mlx_raw_vector(store, "model.norm.weight", hidden)?,
        })
    }

    /// Dequantized embedding rows for `ids`: `[ids.len(), hidden]` bf16,
    /// exactly `nn.QuantizedEmbedding.__call__` (row gather + dequantize).
    fn embed_rows(&self, ids: &[u32]) -> Result<MlxArray, MlxPipelineError> {
        if ids.is_empty() {
            return Err(MlxPipelineError::Layout("empty token id list".to_string()));
        }
        for &id in ids {
            if id as usize >= self.cfg.vocab_size {
                return Err(MlxPipelineError::Layout(format!(
                    "token id {id} outside vocab {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let indices = MlxArray::from_u32(&[ids.len()], ids)?;
        let w = self.embed.w.take(&indices, 0)?;
        let scales = self.embed.scales.take(&indices, 0)?;
        let biases = self.embed.biases.take(&indices, 0)?;
        Ok(w.dequantize(
            &scales,
            Some(&biases),
            Some(self.embed.group_size),
            Some(self.embed.bits),
        )?)
    }

    /// Tied-embedding logits: quantized matmul of `[rows, hidden]` hidden
    /// states against the quantized `[vocab, hidden]` table (transposed) ->
    /// `[rows, vocab]` bf16.
    fn logits(&self, hidden_states: &MlxArray) -> Result<MlxArray, MlxError> {
        self.embed.matmul(hidden_states)
    }

    /// Greedy pick from `[1, hidden]` hidden states: logits matmul + argmax
    /// stay on the GPU; the only host traffic is the 4-byte token id (the
    /// step's explicit eval). mlx argmax tie-breaks to the first index like
    /// `np.argmax`, so picks match the CPU reference semantics.
    fn argmax(&self, hidden_states: &MlxArray) -> Result<u32, MlxPipelineError> {
        let logits = self.logits(hidden_states)?;
        let pick = logits.argmax(-1, false)?;
        Ok(pick.to_u32_vec()?[0])
    }

    /// Full forward over prompt embeds `[seq, hidden]` bf16, filling
    /// `cache`; returns the final-normed hidden states
    /// `[1, seq, hidden]` bf16.
    fn prefill(
        &self,
        embeds: MlxArray,
        cache: &mut MlxKvCache,
    ) -> Result<MlxArray, MlxPipelineError> {
        let hidden = self.cfg.hidden_size;
        if embeds.ndim()? != 2 || embeds.dim(1)? != hidden {
            return Err(MlxPipelineError::Layout(format!(
                "prefill embeds must be rank-2 [seq, {hidden}]"
            )));
        }
        let seq = embeds.dim(0)?;
        let mut x = embeds.reshape(&[1, seq, hidden])?;
        for (i, lw) in self.layers.iter().enumerate() {
            x = self.layer_forward(i, lw, x, cache, Pass::Prefill)?;
        }
        Ok(x.rms_norm(Some(&self.final_norm), self.cfg.rms_norm_eps)?)
    }

    /// Single-token forward at absolute `position`; `embed` is
    /// `[1, hidden]` bf16. Returns the final-normed hidden row
    /// `[1, 1, hidden]` bf16.
    fn step(
        &self,
        embed: &MlxArray,
        position: u32,
        cache: &mut MlxKvCache,
    ) -> Result<MlxArray, MlxPipelineError> {
        let hidden = self.cfg.hidden_size;
        if embed.ndim()? != 2 || embed.dim(1)? != hidden || embed.dim(0)? != 1 {
            return Err(MlxPipelineError::Layout(format!(
                "step embed must be [1, {hidden}]"
            )));
        }
        let mut x = embed.reshape(&[1, 1, hidden])?;
        for (i, lw) in self.layers.iter().enumerate() {
            x = self.layer_forward(i, lw, x, cache, Pass::Step { position })?;
        }
        Ok(x.rms_norm(Some(&self.final_norm), self.cfg.rms_norm_eps)?)
    }

    /// One pre-LN layer. `x` is `[1, seq, hidden]` bf16.
    fn layer_forward(
        &self,
        index: usize,
        lw: &MlxDecoderLayer,
        x: MlxArray,
        cache: &mut MlxKvCache,
        pass: Pass,
    ) -> Result<MlxArray, MlxError> {
        let cfg = &self.cfg;
        let (q_heads, kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let seq = pass.seq_len(&x)?;
        let hidden = cfg.hidden_size;
        let scale = (head_dim as f64).powf(-0.5) as f32;

        // Attention half: RMSNorm -> quantized projections -> per-head
        // q/k RMSNorm -> transpose to head-major -> RoPE -> cache append ->
        // SDPA (causal at prefill, unmasked for single-token steps).
        let normed = x.rms_norm(Some(&lw.input_ln), cfg.rms_norm_eps)?;
        let flat = normed.reshape(&[seq, hidden])?;
        let q = lw.q.matmul(&flat)?.reshape(&[1, seq, q_heads, head_dim])?;
        let k = lw.k.matmul(&flat)?.reshape(&[1, seq, kv_heads, head_dim])?;
        let v = lw.v.matmul(&flat)?.reshape(&[1, seq, kv_heads, head_dim])?;
        let q = q.rms_norm(Some(&lw.q_norm), cfg.rms_norm_eps)?;
        let k = k.rms_norm(Some(&lw.k_norm), cfg.rms_norm_eps)?;
        let q = q.transpose(&[0, 2, 1, 3])?;
        let k = k.transpose(&[0, 2, 1, 3])?;
        let v = v.transpose(&[0, 2, 1, 3])?;
        let offset = pass.rope_offset();
        let q = q.rope(head_dim, cfg.rope_theta, 1.0, offset)?;
        let k = k.rope(head_dim, cfg.rope_theta, 1.0, offset)?;
        cache.append(index, k, v)?;
        let (k_cached, v_cached) = cache.layer(index);
        let attn = q.scaled_dot_product_attention(k_cached, v_cached, scale, pass.mask())?;
        let attn = attn
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, seq, q_heads * head_dim])?;
        let o =
            lw.o.matmul(&attn.reshape(&[seq, q_heads * head_dim])?)?
                .reshape(&[1, seq, hidden])?;
        let x = x.add(&o)?;

        // MLP half: RMSNorm -> gate/up -> SiLU(gate) * up -> down.
        let normed = x.rms_norm(Some(&lw.post_ln), cfg.rms_norm_eps)?;
        let flat = normed.reshape(&[seq, hidden])?;
        let gate = lw.gate.matmul(&flat)?;
        let up = lw.up.matmul(&flat)?;
        let activated = gate.silu()?.multiply(&up)?;
        let out = lw.down.matmul(&activated)?.reshape(&[1, seq, hidden])?;
        x.add(&out)
    }
}

/// Per-layer KV cache of head-major `[1, kv_heads, cached_len, head_dim]`
/// bf16 arrays, grown per append with `concatenate` along axis 2 (the mlx
/// `KVCache.update_and_fetch` layout).
struct MlxKvCache {
    layers: Vec<Option<(MlxArray, MlxArray)>>,
}

impl MlxKvCache {
    fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| None).collect(),
        }
    }

    fn append(&mut self, index: usize, k: MlxArray, v: MlxArray) -> Result<(), MlxError> {
        let slot = &mut self.layers[index];
        *slot = Some(match slot.take() {
            None => (k, v),
            Some((old_k, old_v)) => (
                MlxArray::concatenate(&[&old_k, &k], 2)?,
                MlxArray::concatenate(&[&old_v, &v], 2)?,
            ),
        });
        Ok(())
    }

    /// Cached (k, v) of a layer; every call site appends before attending.
    fn layer(&self, index: usize) -> (&MlxArray, &MlxArray) {
        self.layers[index]
            .as_ref()
            .map(|(k, v)| (k, v))
            .expect("cache slot must be appended before attention")
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A loaded Qwen3-ASR checkpoint ready to transcribe on MLX/Metal.
///
/// Mirrors [`crate::models::qwen3_asr::pipeline::Qwen3AsrPipeline`]:
/// `load(checkpoint_dir)` + `transcribe(wav, TranscriptionRequest)` with the
/// same semantics; the 8-bit retry is a second pipeline on the other
/// checkpoint directory (see [`mlx_transcribe_with_retry`]).
pub struct MlxAsrPipeline {
    tokenizer: Tokenizer,
    tower: MlxTower,
    decoder: MlxDecoder,
    audio_token_id: u32,
    output_dim: usize,
    mel_config: MelConfig,
}

impl std::fmt::Debug for MlxAsrPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxAsrPipeline")
            .field("decoder", &self.decoder.cfg)
            .field("audio_token_id", &self.audio_token_id)
            .field("output_dim", &self.output_dim)
            .finish_non_exhaustive()
    }
}

impl MlxAsrPipeline {
    /// Load a checkpoint snapshot directory (4-bit or 8-bit — the retry is
    /// just a second [`MlxAsrPipeline`] on the other directory).
    pub fn load(checkpoint_dir: &Path) -> Result<Self, MlxPipelineError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| MlxPipelineError::ConfigMismatch(format!("read config.json: {e}")))?;
        let cfg: Qwen3AsrCheckpointConfig = parse_config(&raw)?;
        let thinker = &cfg.thinker_config;
        let tokenizer = Tokenizer::from_checkpoint(checkpoint_dir)?;
        let tower = MlxTower::from_store(&thinker.audio_config, &store)?;
        let decoder = MlxDecoder::from_store(&thinker.text_config, &store)?;
        let mel_config = MelConfig::default();
        if mel_config.n_mels != thinker.audio_config.num_mel_bins {
            return Err(MlxPipelineError::ConfigMismatch(format!(
                "mel config n_mels {} != checkpoint num_mel_bins {}",
                mel_config.n_mels, thinker.audio_config.num_mel_bins
            )));
        }
        Ok(Self {
            tokenizer,
            tower,
            decoder,
            audio_token_id: thinker.audio_token_id,
            output_dim: thinker.audio_config.output_dim,
            mel_config,
        })
    }

    /// Transcribe one 16 kHz mono wav file.
    pub fn transcribe(
        &self,
        wav_path: &Path,
        request: crate::models::qwen3_asr::pipeline::TranscriptionRequest<'_>,
    ) -> Result<String, MlxPipelineError> {
        let language = match request.language {
            None => None,
            Some(input) => Some(resolved_language(input)?),
        };
        let outcome = self.generate(wav_path, language.as_deref(), request.context)?;
        finish_transcription(outcome)
    }

    /// Mel -> tower -> prompt -> greedy generation outcome.
    fn generate(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<GenerationOutcome, MlxPipelineError> {
        let wav = load_wav(wav_path)?;
        let mel = mel_spectrogram(&wav.samples, &self.mel_config)?;
        // The MLX tower consumes the frequency-major [n_mels, frames] layout
        // mel_spectrogram already produces (the CPU tower needs the
        // time-major transpose; here none is required).
        let (features, pads) = self.tower.forward(&mel.data, mel.frames)?;
        debug_assert_eq!(pads, frames_to_audio_pads(mel.frames));
        let prompt = build_asr_prompt(context, language, pads);
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let eos_ids = resolve_eos_ids(&self.tokenizer);
        let stream = self.greedy_token_stream(
            &prompt_ids,
            &features,
            Some(&eos_ids),
            GenerationSettings::default(),
        )?;
        let text = self.tokenizer.decode(&stream.tokens, true)?;
        let loop_detected = detect_repetition_loop(&text);
        Ok(GenerationOutcome {
            tokens: stream.tokens,
            text,
            finish_reason: stream.finish_reason,
            loop_detected,
        })
    }

    /// Greedy token stream over the merged prompt + audio embeds.
    ///
    /// `eos_ids: None` produces the FORCED stream (EOS tokens kept
    /// in-stream until the budget) — the golden fixture's argmax stream
    /// semantics; `Some` mirrors the production eos-aware loop.
    fn greedy_token_stream(
        &self,
        prompt_ids: &[u32],
        features: &[f32],
        eos_ids: Option<&BTreeSet<u32>>,
        settings: GenerationSettings,
    ) -> Result<TokenStream, MlxPipelineError> {
        if prompt_ids.is_empty() {
            return Err(MlxPipelineError::EmptyPrompt);
        }
        let hidden = self.decoder.cfg.hidden_size;
        let n_audio = features.len() / self.output_dim;
        if !features.len().is_multiple_of(self.output_dim) {
            return Err(MlxPipelineError::Layout(
                "tower features are not a multiple of output_dim".to_string(),
            ));
        }

        // Audio features cast f32 -> bf16, exactly `audio_features
        // .astype(inputs_embeds.dtype)` (round-to-nearest-even on both
        // sides).
        let feature_rows: Vec<bf16> = features.iter().map(|&v| bf16::from_f32(v)).collect();

        // Prompt embedding rows (dequantized on device), merged with the
        // feature rows on the host: every `<|audio_pad|>` position takes the
        // next feature row while features remain; surplus pads keep their
        // embedding (`_build_inputs_embeds`).
        let embed_rows = self.decoder.embed_rows(prompt_ids)?;
        let dequantized = embed_rows.to_bf16_vec()?;
        let mut merged: Vec<bf16> = Vec::with_capacity(prompt_ids.len() * hidden);
        let mut audio_idx = 0usize;
        for (row, &id) in prompt_ids.iter().enumerate() {
            if id == self.audio_token_id && audio_idx < n_audio {
                merged.extend_from_slice(
                    &feature_rows[audio_idx * self.output_dim..(audio_idx + 1) * self.output_dim],
                );
                audio_idx += 1;
            } else {
                let src = row * hidden..(row + 1) * hidden;
                merged.extend_from_slice(&dequantized[src]);
            }
        }
        let embeds = MlxArray::from_bf16(&[prompt_ids.len(), hidden], &merged)?;

        // Prefill + first-token argmax at the last prompt position.
        let mut cache = MlxKvCache::new(self.decoder.cfg.num_hidden_layers);
        let final_hidden = self.decoder.prefill(embeds, &mut cache)?;
        let prompt_len = prompt_ids.len();
        let last = final_hidden
            .slice(
                &[0, prompt_len - 1, 0],
                &[1, prompt_len, hidden],
                &[1, 1, 1],
            )?
            .reshape(&[1, hidden])?;
        let mut y = self.decoder.argmax(&last)?;

        // Decode steps: positions continue sequentially; each step's logits
        // readback is the step's explicit eval.
        let mut tokens = Vec::with_capacity(settings.max_new_tokens.min(1024));
        let mut finish_reason = FinishReason::MaxNewTokens;
        loop {
            if let Some(eos) = eos_ids
                && eos.contains(&y)
            {
                finish_reason = FinishReason::Eos;
                break;
            }
            if tokens.len() >= settings.max_new_tokens {
                break;
            }
            tokens.push(y);
            let emb = self.decoder.embed_rows(&[y])?;
            let position = (prompt_len + tokens.len() - 1) as u32;
            let step_hidden = self.decoder.step(&emb, position, &mut cache)?;
            y = self.decoder.argmax(&step_hidden.reshape(&[1, hidden])?)?;
        }

        Ok(TokenStream {
            tokens,
            finish_reason,
        })
    }
}

/// Turn a generation outcome into final transcription text.
fn finish_transcription(outcome: GenerationOutcome) -> Result<String, MlxPipelineError> {
    let text = parse_qwen_output(&outcome.text).trim().to_string();
    if text.is_empty() {
        return Err(MlxPipelineError::EmptyTranscription);
    }
    Ok(text)
}

/// Whether a 4-bit outcome warrants the 8-bit retry (pure, unit-testable).
pub fn should_retry(outcome: &GenerationOutcome) -> bool {
    outcome.loop_detected
}

/// Transcribe with the 4-bit -> 8-bit retry: run `primary`; when the
/// repetition-loop detector trips and a `retry` pipeline is supplied, rerun
/// on it. Retry failure falls back to the primary result (matching the CPU
/// pipeline's best-effort semantics).
pub fn mlx_transcribe_with_retry(
    primary: &MlxAsrPipeline,
    retry: Option<&MlxAsrPipeline>,
    wav_path: &Path,
    request: crate::models::qwen3_asr::pipeline::TranscriptionRequest<'_>,
) -> Result<String, MlxPipelineError> {
    let language = match request.language {
        None => None,
        Some(input) => Some(resolved_language(input)?),
    };
    let outcome = primary.generate(wav_path, language.as_deref(), request.context)?;
    if !should_retry(&outcome) {
        return finish_transcription(outcome);
    }
    match retry {
        Some(p8) => match p8.transcribe(wav_path, request) {
            Ok(text) => Ok(text),
            Err(_) => finish_transcription(outcome),
        },
        None => finish_transcription(outcome),
    }
}

// ---------------------------------------------------------------------------
// Weight import helpers
// ---------------------------------------------------------------------------

/// Loads one unquantized bf16 checkpoint tensor as an mlx array (bit-exact
/// byte import) after dtype + shape validation.
fn mlx_bf16_tensor(
    store: &WeightStore,
    name: &str,
    expected: &[usize],
) -> Result<MlxArray, MlxPipelineError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| MlxPipelineError::Layout(format!("missing checkpoint tensor {name}")))?;
    if format!("{:?}", info.dtype) != "BF16" {
        return Err(MlxPipelineError::Layout(format!(
            "{name}: dtype {:?}, expected BF16 (unquantized tower)",
            info.dtype
        )));
    }
    if info.shape.as_slice() != expected {
        return Err(MlxPipelineError::Layout(format!(
            "{name}: shape {:?}, expected {expected:?}",
            info.shape
        )));
    }
    let bytes = store.tensor_bytes(name)?;
    Ok(MlxArray::from_bytes(bytes, MlxDtype::Bf16, expected)?)
}

/// Loads a plain (unquantized) f16/bf16/f32 checkpoint vector verbatim.
fn mlx_raw_vector(
    store: &WeightStore,
    name: &str,
    len: usize,
) -> Result<MlxArray, MlxPipelineError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| MlxPipelineError::Layout(format!("missing checkpoint tensor {name}")))?;
    let dtype = match format!("{:?}", info.dtype).as_str() {
        "F16" => MlxDtype::F16,
        "BF16" => MlxDtype::Bf16,
        "F32" => MlxDtype::F32,
        other => {
            return Err(MlxPipelineError::Layout(format!(
                "{name}: unsupported norm dtype {other}"
            )));
        }
    };
    if info.shape.as_slice() != [len] {
        return Err(MlxPipelineError::Layout(format!(
            "{name}: shape {:?}, expected [{len}]",
            info.shape
        )));
    }
    let bytes = store.tensor_bytes(name)?;
    Ok(MlxArray::from_bytes(bytes, dtype, &[len])?)
}

// ---------------------------------------------------------------------------
// Geometry helpers (private re-derivations of the formulas locked in the
// encoder module's tests; the pad-count relation itself is the pub
// `frames_to_audio_pads`)
// ---------------------------------------------------------------------------

/// Time-axis output length of one 3x3 stride-2 pad-1 convolution:
/// `floor((t - 1) / 2) + 1` (`t >= 1`).
fn conv_time_out(t: usize) -> usize {
    (t - 1) / 2 + 1
}

/// Mel-bin count surviving the three convolutions (128 -> 16).
fn conv_freq_after_front_end(n_mels: usize) -> usize {
    let mut h = n_mels;
    for _ in 0..3 {
        h = conv_time_out(h);
    }
    h
}

/// Time length after the full conv front-end for one chunk.
fn chunk_time_after_front_end(chunk_len: usize) -> usize {
    conv_time_out(conv_time_out(conv_time_out(chunk_len)))
}

/// Conv chunk lengths over `frames` (chunk size 100 = `n_window * 2`):
/// full chunks first, final remainder unless an exact multiple.
fn mel_chunk_lengths(frames: usize, chunk_size: usize) -> Vec<usize> {
    let count = frames.div_ceil(chunk_size);
    (0..count)
        .map(|j| {
            if j + 1 == count {
                let remainder = frames % chunk_size;
                if remainder == 0 {
                    chunk_size
                } else {
                    remainder
                }
            } else {
                chunk_size
            }
        })
        .collect()
}

/// Attention block ranges partitioning `0..total` into `window`-sized
/// blocks plus a remainder (the reference `cu_chunk_lens` loop).
fn attention_block_ranges(total: usize, window: usize) -> Vec<(usize, usize)> {
    assert!(window >= 1, "attention window must be >= 1");
    let mut blocks = Vec::new();
    let mut covered = 0;
    for _ in 0..total / window {
        blocks.push((covered, covered + window));
        covered += window;
    }
    if !total.is_multiple_of(window) {
        blocks.push((covered, total));
    }
    blocks
}

/// Additive block mask: 0 inside blocks, -1e9 outside (the reference
/// `_create_block_attention_mask`).
fn block_mask_f32(total: usize, blocks: &[(usize, usize)]) -> Vec<f32> {
    let mut mask = vec![-1e9f32; total * total];
    for &(s, e) in blocks {
        for i in s..e {
            for j in s..e {
                mask[i * total + j] = 0.0;
            }
        }
    }
    mask
}

/// Sinusoidal position embedding rows `[seqlen, channels]` f32 (row `p` is
/// `[sin(p * inv_0..), cos(p * inv_0..)]`, `inv_i = exp(-ln(10000) * i /
/// (channels/2 - 1))`) — same f32 recipe as the CPU encoder port.
fn sinusoidal_position_embedding(seqlen: usize, channels: usize) -> Vec<f32> {
    debug_assert!(channels.is_multiple_of(2));
    let half = channels / 2;
    let increment = (10_000f64).ln() / (half - 1) as f64;
    let increment = -(increment as f32);
    let mut inv = vec![0f32; half];
    for (i, inv) in inv.iter_mut().enumerate() {
        *inv = (increment * i as f32).exp();
    }
    let mut out = vec![0f32; seqlen * channels];
    for p in 0..seqlen {
        for i in 0..half {
            let scaled = p as f32 * inv[i];
            out[p * channels + i] = scaled.sin();
            out[p * channels + half + i] = scaled.cos();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    const ASR_REPO_4BIT: &str = "models--mlx-community--Qwen3-ASR-1.7B-4bit";
    const ASR_REPO_8BIT: &str = "models--mlx-community--Qwen3-ASR-1.7B-8bit";

    fn hf_cache() -> PathBuf {
        PathBuf::from(
            std::env::var("AIMD_TEST_HF_CACHE")
                .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
        )
    }

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

    /// Loads the 4-bit MLX pipeline once per process and shares it across
    /// tests (~2 GB import dominates the suite runtime).
    fn shared_pipeline() -> Option<&'static MlxAsrPipeline> {
        static PIPELINE: OnceLock<Option<MlxAsrPipeline>> = OnceLock::new();
        PIPELINE
            .get_or_init(|| {
                let Some(dir) = asr_4bit_snapshot() else {
                    eprintln!("skipping: {ASR_REPO_4BIT} not in HF cache");
                    return None;
                };
                match MlxAsrPipeline::load(&dir) {
                    Ok(pipeline) => Some(pipeline),
                    Err(e) => {
                        eprintln!("skipping: MLX pipeline load failed: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden")
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect()
    }

    // ---------- pure helpers ----------

    #[test]
    fn mlx_backend_geometry_helpers_match_encoder_formulas() {
        assert_eq!(conv_time_out(100), 50);
        assert_eq!(conv_time_out(50), 25);
        assert_eq!(conv_time_out(25), 13);
        assert_eq!(conv_time_out(1), 1);
        assert_eq!(conv_freq_after_front_end(128), 16);
        assert_eq!(chunk_time_after_front_end(100), 13);
        assert_eq!(chunk_time_after_front_end(61), 8);
        assert_eq!(mel_chunk_lengths(150, 100), vec![100, 50]);
        assert_eq!(mel_chunk_lengths(100, 100), vec![100]);
        assert_eq!(mel_chunk_lengths(61, 100), vec![61]);
        assert_eq!(attention_block_ranges(117, 104), vec![(0, 104), (104, 117)]);
        assert_eq!(attention_block_ranges(104, 104), vec![(0, 104)]);
        // The pad-count relation is re-exported from the encoder module.
        assert_eq!(frames_to_audio_pads(150), 20);
        assert_eq!(frames_to_audio_pads(900), 117);
    }

    #[test]
    fn mlx_backend_block_mask_matches_reference_layout() {
        let mask = block_mask_f32(5, &[(0, 2), (2, 5)]);
        for i in 0..5 {
            for j in 0..5 {
                let expected = if (i < 2) == (j < 2) { 0.0 } else { -1e9 };
                assert_eq!(mask[i * 5 + j], expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn mlx_backend_sinusoidal_embedding_known_values() {
        let table = sinusoidal_position_embedding(3, 8);
        assert_eq!(table.len(), 24);
        // Row 0: sin 0 / cos 1; row 1, i=0: sin(1) / cos(1).
        assert!(table[..4].iter().all(|v| v.abs() < 1e-7));
        assert!((table[4..8].iter().all(|v| (v - 1.0).abs() < 1e-7)));
        assert!((table[8] - 1.0f32.sin()).abs() < 1e-6);
        assert!((table[12] - 1.0f32.cos()).abs() < 1e-6);
        // Prefix property.
        let short = sinusoidal_position_embedding(2, 8);
        assert_eq!(&table[..16], &short[..]);
    }

    // ---------- error paths ----------

    #[test]
    fn mlx_backend_missing_checkpoint_reports_typed_error() {
        let err = MlxAsrPipeline::load(Path::new("/definitely/not/here-aimd-mlx"))
            .expect_err("missing checkpoint must fail");
        assert!(
            matches!(err, MlxPipelineError::Weights(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn mlx_backend_language_and_prompt_guards() {
        // These guards fire before any mlx work, so no checkpoint is needed.
        assert!(matches!(
            resolved_language("xx"),
            Err(MlxPipelineError::UnsupportedLanguage(_))
        ));
        assert_eq!(resolved_language("zh").unwrap(), "Chinese");
    }

    #[test]
    fn mlx_backend_finish_transcription_parses_and_rejects_empty() {
        let outcome = GenerationOutcome {
            tokens: vec![1, 2],
            text: "language Chinese<asr_text>你好".to_string(),
            finish_reason: FinishReason::Eos,
            loop_detected: false,
        };
        assert_eq!(finish_transcription(outcome.clone()).unwrap(), "你好");
        assert!(!should_retry(&outcome));
        let looping = GenerationOutcome {
            loop_detected: true,
            ..outcome
        };
        assert!(should_retry(&looping));
        let empty = GenerationOutcome {
            tokens: vec![],
            text: String::new(),
            finish_reason: FinishReason::Eos,
            loop_detected: false,
        };
        assert!(matches!(
            finish_transcription(empty.clone()),
            Err(MlxPipelineError::EmptyTranscription)
        ));
        assert!(!should_retry(&empty));
    }

    // ---------- checkpoint-backed parity ----------

    /// Tower parity on the committed golden case: the MLX tower (conv via
    /// mlx conv2d, attention via mlx SDPA) vs the mlx-audio-exported
    /// `stage_final_projection` reference, gated with the same tolerances as
    /// the CPU encoder goldens (atol 1e-3 / rtol 1e-2).
    #[test]
    fn mlx_backend_tower_matches_golden_final_projection() {
        const GOLDEN_ATOL: f64 = 1e-3;
        const GOLDEN_RTOL: f64 = 1e-2;
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let case = golden_dir().join("asr_encoder/cases/sine_1_5s");
        if !case.is_dir() {
            eprintln!("skipping: golden encoder case absent");
            return;
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(case.join("meta.json")).unwrap()).unwrap();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let pads = meta["audio_pad_count"].as_u64().unwrap() as usize;
        let mel = read_f32(&case.join("mel.f32")); // time-major [frames, n_mels]
        assert_eq!(mel.len(), frames * pipeline.mel_config.n_mels);
        // The MLX tower consumes the frequency-major [n_mels, frames] layout
        // mel_spectrogram emits (the golden fixture stores the CPU-port's
        // time-major input).
        let n_mels = pipeline.mel_config.n_mels;
        let mut mel_fm = vec![0f32; mel.len()];
        for t in 0..frames {
            for f in 0..n_mels {
                mel_fm[f * frames + t] = mel[t * n_mels + f];
            }
        }

        let (features, got_pads) = pipeline.tower.forward(&mel_fm, frames).unwrap();
        assert_eq!(got_pads, pads);
        let reference = read_f32(&case.join("stage_final_projection.f32"));
        assert_eq!(features.len(), reference.len());
        let mut max_abs = 0f64;
        let mut violations = 0usize;
        for (&a, &r) in features.iter().zip(&reference) {
            let (a, r) = (f64::from(a), f64::from(r));
            let abs = (a - r).abs();
            let rel = abs / r.abs().max(f64::MIN_POSITIVE);
            max_abs = max_abs.max(abs);
            if abs > GOLDEN_ATOL && rel > GOLDEN_RTOL {
                violations += 1;
            }
        }
        eprintln!("mlx tower golden: max_abs={max_abs:.3e} violations={violations}");
        assert_eq!(violations, 0, "tower output beyond the golden gate");
    }

    /// Decoder parity: the forced 16-step greedy argmax stream over the
    /// golden synthetic input must match the mlx-audio export token for
    /// token (both sides run the same quantized bf16 Metal graph).
    #[test]
    fn mlx_backend_decoder_golden_argmax_stream() {
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let Ok(raw) = std::fs::read(golden_dir().join("asr_decoder/manifest.json")) else {
            eprintln!("skipping: golden decoder fixture absent");
            return;
        };
        let manifest: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let parse_ids = |key: &str| -> Vec<u32> {
            manifest[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        let prompt_ids = parse_ids("prompt_ids");
        let expected = parse_ids("greedy_tokens");
        let features = read_f32(&golden_dir().join("asr_decoder/features.f32"));

        let stream = pipeline
            .greedy_token_stream(
                &prompt_ids,
                &features,
                None,
                GenerationSettings { max_new_tokens: 16 },
            )
            .expect("forced argmax stream");
        assert_eq!(
            stream.tokens, expected,
            "forced argmax stream must match MLX"
        );
        assert_eq!(stream.finish_reason, FinishReason::MaxNewTokens);
    }

    // ---------- end-to-end (macOS TTS bench clips) ----------

    /// Generates `/tmp/aimd_bench_{zh,en}.wav` via macOS `say` + ffmpeg when
    /// absent. Returns false when the tools are unavailable.
    fn ensure_bench_clips() -> bool {
        let ok = |s: std::io::Result<std::process::ExitStatus>| {
            s.map(|st| st.success()).unwrap_or(false)
        };
        let zh = PathBuf::from("/tmp/aimd_bench_zh.wav");
        let en = PathBuf::from("/tmp/aimd_bench_en.wav");
        if zh.is_file() && en.is_file() {
            return true;
        }
        let tmp = std::env::temp_dir();
        let aiff_zh = tmp.join("aimd_bench_zh.aiff");
        let aiff_en = tmp.join("aimd_bench_en.aiff");
        if !zh.is_file() {
            if !ok(std::process::Command::new("say")
                .args(["-v", "Eddy (Chinese (China mainland))", "-o"])
                .arg(&aiff_zh)
                .arg("今天天气真好，上下文工程让一切变得更简单。")
                .status())
            {
                return false;
            }
            if !ok(ffmpeg_to_wav(&aiff_zh, &zh)) {
                return false;
            }
        }
        if !en.is_file() {
            if !ok(std::process::Command::new("say")
                .args(["-o"])
                .arg(&aiff_en)
                .arg("Today's weather is great, and context engineering just got simpler.")
                .status())
            {
                return false;
            }
            if !ok(ffmpeg_to_wav(&aiff_en, &en)) {
                return false;
            }
        }
        zh.is_file() && en.is_file()
    }

    fn ffmpeg_to_wav(input: &Path, output: &Path) -> std::io::Result<std::process::ExitStatus> {
        std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(input)
            .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(output)
            .status()
    }

    /// E2E on the recorded bench clips: the transcription text must match
    /// the CPU backend outputs recorded in docs/performance.md (zh includes
    /// the 上下文忠诚 mishearing; en the context-engineering sentence).
    /// Regenerated clips only assert non-empty output.
    #[test]
    fn mlx_backend_transcribes_bench_clips() {
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let recorded = PathBuf::from("/tmp/aimd_bench_zh.wav").is_file()
            && PathBuf::from("/tmp/aimd_bench_en.wav").is_file();
        if !ensure_bench_clips() {
            eprintln!("skipping: bench clips unavailable and say/ffmpeg generation failed");
            return;
        }
        let zh = pipeline
            .transcribe(
                Path::new("/tmp/aimd_bench_zh.wav"),
                crate::models::qwen3_asr::pipeline::TranscriptionRequest {
                    language: Some("zh"),
                    context: None,
                },
            )
            .expect("zh transcription");
        eprintln!("zh: {zh}");
        assert!(!zh.trim().is_empty());
        if recorded {
            assert!(zh.contains("上下文忠诚"), "zh text drifted: {zh}");
        }

        let en = pipeline
            .transcribe(
                Path::new("/tmp/aimd_bench_en.wav"),
                crate::models::qwen3_asr::pipeline::TranscriptionRequest {
                    language: Some("en"),
                    context: None,
                },
            )
            .expect("en transcription");
        eprintln!("en: {en}");
        assert!(!en.trim().is_empty());
        if recorded {
            assert!(
                en.contains("context engineering just got simpler"),
                "en text drifted: {en}"
            );
        }
    }

    /// A pure 440 Hz sine is noise to the model: it must emit EOS
    /// immediately and the pipeline must report `EmptyTranscription`
    /// (every stage runs and composes).
    #[test]
    fn mlx_backend_sine_fixture_reports_empty_transcription() {
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let samples: Vec<f32> = (0..24000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let tmp = std::env::temp_dir().join("aimd_mlx_pipeline_sine.wav");
        write_s16_wav(&tmp, &samples);
        let err = pipeline
            .transcribe(
                &tmp,
                crate::models::qwen3_asr::pipeline::TranscriptionRequest::default(),
            )
            .expect_err("noise input must yield EmptyTranscription");
        assert!(matches!(err, MlxPipelineError::EmptyTranscription), "{err}");
    }

    /// The 8-bit retry checkpoint loads through the same path.
    #[test]
    fn mlx_backend_loads_8bit_checkpoint() {
        let Some(dir) = snapshot_dir(ASR_REPO_8BIT, &["config.json", "model.safetensors"]) else {
            eprintln!("skipping: {ASR_REPO_8BIT} not in HF cache");
            return;
        };
        let pipeline = MlxAsrPipeline::load(&dir).expect("8-bit checkpoint loads");
        assert_eq!(pipeline.decoder.cfg.num_hidden_layers, 28);
        assert_eq!(pipeline.decoder.embed.bits, 8);
    }

    fn write_s16_wav(path: &Path, samples: &[f32]) {
        let mut data = Vec::with_capacity(44 + samples.len() * 2);
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + samples.len() * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVEfmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&16000u32.to_le_bytes());
        data.extend_from_slice(&32000u32.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((samples.len() * 2) as u32).to_le_bytes());
        for s in samples {
            let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            data.extend_from_slice(&pcm.to_le_bytes());
        }
        std::fs::write(path, data).expect("write wav");
    }
}
