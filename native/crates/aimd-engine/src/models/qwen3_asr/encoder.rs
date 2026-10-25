//! Qwen3-ASR audio tower (encoder) — WS-06a.
//!
//! Faithful CPU port of `mlx_audio.stt.models.qwen3_asr.qwen3_asr.AudioEncoder`
//! (the porting source of truth): 128-bin log-mel in, `[pads, output_dim]`
//! audio features out, ready for the WS-06b text decoder.
//!
//! # Pipeline (single sequence; the inference path is batch 1)
//!
//! 1. The mel time axis is cut into chunks of `n_window * 2` frames (100).
//!    Each chunk runs through the conv front-end **independently** — the conv
//!    never sees across chunk boundaries. The reference zero-pads every chunk
//!    to the longest chunk length, convolves the stacked batch, and trims
//!    each row to its own output length; the port reproduces that pad/trim
//!    exactly (it is not skippable: the conv biases + GELU make the padded
//!    tail rows nonzero, and they leak into the last valid rows of the next
//!    convolution).
//! 2. Per chunk: the conv output `[freq', time', 480]` (NHWC, MLX layout) is
//!    flattened time-major with index `c * freq' + f` (the
//!    `transpose(0, 2, 3, 1).reshape(b, t, c * f)` of the reference), then
//!    `conv_out` (no bias) projects to `d_model`, and the sinusoidal
//!    position embedding (which restarts at position 0 for every chunk) is
//!    added.
//! 3. Chunk outputs are concatenated into the encoder stream of
//!    `frames_to_audio_pads(frames)` rows.
//! 4. Attention is block-diagonal: the stream is split into blocks of
//!    `max_chunk_len_after_cnn * (n_window_infer / (n_window * 2))` rows
//!    (104 for the 1.7B checkpoint) and each block attends only within
//!    itself. The reference implements this as an additive `-1e9` mask; the
//!    masked scores underflow `exp` to exactly zero in f32, so attending
//!    block-by-block is bit-equivalent (`encoder_attention_blocks_*`).
//! 5. 24 pre-norm encoder layers: biased LayerNorm (eps 1e-5) -> MHA
//!    (biased q/k/v/out projections, query scaled by `head_dim^-0.5`) ->
//!    residual, then biased LayerNorm -> fc1 -> exact-erf GELU -> fc2 ->
//!    residual. Finally `ln_post` -> GELU(`proj1`) -> `proj2`.
//!
//! # Numeric contract
//!
//! MLX runs the tower with **float32 activations**: the mel input is f32 and
//! every bf16 weight is promoted per op, so this port stores the bf16
//! weights verbatim (loaded through [`aimd_weights::WeightStore`] raw tensor
//! access) and upcasts to f32 inside the inner loops. All accumulation is
//! f32; there is exactly one rounding per MLX op output because both sides
//! round to f32. The golden parity bound is documented on the tests below
//! and in `native/golden/asr_encoder/manifest.json`.
//!
//! The tower tensors are UNQUANTIZED bf16 (`audio_tower.*` has no
//! `.scales`/`.biases` triples — only the 4-bit text decoder is quantized).
//!
//! # Frames -> audio pad tokens
//!
//! [`frames_to_audio_pads`] is the exact
//! `_get_feat_extract_output_lengths` relation used by the mlx-audio
//! preprocessing to size the `<|audio_pad|>` prompt block; it is locked
//! against the committed tokenizer fixtures (`audio_pad_count`) and the
//! encoder goldens.

use aimd_weights::{WeightStore, WeightsError};
use half::bf16;

use super::config::AudioTowerConfig;

/// LayerNorm epsilon of mlx `nn.LayerNorm` (the default used by the tower).
const LAYER_NORM_EPS: f32 = 1e-5;

/// Chunk length of the conv front-end: `n_window * 2`, hard-coded as `100`
/// inside mlx-audio's `_get_feat_extract_output_lengths`.
const CONV_CHUNK_FRAMES: usize = 100;

/// After-cnn frames produced by one full 100-frame chunk (50 -> 25 -> 13).
const FULL_CHUNK_AFTER_CNN: usize = 13;

/// Errors surfaced by the audio tower.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// A checkpoint tensor is missing, has the wrong dtype, or the wrong
    /// shape.
    #[error("invalid audio-tower tensor {name}: {reason}")]
    InvalidTensor { name: String, reason: String },
    /// An input rejected before any compute (empty mel, length mismatch).
    #[error("invalid audio-tower input: {0}")]
    InvalidInput(String),
    /// A configuration the tower cannot honor (unsupported activation,
    /// degenerate windowing).
    #[error("unsupported audio-tower configuration: {0}")]
    Config(String),
    /// Propagated checkpoint I/O or parse failure.
    #[error(transparent)]
    Weights(#[from] WeightsError),
}

/// Stage activations of a tower forward pass, handed to the
/// [`AudioTower::forward_traced`] callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Hidden states entering the encoder layers (post conv front-end +
    /// `conv_out` + positional embedding + per-chunk trim): `[pads, d_model]`.
    ConvFrontEnd,
    /// Hidden states after encoder layer `index`: `[pads, d_model]`.
    Layer(usize),
    /// Tower output after `ln_post` -> GELU(`proj1`) -> `proj2`:
    /// `[pads, output_dim]`.
    FinalProjection,
}

/// Output of one audio-tower forward pass.
#[derive(Clone, Debug)]
pub struct AudioTowerOutput {
    /// Row-major `[pads, output_dim]` f32 features for the decoder.
    pub features: Vec<f32>,
    /// Feature rows; equals [`frames_to_audio_pads`] of the input frames.
    pub pads: usize,
    /// Inclusive-exclusive attention block ranges over the stream (parity
    /// and debug visibility of the windowed-attention split).
    pub attention_blocks: Vec<(usize, usize)>,
}

/// One encoder layer's weights, all bf16 verbatim from the checkpoint.
#[derive(Clone, Debug)]
struct EncoderLayerWeights {
    q_w: Vec<bf16>,
    q_b: Vec<bf16>,
    k_w: Vec<bf16>,
    k_b: Vec<bf16>,
    v_w: Vec<bf16>,
    v_b: Vec<bf16>,
    o_w: Vec<bf16>,
    o_b: Vec<bf16>,
    attn_ln_w: Vec<bf16>,
    attn_ln_b: Vec<bf16>,
    fc1_w: Vec<bf16>,
    fc1_b: Vec<bf16>,
    fc2_w: Vec<bf16>,
    fc2_b: Vec<bf16>,
    final_ln_w: Vec<bf16>,
    final_ln_b: Vec<bf16>,
}

/// The full audio tower: conv front-end + 24 windowed encoder layers +
/// projection head.
#[derive(Clone, Debug)]
pub struct AudioTower {
    cfg: AudioTowerConfig,
    conv2d1_w: Vec<bf16>,
    conv2d1_b: Vec<bf16>,
    conv2d2_w: Vec<bf16>,
    conv2d2_b: Vec<bf16>,
    conv2d3_w: Vec<bf16>,
    conv2d3_b: Vec<bf16>,
    conv_out_w: Vec<bf16>,
    layers: Vec<EncoderLayerWeights>,
    ln_post_w: Vec<bf16>,
    ln_post_b: Vec<bf16>,
    proj1_w: Vec<bf16>,
    proj1_b: Vec<bf16>,
    proj2_w: Vec<bf16>,
    proj2_b: Vec<bf16>,
    /// `d_model / encoder_attention_heads`.
    head_dim: usize,
    /// `head_dim^-0.5`, applied to the projected queries (exact power of
    /// two for every supported head size).
    q_scale: f32,
}

impl AudioTower {
    /// Loads the tower from a checkpoint [`WeightStore`].
    ///
    /// Every `audio_tower.*` tensor is validated for dtype (BF16) and shape
    /// before any compute; a mismatch is a typed
    /// [`EncoderError::InvalidTensor`], so a checkpoint revision that drifts
    /// structurally fails fast instead of producing garbage.
    pub fn from_weight_store(
        cfg: &AudioTowerConfig,
        store: &WeightStore,
    ) -> Result<Self, EncoderError> {
        if cfg.activation_function != "gelu" {
            return Err(EncoderError::Config(format!(
                "activation_function {:?}, only exact-erf \"gelu\" is implemented",
                cfg.activation_function
            )));
        }
        if cfg.d_model == 0
            || cfg.encoder_attention_heads == 0
            || !cfg.d_model.is_multiple_of(cfg.encoder_attention_heads)
        {
            return Err(EncoderError::Config(format!(
                "d_model {} not divisible by encoder_attention_heads {}",
                cfg.d_model, cfg.encoder_attention_heads
            )));
        }
        if cfg.infer_window() < cfg.n_window * 2 {
            return Err(EncoderError::Config(format!(
                "n_window_infer {} < 2 * n_window {} would collapse attention windowing",
                cfg.infer_window(),
                cfg.n_window * 2
            )));
        }
        if cfg.encoder_layers == 0 {
            return Err(EncoderError::Config(
                "encoder_layers must be >= 1".to_string(),
            ));
        }

        let head_dim = cfg.d_model / cfg.encoder_attention_heads;
        let freq_after_conv = conv_freq_after_front_end(cfg.num_mel_bins);
        let conv_in = cfg.downsample_hidden_size * freq_after_conv;

        // Conv weights are MLX NHWC `[o, kh, kw, c]`; bias `[o]`.
        let conv2d1_w = bf16_tensor(
            store,
            "audio_tower.conv2d1.weight",
            &[cfg.downsample_hidden_size, 3, 3, 1],
        )?;
        let conv2d1_b = bf16_tensor(
            store,
            "audio_tower.conv2d1.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let conv2d2_w = bf16_tensor(
            store,
            "audio_tower.conv2d2.weight",
            &[cfg.downsample_hidden_size, 3, 3, cfg.downsample_hidden_size],
        )?;
        let conv2d2_b = bf16_tensor(
            store,
            "audio_tower.conv2d2.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let conv2d3_w = bf16_tensor(
            store,
            "audio_tower.conv2d3.weight",
            &[cfg.downsample_hidden_size, 3, 3, cfg.downsample_hidden_size],
        )?;
        let conv2d3_b = bf16_tensor(
            store,
            "audio_tower.conv2d3.bias",
            &[cfg.downsample_hidden_size],
        )?;
        let conv_out_w = bf16_tensor(
            store,
            "audio_tower.conv_out.weight",
            &[cfg.d_model, conv_in],
        )?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for index in 0..cfg.encoder_layers {
            layers.push(load_layer(store, index, cfg.d_model, cfg.encoder_ffn_dim)?);
        }

        let ln_post_w = bf16_tensor(store, "audio_tower.ln_post.weight", &[cfg.d_model])?;
        let ln_post_b = bf16_tensor(store, "audio_tower.ln_post.bias", &[cfg.d_model])?;
        let proj1_w = bf16_tensor(
            store,
            "audio_tower.proj1.weight",
            &[cfg.d_model, cfg.d_model],
        )?;
        let proj1_b = bf16_tensor(store, "audio_tower.proj1.bias", &[cfg.d_model])?;
        let proj2_w = bf16_tensor(
            store,
            "audio_tower.proj2.weight",
            &[cfg.output_dim, cfg.d_model],
        )?;
        let proj2_b = bf16_tensor(store, "audio_tower.proj2.bias", &[cfg.output_dim])?;

        Ok(Self {
            cfg: cfg.clone(),
            conv2d1_w,
            conv2d1_b,
            conv2d2_w,
            conv2d2_b,
            conv2d3_w,
            conv2d3_b,
            conv_out_w,
            layers,
            ln_post_w,
            ln_post_b,
            proj1_w,
            proj1_b,
            proj2_w,
            proj2_b,
            head_dim,
            q_scale: (head_dim as f32).sqrt().recip(),
        })
    }

    /// Tower configuration.
    pub fn config(&self) -> &AudioTowerConfig {
        &self.cfg
    }

    /// Runs the tower over a log-mel spectrogram.
    ///
    /// `mel` is row-major `[frames, num_mel_bins]` f32 (time-major; the
    /// `audio` crate emits frequency-major, so the pipeline transposes). It
    /// must contain exactly the true (mask-covered) frames: any
    /// multiple-of-`2 * n_window` zero padding applied by the feature
    /// extractor must be sliced off by the caller, because the pad count is
    /// derived from the true frame count only.
    pub fn forward(&self, mel: &[f32], frames: usize) -> Result<AudioTowerOutput, EncoderError> {
        self.forward_traced(mel, frames, &mut |_, _| {})
    }

    /// [`AudioTower::forward`] with a per-stage activation callback
    /// (golden parity harness). The callback receives row-major stage
    /// buffers; stages are produced in graph order.
    pub fn forward_traced(
        &self,
        mel: &[f32],
        frames: usize,
        trace: &mut dyn FnMut(Stage, &[f32]),
    ) -> Result<AudioTowerOutput, EncoderError> {
        let n_mels = self.cfg.num_mel_bins;
        let d = self.cfg.d_model;
        if frames == 0 {
            return Err(EncoderError::InvalidInput(
                "frames must be >= 1".to_string(),
            ));
        }
        if mel.len() != frames * n_mels {
            return Err(EncoderError::InvalidInput(format!(
                "mel holds {} values, expected frames ({frames}) * num_mel_bins ({n_mels})",
                mel.len()
            )));
        }

        // 1) Frequency-major view [n_mels, frames] for the conv front-end.
        let mut mel_fm = vec![0f32; frames * n_mels];
        for t in 0..frames {
            for f in 0..n_mels {
                mel_fm[f * frames + t] = mel[t * n_mels + f];
            }
        }

        // 2) Conv chunks (zero-padded to the longest chunk, trimmed back to
        //    the true chunk's output length), conv_out, per-chunk positional
        //    embedding; concatenate the kept rows.
        let chunk_size = self.cfg.n_window * 2;
        let chunk_lengths = mel_chunk_lengths(frames, chunk_size);
        let max_chunk_len = chunk_lengths.iter().copied().max().unwrap_or(0);
        let mut hidden: Vec<f32> = Vec::new();
        let mut start = 0;
        for &clen in &chunk_lengths {
            let rows = self.conv_chunk(&mel_fm, frames, clen, max_chunk_len, start);
            hidden.extend_from_slice(&rows);
            start += clen;
        }
        let total = hidden.len() / d;
        trace(Stage::ConvFrontEnd, &hidden);

        // 3) Block-diagonal attention ranges.
        let window = self.attention_window_for(&chunk_lengths);
        let blocks = attention_block_ranges(total, window);

        // 4) Encoder layers.
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = self.encoder_layer(layer, hidden, &blocks);
            trace(Stage::Layer(index), &hidden);
        }

        // 5) Projection head: ln_post -> proj1 -> GELU -> proj2.
        let out = layer_norm_rows(&hidden, &self.ln_post_w, &self.ln_post_b, d);
        let out = linear(&out, &self.proj1_w, Some(&self.proj1_b), d, d);
        let mut out = out;
        gelu_in_place(&mut out);
        let out = linear(
            &out,
            &self.proj2_w,
            Some(&self.proj2_b),
            self.cfg.output_dim,
            d,
        );
        trace(Stage::FinalProjection, &out);

        Ok(AudioTowerOutput {
            pads: total,
            features: out,
            attention_blocks: blocks,
        })
    }

    /// Per-forward attention window: `max chunk-after-cnn length *
    /// (n_window_infer / chunk_size)`, mirroring the reference.
    fn attention_window_for(&self, chunk_lengths: &[usize]) -> usize {
        let max_chunk_after_cnn = chunk_lengths
            .iter()
            .map(|&len| chunk_time_after_front_end(len))
            .max()
            .unwrap_or(0);
        max_chunk_after_cnn * (self.cfg.infer_window() / (self.cfg.n_window * 2))
    }

    /// Conv front-end + `conv_out` + positional embedding for one mel chunk.
    ///
    /// `mel_fm` is the frequency-major `[num_mel_bins, frames]` buffer and
    /// `start..start + clen` the chunk's true time span. The conv runs on the
    /// chunk **padded with zeros to `max_chunk_len`** and the result is
    /// trimmed back to the true chunk's output length
    /// `chunk_time_after_front_end(clen)` — exactly the reference graph. The
    /// pad/trim is NOT skippable: every conv has a bias and every stage is
    /// followed by GELU, so the padded tail produces `gelu(bias)` rows that
    /// leak into the last valid rows of the next convolution's receptive
    /// field. Returns the chunk's encoder-input rows, flattened
    /// `[valid_len, d_model]` (positions restart at 0).
    fn conv_chunk(
        &self,
        mel_fm: &[f32],
        frames: usize,
        clen: usize,
        max_chunk_len: usize,
        start: usize,
    ) -> Vec<f32> {
        let n_mels = self.cfg.num_mel_bins;
        let channels = self.cfg.downsample_hidden_size;

        // Zero-padded NHWC chunk [n_mels, max_chunk_len, 1].
        let mut x = vec![0f32; n_mels * max_chunk_len];
        for f in 0..n_mels {
            x[f * max_chunk_len..f * max_chunk_len + clen]
                .copy_from_slice(&mel_fm[f * frames + start..f * frames + start + clen]);
        }

        x = conv2d_3x3_s2_p1(
            &x,
            n_mels,
            max_chunk_len,
            1,
            &self.conv2d1_w,
            &self.conv2d1_b,
            channels,
        );
        gelu_in_place(&mut x);
        let (h1, w1) = (conv_time_out(n_mels), conv_time_out(max_chunk_len));
        x = conv2d_3x3_s2_p1(
            &x,
            h1,
            w1,
            channels,
            &self.conv2d2_w,
            &self.conv2d2_b,
            channels,
        );
        gelu_in_place(&mut x);
        let (h2, w2) = (conv_time_out(h1), conv_time_out(w1));
        x = conv2d_3x3_s2_p1(
            &x,
            h2,
            w2,
            channels,
            &self.conv2d3_w,
            &self.conv2d3_b,
            channels,
        );
        gelu_in_place(&mut x);
        let (h3, w3) = (conv_time_out(h2), conv_time_out(w2));

        // Flatten time-major with index `c * freq' + f`, then conv_out.
        let flat_in = channels * h3;
        let mut flat = vec![0f32; w3 * flat_in];
        for j in 0..w3 {
            for c in 0..channels {
                for f in 0..h3 {
                    flat[j * flat_in + c * h3 + f] = x[(f * w3 + j) * channels + c];
                }
            }
        }
        let rows = linear(&flat, &self.conv_out_w, None, self.cfg.d_model, flat_in);

        // Positional embedding restarts at 0 for every chunk; the reference
        // adds it over the padded length and trims with the rows, which is
        // the same as adding it to the kept prefix only.
        let pos = sinusoidal_position_embedding(w3, self.cfg.d_model);
        let mut rows = rows;
        for (value, p) in rows.iter_mut().zip(pos) {
            *value += p;
        }

        // Trim the padded tail: keep the true chunk's output rows.
        let valid = chunk_time_after_front_end(clen);
        rows.truncate(valid * self.cfg.d_model);
        rows
    }

    /// One pre-norm encoder layer with block-diagonal attention.
    fn encoder_layer(
        &self,
        layer: &EncoderLayerWeights,
        hidden: Vec<f32>,
        blocks: &[(usize, usize)],
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let total = hidden.len() / d;
        let heads = self.cfg.encoder_attention_heads;

        // Pre-norm attention half.
        let normed = layer_norm_rows(&hidden, &layer.attn_ln_w, &layer.attn_ln_b, d);
        let mut q = linear(&normed, &layer.q_w, Some(&layer.q_b), d, d);
        for value in &mut q {
            *value *= self.q_scale;
        }
        let k = linear(&normed, &layer.k_w, Some(&layer.k_b), d, d);
        let v = linear(&normed, &layer.v_w, Some(&layer.v_b), d, d);

        let mut attn = vec![0f32; total * d];
        let mut scores = vec![0f32; total.max(1)];
        for &(s, e) in blocks {
            block_attention(
                &q,
                &k,
                &v,
                &mut attn,
                &mut scores,
                s,
                e,
                heads,
                self.head_dim,
            );
        }
        let attn_out = linear(&attn, &layer.o_w, Some(&layer.o_b), d, d);

        let mut hidden = hidden;
        add_in_place(&mut hidden, &attn_out);

        // Pre-norm feed-forward half.
        let h = layer_norm_rows(&hidden, &layer.final_ln_w, &layer.final_ln_b, d);
        let h = linear(
            &h,
            &layer.fc1_w,
            Some(&layer.fc1_b),
            self.cfg.encoder_ffn_dim,
            d,
        );
        let mut h = h;
        gelu_in_place(&mut h);
        let h = linear(
            &h,
            &layer.fc2_w,
            Some(&layer.fc2_b),
            d,
            self.cfg.encoder_ffn_dim,
        );
        add_in_place(&mut hidden, &h);
        hidden
    }
}

// ---------------------------------------------------------------------------
// Numeric helpers (private; aimd-tensor ops are f16-only and the tower needs
// biased LayerNorm, exact-erf GELU and bf16-weight linears)
// ---------------------------------------------------------------------------

/// Weighted sum of an f32 slice and a bf16 row (checkpoint weights are
/// upcast per element; f32 accumulate, sequential order — deterministic).
#[inline]
fn dot_f32_bf16(x: &[f32], w: &[bf16]) -> f32 {
    let mut acc = 0f32;
    for (&a, &b) in x.iter().zip(w.iter()) {
        acc += a * b.to_f32();
    }
    acc
}

/// 2-D convolution with a 3x3 kernel, stride 2, padding 1 over an NHWC
/// input (`x`: `[h, w, c]`; `weight`: MLX `[o, kh, kw, c]`; `bias`: `[o]`),
/// returning `[h_out, w_out, o]` with
/// `h_out = floor((h - 1) / 2) + 1` (same formula per axis). Zero padding on
/// every edge; f32 accumulate over upcast bf16 weights.
///
/// Parallel over flattened `(h_out, o)` tasks; each output element has one
/// fixed serial accumulation order (di, dj, c), so the split never changes
/// results.
fn conv2d_3x3_s2_p1(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    weight: &[bf16],
    bias: &[bf16],
    o: usize,
) -> Vec<f32> {
    debug_assert!(h >= 1 && w >= 1 && c >= 1 && o >= 1);
    debug_assert_eq!(weight.len(), o * 9 * c);
    debug_assert_eq!(bias.len(), o);
    debug_assert_eq!(x.len(), h * w * c);
    let h_out = conv_time_out(h);
    let w_out = conv_time_out(w);
    let mut out = vec![0f32; h_out * w_out * o];
    par_rows(&mut out, h_out, &|i, out_row| {
        for j in 0..w_out {
            for ch in 0..o {
                let mut acc = bias[ch].to_f32();
                for di in 0..3 {
                    let ii = 2 * i + di; // input row is ii - 1 after padding
                    if ii == 0 || ii > h {
                        continue;
                    }
                    for dj in 0..3 {
                        let jj = 2 * j + dj;
                        if jj == 0 || jj > w {
                            continue;
                        }
                        let x_row =
                            &x[((ii - 1) * w + (jj - 1)) * c..((ii - 1) * w + (jj - 1)) * c + c];
                        let w_row =
                            &weight[((ch * 3 + di) * 3 + dj) * c..((ch * 3 + di) * 3 + dj) * c + c];
                        acc += dot_f32_bf16(x_row, w_row);
                    }
                }
                out_row[j * o + ch] = acc;
            }
        }
    });
    out
}

/// Row-major linear layer over bf16 weights: `out[r, o] =
/// bias[o] + sum_i x[r, i] * w[o, i]` (MLX `nn.Linear` layout
/// `[out_features, in_features]`). Output rows are computed independently,
/// so the row split across threads cannot change any result.
fn linear(
    x: &[f32],
    weight: &[bf16],
    bias: Option<&[bf16]>,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    debug_assert_eq!(weight.len(), out_features * in_features);
    debug_assert!(bias.is_none_or(|b| b.len() == out_features));
    debug_assert_eq!(x.len() % in_features, 0);
    let rows = x.len() / in_features;
    let mut out = vec![0f32; rows * out_features];
    if rows == 0 || out_features == 0 {
        return out;
    }
    par_rows(&mut out, rows, &|row, out_row| {
        let xrow = &x[row * in_features..(row + 1) * in_features];
        for o in 0..out_features {
            let wrow = &weight[o * in_features..(o + 1) * in_features];
            let mut acc = bias.map_or(0.0, |b| b[o].to_f32());
            acc += dot_f32_bf16(xrow, wrow);
            out_row[o] = acc;
        }
    });
    out
}

/// Biased LayerNorm over the last dimension of every row (`mlx.nn.LayerNorm`:
/// biased variance, eps 1e-5, `y = (x - mean) / sqrt(var + eps) * w + b`).
fn layer_norm_rows(x: &[f32], weight: &[bf16], bias: &[bf16], d: usize) -> Vec<f32> {
    debug_assert_eq!(x.len() % d, 0);
    debug_assert_eq!(weight.len(), d);
    debug_assert_eq!(bias.len(), d);
    let rows = x.len() / d;
    let mut out = vec![0f32; x.len()];
    if d == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + LAYER_NORM_EPS).sqrt();
        let orow = &mut out[r * d..(r + 1) * d];
        for (i, &value) in row.iter().enumerate() {
            orow[i] = (value - mean) * inv * weight[i].to_f32() + bias[i].to_f32();
        }
    }
    out
}

/// Exact-erf GELU in place (`mlx.nn.gelu`):
/// `0.5 * x * (1 + erf(x / sqrt(2)))`.
fn gelu_in_place(x: &mut [f32]) {
    for value in x.iter_mut() {
        let v = f64::from(*value);
        let g = 0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2));
        *value = g as f32;
    }
}

/// `erf(x)` in f64: Maclaurin series for `|x| < 3`, and the DLMF 7.9.1
/// continued fraction for `erfc` beyond (the series alone would cancel
/// catastrophically at large `|x|`). Accuracy is far inside one f32 ulp of
/// the reference's erf, which is all the parity gate can observe.
fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x < 3.0 {
        // erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1)),
        // evaluated by the term ratio term_{n+1} = -term_n * x^2 / (n + 1).
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 1.0f64;
        loop {
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() || n > 100.0 {
                break;
            }
            n += 1.0;
        }
        return sum * (2.0 / std::f64::consts::PI.sqrt());
    }
    // erfc(x) = exp(-x^2) / (sqrt(pi) * F), F = z + (1/2)/(z + 1/(z + (3/2)/(z + ...)))
    // (numerators k/2, denominators z), evaluated backward with 64 terms;
    // erf = 1 - erfc.
    let z = x;
    let mut tail = 0f64;
    for k in (1..=64).rev() {
        tail = (k as f64 / 2.0) / (z + tail);
    }
    let f = z + tail;
    1.0 - (-x * x).exp() / (std::f64::consts::PI.sqrt() * f)
}

/// Block-wise multi-head self-attention: rows `s..e` attend only within
/// `s..e` (equivalent to the reference's additive -1e9 block mask because
/// masked scores underflow `exp` to zero in f32). `q` is already scaled.
///
/// Layouts: `q`/`k`/`v`/`attn` are row-major `[total, heads * head_dim]`
/// with per-head head-major blocks (`row * heads + head` then `head_dim`).
#[allow(clippy::too_many_arguments)]
fn block_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn: &mut [f32],
    scores: &mut [f32],
    s: usize,
    e: usize,
    heads: usize,
    head_dim: usize,
) {
    let total_heads = heads * head_dim;
    for head in 0..heads {
        for i in s..e {
            let q_row =
                &q[i * total_heads + head * head_dim..i * total_heads + (head + 1) * head_dim];
            let mut max = f32::NEG_INFINITY;
            for (j, score) in scores[s..e].iter_mut().enumerate() {
                let j = s + j;
                let k_row =
                    &k[j * total_heads + head * head_dim..j * total_heads + (head + 1) * head_dim];
                let dot = dot_f32_f32(q_row, k_row);
                *score = dot;
                max = max.max(dot);
            }
            let mut sum = 0f32;
            for score in &mut scores[s..e] {
                *score = (*score - max).exp();
                sum += *score;
            }
            let inv = 1.0 / sum;
            let out_row = &mut attn
                [i * total_heads + head * head_dim..i * total_heads + (head + 1) * head_dim];
            for out in out_row.iter_mut() {
                *out = 0.0;
            }
            for (idx, &p) in scores[s..e].iter().enumerate() {
                let j = s + idx;
                let p = p * inv;
                let v_row =
                    &v[j * total_heads + head * head_dim..j * total_heads + (head + 1) * head_dim];
                for (out, &val) in out_row.iter_mut().zip(v_row) {
                    *out += p * val;
                }
            }
        }
    }
}

#[inline]
fn dot_f32_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

fn add_in_place(target: &mut [f32], other: &[f32]) {
    debug_assert_eq!(target.len(), other.len());
    for (t, &o) in target.iter_mut().zip(other) {
        *t += o;
    }
}

/// Sinusoidal position embedding (`SinusoidalPositionEmbedding`, f32 like
/// the reference): row `p` is
/// `[sin(p * inv_0..inv_{h-1}), cos(p * inv_0..inv_{h-1})]` with
/// `inv_i = exp(-ln(10000) * i / (channels/2 - 1))` computed in f32 (the
/// f64 increment is downcast once, matching MLX scalar promotion).
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

/// Runs `f(row, row_slice)` for every one of `rows` equal-size slices of
/// `buf`, splitting them across at most one thread per available core when
/// the work justifies it (serial otherwise). Every row is computed
/// independently, so the split never changes results — this is a pure
/// throughput knob for the golden fixtures, replaced by NEON microkernels in
/// the perf phase.
fn par_rows<T, F>(buf: &mut [T], rows: usize, f: &F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    debug_assert!(rows == 0 || buf.len().is_multiple_of(rows));
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if rows == 0 {
        return;
    }
    let row_len = buf.len() / rows;
    if threads <= 1 || rows < 2 * threads {
        for row in 0..rows {
            f(row, &mut buf[row * row_len..(row + 1) * row_len]);
        }
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [T] = buf;
        let mut start = 0;
        while start < rows {
            let end = (start + per).min(rows);
            let take = (end - start) * row_len;
            let (span, tail) = std::mem::take(&mut rest).split_at_mut(take);
            rest = tail;
            scope.spawn(move || {
                for row in start..end {
                    let offset = (row - start) * row_len;
                    f(row, &mut span[offset..offset + row_len]);
                }
            });
            start = end;
        }
    });
}

// ---------------------------------------------------------------------------
// Geometry (documented, unit-locked relations)
// ---------------------------------------------------------------------------

/// Time-axis output length of one 3x3 stride-2 pad-1 convolution:
/// `floor((t - 1) / 2) + 1` (`t >= 1`).
fn conv_time_out(t: usize) -> usize {
    (t - 1) / 2 + 1
}

/// Mel-bin count surviving the three convolutions, e.g. `128 -> 16`.
fn conv_freq_after_front_end(n_mels: usize) -> usize {
    let mut h = n_mels;
    for _ in 0..3 {
        h = conv_time_out(h);
    }
    h
}

/// Time-axis length after the full conv front-end for one chunk
/// (`conv_time_out` three times).
fn chunk_time_after_front_end(chunk_len: usize) -> usize {
    conv_time_out(conv_time_out(conv_time_out(chunk_len)))
}

/// Conv chunk lengths over `frames` with chunk size `chunk_size`: full
/// chunks first, and a final remainder chunk unless `frames` is an exact
/// multiple (`frames >= 1`).
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

/// Attention block ranges for one sequence, mirroring the reference's
/// `cu_chunk_lens` loop: `window`-sized full blocks followed by a remainder
/// block. `window >= 1`; the ranges partition `0..total` in order.
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

/// Mel frames -> `<|audio_pad|>` token count, the exact
/// `_get_feat_extract_output_lengths` relation of mlx-audio (hard-coded to
/// its 100-frame conv chunking):
///
/// ```text
/// rem  = frames % 100
/// a1   = floor((rem - 1) / 2) + 1     // conv2d1 time output
/// a2   = floor((a1 - 1) / 2) + 1      // conv2d2
/// a3   = floor((a2 - 1) / 2) + 1      // conv2d3
/// pads = a3 + (frames / 100) * 13     // each full 100-frame chunk -> 13
/// ```
///
/// Floor division uses python semantics, so `rem == 0` contributes 0. The
/// prompt builder (WS-06b) consumes this to size the audio block before the
/// encoder runs; it is locked against the committed tokenizer fixtures and
/// the encoder goldens.
pub fn frames_to_audio_pads(frames: usize) -> usize {
    let rem = (frames % CONV_CHUNK_FRAMES) as i64;
    let a1 = (rem - 1).div_euclid(2) + 1;
    let a2 = (a1 - 1).div_euclid(2) + 1;
    let a3 = (a2 - 1).div_euclid(2) + 1;
    a3.max(0) as usize + (frames / CONV_CHUNK_FRAMES) * FULL_CHUNK_AFTER_CNN
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

/// Loads one raw bf16 tensor and validates dtype + shape.
///
/// The dtype check compares the safetensors `Dtype` through its `Debug`
/// rendering to avoid pulling the `safetensors` crate into `aimd-engine`
/// (the type is only nameable with the dependency).
fn bf16_tensor(
    store: &WeightStore,
    name: &str,
    expected: &[usize],
) -> Result<Vec<bf16>, EncoderError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| EncoderError::InvalidTensor {
            name: name.to_string(),
            reason: "tensor not found in checkpoint".to_string(),
        })?;
    if format!("{:?}", info.dtype) != "BF16" {
        return Err(EncoderError::InvalidTensor {
            name: name.to_string(),
            reason: format!("dtype {:?}, expected BF16 (unquantized tower)", info.dtype),
        });
    }
    if info.shape.as_slice() != expected {
        return Err(EncoderError::InvalidTensor {
            name: name.to_string(),
            reason: format!("shape {:?}, expected {expected:?}", info.shape),
        });
    }
    let expected_bytes: usize = expected.iter().product::<usize>() * 2;
    let bytes = store.tensor_bytes(name)?;
    if bytes.len() != expected_bytes {
        return Err(EncoderError::InvalidTensor {
            name: name.to_string(),
            reason: format!("byte length {}, expected {expected_bytes}", bytes.len()),
        });
    }
    let (pairs, rest) = bytes.as_chunks::<2>();
    if !rest.is_empty() {
        return Err(EncoderError::InvalidTensor {
            name: name.to_string(),
            reason: "byte length is not a multiple of 2".to_string(),
        });
    }
    Ok(pairs
        .iter()
        .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)))
        .collect())
}

fn load_layer(
    store: &WeightStore,
    index: usize,
    d_model: usize,
    ffn_dim: usize,
) -> Result<EncoderLayerWeights, EncoderError> {
    let prefix = format!("audio_tower.layers.{index}.");
    let linear = |name: &str, out: usize, inp: usize| -> Result<Vec<bf16>, EncoderError> {
        bf16_tensor(store, &format!("{prefix}{name}.weight"), &[out, inp])
    };
    let vector = |name: &str, out: usize| -> Result<Vec<bf16>, EncoderError> {
        bf16_tensor(store, &format!("{prefix}{name}"), &[out])
    };
    Ok(EncoderLayerWeights {
        q_w: linear("self_attn.q_proj", d_model, d_model)?,
        q_b: vector("self_attn.q_proj.bias", d_model)?,
        k_w: linear("self_attn.k_proj", d_model, d_model)?,
        k_b: vector("self_attn.k_proj.bias", d_model)?,
        v_w: linear("self_attn.v_proj", d_model, d_model)?,
        v_b: vector("self_attn.v_proj.bias", d_model)?,
        o_w: linear("self_attn.out_proj", d_model, d_model)?,
        o_b: vector("self_attn.out_proj.bias", d_model)?,
        attn_ln_w: vector("self_attn_layer_norm.weight", d_model)?,
        attn_ln_b: vector("self_attn_layer_norm.bias", d_model)?,
        fc1_w: linear("fc1", ffn_dim, d_model)?,
        fc1_b: vector("fc1.bias", ffn_dim)?,
        fc2_w: linear("fc2", d_model, ffn_dim)?,
        fc2_b: vector("fc2.bias", d_model)?,
        final_ln_w: vector("final_layer_norm.weight", d_model)?,
        final_ln_b: vector("final_layer_norm.bias", d_model)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    /// Golden parity gate: an element passes if `abs <= GOLDEN_ATOL` or
    /// `rel <= GOLDEN_RTOL`. The tower weights are bit-identical on both
    /// sides (raw bf16 consumed verbatim), so the residual is dominated by
    /// f32 matmul/conv accumulation order, not weight precision. Observed
    /// across all committed cases (`--nocapture`): worst absolute diff
    /// 7.8e-5 on the interior stages (`conv_front_end`..`layer_23`) and
    /// 8.5e-7 on the final projection (the head's LayerNorm re-normalizes
    /// the drift); worst relative diff ~0.19, occurring only at near-zero
    /// elements that pass via the abs branch. The gate keeps a >=12x
    /// headroom on the observed absolute bound; the same constants are
    /// recorded in `native/golden/asr_encoder/manifest.json`.
    const GOLDEN_ATOL: f64 = 1e-3;
    const GOLDEN_RTOL: f64 = 1e-2;

    // ---------- geometry helpers ----------

    #[test]
    fn encoder_conv_time_and_freq_shapes() {
        // floor((t - 1)/2) + 1
        assert_eq!(conv_time_out(1), 1);
        assert_eq!(conv_time_out(2), 1);
        assert_eq!(conv_time_out(3), 2);
        assert_eq!(conv_time_out(100), 50);
        assert_eq!(conv_time_out(50), 25);
        assert_eq!(conv_time_out(25), 13);
        // mel freq axis: 128 -> 64 -> 32 -> 16
        assert_eq!(conv_freq_after_front_end(128), 16);
        // per-chunk time chain
        assert_eq!(chunk_time_after_front_end(100), 13);
        assert_eq!(chunk_time_after_front_end(50), 7);
        assert_eq!(chunk_time_after_front_end(70), 9);
        assert_eq!(chunk_time_after_front_end(61), 8);
        assert_eq!(chunk_time_after_front_end(1), 1);
        // 7680 = downsample_hidden_size * freq_after_conv (checkpoint fact)
        assert_eq!(480 * conv_freq_after_front_end(128), 7680);
    }

    #[test]
    fn encoder_mel_chunk_lengths_table() {
        assert_eq!(mel_chunk_lengths(150, 100), vec![100, 50]);
        assert_eq!(mel_chunk_lengths(270, 100), vec![100, 100, 70]);
        assert_eq!(mel_chunk_lengths(900, 100), vec![100; 9]);
        assert_eq!(mel_chunk_lengths(100, 100), vec![100]);
        assert_eq!(mel_chunk_lengths(200, 100), vec![100, 100]);
        assert_eq!(mel_chunk_lengths(61, 100), vec![61]);
        assert_eq!(mel_chunk_lengths(1, 100), vec![1]);
    }

    #[test]
    fn encoder_frames_to_audio_pads_formula() {
        // rem-only cases (frames < 100)
        assert_eq!(frames_to_audio_pads(1), 1);
        assert_eq!(frames_to_audio_pads(61), 8);
        assert_eq!(frames_to_audio_pads(99), 13);
        // whole chunks + remainder
        assert_eq!(frames_to_audio_pads(100), 13);
        assert_eq!(frames_to_audio_pads(150), 20);
        assert_eq!(frames_to_audio_pads(180), 23);
        assert_eq!(frames_to_audio_pads(200), 26);
        assert_eq!(frames_to_audio_pads(270), 35);
        assert_eq!(frames_to_audio_pads(400), 52);
        assert_eq!(frames_to_audio_pads(900), 117);
        assert_eq!(frames_to_audio_pads(1000), 130);
        // exact multiples contribute 0 from the remainder branch
        assert_eq!(frames_to_audio_pads(0), 0);
        // consistency with the generic conv chain for the 100-frame chunking
        for frames in [1usize, 2, 61, 99, 100, 101, 150, 270, 400, 900, 1500] {
            let generic = (0..frames.div_ceil(100))
                .map(|j| {
                    let len = if j + 1 == frames.div_ceil(100) {
                        let r = frames % 100;
                        if r == 0 { 100 } else { r }
                    } else {
                        100
                    };
                    chunk_time_after_front_end(len)
                })
                .sum::<usize>();
            assert_eq!(frames_to_audio_pads(frames), generic, "frames {frames}");
        }
    }

    #[test]
    fn encoder_attention_blocks_match_naive_partition() {
        // Naive mirror of the reference cu_chunk_lens loop, written row-wise:
        // row i belongs to block i / window.
        let naive = |total: usize, window: usize| -> Vec<(usize, usize)> {
            if total == 0 {
                return Vec::new();
            }
            let count = total.div_ceil(window);
            (0..count)
                .map(|b| {
                    let s = b * window;
                    (s, (s + window).min(total))
                })
                .collect()
        };
        for &(total, window) in &[
            (1usize, 1usize),
            (5, 8),
            (20, 104),
            (104, 104),
            (105, 104),
            (117, 104),
            (208, 104),
            (209, 104),
            (1000, 104),
        ] {
            assert_eq!(attention_block_ranges(total, window), naive(total, window));
        }
        // Partition property: contiguous, ordered, covering 0..total.
        let blocks = attention_block_ranges(117, 104);
        assert_eq!(blocks, vec![(0, 104), (104, 117)]);
        let mut covered = 0;
        for (s, e) in &blocks {
            assert_eq!(*s, covered);
            assert!(e > s && e - s <= 104);
            covered = *e;
        }
        assert_eq!(covered, 117);
    }

    // ---------- numeric helpers ----------

    #[test]
    fn encoder_gelu_and_erf_known_values() {
        let erf_values = [
            (0.25, 0.2763263901682369),
            (0.5, 0.5204998778130465),
            (1.0, 0.8427007929497149),
            (2.0, 0.9953222650189527),
            (3.0, 0.9999779095030014),
            (4.0, 0.9999999845827421),
            (5.0, 0.9999999999983965),
        ];
        for (x, expected) in erf_values {
            let got = erf(x);
            assert!(
                (got - expected).abs() < 1e-12,
                "erf({x}) = {got}, expected {expected}"
            );
            assert_eq!(erf(-x), -got, "erf is odd at {x}");
        }
        assert_eq!(erf(0.0), 0.0);
        assert!(erf(6.0) > 1.0 - 1e-15);

        // gelu(x) = x * Phi(x) with the standard normal CDF values.
        let gelu_values = [
            (0.0f64, 0.0f64),
            (1.0, 0.8413447460685429),
            (-1.0, -0.15865525393145707),
            (2.0, 1.9544997361036416),
            (-2.0, -0.045500263896358424),
            (0.5, 0.34573123063700655),
        ];
        for (x, expected) in gelu_values {
            let g = 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2));
            assert!(
                (g - expected).abs() < 1e-12,
                "gelu({x}) = {g}, expected {expected}"
            );
        }
        // The in-place f32 version agrees with the f64 reference inside f32
        // rounding.
        let mut values = [-3.0f32, -1.0, 0.0, 0.5, 1.0, 2.5];
        let reference: Vec<f64> = values
            .iter()
            .map(|&v| 0.5 * f64::from(v) * (1.0 + erf(f64::from(v) / std::f64::consts::SQRT_2)))
            .collect();
        gelu_in_place(&mut values);
        for (got, expected) in values.iter().zip(reference) {
            assert!((f64::from(*got) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn encoder_layer_norm_matches_hand_values() {
        // x = [3, 4]: mean 3.5, biased var 0.25. Without affine:
        // out = (x - 3.5) / sqrt(0.25 + 1e-5).
        let x = [3.0f32, 4.0];
        let one = [bf16::from_f32(1.0), bf16::from_f32(1.0)];
        let zero = [bf16::from_f32(0.0), bf16::from_f32(0.0)];
        let out = layer_norm_rows(&x, &one, &zero, 2);
        let inv = 1.0 / (0.25f32 + LAYER_NORM_EPS).sqrt();
        assert!((out[0] - (-0.5 * inv)).abs() < 1e-6);
        assert!((out[1] - 0.5 * inv).abs() < 1e-6);
        // Affine: w = 2, b = 1 doubles and shifts.
        let two = [bf16::from_f32(2.0), bf16::from_f32(2.0)];
        let one_bias = [bf16::from_f32(1.0), bf16::from_f32(1.0)];
        let out = layer_norm_rows(&x, &two, &one_bias, 2);
        assert!((out[0] - (-0.5 * inv * 2.0 + 1.0)).abs() < 1e-6);
        // Rows are independent; a constant row normalizes to ~0 (bias aside).
        let rows = [5.0f32, 5.0, 5.0, 5.0];
        let out = layer_norm_rows(&rows, &one, &zero, 2);
        assert!(out[..2].iter().all(|v| v.abs() < 1e-5));
        assert!(out[2..].iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn encoder_sinusoidal_position_embedding_values() {
        let channels = 8;
        let table = sinusoidal_position_embedding(4, channels);
        assert_eq!(table.len(), 4 * channels);
        // Row 0: sin(0) = 0 for the first half, cos(0) = 1 for the second.
        assert!(table[..4].iter().all(|v| v.abs() < 1e-7));
        assert!(table[4..8].iter().all(|v| (v - 1.0).abs() < 1e-7));
        // inv_i = exp(-ln(10000) * i / 3): i = 0 -> 1 (sin(1) at p = 1),
        // i = 3 -> 1e-4 (cos(1e-4) at p = 1).
        let row1 = &table[channels..2 * channels];
        assert!((row1[0] - 1.0f32.sin()).abs() < 1e-6);
        assert!((row1[4] - 1.0f32.cos()).abs() < 1e-6);
        let expected_last = (1.0f32 * 1e-4f32).cos();
        assert!((row1[7] - expected_last).abs() < 1e-9);
        // Position tables are pure functions of the row index, so any call
        // with a shorter length is a prefix of a longer one.
        let short = sinusoidal_position_embedding(2, channels);
        assert_eq!(&table[..2 * channels], &short[..]);
    }

    #[test]
    fn encoder_block_attention_matches_naive_masked_reference() {
        // Full-matrix additive -1e9 block mask vs the block-wise kernel:
        // they must agree to f32 rounding (masked scores underflow to 0).
        let (total, heads, head_dim) = (7usize, 2usize, 4usize);
        let block = 3usize; // blocks [0,3), [3,6), [6,7)
        let dim = heads * head_dim;
        let mut rng_state = 0x5EED_u64;
        let mut next = || {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let mut q = vec![0f32; total * dim];
        let mut k = vec![0f32; total * dim];
        let mut v = vec![0f32; total * dim];
        for value in q.iter_mut().chain(k.iter_mut()).chain(v.iter_mut()) {
            *value = next();
        }

        // Block-wise kernel (queries pre-scaled like the real forward: the
        // kernel itself applies no scale).
        let scale = (head_dim as f32).sqrt().recip();
        for value in &mut q {
            *value *= scale;
        }
        let mut attn = vec![0f32; total * dim];
        let mut scores = vec![0f32; total];
        for (s, e) in attention_block_ranges(total, block) {
            block_attention(&q, &k, &v, &mut attn, &mut scores, s, e, heads, head_dim);
        }

        // Naive masked reference (additive -1e9 outside blocks). The scale
        // is exact (power of two), so pre-scaled q and post-scaled dots
        // agree bit-for-bit.
        let mut reference = vec![0f32; total * dim];
        for i in 0..total {
            for h in 0..heads {
                let mut raw = vec![0f32; total];
                for j in 0..total {
                    let mut dot = 0f32;
                    for d in 0..head_dim {
                        dot += q[i * dim + h * head_dim + d] * k[j * dim + h * head_dim + d];
                    }
                    raw[j] = dot + if j / block == i / block { 0.0 } else { -1e9 };
                }
                let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut raw {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for j in 0..total {
                        acc += (raw[j] / sum) * v[j * dim + h * head_dim + d];
                    }
                    reference[i * dim + h * head_dim + d] = acc;
                }
            }
        }
        for (a, r) in attn.iter().zip(&reference) {
            assert!((a - r).abs() < 1e-5, "{a} vs {r}");
        }
    }

    #[test]
    fn encoder_conv_matches_naive_reference() {
        // 3x3 stride 2 pad 1 over a small [h, w, c] input, including the
        // odd-size and padding-edge cases.
        let (h, w, c, o) = (5usize, 7usize, 3usize, 4usize);
        let mut input = vec![0f32; h * w * c];
        for (i, value) in input.iter_mut().enumerate() {
            *value = ((i % 13) as f32 - 6.0) * 0.21;
        }
        let mut weight = vec![bf16::ZERO; o * 9 * c];
        for (i, value) in weight.iter_mut().enumerate() {
            *value = bf16::from_f32(((i % 7) as f32 - 3.0) * 0.13);
        }
        let bias: Vec<bf16> = (0..o)
            .map(|i| bf16::from_f32(i as f32 * 0.05 - 0.1))
            .collect();

        let out = conv2d_3x3_s2_p1(&input, h, w, c, &weight, &bias, o);
        let (h2, w2) = (conv_time_out(h), conv_time_out(w));
        assert_eq!(out.len(), h2 * w2 * o);

        for i in 0..h2 {
            for j in 0..w2 {
                for ch in 0..o {
                    let mut acc = bias[ch].to_f32();
                    for di in 0..3 {
                        for dj in 0..3 {
                            let ii = 2 * i + di;
                            let jj = 2 * j + dj;
                            if ii == 0 || ii > h || jj == 0 || jj > w {
                                continue; // zero padding
                            }
                            for cc in 0..c {
                                acc += input[((ii - 1) * w + (jj - 1)) * c + cc]
                                    * weight[((ch * 3 + di) * 3 + dj) * c + cc].to_f32();
                            }
                        }
                    }
                    let got = out[(i * w2 + j) * o + ch];
                    assert!((got - acc).abs() < 1e-4, "({i},{j},{ch}): {got} vs {acc}");
                }
            }
        }
    }

    #[test]
    fn encoder_linear_matches_naive_and_handles_bias() {
        let (rows, k, n) = (3usize, 5usize, 4usize);
        let x: Vec<f32> = (0..rows * k).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let weight: Vec<bf16> = (0..n * k)
            .map(|i| bf16::from_f32((i as f32) * 0.11 - 0.5))
            .collect();
        let bias: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.07)).collect();

        let out = linear(&x, &weight, Some(&bias), n, k);
        for r in 0..rows {
            for o in 0..n {
                let mut acc = bias[o].to_f32();
                for i in 0..k {
                    acc += x[r * k + i] * weight[o * k + i].to_f32();
                }
                assert!((out[r * n + o] - acc).abs() < 1e-4);
            }
        }
        // No-bias variant starts from zero.
        let out = linear(&x, &weight, None, n, k);
        for r in 0..rows {
            for o in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[r * k + i] * weight[o * k + i].to_f32();
                }
                assert!((out[r * n + o] - acc).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn encoder_forward_rejects_bad_input() {
        let Some(tower) = shared_tower() else {
            eprintln!("skipping: checkpoint not available");
            return;
        };
        assert!(matches!(
            tower.forward(&[], 0),
            Err(EncoderError::InvalidInput(_))
        ));
        let mel = vec![0f32; 10 * tower.config().num_mel_bins];
        assert!(matches!(
            tower.forward(&mel, 5),
            Err(EncoderError::InvalidInput(_))
        ));
    }

    // ---------- golden fixtures ----------

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn golden_dir() -> PathBuf {
        repo_root().join("native/golden/asr_encoder")
    }

    fn checkpoint_snapshot() -> Result<PathBuf, String> {
        let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
        let hub = Path::new(&home)
            .join(".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-4bit/snapshots");
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(&hub)
            .map_err(|e| format!("HF cache missing ({e})"))?
            .flatten()
            .map(|entry| entry.path())
            .filter(|p| {
                p.is_dir()
                    && p.read_dir().is_ok_and(|mut d| {
                        d.any(|e| {
                            e.ok().is_some_and(|e| {
                                e.path().extension().is_some_and(|ext| ext == "safetensors")
                            })
                        })
                    })
            })
            .collect();
        candidates.sort();
        candidates
            .into_iter()
            .next()
            .ok_or_else(|| "no checkpoint snapshot".to_string())
    }

    /// Shared tower: the checkpoint is ~625 MB of bf16, so the golden tests
    /// load it once and reuse it. `None` means "skip, no local cache".
    fn shared_tower() -> Option<&'static AudioTower> {
        static TOWER: OnceLock<Result<AudioTower, String>> = OnceLock::new();
        let built = TOWER.get_or_init(|| {
            let snapshot = checkpoint_snapshot()?;
            let store = aimd_weights::load_checkpoint(&snapshot).map_err(|e| e.to_string())?;
            let raw = std::fs::read(snapshot.join("config.json")).map_err(|e| e.to_string())?;
            let cfg = super::super::config::parse_config(&raw).map_err(|e| e.to_string())?;
            AudioTower::from_weight_store(&cfg.thinker_config.audio_config, &store)
                .map_err(|e| e.to_string())
        });
        match built {
            Ok(tower) => Some(tower),
            Err(reason) => {
                eprintln!("skipping: {reason}");
                None
            }
        }
    }

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// Compares one stage and prints the observed worst-case diffs (visible
    /// with `--nocapture`); panics with full detail beyond the gate.
    fn compare_stage(actual: &[f32], reference: &[f32], label: &str) {
        assert_eq!(actual.len(), reference.len(), "{label}: length mismatch");
        let mut max_abs = 0f64;
        let mut max_rel = 0f64;
        let mut worst_abs = 0usize;
        let mut violations = 0usize;
        for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
            let (a, r) = (f64::from(a), f64::from(r));
            let abs = (a - r).abs();
            let rel = abs / r.abs().max(f64::MIN_POSITIVE);
            if abs > max_abs {
                max_abs = abs;
                worst_abs = i;
            }
            if rel > max_rel {
                max_rel = rel;
            }
            if abs > GOLDEN_ATOL && rel > GOLDEN_RTOL {
                violations += 1;
            }
        }
        eprintln!("{label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert_eq!(
            violations, 0,
            "{label}: {violations} element(s) beyond the gate; worst abs {max_abs:.3e} at {worst_abs}"
        );
    }

    fn stage_file_name(stage: Stage) -> String {
        match stage {
            Stage::ConvFrontEnd => "stage_conv_front_end.f32".to_string(),
            Stage::Layer(index) => format!("stage_layer_{index:02}.f32"),
            Stage::FinalProjection => "stage_final_projection.f32".to_string(),
        }
    }

    fn encoder_golden_case(name: &str) {
        let Some(tower) = shared_tower() else {
            return;
        };
        let case_dir = golden_dir().join("cases").join(name);
        if !case_dir.is_dir() {
            eprintln!("skipping: golden case {name} not committed");
            return;
        }
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(case_dir.join("meta.json")).unwrap()).unwrap();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let pads = meta["audio_pad_count"].as_u64().unwrap() as usize;
        let cu: Vec<usize> = meta["geometry"]["cu_seqlens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected_blocks: Vec<(usize, usize)> = cu.windows(2).map(|w| (w[0], w[1])).collect();

        let mel = read_f32(&case_dir.join("mel.f32"));
        assert_eq!(mel.len(), frames * tower.config().num_mel_bins);

        let mut stages: Vec<(Stage, Vec<f32>)> = Vec::new();
        let out = tower
            .forward_traced(&mel, frames, &mut |stage, data| {
                stages.push((stage, data.to_vec()))
            })
            .unwrap();

        // Geometry lock: the Rust chunk/block bookkeeping must reproduce the
        // exported reference geometry exactly.
        assert_eq!(out.pads, pads, "{name}: pad count");
        assert_eq!(out.pads, frames_to_audio_pads(frames), "{name}: formula");
        assert_eq!(out.attention_blocks, expected_blocks, "{name}: blocks");

        for (stage, data) in &stages {
            let path = case_dir.join(stage_file_name(*stage));
            if !path.is_file() {
                continue; // only the four probed layers are committed
            }
            let reference = read_f32(&path);
            compare_stage(
                data,
                &reference,
                &format!("{name}:{}", stage_file_name(*stage)),
            );
        }
        // The returned features are the final-projection stage.
        let final_reference = read_f32(&case_dir.join("stage_final_projection.f32"));
        compare_stage(&out.features, &final_reference, &format!("{name}:features"));
    }

    #[test]
    fn encoder_golden_sine_1_5s() {
        encoder_golden_case("sine_1_5s");
    }

    #[test]
    fn encoder_golden_multi_tone_2_7s() {
        encoder_golden_case("multi_tone_2_7s");
    }

    #[test]
    fn encoder_golden_sine_9s() {
        encoder_golden_case("sine_9s");
    }

    /// Locks the frames -> pads relation against the committed tokenizer
    /// fixtures (`audio_pad_count`), e.g. the sine 1.5 s recipe -> 20 pads.
    #[test]
    fn encoder_frames_to_pads_matches_tokenizer_fixtures() {
        let cases_dir = repo_root().join("native/golden/tokenizer/cases");
        let Ok(entries) = std::fs::read_dir(&cases_dir) else {
            eprintln!("skipping: tokenizer fixtures not committed");
            return;
        };
        let mut checked = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "json")
                || !path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("asr_"))
            {
                continue;
            }
            let case: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let recipe = &case["audio_recipe"];
            let duration = recipe["duration_s"].as_f64().unwrap();
            // The generator computes n = int(16000 * duration); mel frames
            // are floor(n / hop) with hop 160.
            let samples = (duration * 16000.0) as usize;
            let frames = samples / 160;
            let expected = case["audio_pad_count"].as_u64().unwrap() as usize;
            assert_eq!(
                frames_to_audio_pads(frames),
                expected,
                "case {} (duration {duration})",
                case["case"].as_str().unwrap_or("?")
            );
            checked += 1;
        }
        assert!(
            checked >= 4,
            "expected at least 4 ASR tokenizer fixtures, got {checked}"
        );
    }

    // ---------- hermetic synthetic checkpoint ----------

    /// Deterministic mini tower (8 mel bins, d_model 8, 2 layers, heads 2,
    /// n_window 2) written as a temp safetensors checkpoint. Exercises the
    /// full forward path — chunks [4, 4, 1], attention blocks [0, 2) + [2, 3)
    /// — without the multi-GB HF cache.
    /// Deliberate checkpoint corruptions for the guard tests.
    #[derive(Clone, Copy)]
    enum Corruption {
        /// Inflate the shape of a tensor (rejected by the safetensors
        /// header validation inside `load_checkpoint`).
        Shape(&'static str),
        /// Store a tensor under the F16 header (loads fine; rejected by the
        /// tower's dtype guard).
        Dtype(&'static str),
    }

    fn write_mini_checkpoint(dir: &Path, corrupt: Option<Corruption>) -> AudioTowerConfig {
        std::fs::create_dir_all(dir).unwrap();
        let mut tensors: Vec<(String, Vec<usize>, String, Vec<u8>)> = Vec::new();
        {
            let mut push = |name: &str, shape: Vec<usize>| {
                let count: usize = shape.iter().product();
                let values: Vec<bf16> = (0..count)
                    .map(|i| bf16::from_f32(((i % 11) as f32 - 5.0) * 0.11))
                    .collect();
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dtype = match corrupt {
                    Some(Corruption::Dtype(n)) if n == name => "F16".to_string(),
                    _ => "BF16".to_string(),
                };
                let shape = match corrupt {
                    Some(Corruption::Shape(n)) if n == name => {
                        shape.iter().map(|d| d + 3).collect()
                    }
                    _ => shape,
                };
                tensors.push((name.to_string(), shape, dtype, bytes));
            };

            push("audio_tower.conv2d1.weight", vec![4, 3, 3, 1]);
            push("audio_tower.conv2d1.bias", vec![4]);
            push("audio_tower.conv2d2.weight", vec![4, 3, 3, 4]);
            push("audio_tower.conv2d2.bias", vec![4]);
            push("audio_tower.conv2d3.weight", vec![4, 3, 3, 4]);
            push("audio_tower.conv2d3.bias", vec![4]);
            push("audio_tower.conv_out.weight", vec![8, 4]);
            for index in 0..2 {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    push(
                        &format!("audio_tower.layers.{index}.self_attn.{proj}.weight"),
                        vec![8, 8],
                    );
                    push(
                        &format!("audio_tower.layers.{index}.self_attn.{proj}.bias"),
                        vec![8],
                    );
                }
                push(
                    &format!("audio_tower.layers.{index}.self_attn_layer_norm.weight"),
                    vec![8],
                );
                push(
                    &format!("audio_tower.layers.{index}.self_attn_layer_norm.bias"),
                    vec![8],
                );
                push(
                    &format!("audio_tower.layers.{index}.fc1.weight"),
                    vec![16, 8],
                );
                push(&format!("audio_tower.layers.{index}.fc1.bias"), vec![16]);
                push(
                    &format!("audio_tower.layers.{index}.fc2.weight"),
                    vec![8, 16],
                );
                push(&format!("audio_tower.layers.{index}.fc2.bias"), vec![8]);
                push(
                    &format!("audio_tower.layers.{index}.final_layer_norm.weight"),
                    vec![8],
                );
                push(
                    &format!("audio_tower.layers.{index}.final_layer_norm.bias"),
                    vec![8],
                );
            }
            push("audio_tower.ln_post.weight", vec![8]);
            push("audio_tower.ln_post.bias", vec![8]);
            push("audio_tower.proj1.weight", vec![8, 8]);
            push("audio_tower.proj1.bias", vec![8]);
            push("audio_tower.proj2.weight", vec![6, 8]);
            push("audio_tower.proj2.bias", vec![6]);
        }

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, dtype, bytes) in &tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_json = serde_json::to_string(&header).unwrap();
        let mut file = (header_json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header_json.as_bytes());
        file.extend_from_slice(&data);
        std::fs::write(dir.join("mini.safetensors"), &file).unwrap();

        AudioTowerConfig {
            d_model: 8,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 16,
            num_mel_bins: 8,
            conv_chunksize: 500,
            downsample_hidden_size: 4,
            n_window: 2,
            n_window_infer: Some(8),
            output_dim: 6,
            activation_function: "gelu".to_string(),
        }
    }

    fn load_mini_tower(dir: &Path, cfg: &AudioTowerConfig) -> AudioTower {
        let store = aimd_weights::load_checkpoint(dir).unwrap();
        AudioTower::from_weight_store(cfg, &store).unwrap()
    }

    #[test]
    fn encoder_synthetic_tower_end_to_end() {
        let dir = std::env::temp_dir().join("aimd-encoder-test-mini");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = write_mini_checkpoint(&dir, None);
        let tower = load_mini_tower(&dir, &cfg);
        assert_eq!(conv_freq_after_front_end(cfg.num_mel_bins), 1); // 8 -> 4 -> 2 -> 1
        assert_eq!(tower.head_dim, 4);

        // 9 frames -> chunks [4, 4, 1] -> after-cnn [1, 1, 1] -> 3 pads;
        // window = 1 * (8 / 4) = 2 -> blocks [0, 2) + [2, 3).
        let frames = 9usize;
        let mel: Vec<f32> = (0..frames * 8)
            .map(|i| ((i % 5) as f32) * 0.3 - 0.6)
            .collect();
        let out = tower.forward(&mel, frames).unwrap();
        assert_eq!(out.pads, 3);
        assert_eq!(out.attention_blocks, vec![(0, 2), (2, 3)]);
        assert_eq!(out.features.len(), 3 * cfg.output_dim);
        assert!(out.features.iter().all(|v| v.is_finite()));

        // Exact multiple: 8 frames -> chunks [4, 4] -> 2 pads, one block.
        let frames = 8usize;
        let mel: Vec<f32> = (0..frames * 8)
            .map(|i| ((i % 7) as f32) * 0.2 - 0.5)
            .collect();
        let out = tower.forward(&mel, frames).unwrap();
        assert_eq!(out.pads, 2);
        assert_eq!(out.attention_blocks, vec![(0, 2)]);
        assert!(out.features.iter().all(|v| v.is_finite()));

        // Determinism: the same input reproduces the same values.
        let again = tower.forward(&mel, frames).unwrap();
        assert_eq!(again.features, out.features);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encoder_from_weight_store_rejects_bad_tensors() {
        // Wrong dtype header (F16 where the tower requires BF16): the store
        // loads, the tower's dtype guard must reject.
        let dir = std::env::temp_dir().join("aimd-encoder-test-mini-dtype");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg =
            write_mini_checkpoint(&dir, Some(Corruption::Dtype("audio_tower.conv_out.weight")));
        let store = aimd_weights::load_checkpoint(&dir).unwrap();
        let err = AudioTower::from_weight_store(&cfg, &store).unwrap_err();
        assert!(matches!(err, EncoderError::InvalidTensor { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);

        // Inflated shape: rejected by the safetensors header validation.
        let dir = std::env::temp_dir().join("aimd-encoder-test-mini-shape");
        let _ = std::fs::remove_dir_all(&dir);
        let _cfg =
            write_mini_checkpoint(&dir, Some(Corruption::Shape("audio_tower.conv_out.weight")));
        assert!(aimd_weights::load_checkpoint(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);

        // Missing tensors entirely (an empty-but-valid safetensors file).
        let dir = std::env::temp_dir().join("aimd-encoder-test-mini-empty");
        let _ = std::fs::remove_dir_all(&dir);
        let cfg = write_mini_checkpoint(&dir, None);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let header = "{}";
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        std::fs::write(dir.join("mini.safetensors"), &file).unwrap();
        let store = aimd_weights::load_checkpoint(&dir).unwrap();
        let err = AudioTower::from_weight_store(&cfg, &store).unwrap_err();
        assert!(matches!(err, EncoderError::InvalidTensor { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Weight-name mapping lock: every `audio_tower.*` tensor of the real
    /// checkpoint must be BF16 and carry the expected shape (cheap: mmap
    /// only, no forward).
    #[test]
    fn encoder_checkpoint_weight_mapping() {
        let Ok(snapshot) = checkpoint_snapshot() else {
            eprintln!("skipping: checkpoint not available");
            return;
        };
        let store = aimd_weights::load_checkpoint(&snapshot).unwrap();
        let expected: &[(&str, &[usize])] = &[
            ("audio_tower.conv2d1.weight", &[480, 3, 3, 1]),
            ("audio_tower.conv2d1.bias", &[480]),
            ("audio_tower.conv2d2.weight", &[480, 3, 3, 480]),
            ("audio_tower.conv2d2.bias", &[480]),
            ("audio_tower.conv2d3.weight", &[480, 3, 3, 480]),
            ("audio_tower.conv2d3.bias", &[480]),
            ("audio_tower.conv_out.weight", &[1024, 7680]),
            ("audio_tower.ln_post.weight", &[1024]),
            ("audio_tower.ln_post.bias", &[1024]),
            ("audio_tower.proj1.weight", &[1024, 1024]),
            ("audio_tower.proj1.bias", &[1024]),
            ("audio_tower.proj2.weight", &[2048, 1024]),
            ("audio_tower.proj2.bias", &[2048]),
        ];
        for (name, shape) in expected {
            let info = store
                .tensor_info(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(info.shape.as_slice(), *shape, "{name}");
            assert_eq!(format!("{:?}", info.dtype), "BF16", "{name}");
        }
        for index in 0..24 {
            for tensor in [
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "self_attn_layer_norm.weight",
                "self_attn_layer_norm.bias",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
                "final_layer_norm.weight",
                "final_layer_norm.bias",
            ] {
                let name = format!("audio_tower.layers.{index}.{tensor}");
                assert!(store.tensor_info(&name).is_some(), "missing {name}");
            }
        }
        // The tower is unquantized: no scales/biases triples under it. A
        // deviation here is a STOP condition per the checkpoint contract.
        assert!(
            store
                .quantized_bases()
                .iter()
                .all(|base| !base.starts_with("audio_tower.")),
            "audio_tower must have no quantization triples"
        );
    }
}
