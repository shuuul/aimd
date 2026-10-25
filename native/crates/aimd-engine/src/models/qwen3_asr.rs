//! Qwen3-ASR-1.7B graph (WS-06).
//!
//! Architecture (from `mlx-community/Qwen3-ASR-1.7B-4bit/config.json`):
//! 128-mel front-end -> conv downsample (`downsample_hidden_size=480`,
//! `conv_chunksize=500`) -> 24-layer windowed encoder (d_model 1024, 16
//! heads, FFN 4096, GELU, `n_window=50` / `n_window_infer=800`) -> output
//! projection to 2048 -> Qwen3 dense decoder (28 layers, hidden 2048, GQA
//! 16/8, head_dim 128, SwiGLU 6144, RoPE theta 1e6, tied embeddings, vocab
//! 151936).
//!
//! Generation: greedy; repetition-loop detector retries the same audio with
//! the 8-bit checkpoint when triggered.

/// Placeholder for the WS-06 model surface.
pub struct Qwen3AsrStub;
