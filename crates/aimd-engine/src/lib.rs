//! aimd native engine: tokenizer, audio front-end, and MLX model graphs.
//!
//! The default `mlx` feature enables model graphs through `aimd-mlx`
//! (Metal on macOS and MLX CUDA on Linux). MLX also provides its own CPU
//! device for testing with `AIMD_MLX_DEVICE=cpu`. Audio loading, tokenizer
//! helpers, and model-output parsing are available without the feature.
//!
//! # Layout
//!
//! - [`tokenizer`] (WS-04): wraps the `tokenizers` crate over the two
//!   checkpoints' `tokenizer.json` and builds the exact chat-template
//!   token sequences the Python backends produce today.
//! - [`audio`] (WS-05): wav loading + 128-bin mel matching the Qwen3-ASR
//!   feature extractor, validated against exported references.
//! - [`models`] (WS-06/WS-07): `qwen3_asr` and `unlimited_ocr` graphs and
//!   generation loops on MLX (greedy, logits post-processing hooks).

pub mod audio;
pub mod models;
pub mod parallel;
pub mod tokenizer;
pub mod weights;
