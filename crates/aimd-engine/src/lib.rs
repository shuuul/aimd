//! aimd native engine: tokenizer, audio front-end, and MLX model graphs.
//!
//! The engine is MLX-only: every model graph runs through the `aimd-mlx`
//! shim (Metal on macOS, the `mlx` CUDA backend on Linux). There is no
//! hand-written CPU model path; the only CPU execution is MLX's own CPU
//! device, selected with `AIMD_MLX_DEVICE=cpu` or automatically when no
//! Metal device exists (see `aimd_mlx::device_is_cpu`). The `mlx` cargo feature is an empty alias kept so
//! `--features mlx` invocations keep building.
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
