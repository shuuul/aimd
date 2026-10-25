//! aimd native engine: tokenizer, audio front-end, and model graphs.
//!
//! # Layout
//!
//! - [`tokenizer`] (WS-04): wraps the `tokenizers` crate over the two
//!   checkpoints' `tokenizer.json` and builds the exact chat-template
//!   token sequences the Python backends produce today.
//! - [`audio`] (WS-05): wav loading + 128-bin mel matching the Qwen3-ASR
//!   feature extractor, validated against exported references.
//! - [`models`] (WS-06/WS-07): `qwen3_asr` and `unlimited_ocr` graphs and
//!   generation loops (greedy, logits post-processing hooks).

pub mod audio;
pub mod models;
pub mod tokenizer;
