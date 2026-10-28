//! Model graphs and generation loops (WS-06 / WS-07), all on the MLX
//! backend: [`qwen3_asr`] (speech), [`qwen3_aligner`] (forced alignment), [`unlimited_ocr`]
//! (documents), and [`pp_ocrv6`] (det + rec line OCR).

pub(crate) mod kv_cache;
pub mod pp_ocrv6;
pub mod qwen3_aligner;
pub mod qwen3_asr;
pub mod unlimited_ocr;
