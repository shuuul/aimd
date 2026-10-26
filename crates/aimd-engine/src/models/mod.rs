//! Model graphs and generation loops (WS-06 / WS-07), all on the MLX
//! backend: [`qwen3_asr`] (speech), [`unlimited_ocr`] and [`paddleocr_vl`]
//! (documents), and [`pp_ocrv6`] (det + rec line OCR).

pub(crate) mod kv_cache;
pub mod paddleocr_vl;
pub mod pp_ocrv6;
pub mod qwen3_asr;
pub mod unlimited_ocr;
