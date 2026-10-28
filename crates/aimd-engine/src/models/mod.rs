//! Model graphs and generation loops (WS-06 / WS-07), all on the MLX
//! backend: [`qwen3_asr`] (speech), [`qwen3_aligner`] (forced alignment), [`unlimited_ocr`], [`paddleocr_vl`] and [`minicpm_ocr`]
//! (documents), [`pp_ocrv6`] (det + rec line OCR), and [`pp_doclayout_v3`]
//! (layout regions).

pub(crate) mod kv_cache;
pub mod minicpm_ocr;
pub mod paddleocr_vl;
pub mod pp_doclayout_v3;
pub mod pp_ocrv6;
pub mod qwen3_aligner;
pub mod qwen3_asr;
pub mod unlimited_ocr;
