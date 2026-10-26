//! Model graphs and generation loops (WS-06 / WS-07), all on the MLX
//! backend: [`qwen3_asr`] (speech) and [`unlimited_ocr`] (documents).

pub(crate) mod kv_cache;
pub mod qwen3_asr;
pub mod unlimited_ocr;
