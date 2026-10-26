//! PaddleOCR-VL-1.6 (0.9B: NaViT vision tower + ERNIE-4.5-0.3B decoder)
//! on the MLX backend, op for op with mlx-vlm 0.7.10 `paddleocr_vl`.
//!
//! Weights come from the official dense bf16 `PaddlePaddle/PaddleOCR-VL-1.6`
//! snapshot; `quant_bits = Some(4 | 6 | 8)` quantizes at load exactly like
//! `mlx_vlm.convert` (4-bit is bit-exact with
//! `mlx-community/PaddleOCR-VL-1.6-4bit`).
//!
//! # Module layout
//!
//! - [`config`]: typed `config.json` / `preprocessor_config.json` /
//!   `generation_config.json`.
//! - [`preprocess`]: `smart_resize`, PIL-exact bicubic resize (shared with
//!   Unlimited-OCR), the u8 -> bf16 pixel table, patchify, the `OCR:`
//!   prompt text with its image-placeholder run, and `get_rope_index`.
//! - `weights`: sanitize mapping (fused vision `qkv`, NHWC conv weight) and
//!   the convert quantization predicate.
//! - [`vision`]: patch embed, bilinear position embedding, 2-D rotary
//!   encoder, `post_layernorm`, 2x2-merge projector.
//! - [`decoder`]: ERNIE blocks with sectioned 3-D mRoPE (Metal-kernel
//!   numerics), KV cache, boolean causal prefill mask, `lm_head`.
//! - [`generate`]: mlx-vlm `generate_step` greedy loop (chunked prefill,
//!   `cache_offset + rope_delta` decode positions, pipelined decode).
//! - [`spotting`]: the `Spotting:` task (README x2 LANCZOS upscale, larger
//!   `max_pixels`) and the `<|LOC_n|>` output parser.
//! - [`pipeline`]: [`pipeline::PaddleOcrVlPipeline`] wiring all of it;
//!   [`pipeline::VlTask`] picks `OCR:` (default) or `Spotting:`.
//!
//! # Reference implementations
//!
//! - `mlx_vlm/models/paddleocr_vl/{paddleocr_vl.py, vision.py, language.py,
//!   processing_paddleocr_vl.py}`, `mlx_vlm/models/{rope_utils.py,
//!   interpolate.py}`, `mlx_vlm/generate/ar.py` (mlx-vlm 0.7.10).
//! - Goldens: `scripts/export_paddleocr_vl_reference.py` ->
//!   `tests/golden/paddleocr_vl/`.

#[cfg(test)]
mod checkpoint_tests;
pub mod config;
pub mod decoder;
pub mod generate;
pub mod pipeline;
pub mod preprocess;
pub mod spotting;
#[cfg(test)]
pub(crate) mod test_support;
pub mod vision;
pub(crate) mod weights;
