//! MiniCPM-OCR 1.0 (SigLIP NaViT vision tower + Qwen3.5-0.8B hybrid
//! decoder) on the MLX backend, aligned with the original implementation:
//! transformers `minicpm_ocr` / `qwen3_5` numerics, the official SDK's
//! region pipeline, and vLLM's DSpark speculative decoding (spec 009).
//!
//! Weights come from the official bf16 `openbmb/MiniCPM-OCR-1.0`
//! snapshot; `quant_bits = Some(4 | 6 | 8)` quantizes the decoder's
//! attention and MLP projections at load exactly like `mlx_vlm.convert`.
//!
//! # Module layout
//!
//! - [`config`]: typed `config.json` / `preprocessor_config.json`.
//! - [`preprocess`]: HF slicing (overview + refine grid), torchvision
//!   antialiased bicubic resizes, the pixel table, patch rows, position buckets and the
//!   `<image>` / `<slice>` prompt layout.
//! - [`vision`]: packed patch embedding, bucketed position table, SigLIP
//!   encoder, `post_layernorm`, 2x2 merger (exact GELU).
//! - [`decoder`]: full attention (gated, partial RoPE) and Gated DeltaNet
//!   layers with their caches, tied `lm_head`.
//! - [`generate`]: greedy loop with the SDK's no-repeat n-gram guard, and
//!   DSpark speculative decoding ([`dspark`] drafter, greedy verification).
//! - [`pipeline`]: [`pipeline::MiniCpmOcrPipeline`], one image -> text.
//! - [`regions`] (+ [`markdown`], [`latex`], [`otsl`]): the SDK region
//!   driver and page assembly; [`document`]: layout + regions + recogniser.
//!
//! # Reference implementations
//!
//! - transformers 5.21 `minicpm_ocr` (model, image processor, processor)
//!   and `qwen3_5`: the numerical source of truth.
//! - The MiniCPM-OCR SDK (`sdk/minicpmvocr`): regions and page assembly.
//! - vLLM `qwen3_dspark` / `dspark` speculator: the drafter.
//! - Goldens: `scripts/export_minicpm_ocr_reference.py` ->
//!   `tests/golden/minicpm_ocr/`, `scripts/export_minicpm_regions_reference.py`
//!   -> `tests/golden/minicpm_regions/`.

#[cfg(test)]
mod checkpoint_tests;
pub mod config;
pub mod decoder;
pub mod document;
pub mod dspark;
pub mod generate;
pub mod latex;
pub mod markdown;
pub mod otsl;
pub mod pipeline;
pub mod preprocess;
pub mod regions;
#[cfg(test)]
mod regions_tests;
#[cfg(test)]
pub(crate) mod test_support;
pub mod vision;
