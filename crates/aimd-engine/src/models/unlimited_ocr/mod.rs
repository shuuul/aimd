//! Unlimited-OCR graph (WS-07) on the MLX Metal backend.
//!
//! # Module layout
//!
//! - [`config`]: typed checkpoint config (vision/projector/language +
//!   gundam knobs), parsed from the checkpoint `config.json`. Shared by
//!   all submodules; do not duplicate.
//! - [`preprocess`]: host-side gundam preprocessing (1024 global view +
//!   640 local crops, PIL-exact bicubic resize, bf16 normalization) and
//!   the f32 interpolation tables (SAM/CLIP position embeddings, SAM
//!   relative-position tables).
//! - `mlx_weights`: verbatim safetensors -> MLX uploads (dense bf16 and
//!   packed 4-bit affine modules, no eager dequantization).
//! - [`mlx_vision`]: SAM ViT-B (window attention, global blocks
//!   2/5/8/11, decomposed rel-pos bias) + CLIP-L + quantized projector
//!   2048 -> 1280 + mlx-vlm token shaping (`image_newline` columns,
//!   `view_separator`), op-for-op with mlx-vlm `deepseekocr`.
//! - [`mlx_decoder`]: DeepSeek-V2-lite decoder (12 layers, hidden 1280,
//!   LlamaAttention, layer 0 dense then 64 routed experts top-6 with
//!   softmax scoring + 2 shared experts via `gather_qmm`) with the
//!   Python-exact `RingSlidingKVCache` (window 128).
//! - `dash`: Unlimited-OCR-2 DASH decoder state (config, `PromptPages`,
//!   router weights, block-summary cache, per-forward page2d RoPE and
//!   routing candidates); the attention kernels live in `aimd-mlx`.
//! - [`generate`]: image-feature splice, mlx-vlm `generate_step` loop
//!   (2048-token chunked prefill, pipelined greedy decode), the
//!   sliding-window no-repeat n-gram guard (n=35, window=128, ported from
//!   `src/aimd/plugins/ocr/models/mlx.py`), EOS handling and the markdown
//!   normalization port from `src/aimd/plugins/ocr/models/unlimited.py`.
//! - [`pipeline`]: [`pipeline::UnlimitedOcrPipeline`] wires image decode
//!   -> preprocessing -> vision features -> `<image>document parsing.`
//!   prompt ids (bos + one contiguous `<image>` run sized by
//!   [`pipeline::gundam_image_token_count`]) -> embedding splice ->
//!   guarded greedy generation -> strip + markdown normalization.
//! - [`markdown2`]: Unlimited-OCR 2 raw layout blocks -> Markdown (port of
//!   Baidu's `process_unlimitedOCR2.py`), plus page/block accessors.
//!
//! # Reference implementations
//!
//! - MLX graph: `.venv/lib/python3.12/site-packages/mlx_vlm/models/
//!   unlimited_ocr/{unlimitedocr.py, language.py,
//!   processing_unlimitedocr.py, config.py}` (+ `../deepseekocr/` for the
//!   base vision/decoder classes) — the porting source of truth.
//! - Upstream: `baidu/Unlimited-OCR` snapshot (deepencoder.py,
//!   modeling_deepseekv2.py).
//!
//! `mlx_vision` must not import `mlx_decoder`/`generate` and vice versa;
//! only `pipeline` composes them.

pub mod config;
pub mod dash;
pub mod generate;
pub mod markdown2;
pub mod mlx_decoder;
pub mod mlx_vision;
pub(crate) mod mlx_weights;
pub mod pipeline;
pub mod preprocess;
#[cfg(test)]
pub(crate) mod test_support;
