//! PP-DocLayoutV3 layout detection (spec 009 WS-04) on MLX, NHWC f32,
//! matching the MiniCPM-OCR SDK's torch layout provider
//! (`layouts/ppdoclayout3_torch.py`: transformers'
//! `PPDocLayoutV3ForObjectDetection` in f32 plus the SDK's numpy
//! post-process).
//!
//! # Module layout
//!
//! - [`config`]: the supported geometry, checked against `config.json`.
//! - [`preprocess`]: RGB -> BGR, OpenCV-exact `INTER_CUBIC` resize to
//!   800x800, `/ 255`.
//! - `layers`: MLP heads, layer norms, self-attention, 2x upsamplers on top
//!   of the PP-OCRv6 weight import (batch norms folded at load).
//! - `backbone`: HGNetV2-L.
//! - [`encoder`]: hybrid encoder (AIFI + FPN/PAN + mask prototypes).
//! - `decoder`: deformable decoder layer (`MlxArray::grid_sample`).
//! - [`model`]: query selection, mask-enhanced init, 6 refinement layers,
//!   class / box / reading-order heads.
//! - [`postprocess`]: the SDK post-process and [`postprocess::LayoutRegion`].
//! - [`pipeline`]: [`pipeline::PpDocLayoutV3`] (`load`, `detect`).
//!
//! # Reference
//!
//! `scripts/export_pp_doclayout_v3_reference.py` runs transformers 5.21
//! (torch 2.14, f32, CPU) behind the SDK preprocessing and post-process and
//! writes `tests/golden/pp_doclayout_v3/`.

pub(crate) mod backbone;
pub mod config;
pub(crate) mod decoder;
pub mod encoder;
pub(crate) mod layers;
pub mod model;
pub mod pipeline;
pub mod postprocess;
pub mod preprocess;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
