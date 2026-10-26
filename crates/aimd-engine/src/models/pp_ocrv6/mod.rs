//! PP-OCRv6 medium (spec 004 WS-02): DB text detection + CTC line
//! recognition on MLX, NHWC f32, matching mlx4ocr 0.1.2 (non-compiled
//! path) op for op.
//!
//! # Module layout
//!
//! - [`config`]: typed det/rec configs from the HF safetensors snapshots
//!   (`config.json`, rec `preprocessor_config.json` character table).
//! - `layers`: Paddle -> MLX weight import (OIHW -> OHWI on the host,
//!   batch norms folded at load with mlx4ocr's MLX ops), `Conv`/`Linear`,
//!   the LCNetV4 block and the branch stem shared by det and rec.
//! - [`det`]: PPLCNetV4 backbone -> RepLKPAN + IntraCL neck -> DB head.
//! - [`rec`]: PPLCNetV4 backbone + AvgPool -> LightSVTR encoder -> CTC
//!   head + softmax; swaps the HF export's `head.encoder.conv_block.0/.1`
//!   back into place (the fix mlx4ocr applies from the Paddle pickle).
//! - [`image_ops`]: host image ops bit-exact with OpenCV 4.13 (BGR decode,
//!   `INTER_LINEAR` resize, `getPerspectiveTransform` + `warpPerspective`
//!   `INTER_CUBIC`/`BORDER_REPLICATE`, `np.rot90`).
//! - [`db`]: DB post-process with ported `findContours`, `minAreaRect`,
//!   `boxPoints`, `fillPoly` + `mean`, shapely area/length and the
//!   pyclipper round offset; reading-order sort.
//! - [`ctc`]: greedy CTC decode, numpy `argsort` (introsort) and the
//!   pairwise f32 mean used for line scores.
//! - [`pipeline`]: [`pipeline::PpOcrV6Pipeline`] (decode -> det -> boxes ->
//!   crops -> batched rec -> drop < 0.5 -> markdown).
//!
//! # Reference implementations
//!
//! - mlx4ocr 0.1.2 (PyPI): `models/{common,det,rec}`, `preprocess/`,
//!   `postprocess/{db,ctc}.py`, `pipeline/{ocr,crop}.py` — the porting
//!   source of truth (`scripts/export_ppocrv6_reference.py` pins it and
//!   regenerates `tests/golden/ppocrv6/`).
//! - OpenCV 4.13 `imgproc` (`resize.cpp`, `imgwarp.cpp`, `contours_new.cpp`,
//!   `convhull.cpp`, `rotcalipers.cpp`, `drawing.cpp`), Clipper 6.4.2
//!   (pyclipper 1.4.0), GEOS (shapely 2.1.2), numpy 2.4 `npysort`.

pub mod config;
pub mod ctc;
pub mod db;
pub mod det;
pub mod image_ops;
pub(crate) mod layers;
pub mod pipeline;
pub mod rec;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
