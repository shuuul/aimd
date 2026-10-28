//! MiniCPM-OCR image processor (4x mode) and prompt layout.
//!
//! Slicing follows transformers' `MiniCPMOcrImageProcessor`
//! (`small_image_upsize="always"`, no adaptive upsize):
//!
//! - grid: `multiple = min(ceil(w * h / 448^2), 9)`; no slicing when
//!   `multiple <= 1`; otherwise the `(cols, rows)` among `multiple - 1 ..=
//!   multiple + 1` slices minimizing `|ln(w / h) - ln(cols / rows)|`, ties
//!   going to more rows;
//! - `find_best_resize`: scale to the 448^2 budget when larger (or always
//!   for an unsliced image), `int()` truncation, then `ensure_divide` to a
//!   multiple of 28 (`max(round(len / 28) * 28, 28)`, Python's
//!   round-half-even);
//! - a sliced image keeps an overview (`source`, never upscaled) and a
//!   `refine` resize cut into the grid, row-major.
//!
//! Resizes are torchvision's uint8 antialiased bicubic (`tvF.resize(...,
//! antialias=True)` in transformers' `TorchvisionBackend`, which vLLM
//! serving uses): [`resize_bicubic_aa`].
//! Pixels are `(v / 255 - 0.5) / 0.5` in f32, then bf16. Each crop becomes
//! `[L, 14 * 14 * 3]` patch rows in `(py, px, c)` order, the layout of
//! mlx-vlm's packed patch-embedding matmul.
//!
//! The prompt places `<image_id>0</image_id><image>` + one `<|image_pad|>`
//! per merged token (`gh * gw / 4`) + `</image>` for the overview, then
//! each slice as `<slice>...</slice>`, rows joined by `"\n"`, in place of
//! the chat template's single `<|image_pad|>`.

use half::bf16;

use super::config::PreprocessorConfig;
use crate::models::unlimited_ocr::preprocess::PreprocessError;

/// Side of the learned position table (`image_size / patch_size`).
pub const POS_SIDE: usize = 70;

/// Merged visual tokens per crop in 4x mode: one per 2 x 2 patches.
pub const MERGE: usize = 2;

/// `<|image_pad|>`.
pub const IMAGE_PAD: &str = "<|image_pad|>";

/// `max(round(len / d) * d, d)` with Python's round-half-even.
pub fn ensure_divide(len: f64, d: usize) -> usize {
    (((len / d as f64).round_ties_even() as usize) * d).max(d)
}

/// `find_best_resize` -> `(width, height)`, aligned to `patch * 2`.
pub fn find_best_resize(
    w: f64,
    h: f64,
    scale: usize,
    patch: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut w, mut h) = (w, h);
    if w * h > (scale * scale) as f64 || allow_upscale {
        let ratio = w / h;
        h = (scale as f64 / ratio.sqrt()).trunc();
        w = (h * ratio).trunc();
    }
    let align = patch * MERGE;
    (ensure_divide(w, align), ensure_divide(h, align))
}

/// The slicing grid `(cols, rows)`, `None` for an unsliced image.
pub fn sliced_grid(w: usize, h: usize, scale: usize, max_slices: usize) -> Option<(usize, usize)> {
    let log_ratio = (w as f64 / h as f64).ln();
    let ratio = (w * h) as f64 / (scale * scale) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slices);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1usize, 1usize);
    let mut best_err = f64::INFINITY;
    for n in [multiple - 1, multiple, multiple + 1] {
        if n == 1 || n > max_slices {
            continue;
        }
        for rows in 1..=n {
            if !n.is_multiple_of(rows) {
                continue;
            }
            let cols = n / rows;
            let err = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
            if err < best_err {
                best = (cols, rows);
                best_err = err;
            } else if err == best_err && rows > best.1 {
                best = (cols, rows);
            }
        }
    }
    Some(best)
}

/// Resize targets of one image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlicePlan {
    /// `(cols, rows)`, `None` when unsliced.
    pub grid: Option<(usize, usize)>,
    /// Overview `(width, height)`.
    pub source: (usize, usize),
    /// Refine `(width, height)` (cut into `grid`), sliced images only.
    pub refine: Option<(usize, usize)>,
}

/// The slicing plan of a `w x h` image.
pub fn slice_plan(w: usize, h: usize, cfg: &PreprocessorConfig) -> SlicePlan {
    let (scale, patch) = (cfg.scale_resolution, cfg.patch_size);
    let grid = if cfg.slice_mode {
        sliced_grid(w, h, scale, cfg.max_slice_nums)
    } else {
        None
    };
    let Some((cols, rows)) = grid else {
        return SlicePlan {
            grid: None,
            source: find_best_resize(w as f64, h as f64, scale, patch, true),
            refine: None,
        };
    };
    let (rw, rh) = (ensure_divide(w as f64, cols), ensure_divide(h as f64, rows));
    let (cw, ch) = find_best_resize(
        rw as f64 / cols as f64,
        rh as f64 / rows as f64,
        scale,
        patch,
        true,
    );
    SlicePlan {
        grid: Some((cols, rows)),
        source: find_best_resize(w as f64, h as f64, scale, patch, false),
        refine: Some((cw * cols, ch * rows)),
    }
}

/// One RGB8 crop fed to the vision tower.
#[derive(Clone, Debug)]
pub struct Crop {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Crop {
    /// Patch grid `(gh, gw)`.
    pub fn grid(&self, patch: usize) -> (usize, usize) {
        (self.height / patch, self.width / patch)
    }
}

/// Overview + slices (row-major) of an RGB8 HWC image, and its plan.
pub fn crops_rgb8(
    rgb: &[u8],
    width: usize,
    height: usize,
    cfg: &PreprocessorConfig,
) -> Result<(Vec<Crop>, SlicePlan), PreprocessError> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 {
        return Err(PreprocessError::Layout(format!(
            "rgb buffer {} != {width}x{height}x3",
            rgb.len()
        )));
    }
    let plan = slice_plan(width, height, cfg);
    let resize = |(w, h): (usize, usize)| -> Result<Crop, PreprocessError> {
        let rgb = resize_bicubic_aa(rgb, width, height, w, h);
        Ok(Crop {
            rgb,
            width: w,
            height: h,
        })
    };
    let mut crops = vec![resize(plan.source)?];
    if let (Some((cols, rows)), Some(refine)) = (plan.grid, plan.refine) {
        let full = resize(refine)?;
        let (cw, ch) = (full.width / cols, full.height / rows);
        for r in 0..rows {
            for c in 0..cols {
                let mut out = Vec::with_capacity(cw * ch * 3);
                for y in r * ch..(r + 1) * ch {
                    let start = (y * full.width + c * cw) * 3;
                    out.extend_from_slice(&full.rgb[start..start + cw * 3]);
                }
                crops.push(Crop {
                    rgb: out,
                    width: cw,
                    height: ch,
                });
            }
        }
    }
    Ok((crops, plan))
}

/// u8 -> bf16 of `(v / 255 - mean) / std` in f32 (numpy f32 math, then
/// mlx's round-to-nearest-even bf16 cast).
pub fn pixel_lut(mean: f32, std: f32) -> [bf16; 256] {
    let mut lut = [bf16::ZERO; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        *slot = bf16::from_f32((v as f32 / 255.0 - mean) / std);
    }
    lut
}

/// `[gh * gw, patch * patch * 3]` bf16 patch rows of a crop in
/// `(py, px, c)` order.
pub fn patch_rows(crop: &Crop, patch: usize, luts: &[[bf16; 256]; 3]) -> Vec<bf16> {
    let (gh, gw) = crop.grid(patch);
    let mut out = Vec::with_capacity(gh * gw * patch * patch * 3);
    for i in 0..gh {
        for j in 0..gw {
            for py in 0..patch {
                let row = (i * patch + py) * crop.width + j * patch;
                for px in 0..patch {
                    let base = (row + px) * 3;
                    for (c, lut) in luts.iter().enumerate() {
                        out.push(lut[crop.rgb[base + c] as usize]);
                    }
                }
            }
        }
    }
    out
}

/// `np.arange(1 / side, 1.0, 1 / side)` (f64, numpy's `start + i * delta`
/// fill) cast to f32.
pub fn position_boundaries(side: usize) -> Vec<f32> {
    let step = 1.0 / side as f64;
    let len = ((1.0 - step) / step).ceil() as usize;
    let delta = (step + step) - step;
    (0..len).map(|i| (step + i as f64 * delta) as f32).collect()
}

/// Position-table bucket of each of `n` patches along one axis:
/// `sum(clip(arange(n) / n, 0, 1 - 1e-6) >= boundaries)` in f32.
pub fn position_buckets(n: usize, boundaries: &[f32]) -> Vec<u32> {
    let n = n.max(1);
    let hi = 1.0f32 - 1e-6;
    (0..n)
        .map(|i| {
            let frac = (i as f32 / n as f32).clamp(0.0, hi);
            boundaries.iter().filter(|&&b| frac >= b).count() as u32
        })
        .collect()
}

/// Row-major position ids `bucket_h * side + bucket_w` of a `gh x gw` grid.
pub fn position_ids(gh: usize, gw: usize, side: usize) -> Vec<u32> {
    let b = position_boundaries(side);
    let (bh, bw) = (position_buckets(gh, &b), position_buckets(gw, &b));
    bh.iter()
        .flat_map(|&h| bw.iter().map(move |&w| h * side as u32 + w))
        .collect()
}

/// Merged-token count of a `gh x gw` crop.
pub fn crop_tokens(gh: usize, gw: usize) -> usize {
    gh * gw / (MERGE * MERGE)
}

/// The image placeholder text for an overview + `(cols, rows)` slices,
/// given each crop's merged-token count (overview first).
pub fn image_placeholder(tokens: &[usize], grid: Option<(usize, usize)>) -> String {
    let pads = |n: usize| IMAGE_PAD.repeat(n);
    let mut s = format!("<image_id>0</image_id><image>{}</image>", pads(tokens[0]));
    if let Some((cols, rows)) = grid {
        let mut lines = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut line = String::new();
            for c in 0..cols {
                line.push_str(&format!(
                    "<slice>{}</slice>",
                    pads(tokens[1 + r * cols + c])
                ));
            }
            lines.push(line);
        }
        s.push_str(&lines.join("\n"));
    }
    s
}

/// The chat-formatted prompt (non-thinking prefix) around one image.
pub fn build_prompt_text(placeholder: &str, task: &str) -> String {
    format!(
        "<|im_start|>user\n{placeholder}\n{task}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// torch's antialiased bicubic `interpolate` on a uint8 RGB8 HWC image
/// (`_upsample_bicubic2d_aa`, uint8 path): Pillow-style separable passes
/// (horizontal, then vertical; a pass whose size is unchanged is skipped)
/// through a uint8 intermediate, with `a = -0.5` and int16 fixed-point
/// weights whose precision is the largest that keeps the biggest weight
/// below `2^15`.
pub fn resize_bicubic_aa(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    debug_assert_eq!(src.len(), in_w * in_h * 3);
    let mut cur = src.to_vec();
    if out_w != in_w {
        let (w, p) = aa_weights(in_w, out_w);
        let mut out = vec![0u8; out_w * in_h * 3];
        for y in 0..in_h {
            let row = &cur[y * in_w * 3..(y + 1) * in_w * 3];
            for (x, (lo, ws)) in w.iter().enumerate() {
                for c in 0..3 {
                    let acc = ws.iter().enumerate().fold(1i64 << (p - 1), |a, (j, &k)| {
                        a + row[(lo + j) * 3 + c] as i64 * k as i64
                    });
                    out[(y * out_w + x) * 3 + c] = (acc >> p).clamp(0, 255) as u8;
                }
            }
        }
        cur = out;
    }
    if out_h != in_h {
        let (w, p) = aa_weights(in_h, out_h);
        let stride = out_w * 3;
        let mut out = vec![0u8; out_h * stride];
        for (y, (lo, ws)) in w.iter().enumerate() {
            for i in 0..stride {
                let acc = ws.iter().enumerate().fold(1i64 << (p - 1), |a, (j, &k)| {
                    a + cur[(lo + j) * stride + i] as i64 * k as i64
                });
                out[y * stride + i] = (acc >> p).clamp(0, 255) as u8;
            }
        }
        cur = out;
    }
    cur
}

/// Per output index the first source index and int16 weights, plus the
/// shared fixed-point precision (`_compute_index_ranges_int16_weights`).
fn aa_weights(in_size: usize, out_size: usize) -> (Vec<(usize, Vec<i16>)>, u32) {
    fn cubic(x: f64) -> f64 {
        const A: f64 = -0.5;
        let x = x.abs();
        if x < 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * A
        } else {
            0.0
        }
    }
    let scale = in_size as f64 / out_size as f64;
    let (support, inv) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let rows: Vec<(usize, Vec<f64>)> = (0..out_size)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let mut w: Vec<f64> = (lo..hi)
                .map(|j| cubic((j as f64 - center + 0.5) * inv))
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (lo, w)
        })
        .collect();
    let w_max = rows
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0.0f64, f64::max);
    let mut p = 0u32;
    while p < 22 && ((0.5 + w_max * (1u64 << (p + 1)) as f64) as i64) < (1 << 15) {
        p += 1;
    }
    let one = (1u64 << p) as f64;
    let rows = rows
        .into_iter()
        .map(|(lo, w)| {
            let w = w
                .iter()
                .map(|&v| (v * one + if v >= 0.0 { 0.5 } else { -0.5 }) as i16)
                .collect();
            (lo, w)
        })
        .collect();
    (rows, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minicpm_ocr_ensure_divide_rounds_half_even() {
        assert_eq!(ensure_divide(42.0, 28), 56);
        assert_eq!(ensure_divide(14.0, 28), 28);
        assert_eq!(ensure_divide(70.0, 28), 56);
        assert_eq!(ensure_divide(1.0, 28), 28);
    }

    #[test]
    fn minicpm_ocr_placeholder_rows_join_with_newlines() {
        let s = image_placeholder(&[1, 1, 1, 1, 1], Some((2, 2)));
        let p = IMAGE_PAD;
        assert_eq!(
            s,
            format!(
                "<image_id>0</image_id><image>{p}</image><slice>{p}</slice><slice>{p}</slice>\n<slice>{p}</slice><slice>{p}</slice>"
            )
        );
    }
}
