//! Unlimited-OCR host-side preprocessing (backend-agnostic): gundam image
//! preprocessing, pixel normalization, and the weight-only interpolation
//! tables the MLX vision graph uploads once per target grid.
//!
//! Porting sources (verified against the local
//! `mlx-community/Unlimited-OCR-4bit` snapshot and mlx-vlm 0.7.0):
//!
//! - gundam mode (`base_size` 1024, `image_size` 640, `cropping=True`) from
//!   `mlx_vlm/models/unlimited_ocr/processing_unlimitedocr.py` as driven by
//!   `src/aimd/plugins/ocr/models/mlx.py`;
//! - Pillow 12.3.0 `src/libImaging/Resample.c` (bicubic, fixed-point 8bpc)
//!   plus `ImageOps.pad`/`contain` for the resize contract — PIL-exact:
//!   double-precision coefficient tables normalized to sum 1, fixed-point
//!   i32 accumulation with `PRECISION_BITS = 22`, horizontal pass then
//!   vertical into a uint8 intermediate, gray 127 `ImageOps.pad` canvas
//!   with banker's-rounded offsets;
//! - `mlx_vlm/models/kernels.py::bicubic_interpolate` (antialiased Metal
//!   kernel) for the SAM pos-embed resize, `sam.py::get_rel_pos` for the
//!   relative-position tables, and `deepseekocr/vision.py::_get_abs_pos`
//!   (`nn.Upsample` cubic, with its layout quirk) for the CLIP pos embed.
//!
//! The interpolation tables depend only on weights and the target grid
//! (40x40 SAM tokens for a 640 crop), so the MLX graph computes them here
//! once and caches them instead of re-running the kernels per image; every
//! table is rounded to bf16 exactly where the reference `.astype(dtype)`s.
//!
//! # Token-count formula
//!
//! Gundam mode (`patch_size` 16, `downsample_ratio` 4, base 1024, crop
//! 640): `num_queries_base = ceil(1024/16/4) = 16`,
//! `num_queries = ceil(640/16/4) = 10`. The prompt carries
//! `(16 + 1) * 16 + 1 = 273` global image tokens plus `(10*w + 1) * 10*h`
//! crop tokens for a `w x h` crop grid.

use half::bf16;

/// Gundam crop tile edge (`image_size` in the gundam-mode processor call).
pub(crate) const GUNDAM_CROP_PX: usize = 640;
/// Gundam global view edge (`base_size`).
pub(crate) const GUNDAM_BASE_PX: usize = 1024;

/// Preprocessing failures.
#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("preprocess layout: {0}")]
    Layout(String),
}

/// Round-to-nearest-even f32 -> bf16 -> f32 (an MLX `.astype(bfloat16)`).
#[cfg(test)]
#[inline]
pub(crate) fn bfr(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

// ---------------------------------------------------------------------------
// PIL-exact bicubic / Lanczos resize (Pillow Resample.c, 8bpc fixed point)
// ---------------------------------------------------------------------------

/// Pillow resampling filters ported here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PilFilter {
    /// `Image.BICUBIC` (`a = -0.5`, support 2).
    Bicubic,
    /// `Image.LANCZOS` (3-lobe windowed sinc, support 3).
    Lanczos,
}

impl PilFilter {
    fn support(self) -> f64 {
        match self {
            PilFilter::Bicubic => 2.0,
            PilFilter::Lanczos => 3.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            PilFilter::Bicubic => bicubic_filter(x),
            PilFilter::Lanczos => lanczos_filter(x),
        }
    }
}

/// Pillow `sinc_filter`.
fn sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

/// Pillow `lanczos_filter` (truncated sinc, `-3 <= x < 3`).
fn lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc_filter(x) * sinc_filter(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow's bicubic filter (`a = -0.5`), double precision.
fn bicubic_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    } else {
        0.0
    }
}

/// Coefficient rows for one resampled axis: `kk[(xx, tap)]` (double
/// precision, normalized to sum 1 over in-bounds taps) and per-output
/// `bounds = [xmin, tap_count]`.
struct ResizeCoeffs {
    kk: Vec<f64>,
    bounds: Vec<i32>,
    ksize: usize,
}

impl ResizeCoeffs {
    /// Pillow `precompute_coeffs` (same double-precision op order).
    fn new(in_size: usize, out_size: usize, filter: PilFilter) -> Result<Self, PreprocessError> {
        if in_size == 0 || out_size == 0 {
            return Err(PreprocessError::Layout("resize axis of size 0".to_string()));
        }
        let scale = in_size as f64 / out_size as f64;
        let filterscale = scale.max(1.0);
        let ss = 1.0 / filterscale;
        let support = filter.support() * filterscale;
        let ksize = (support.ceil() as usize) * 2 + 1;
        let mut kk = vec![0f64; out_size * ksize];
        let mut bounds = vec![0i32; out_size * 2];
        for (xx, slot) in kk.chunks_exact_mut(ksize).enumerate() {
            let center = (xx as f64 + 0.5) * scale;
            let mut xmin = (center - support + 0.5) as i64;
            if xmin < 0 {
                xmin = 0;
            }
            let mut xmax = (center + support + 0.5) as i64;
            if xmax > in_size as i64 {
                xmax = in_size as i64;
            }
            if xmax < xmin {
                return Err(PreprocessError::Layout(format!(
                    "empty resize tap window at {xx} ({in_size}->{out_size})"
                )));
            }
            let count = (xmax - xmin) as usize;
            let mut ww = 0f64;
            for (i, k) in slot.iter_mut().take(count).enumerate() {
                let w = filter.eval(((i as i64 + xmin) as f64 - center + 0.5) * ss);
                *k = w;
                ww += w;
            }
            if ww != 0.0 {
                for k in slot.iter_mut().take(count) {
                    *k /= ww;
                }
            }
            bounds[xx * 2] = xmin as i32;
            bounds[xx * 2 + 1] = count as i32;
        }
        Ok(Self { kk, bounds, ksize })
    }

    /// Pillow `normalize_coeffs_8bpc`: `(k * 2^22) +/- 0.5` then C-trunc
    /// (round half away from zero; the sign-dependent branch matters for
    /// bicubic's negative lobes — verified pixel-exact vs Pillow 12.3).
    fn fixed_point(&self) -> Vec<i32> {
        self.kk
            .iter()
            .map(|&k| {
                let v = k * (1u64 << 22) as f64;
                if v >= 0.0 {
                    (v + 0.5) as i32
                } else {
                    (v - 0.5) as i32
                }
            })
            .collect()
    }
}

const PRECISION_BITS: u32 = 22;

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Pillow `Image.resize(.., BICUBIC)` on an RGB8 HWC image.
pub(crate) fn resize_bicubic_pil(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
) -> Result<Vec<u8>, PreprocessError> {
    resize_pil(
        src,
        in_w,
        in_h,
        out_w,
        out_h,
        PilFilter::Bicubic,
        PassOrder::Pillow12,
    )
}

/// Which Pillow release's separable pass order to reproduce. The order
/// changes the uint8 intermediate, so it must follow the reference oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassOrder {
    /// Pillow <= 12.x: always horizontal, then vertical.
    Pillow12,
    /// Pillow 13: vertical first when the height shrinks by more than twice
    /// the width (`horizontal_first` in `ImagingResampleInner`).
    Pillow13,
}

/// Pillow `Image.resize` on an RGB8 HWC image, two separable fixed-point
/// passes through a uint8 intermediate exactly like `ImagingResampleInner`,
/// in the pass order of the given Pillow release.
pub(crate) fn resize_pil(
    src: &[u8],
    in_w: usize,
    in_h: usize,
    out_w: usize,
    out_h: usize,
    filter: PilFilter,
    order: PassOrder,
) -> Result<Vec<u8>, PreprocessError> {
    debug_assert_eq!(src.len(), in_w * in_h * 3);
    let (dh, dw) = (in_h as i64 - out_h as i64, in_w as i64 - out_w as i64);
    let horizontal_first = order == PassOrder::Pillow12 || !(dh > 0 && dh > dw * 2);
    let mut cur = src.to_vec();
    if horizontal_first {
        if out_w != in_w {
            cur = resize_horizontal(&cur, in_w, in_h, out_w, filter)?;
        }
        if out_h != in_h {
            cur = resize_vertical(&cur, out_w, in_h, out_h, filter)?;
        }
    } else {
        if out_h != in_h {
            cur = resize_vertical(&cur, in_w, in_h, out_h, filter)?;
        }
        if out_w != in_w {
            cur = resize_horizontal(&cur, in_w, out_h, out_w, filter)?;
        }
    }
    Ok(cur)
}

/// Horizontal pass on an RGB8 HWC image (`w -> out_w`): Pillow
/// `_ImagingResampleHorizontal_8bpc`, threaded by rows. The i32
/// fixed-point sums are exact, so the tap order does not matter.
fn resize_horizontal(
    src: &[u8],
    w: usize,
    h: usize,
    out_w: usize,
    filter: PilFilter,
) -> Result<Vec<u8>, PreprocessError> {
    let co = ResizeCoeffs::new(w, out_w, filter)?;
    let kf = co.fixed_point();
    let mut out = vec![0u8; h * out_w * 3];
    crate::parallel::for_row_bands(&mut out, out_w * 3, 16, |first, band| {
        for (r, dst) in band.chunks_exact_mut(out_w * 3).enumerate() {
            let row = &src[(first + r) * w * 3..(first + r + 1) * w * 3];
            for (xx, px) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let xmin = co.bounds[xx * 2] as usize;
                let n = co.bounds[xx * 2 + 1] as usize;
                let k = &kf[xx * co.ksize..xx * co.ksize + n];
                let taps = &row[xmin * 3..(xmin + n) * 3];
                let mut ss = [1i32 << (PRECISION_BITS - 1); 3];
                for (&kt, t) in k.iter().zip(taps.as_chunks::<3>().0) {
                    for c in 0..3 {
                        ss[c] += t[c] as i32 * kt;
                    }
                }
                for c in 0..3 {
                    px[c] = clip8(ss[c] >> PRECISION_BITS);
                }
            }
        }
    });
    Ok(out)
}

/// Vertical pass on an RGB8 HWC image (`h -> out_h`): Pillow
/// `_ImagingResampleVertical_8bpc`, threaded by output rows.
fn resize_vertical(
    src: &[u8],
    w: usize,
    h: usize,
    out_h: usize,
    filter: PilFilter,
) -> Result<Vec<u8>, PreprocessError> {
    let co = ResizeCoeffs::new(h, out_h, filter)?;
    let kf = co.fixed_point();
    let row_len = w * 3;
    let mut out = vec![0u8; out_h * row_len];
    crate::parallel::for_row_bands(&mut out, row_len, 16, |first, band| {
        let mut ss = vec![0i32; row_len];
        for (r, dst) in band.chunks_exact_mut(row_len).enumerate() {
            let yy = first + r;
            let ymin = co.bounds[yy * 2] as usize;
            let n = co.bounds[yy * 2 + 1] as usize;
            let k = &kf[yy * co.ksize..yy * co.ksize + n];
            ss.fill(1i32 << (PRECISION_BITS - 1));
            for (t, &kt) in k.iter().enumerate() {
                let row = &src[(ymin + t) * row_len..(ymin + t + 1) * row_len];
                for (acc, &v) in ss.iter_mut().zip(row) {
                    *acc += v as i32 * kt;
                }
            }
            for (o, &acc) in dst.iter_mut().zip(&ss) {
                *o = clip8(acc >> PRECISION_BITS);
            }
        }
    });
    Ok(out)
}

/// Python 3 `round` (round half to even) for the positive size values.
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    // Ties go to the even neighbour.
    if diff > 0.5 || (diff == 0.5 && f as i64 % 2 != 0) {
        f as i64 + 1
    } else {
        f as i64
    }
}

// ---------------------------------------------------------------------------
// Gundam preprocessing (dynamic_preprocess + ImageOps.pad + ImageTransform)
// ---------------------------------------------------------------------------

/// `sorted(target_ratios, key=lambda x: x[0] * x[1])` of
/// `dynamic_preprocess(min_num=2, max_num=32)` — the exact iteration order
/// CPython 3.12 produces for that set (recorded by the exporter in
/// `manifest.json`; ties in `find_closest_aspect_ratio` depend on it).
const RATIO_ORDER: [(u32, u32); 118] = [
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (6, 1),
    (3, 2),
    (2, 3),
    (1, 6),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (9, 1),
    (3, 3),
    (1, 9),
    (1, 10),
    (5, 2),
    (10, 1),
    (2, 5),
    (1, 11),
    (11, 1),
    (6, 2),
    (1, 12),
    (2, 6),
    (12, 1),
    (3, 4),
    (4, 3),
    (13, 1),
    (1, 13),
    (14, 1),
    (1, 14),
    (7, 2),
    (2, 7),
    (1, 15),
    (5, 3),
    (3, 5),
    (15, 1),
    (8, 2),
    (16, 1),
    (4, 4),
    (1, 16),
    (2, 8),
    (1, 17),
    (17, 1),
    (18, 1),
    (3, 6),
    (9, 2),
    (1, 18),
    (6, 3),
    (2, 9),
    (1, 19),
    (19, 1),
    (20, 1),
    (5, 4),
    (2, 10),
    (1, 20),
    (4, 5),
    (10, 2),
    (7, 3),
    (1, 21),
    (3, 7),
    (21, 1),
    (2, 11),
    (11, 2),
    (22, 1),
    (1, 22),
    (1, 23),
    (23, 1),
    (1, 24),
    (24, 1),
    (6, 4),
    (3, 8),
    (4, 6),
    (2, 12),
    (8, 3),
    (12, 2),
    (5, 5),
    (1, 25),
    (25, 1),
    (1, 26),
    (2, 13),
    (13, 2),
    (26, 1),
    (9, 3),
    (27, 1),
    (3, 9),
    (1, 27),
    (1, 28),
    (7, 4),
    (14, 2),
    (2, 14),
    (4, 7),
    (28, 1),
    (29, 1),
    (1, 29),
    (3, 10),
    (15, 2),
    (1, 30),
    (2, 15),
    (10, 3),
    (5, 6),
    (30, 1),
    (6, 5),
    (1, 31),
    (31, 1),
    (8, 4),
    (1, 32),
    (4, 8),
    (32, 1),
    (16, 2),
    (2, 16),
];

/// The same order for `max_num=16`, Unlimited-OCR-2's single-page
/// gundam limit (`max_crops`). Not a filter of [`RATIO_ORDER`]: the set's
/// iteration order depends on its contents.
const RATIO_ORDER_16: [(u32, u32); 49] = [
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
    (11, 1),
    (1, 11),
    (12, 1),
    (3, 4),
    (4, 3),
    (1, 12),
    (6, 2),
    (2, 6),
    (13, 1),
    (1, 13),
    (14, 1),
    (2, 7),
    (1, 14),
    (7, 2),
    (1, 15),
    (5, 3),
    (15, 1),
    (3, 5),
    (1, 16),
    (2, 8),
    (16, 1),
    (8, 2),
    (4, 4),
];

/// Pillow `find_closest_aspect_ratio` (image_size = 640) over `ratios`.
fn find_closest_aspect_ratio(
    aspect: f64,
    width: u32,
    height: u32,
    ratios: &[(u32, u32)],
) -> (u32, u32) {
    let mut best_ratio_diff = f64::INFINITY;
    let mut best_ratio = (1u32, 1u32);
    let area = width as f64 * height as f64;
    for &(i, j) in ratios {
        let target_aspect_ratio = i as f64 / j as f64;
        let ratio_diff = (aspect - target_aspect_ratio).abs();
        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            best_ratio = (i, j);
        } else if ratio_diff == best_ratio_diff
            && area > 0.5 * 640f64 * 640f64 * i as f64 * j as f64
        {
            best_ratio = (i, j);
        }
    }
    best_ratio
}

/// Decoded RGB8 gundam input: the 1024x1024 global view (gray 127 padding)
/// and the 640x640 local crops (empty when the source fits in 640x640).
#[derive(Clone, Debug)]
pub struct GundamPreprocess {
    /// `[1024 * 1024 * 3]` HWC RGB8.
    pub global: Vec<u8>,
    /// Each `[640 * 640 * 3]` HWC RGB8, row-major tile order.
    pub crops: Vec<Vec<u8>>,
    /// `(width_crop_num, height_crop_num)` — the checkpoint's
    /// `images_spatial_crop`.
    pub crop_ratio: (usize, usize),
}

/// Mirror of `UnlimitedOCRProcessor.tokenize_with_images` image handling:
/// `dynamic_preprocess` (min_num=2, max_num=32, image_size=640) crops plus
/// `ImageOps.pad(image, (1024, 1024), color=(127, 127, 127))` global view.
pub fn preprocess_gundam(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<GundamPreprocess, PreprocessError> {
    preprocess_page(rgb, width, height, PageLayout::Gundam32)
}

/// How one page is laid out for the vision towers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageLayout {
    /// v1 gundam: up to 32 local crops plus the global view.
    Gundam32,
    /// Unlimited-OCR-2 single page: up to 16 local crops (`max_crops`).
    Gundam16,
    /// Unlimited-OCR-2 multi-page base mode: the 1024 global view only.
    Base,
}

/// [`preprocess_gundam`] with the crop policy of `layout`.
pub fn preprocess_page(
    rgb: &[u8],
    width: usize,
    height: usize,
    layout: PageLayout,
) -> Result<GundamPreprocess, PreprocessError> {
    if rgb.len() != width * height * 3 {
        return Err(PreprocessError::Layout(format!(
            "rgb buffer {} does not match {width}x{height}x3",
            rgb.len()
        )));
    }
    let (w, h) = (width as u32, height as u32);
    let mut crops = Vec::new();
    let mut crop_ratio = (1usize, 1usize);
    let ratios: &[(u32, u32)] = match layout {
        PageLayout::Gundam32 => &RATIO_ORDER,
        PageLayout::Gundam16 => &RATIO_ORDER_16,
        PageLayout::Base => &[],
    };
    if !ratios.is_empty() && (w > GUNDAM_CROP_PX as u32 || h > GUNDAM_CROP_PX as u32) {
        let aspect = w as f64 / h as f64;
        let (rw, rh) = find_closest_aspect_ratio(aspect, w, h, ratios);
        crop_ratio = (rw as usize, rh as usize);
        let target_w = GUNDAM_CROP_PX * rw as usize;
        let target_h = GUNDAM_CROP_PX * rh as usize;
        let resized = resize_bicubic_pil(rgb, width, height, target_w, target_h)?;
        for i in 0..rw as usize * rh as usize {
            let col = i % (target_w / GUNDAM_CROP_PX);
            let row = i / (target_w / GUNDAM_CROP_PX);
            let mut tile = vec![0u8; GUNDAM_CROP_PX * GUNDAM_CROP_PX * 3];
            for y in 0..GUNDAM_CROP_PX {
                let src = ((row * GUNDAM_CROP_PX + y) * target_w + col * GUNDAM_CROP_PX) * 3;
                tile[y * GUNDAM_CROP_PX * 3..(y + 1) * GUNDAM_CROP_PX * 3]
                    .copy_from_slice(&resized[src..src + GUNDAM_CROP_PX * 3]);
            }
            crops.push(tile);
        }
    }

    // ImageOps.pad == ImageOps.contain (aspect-preserving BICUBIC resize)
    // pasted centered onto the (127, 127, 127) canvas.
    let im_ratio = w as f64 / h as f64;
    // Square base canvas: target (GUNDAM_BASE_PX, GUNDAM_BASE_PX).
    let dest_ratio = 1.0f64;
    let (rw, rh) = if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let new_height = py_round(h as f64 / w as f64 * GUNDAM_BASE_PX as f64);
            if new_height != GUNDAM_BASE_PX as i64 {
                (GUNDAM_BASE_PX, new_height as usize)
            } else {
                (GUNDAM_BASE_PX, GUNDAM_BASE_PX)
            }
        } else {
            let new_width = py_round(w as f64 / h as f64 * GUNDAM_BASE_PX as f64);
            if new_width != GUNDAM_BASE_PX as i64 {
                (new_width as usize, GUNDAM_BASE_PX)
            } else {
                (GUNDAM_BASE_PX, GUNDAM_BASE_PX)
            }
        }
    } else {
        (GUNDAM_BASE_PX, GUNDAM_BASE_PX)
    };
    let resized = resize_bicubic_pil(rgb, width, height, rw, rh)?;
    let mut global = vec![127u8; GUNDAM_BASE_PX * GUNDAM_BASE_PX * 3];
    if rw != GUNDAM_BASE_PX {
        let x = py_round((GUNDAM_BASE_PX - rw) as f64 * 0.5) as usize;
        for y in 0..rh {
            let dst = (y * GUNDAM_BASE_PX + x) * 3;
            global[dst..dst + rw * 3].copy_from_slice(&resized[(y * rw) * 3..(y * rw + rw) * 3]);
        }
    } else {
        let y = py_round((GUNDAM_BASE_PX - rh) as f64 * 0.5) as usize;
        let dst = y * GUNDAM_BASE_PX * 3;
        global[dst..dst + rh * GUNDAM_BASE_PX * 3].copy_from_slice(&resized);
    }
    Ok(GundamPreprocess {
        global,
        crops,
        crop_ratio,
    })
}

/// `ImageTransform`: HWC u8 -> CHW `((x / 255) - 0.5) / 0.5`, rounded to
/// bf16 exactly like the reference processor's `.astype(mx.bfloat16)`.
#[cfg(test)]
pub(crate) fn normalize_pixels_bf16(u8img: &[u8], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = u8img[(y * w + x) * 3 + c] as f32;
                out[c * h * w + y * w + x] = bfr((v / 255.0 - 0.5) / 0.5);
            }
        }
    }
    out
}

/// `ImageTransform` + the towers' NCHW -> NHWC transpose in one pass:
/// HWC u8 -> HWC bf16 `((x / 255) - 0.5) / 0.5` (same values as
/// `normalize_pixels_bf16`, channel-last like `pixel_values.transpose(0,
/// 2, 3, 1)` feeds SAM).
pub(crate) fn pixels_nhwc_bf16(u8img: &[u8]) -> Vec<bf16> {
    u8img
        .iter()
        .map(|&v| bf16::from_f32((v as f32 / 255.0 - 0.5) / 0.5))
        .collect()
}

// ---------------------------------------------------------------------------
// Interpolation helpers (SAM pos embed bicubic + rel pos linear)
// ---------------------------------------------------------------------------

/// Keys cubic kernel with parameter `a` (torch antialias uses -0.5).
fn cubic_kernel_a(x: f32, a: f32) -> f32 {
    let absx = x.abs();
    let absx2 = absx * absx;
    let absx3 = absx2 * absx;
    if absx <= 1.0 {
        (a + 2.0) * absx3 - (a + 3.0) * absx2 + 1.0
    } else if absx < 2.0 {
        a * absx3 - 5.0 * a * absx2 + 8.0 * a * absx - 4.0 * a
    } else {
        0.0
    }
}

/// Channel-last bicubic interpolation with PyTorch antialias semantics
/// (`F.interpolate(bicubic, antialias=True, align_corners=False)`, as in
/// the mlx_vlm Metal kernel: support 2.0, a = -0.5): f32
/// coordinate math, tap range clamped in-bounds, weight-sum normalization.
/// `src`: `[in_h, in_w, c]`; returns `[out_h, out_w, c]`.
pub(crate) fn bicubic_interp_antialias(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    c_dim: usize,
) -> Vec<f32> {
    let scale_h = out_h as f32 / in_h as f32;
    let scale_w = out_w as f32 / in_w as f32;
    let filter_scale_h = if scale_h < 1.0 { 1.0 / scale_h } else { 1.0 };
    let filter_scale_w = if scale_w < 1.0 { 1.0 / scale_w } else { 1.0 };
    let support = 2.0f32;
    let support_h = support * filter_scale_h;
    let support_w = support * filter_scale_w;
    let mut dst = vec![0f32; out_h * out_w * c_dim];
    for oy in 0..out_h {
        let y_in = ((oy as f32 + 0.5) / out_h as f32) * in_h as f32 - 0.5;
        let y_start = (((y_in - support_h).floor() as i64) + 1).max(0) as usize;
        let y_end = (((y_in + support_h).floor() as i64) + 1).min(in_h as i64) as usize;
        for ox in 0..out_w {
            let x_in = ((ox as f32 + 0.5) / out_w as f32) * in_w as f32 - 0.5;
            let x_start = (((x_in - support_w).floor() as i64) + 1).max(0) as usize;
            let x_end = (((x_in + support_w).floor() as i64) + 1).min(in_w as i64) as usize;
            for c in 0..c_dim {
                let mut result = 0f32;
                let mut weight_sum = 0f32;
                for yp in y_start..y_end {
                    let wy = cubic_kernel_a((yp as f32 - y_in) / filter_scale_h, -0.5);
                    for xp in x_start..x_end {
                        let wx = cubic_kernel_a((xp as f32 - x_in) / filter_scale_w, -0.5);
                        let weight = wy * wx;
                        result += src[(yp * in_w + xp) * c_dim + c] * weight;
                        weight_sum += weight;
                    }
                }
                if weight_sum > 1e-8 {
                    result /= weight_sum;
                }
                dst[(oy * out_w + ox) * c_dim + c] = result;
            }
        }
    }
    dst
}

/// SAM `get_abs_pos_sam` (Baidu `deepencoder.py`): f32 bicubic
/// (antialias). Returns `[tgt, tgt, c]` in f32; the upload rounds it to
/// bf16 like the reference's `.to(dtype)`.
pub(crate) fn sam_pos_embed_interp(
    pos_embed: &[f32],
    src: usize,
    tgt: usize,
    c_dim: usize,
) -> Vec<f32> {
    bicubic_interp_antialias(pos_embed, src, src, tgt, tgt, c_dim)
}

/// CLIP `get_abs_pos` (Baidu `deepencoder.py`) for a `tgt_tokens = 1 +
/// side^2` sequence: the `[src, src, dim]` grid is resampled over its two
/// spatial axes (`view(1, src, src, dim).permute(0, 3, 1, 2)` +
/// `F.interpolate(bicubic, antialias=True, align_corners=False)`) and the
/// class row is kept. `pos` is `[1 + src^2, dim]`; returns `[1 + side^2,
/// dim]` with `side = isqrt(tgt_tokens)` (equal to `pos` when the grid
/// already matches). f32; the upload rounds it to bf16.
pub(crate) fn clip_pos_embed_interp(pos: &[f32], dim: usize, tgt_tokens: usize) -> Vec<f32> {
    let src = (pos.len() / dim - 1).isqrt();
    let side = tgt_tokens.isqrt(); // `int(math.sqrt(tgt_size))`
    if src == side {
        return pos.to_vec();
    }
    let grid = bicubic_interp_antialias(&pos[dim..], src, src, side, side, dim);
    let mut out = Vec::with_capacity((1 + side * side) * dim);
    out.extend_from_slice(&pos[..dim]);
    out.extend_from_slice(&grid);
    out
}

/// SAM `get_rel_pos` for `q_size == k_size` (the only shape this model
/// runs): `F.interpolate(mode="linear", align_corners=False)` of the
/// `[rows, c]` table to `max_rel = 2*q-1` rows when needed (half-pixel
/// source index clamped at 0, f32), then gathered by
/// `relative_coords[i][j] = i - j + (q - 1)` -> `[q, k, c]`.
pub(crate) fn get_rel_pos_table(
    rel_pos: &[f32],
    rel_rows: usize,
    c_dim: usize,
    q_size: usize,
) -> Vec<f32> {
    let max_rel = 2 * q_size - 1;
    let resized: Vec<f32> = if rel_rows != max_rel {
        let scale = rel_rows as f32 / max_rel as f32;
        let mut out = vec![0f32; max_rel * c_dim];
        for m in 0..max_rel {
            let real = ((m as f32 + 0.5) * scale - 0.5).max(0.0);
            let lo_i = real as usize;
            let hi_i = if lo_i < rel_rows - 1 { lo_i + 1 } else { lo_i };
            let l1 = real - lo_i as f32;
            let l0 = 1.0 - l1;
            for c in 0..c_dim {
                let lo = rel_pos[lo_i * c_dim + c];
                let hi = rel_pos[hi_i * c_dim + c];
                out[m * c_dim + c] = l0 * lo + l1 * hi;
            }
        }
        out
    } else {
        rel_pos.to_vec()
    };
    let mut table = vec![0f32; q_size * q_size * c_dim];
    for i in 0..q_size {
        for j in 0..q_size {
            let src_row = (i + q_size - 1 - j) * c_dim;
            let dst = (i * q_size + j) * c_dim;
            table[dst..dst + c_dim].copy_from_slice(&resized[src_row..src_row + c_dim]);
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ocr_vision")
    }

    fn manifest() -> serde_json::Value {
        let raw = std::fs::read_to_string(golden_dir().join("manifest.json")).unwrap();
        serde_json::from_str(&raw).unwrap()
    }

    fn read_bytes(name: &str) -> Vec<u8> {
        std::fs::read(golden_dir().join(name)).unwrap()
    }

    fn read_f32(name: &str) -> Vec<f32> {
        read_bytes(name)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| f32::from_le_bytes(*p))
            .collect()
    }

    pub(crate) fn source_image(name: &str) -> (Vec<u8>, usize, usize) {
        let img = image::load_from_memory(&read_bytes(&format!("images/{name}"))).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        (rgb.into_raw(), w, h)
    }

    /// Deterministic LCG noise image (matches the exporter's make_noise_image).
    fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut state = seed & 0x7FFF_FFFF;
        let mut data = vec![0u8; w * h * 3];
        for b in data.iter_mut() {
            state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0x7FFF_FFFF;
            *b = ((state >> 16) & 0xFF) as u8;
        }
        data
    }

    #[test]
    fn preprocess_pil_resize_matches_pil_fixtures() {
        let m = manifest();
        for fx in m["resize_fixtures"].as_array().unwrap() {
            let (w, h) = (
                fx["source"][0].as_u64().unwrap() as usize,
                fx["source"][1].as_u64().unwrap() as usize,
            );
            let (dw, dh) = (
                fx["target"][0].as_u64().unwrap() as usize,
                fx["target"][1].as_u64().unwrap() as usize,
            );
            let seed = fx["lcg_seed"].as_u64().unwrap() as u32;
            let src = lcg_image(w, h, seed);
            let got = resize_bicubic_pil(&src, w, h, dw, dh).unwrap();
            let want = read_bytes(fx["file"].as_str().unwrap());
            assert_eq!(got.len(), want.len(), "{} size", fx["file"]);
            let diffs = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(diffs, 0, "{}: {diffs} mismatching pixels", fx["file"]);
        }
        eprintln!(
            "vision_pil_resize: {} fixtures bit-exact",
            m["resize_fixtures"].as_array().unwrap().len()
        );
    }

    #[test]
    fn preprocess_gundam_matches_golden() {
        let m = manifest();
        for tag in ["small", "crop"] {
            let img = m["images"][tag].clone();
            let src_name = if tag == "small" {
                "source_small_500x400.png"
            } else {
                "source_crop_1000x500.png"
            };
            let (rgb, w, h) = source_image(src_name);
            let prep = preprocess_gundam(&rgb, w, h).unwrap();
            // Global view sampled rows (u8, bit-exact).
            let rows: Vec<usize> = img["global_dump_rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let want = read_bytes(&format!("{tag}_global_u8.u8"));
            for (ri, &r) in rows.iter().enumerate() {
                let got = &prep.global[r * 1024 * 3..(r + 1) * 1024 * 3];
                let wrow = &want[ri * 1024 * 3..(ri + 1) * 1024 * 3];
                assert_eq!(got, wrow, "{tag}: global row {r}");
            }
            // Crops sampled rows (u8, bit-exact); absent for the no-crop image.
            let Some(crop_files) = img["crop_dump_files"].as_array() else {
                assert!(prep.crops.is_empty(), "{tag}: unexpected crops");
                continue;
            };
            let crop_rows: Vec<usize> = img["crop_dump_rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            for (ci, fname) in crop_files.iter().enumerate() {
                let want = read_bytes(fname.as_str().unwrap());
                for (ri, &r) in crop_rows.iter().enumerate() {
                    let got = &prep.crops[ci][r * 640 * 3..(r + 1) * 640 * 3];
                    let wrow = &want[ri * 640 * 3..(ri + 1) * 640 * 3];
                    assert_eq!(got, wrow, "{tag}: crop {ci} row {r}");
                }
            }
            // Normalized bf16 spot check (channel 0, rows 0..2).
            let norm = normalize_pixels_bf16(&prep.global, 1024, 1024);
            let want = read_f32(&format!("{tag}_global_norm_ch0_rows0_2.f32"));
            assert_eq!(norm[..want.len()], want[..], "{tag}: normalized pixels");
            assert_eq!(
                prep.crop_ratio,
                (
                    img["crop_ratio"][0].as_u64().unwrap() as usize,
                    img["crop_ratio"][1].as_u64().unwrap() as usize
                ),
                "{tag}: crop ratio"
            );
            eprintln!(
                "{tag}: preprocessing bit-exact ({} global rows)",
                rows.len()
            );
        }
    }

    #[test]
    fn preprocess_token_counts_match_manifest() {
        let m = manifest();
        // Token formula: 273 global tokens + (10*w + 1) * 10*h crop tokens.
        for (tag, (wc, hc)) in [("small", (1usize, 1usize)), ("crop", (2usize, 1usize))] {
            let n_tokens = m["images"][tag]["n_tokens"].as_u64().unwrap() as usize;
            let expected = 273
                + if wc * hc > 1 {
                    (10 * wc + 1) * 10 * hc
                } else {
                    0
                };
            assert_eq!(n_tokens, expected, "{tag} token formula");
        }
        // Crop geometry from the committed sources.
        let (rgb, w, h) = source_image("source_crop_1000x500.png");
        let prep = preprocess_gundam(&rgb, w, h).unwrap();
        assert_eq!(prep.crop_ratio, (2, 1));
        assert_eq!(prep.crops.len(), 2);
        assert_eq!(prep.crops[0].len(), 640 * 640 * 3);
        let (rgb, w, h) = source_image("source_small_500x400.png");
        let prep = preprocess_gundam(&rgb, w, h).unwrap();
        assert!(prep.crops.is_empty());
        assert_eq!(prep.crop_ratio, (1, 1));
    }

    #[test]
    fn preprocess_nhwc_pixels_match_normalized_chw() {
        let (rgb, w, h) = source_image("source_small_500x400.png");
        let prep = preprocess_gundam(&rgb, w, h).unwrap();
        let chw = normalize_pixels_bf16(&prep.global, 1024, 1024);
        let nhwc = pixels_nhwc_bf16(&prep.global);
        for (y, x) in [(0usize, 0usize), (17, 900), (511, 3), (1023, 1023)] {
            for c in 0..3 {
                assert_eq!(
                    nhwc[(y * 1024 + x) * 3 + c].to_f32(),
                    chw[c * 1024 * 1024 + y * 1024 + x]
                );
            }
        }
    }
}
