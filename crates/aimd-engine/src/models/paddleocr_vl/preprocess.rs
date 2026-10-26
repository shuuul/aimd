//! PaddleOCR-VL host-side preprocessing: the mlx-vlm `ImageProcessor`
//! (`processing_paddleocr_vl.py`), the processor's prompt expansion, and
//! `LanguageModel.get_rope_index` (3-D mRoPE position ids).
//!
//! Image path (verified bit-exact against mlx-vlm 0.7.10 / Pillow 13.0 by
//! `tests/golden/paddleocr_vl/preprocess.json`):
//!
//! 1. `smart_resize(h, w, factor = patch * merge = 28, min_pixels,
//!    max_pixels)` in f64 with Python's round-half-even;
//! 2. PIL `Image.resize((w', h'), BICUBIC)` — the PIL-exact fixed-point
//!    resampler shared with Unlimited-OCR
//!    ([`crate::models::unlimited_ocr::preprocess::resize_bicubic_pil`]);
//! 3. `(u8 / 255.0 - 0.5) / 0.5` in f64, then the model's
//!    `mx.array(pixel_values, dtype=bfloat16)` — a fixed 256-entry u8 ->
//!    bf16 table ([`pixel_lut`]);
//! 4. patchify `[C, gh, 14, gw, 14] -> [gh * gw, C, 14, 14]` (row-major
//!    patch order), `image_grid_thw = [1, gh, gw]`.

use half::bf16;

use super::config::PreprocessorConfig;
use crate::models::unlimited_ocr::preprocess::{PreprocessError, resize_bicubic_pil};

/// Prompt text before the image run (`chat_template.jinja` for one user
/// turn with one image, as rendered by mlx-vlm `apply_chat_template`).
pub const PROMPT_PREFIX: &str = "<|begin_of_sentence|>User: <|IMAGE_START|>";
/// The image placeholder token (`config.image_token_id`).
pub const IMAGE_PLACEHOLDER: &str = "<|IMAGE_PLACEHOLDER|>";
/// The full-page OCR task prompt (mlx4ocr / PaddleOCR-VL `OCR:`).
pub const OCR_TASK: &str = "OCR:";

/// The prompt string after the processor expands the image placeholder to
/// `n_image` copies: `<|begin_of_sentence|>User: <|IMAGE_START|>{ph × n}
/// <|IMAGE_END|>{task}\nAssistant:\n`. It is tokenized with
/// `add_special_tokens=False`.
pub fn build_prompt_text(task: &str, n_image: usize) -> String {
    let mut s = String::with_capacity(PROMPT_PREFIX.len() + n_image * IMAGE_PLACEHOLDER.len() + 64);
    s.push_str(PROMPT_PREFIX);
    for _ in 0..n_image {
        s.push_str(IMAGE_PLACEHOLDER);
    }
    s.push_str("<|IMAGE_END|>");
    s.push_str(task);
    s.push_str("\nAssistant:\n");
    s
}

/// Python 3 `round` (half to even) on a non-negative f64.
fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 || (diff == 0.5 && (f as i64) % 2 != 0) {
        f as i64 + 1
    } else {
        f as i64
    }
}

/// mlx-vlm `smart_resize`: target `(height, width)`, both multiples of
/// `factor`, with `min_pixels <= h * w <= max_pixels` (approximately) and the
/// aspect ratio preserved. Errors on an aspect ratio above 200.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), PreprocessError> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(PreprocessError::Layout(format!(
            "cannot resize a {width}x{height} image"
        )));
    }
    let f = factor as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f {
        w = py_round(w * f / h) as f64;
        h = f;
    }
    if w < f {
        h = py_round(h * f / w) as f64;
        w = f;
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(PreprocessError::Layout(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        )));
    }
    let mut h_bar = py_round(h / f) as f64 * f;
    let mut w_bar = py_round(w / f) as f64 * f;
    if h_bar * w_bar > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = (h / beta / f).floor() * f;
        w_bar = (w / beta / f).floor() * f;
    } else if h_bar * w_bar < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    if h_bar < f || w_bar < f {
        return Err(PreprocessError::Layout(format!(
            "smart_resize produced {w_bar}x{h_bar} for a {width}x{height} image"
        )));
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// u8 -> bf16 table of `(v / 255.0 - mean) / std` (f64 numpy math, then
/// f64 -> f32 -> bf16 round-to-nearest-even like the mlx conversion chain).
pub fn pixel_lut(mean: f64, std: f64) -> [bf16; 256] {
    let mut lut = [bf16::ZERO; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        let x = (v as f64 / 255.0 - mean) / std;
        *slot = bf16::from_f32(x as f32);
    }
    lut
}

/// One preprocessed image.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    /// `[n_patches, 3, patch, patch]` bf16 pixels (row-major patches).
    pub pixels: Vec<bf16>,
    /// `image_grid_thw` (`t` is always 1 for images).
    pub grid_thw: [usize; 3],
    /// Resized `(height, width)`.
    pub resized_hw: (usize, usize),
}

impl PreparedImage {
    pub fn n_patches(&self) -> usize {
        self.grid_thw[0] * self.grid_thw[1] * self.grid_thw[2]
    }
}

/// Resizes an RGB8 HWC image (`width x height`) to the smart_resize target.
pub fn resize_rgb8(
    rgb: &[u8],
    width: usize,
    height: usize,
    cfg: &PreprocessorConfig,
) -> Result<(Vec<u8>, usize, usize), PreprocessError> {
    if rgb.len() != width * height * 3 {
        return Err(PreprocessError::Layout(format!(
            "rgb buffer {} != {width}x{height}x3",
            rgb.len()
        )));
    }
    let factor = cfg.patch_size * cfg.merge_size;
    let (rh, rw) = smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels)?;
    let resized = resize_bicubic_pil(rgb, width, height, rw, rh)?;
    Ok((resized, rh, rw))
}

/// Full image-processor path on an RGB8 HWC image.
pub fn preprocess_rgb8(
    rgb: &[u8],
    width: usize,
    height: usize,
    cfg: &PreprocessorConfig,
) -> Result<PreparedImage, PreprocessError> {
    if cfg.temporal_patch_size != 1 || cfg.image_mean.len() != 3 || cfg.image_std.len() != 3 {
        return Err(PreprocessError::Layout(
            "unsupported image processor config (temporal patch / mean / std)".into(),
        ));
    }
    let (resized, rh, rw) = resize_rgb8(rgb, width, height, cfg)?;
    let luts: Vec<[bf16; 256]> = (0..3)
        .map(|c| pixel_lut(cfg.image_mean[c], cfg.image_std[c]))
        .collect();
    let p = cfg.patch_size;
    let (gh, gw) = (rh / p, rw / p);
    let mut pixels = Vec::with_capacity(gh * gw * 3 * p * p);
    for i in 0..gh {
        for j in 0..gw {
            for (c, lut) in luts.iter().enumerate() {
                for py in 0..p {
                    let row = (i * p + py) * rw;
                    for px in 0..p {
                        pixels.push(lut[resized[(row + j * p + px) * 3 + c] as usize]);
                    }
                }
            }
        }
    }
    Ok(PreparedImage {
        pixels,
        grid_thw: [1, gh, gw],
        resized_hw: (rh, rw),
    })
}

/// Number of `<|IMAGE_PLACEHOLDER|>` tokens for one image:
/// `t * h * w / merge^2`.
pub fn image_token_count(grid_thw: [usize; 3], merge: usize) -> usize {
    grid_thw[0] * grid_thw[1] * grid_thw[2] / (merge * merge)
}

/// mRoPE position ids for a single-row prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RopeIndex {
    /// `[3][len]` (temporal, height, width) positions.
    pub positions: [Vec<i64>; 3],
    /// `max(position) + 1 - len`; decode position = `cache offset + delta`.
    pub delta: i64,
}

/// `LanguageModel.get_rope_index` for one unpadded row: text runs advance
/// all three axes together; each image block (located at the next
/// `image_token_id` after a `vision_start_token_id`) gets `(t, h, w)`
/// grid positions over the merged `(t, h / merge, w / merge)` grid, offset
/// by the running text length.
pub fn rope_index(
    ids: &[u32],
    image_token_id: u32,
    vision_start_token_id: u32,
    grids: &[[usize; 3]],
    merge: usize,
) -> Result<RopeIndex, PreprocessError> {
    let n = ids.len();
    let image_nums = ids
        .windows(2)
        .filter(|w| w[0] == vision_start_token_id && w[1] == image_token_id)
        .count();
    if image_nums > grids.len() {
        return Err(PreprocessError::Layout(format!(
            "{image_nums} images in the prompt, {} grids",
            grids.len()
        )));
    }
    let mut pos: [Vec<i64>; 3] = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut st = 0usize;
    let next_start = |pos: &[Vec<i64>; 3]| -> i64 {
        pos.iter()
            .filter_map(|axis| axis.iter().copied().max())
            .max()
            .map_or(0, |m| m + 1)
    };
    for grid in grids.iter().take(image_nums) {
        let ed = ids[st..]
            .iter()
            .position(|&t| t == image_token_id)
            .map(|p| p + st)
            .ok_or_else(|| PreprocessError::Layout("image token missing".into()))?;
        let (t, h, w) = (grid[0], grid[1] / merge, grid[2] / merge);
        let st_idx = next_start(&pos);
        let text_len = ed - st;
        for i in 0..text_len {
            for axis in pos.iter_mut() {
                axis.push(st_idx + i as i64);
            }
        }
        let base = st_idx + text_len as i64;
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    pos[0].push(base + ti as i64);
                    pos[1].push(base + hi as i64);
                    pos[2].push(base + wi as i64);
                }
            }
        }
        st = ed + t * h * w;
    }
    if st < n {
        let st_idx = next_start(&pos);
        for i in 0..n - st {
            for axis in pos.iter_mut() {
                axis.push(st_idx + i as i64);
            }
        }
    }
    if pos[0].len() != n {
        return Err(PreprocessError::Layout(format!(
            "rope index covers {} of {n} tokens (image run shorter than its grid)",
            pos[0].len()
        )));
    }
    let max = next_start(&pos) - 1;
    Ok(RopeIndex {
        positions: pos,
        delta: max + 1 - n as i64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn golden_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/paddleocr_vl")
    }

    fn preprocess_golden() -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(golden_dir().join("preprocess.json")).unwrap())
            .unwrap()
    }

    fn fnv1a64(bytes: &[u8]) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        format!("{h:016x}")
    }

    fn u32s(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    #[test]
    fn paddleocr_vl_smart_resize_matches_mlx_vlm() {
        let g = preprocess_golden();
        for case in g["smart_resize"].as_array().unwrap() {
            let c: Vec<i64> = case
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let got = smart_resize(c[0] as usize, c[1] as usize, 28, 112_896, 1_003_520);
            if c[2] < 0 {
                assert!(got.is_err(), "{c:?} should be rejected");
            } else {
                assert_eq!(
                    got.unwrap(),
                    (c[2] as usize, c[3] as usize),
                    "smart_resize({}, {})",
                    c[0],
                    c[1]
                );
            }
        }
    }

    #[test]
    fn paddleocr_vl_pixel_lut_matches_mlx_chain() {
        let g = preprocess_golden();
        let want = u32s(&g["pixel_lut_bf16"]);
        let lut = pixel_lut(0.5, 0.5);
        for (v, (&w, got)) in want.iter().zip(lut.iter()).enumerate() {
            assert_eq!(got.to_bits() as u32, w, "u8 {v}");
        }
    }

    #[test]
    fn paddleocr_vl_preprocess_fixtures_bit_exact() {
        let g = preprocess_golden();
        let cfg = PreprocessorConfig::default();
        for (name, fx) in g["fixtures"].as_object().unwrap() {
            let bytes = std::fs::read(golden_dir().join(fx["image"].as_str().unwrap())).unwrap();
            let rgb = image::load_from_memory(&bytes).unwrap().to_rgb8();
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            let raw = rgb.into_raw();
            let (resized, rh, rw) = resize_rgb8(&raw, w, h, &cfg).unwrap();
            let want_hw: Vec<usize> = fx["resized_hw"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(vec![rh, rw], want_hw, "{name} resized size");
            assert_eq!(
                fnv1a64(&resized),
                fx["resized_rgb8_fnv1a64"].as_str().unwrap(),
                "{name} resized pixels"
            );
            let prep = preprocess_rgb8(&raw, w, h, &cfg).unwrap();
            let grid: Vec<usize> = prep.grid_thw.to_vec();
            let want_grid: Vec<usize> = u32s(&fx["grid_thw"]).iter().map(|&v| v as usize).collect();
            assert_eq!(grid, want_grid, "{name} grid");
            assert_eq!(prep.n_patches() as u64, fx["n_patches"].as_u64().unwrap());
            let bits: Vec<u8> = prep
                .pixels
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            assert_eq!(
                fnv1a64(&bits),
                fx["pixel_bf16_fnv1a64"].as_str().unwrap(),
                "{name} bf16 pixel_values"
            );
        }
    }

    #[test]
    fn paddleocr_vl_prompt_layout_and_rope_index_match_mlx_vlm() {
        let g = preprocess_golden();
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(golden_dir().join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            build_prompt_text(OCR_TASK, 1),
            m["formatted_prompt"].as_str().unwrap()
        );
        for (name, fx) in g["fixtures"].as_object().unwrap() {
            let ids = u32s(&fx["prompt_ids"]);
            let grid: Vec<usize> = u32s(&fx["grid_thw"]).iter().map(|&v| v as usize).collect();
            let grid = [grid[0], grid[1], grid[2]];
            let n_image = image_token_count(grid, 2);
            let run: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == 100_295)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(run.len(), n_image, "{name} image token count");
            assert_eq!(
                run.last().unwrap() - run[0] + 1,
                n_image,
                "{name} contiguous run"
            );
            let idx = rope_index(&ids, 100_295, 101_305, &[grid], 2).unwrap();
            for axis in 0..3 {
                let want: Vec<i64> = fx["position_ids"][axis]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_i64().unwrap())
                    .collect();
                assert_eq!(idx.positions[axis], want, "{name} axis {axis}");
            }
            assert_eq!(
                idx.delta,
                fx["rope_delta"].as_i64().unwrap(),
                "{name} delta"
            );
        }
    }
}
