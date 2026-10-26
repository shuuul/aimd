//! PaddleOCR-VL-1.6 text spotting (`Spotting:` task).
//!
//! Preprocessing follows the checkpoint README (transformers usage):
//! images whose sides are both below 1500 px are upscaled x2 with PIL
//! `LANCZOS` before the processor, which then runs with
//! `max_pixels = 2048 * 28 * 28` instead of the default 1003520.
//!
//! Output grammar (verified on mlx-vlm 0.7.10 greedy runs, see
//! `tests/golden/paddleocr_vl/manifest.json` `spotting`): one line per text
//! item, `text` followed by eight `<|LOC_n|>` tokens — the corners
//! `(x, y)` of a quadrilateral in the order top-left, top-right,
//! bottom-right, bottom-left — and items separated by `\n`. `n` lives on a
//! `0..=1000` grid relative to the processed image (PaddleX
//! `post_process_for_spotting`: `x / 1000.0 * w`). The upscale is an exact
//! x2, so dividing by 1000 normalizes to the original image as well.
//! `<|LOC_n|>` (ids 100297..=101297) are not special tokens, so a
//! `skip_special_tokens` decode keeps them; `<|LOC_BEGIN|>`, `<|LOC_END|>`
//! and `<|LOC_SEP|>` are special and ignored if they ever appear.

use crate::models::unlimited_ocr::preprocess::{PassOrder, PilFilter, PreprocessError, resize_pil};

/// The spotting task prompt.
pub const SPOTTING_TASK: &str = "Spotting:";
/// Both sides below this are upscaled x2 before preprocessing.
pub const SPOTTING_UPSCALE_THRESHOLD: usize = 1500;
/// Processor `max_pixels` for spotting.
pub const SPOTTING_MAX_PIXELS: usize = 2048 * 28 * 28;
/// Side of the `<|LOC_n|>` coordinate grid.
pub const LOC_GRID: f32 = 1000.0;

/// One spotted text item.
#[derive(Clone, Debug, PartialEq)]
pub struct Spot {
    pub text: String,
    /// Corners TL, TR, BR, BL as `(x, y)`, normalized to `0..=1` of the
    /// original image.
    pub quad: [[f32; 2]; 4],
}

/// The README's spotting upscale: `Some((rgb, w * 2, h * 2))` (PIL-exact
/// `LANCZOS`) when both sides are below [`SPOTTING_UPSCALE_THRESHOLD`].
pub fn spotting_upscale(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<Option<(Vec<u8>, usize, usize)>, PreprocessError> {
    if width >= SPOTTING_UPSCALE_THRESHOLD || height >= SPOTTING_UPSCALE_THRESHOLD {
        return Ok(None);
    }
    let (w2, h2) = (width * 2, height * 2);
    Ok(Some((
        resize_pil(
            rgb,
            width,
            height,
            w2,
            h2,
            PilFilter::Lanczos,
            PassOrder::Pillow13,
        )?,
        w2,
        h2,
    )))
}

enum Piece<'a> {
    Text(&'a str),
    Loc(u32),
    Marker,
}

/// Splits decoded text into plain runs, `<|LOC_n|>` values and the
/// `<|LOC_BEGIN|>` / `<|LOC_END|>` / `<|LOC_SEP|>` markers.
fn pieces(raw: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let mut rest = raw;
    while let Some(at) = rest.find("<|LOC_") {
        let tail = &rest[at + "<|LOC_".len()..];
        let Some(end) = tail.find("|>") else {
            break;
        };
        let name = &tail[..end];
        let piece = if !name.is_empty() && name.bytes().all(|b| b.is_ascii_digit()) {
            name.parse().ok().map(Piece::Loc)
        } else if matches!(name, "BEGIN" | "END" | "SEP") {
            Some(Piece::Marker)
        } else {
            None
        };
        let Some(piece) = piece else {
            // Not a location token: keep it as text.
            let cut = at + "<|LOC_".len() + end + "|>".len();
            out.push(Piece::Text(&rest[..cut]));
            rest = &rest[cut..];
            continue;
        };
        if at > 0 {
            out.push(Piece::Text(&rest[..at]));
        }
        out.push(piece);
        rest = &tail[end + "|>".len()..];
    }
    if !rest.is_empty() {
        out.push(Piece::Text(rest));
    }
    out
}

fn grid(v: u32) -> f32 {
    (v as f32 / LOC_GRID).clamp(0.0, 1.0)
}

/// Quad from one item's location values: eight values are four corners;
/// four are an axis-aligned `x0, y0, x1, y1` box; anything else is no quad.
fn quad_of(locs: &[u32]) -> Option<[[f32; 2]; 4]> {
    match *locs {
        [x0, y0, x1, y1, x2, y2, x3, y3] => Some([
            [grid(x0), grid(y0)],
            [grid(x1), grid(y1)],
            [grid(x2), grid(y2)],
            [grid(x3), grid(y3)],
        ]),
        [x0, y0, x1, y1] => Some([
            [grid(x0), grid(y0)],
            [grid(x1), grid(y0)],
            [grid(x1), grid(y1)],
            [grid(x0), grid(y1)],
        ]),
        _ => None,
    }
}

/// Parses a spotting generation (decoded with or without special tokens)
/// into located items: each run of text is closed by the run of
/// `<|LOC_n|>` tokens that follows it. Text is trimmed; items with an
/// empty text or a malformed location run (not 8 or 4 values) are
/// dropped, as is trailing text without locations.
pub fn parse_spotting_output(raw: &str) -> Vec<Spot> {
    let mut spots = Vec::new();
    let mut text = String::new();
    let mut locs: Vec<u32> = Vec::new();
    let mut flush = |text: &mut String, locs: &mut Vec<u32>| {
        if let Some(quad) = quad_of(locs) {
            let t = text.trim();
            if !t.is_empty() {
                spots.push(Spot {
                    text: t.to_string(),
                    quad,
                });
            }
        }
        text.clear();
        locs.clear();
    };
    for piece in pieces(raw) {
        match piece {
            Piece::Text(t) => {
                if !locs.is_empty() {
                    flush(&mut text, &mut locs);
                }
                text.push_str(t);
            }
            Piece::Loc(v) => locs.push(v),
            Piece::Marker => {}
        }
    }
    if !locs.is_empty() {
        flush(&mut text, &mut locs);
    }
    spots
}

/// The spotting text without location tokens or markers (trimmed): the
/// plain-text `markdown` of a spotting run.
pub fn strip_location_tokens(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for piece in pieces(raw) {
        if let Piece::Text(t) = piece {
            out.push_str(t);
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::super::config::PreprocessorConfig;
    use super::super::preprocess::preprocess_rgb8;
    use super::super::test_support::{fixture_image, manifest, preprocess_golden, usizes};
    use super::*;

    fn fnv1a64(bytes: &[u8]) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        format!("{h:016x}")
    }

    /// Deterministic LCG noise (the exporter's `make_noise_image`).
    fn lcg_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut state = seed & 0x7FFF_FFFF;
        let mut data = vec![0u8; w * h * 3];
        for b in data.iter_mut() {
            state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) & 0x7FFF_FFFF;
            *b = ((state >> 16) & 0xFF) as u8;
        }
        data
    }

    /// PIL `LANCZOS` parity on noise (up, odd-ratio and down resizes).
    #[test]
    fn paddleocr_vl_spotting_lanczos_matches_pil() {
        let g = preprocess_golden();
        let cases = g["lanczos"].as_array().unwrap();
        assert!(!cases.is_empty());
        for c in cases {
            let (src, dst) = (usizes(&c["source"]), usizes(&c["target"]));
            let img = lcg_image(src[0], src[1], c["lcg_seed"].as_u64().unwrap() as u32);
            let out = resize_pil(
                &img,
                src[0],
                src[1],
                dst[0],
                dst[1],
                PilFilter::Lanczos,
                PassOrder::Pillow13,
            )
            .unwrap();
            assert_eq!(
                fnv1a64(&out),
                c["rgb8_fnv1a64"].as_str().unwrap(),
                "lanczos {src:?} -> {dst:?}"
            );
        }
    }

    /// README spotting preprocessing on the fixtures: x2 LANCZOS upscale,
    /// grid under the spotting `max_pixels`, and bf16 pixel values, all
    /// bit-exact with the mlx-vlm processor.
    #[test]
    fn paddleocr_vl_spotting_preprocess_bit_exact() {
        let g = preprocess_golden();
        let cfg = PreprocessorConfig {
            max_pixels: SPOTTING_MAX_PIXELS,
            ..PreprocessorConfig::default()
        };
        for (name, fx) in g["spotting"].as_object().unwrap() {
            let (rgb, w, h) = fixture_image(name);
            let (up, uw, uh) = spotting_upscale(&rgb, w, h).unwrap().expect("upscaled");
            assert_eq!(vec![uw, uh], usizes(&fx["upscaled_wh"]), "{name}");
            assert_eq!(
                fnv1a64(&up),
                fx["upscaled_rgb8_fnv1a64"].as_str().unwrap(),
                "{name} LANCZOS x2"
            );
            let prep = preprocess_rgb8(&up, uw, uh, &cfg).unwrap();
            assert_eq!(prep.grid_thw.to_vec(), usizes(&fx["grid_thw"]), "{name}");
            let bits: Vec<u8> = prep
                .pixels
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            assert_eq!(
                fnv1a64(&bits),
                fx["pixel_bf16_fnv1a64"].as_str().unwrap(),
                "{name} pixel_values"
            );
        }
    }

    /// The parser reproduces the exporter's PaddleX-grammar parse of the
    /// reference spotting generations (both precisions).
    #[test]
    fn paddleocr_vl_spotting_parses_reference_generations() {
        let m = manifest();
        for prec in ["bf16", "4bit"] {
            for fx in ["small_text", "page_text"] {
                let s = &m[prec]["fixtures"][fx]["spotting"];
                let text = s["markdown"].as_str().unwrap();
                assert_eq!(strip_location_tokens(text), s["plain"].as_str().unwrap());
                let got = parse_spotting_output(text);
                let want = s["spots"].as_array().unwrap();
                assert!(!want.is_empty(), "{prec} {fx}");
                assert_eq!(got.len(), want.len(), "{prec} {fx} spot count");
                for (g, w) in got.iter().zip(want) {
                    assert_eq!(g.text, w["text"].as_str().unwrap(), "{prec} {fx}");
                    for (k, corner) in w["quad"].as_array().unwrap().iter().enumerate() {
                        for d in 0..2 {
                            let v = corner[d].as_f64().unwrap() as f32;
                            assert!(
                                (g.quad[k][d] - v).abs() <= 1e-6,
                                "{prec} {fx} {:?}: {:?} vs {w}",
                                g.text,
                                g.quad
                            );
                        }
                    }
                }
            }
        }
    }

    const SMALL: &str = "Hello PaddleOCR-VL 2026<|LOC_25|><|LOC_197|><|LOC_728|><|LOC_197|><|LOC_728|><|LOC_393|><|LOC_25|><|LOC_393|>\nnative MLX in Rust.<|LOC_25|><|LOC_563|><|LOC_532|><|LOC_563|><|LOC_532|><|LOC_763|><|LOC_25|><|LOC_763|>";

    fn q(v: [u32; 8]) -> [[f32; 2]; 4] {
        quad_of(&v).unwrap()
    }

    #[test]
    fn paddleocr_vl_spotting_parses_reference_grammar() {
        let spots = parse_spotting_output(SMALL);
        assert_eq!(
            spots,
            vec![
                Spot {
                    text: "Hello PaddleOCR-VL 2026".into(),
                    quad: q([25, 197, 728, 197, 728, 393, 25, 393]),
                },
                Spot {
                    text: "native MLX in Rust.".into(),
                    quad: q([25, 563, 532, 563, 532, 763, 25, 763]),
                },
            ]
        );
        assert_eq!(spots[0].quad[0], [0.025, 0.197]);
        assert_eq!(spots[0].quad[2], [0.728, 0.393]);
        assert_eq!(
            strip_location_tokens(SMALL),
            "Hello PaddleOCR-VL 2026\nnative MLX in Rust."
        );
        // A trailing EOS / special markers do not disturb parsing.
        let wrapped = format!(
            "<|LOC_BEGIN|>{}<|LOC_END|></s>",
            SMALL.replace('\n', "<|LOC_SEP|>\n")
        );
        assert_eq!(parse_spotting_output(&wrapped), spots);
    }

    #[test]
    fn paddleocr_vl_spotting_malformed_items() {
        // Four values: an axis-aligned box.
        let s = parse_spotting_output("abc<|LOC_10|><|LOC_20|><|LOC_30|><|LOC_40|>");
        assert_eq!(
            s[0].quad,
            [[0.01, 0.02], [0.03, 0.02], [0.03, 0.04], [0.01, 0.04]]
        );
        // Wrong counts, empty text and trailing unlocated text are dropped;
        // the following item still parses.
        let raw = "bad<|LOC_1|><|LOC_2|><|LOC_3|>\n<|LOC_1|><|LOC_1|><|LOC_1|><|LOC_1|>\nok<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_0|><|LOC_1000|><|LOC_1000|><|LOC_0|><|LOC_1000|>\ntail";
        let s = parse_spotting_output(raw);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].text, "ok");
        assert_eq!(s[0].quad, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        // Out-of-grid values clamp; non-numeric LOC-like text stays text.
        let s = parse_spotting_output("x <|LOC_FOO|><|LOC_5000|><|LOC_0|><|LOC_0|><|LOC_0|>");
        assert_eq!(s[0].text, "x <|LOC_FOO|>");
        assert_eq!(s[0].quad[0], [1.0, 0.0]);
        assert!(parse_spotting_output("plain OCR text").is_empty());
        assert_eq!(strip_location_tokens(" plain "), "plain");
    }

    #[test]
    fn paddleocr_vl_spotting_upscale_rule() {
        let rgb = vec![200u8; 4 * 3 * 3];
        let (up, w, h) = spotting_upscale(&rgb, 4, 3).unwrap().unwrap();
        assert_eq!((w, h, up.len()), (8, 6, 8 * 6 * 3));
        assert!(up.iter().all(|&v| v == 200));
        assert!(spotting_upscale(&[0; 1500 * 3], 1500, 1).unwrap().is_none());
        assert!(spotting_upscale(&[0; 1499 * 3], 1, 1499).unwrap().is_some());
    }
}
