//! PP-OCRv6 goldens (`tests/golden/ppocrv6/`, mlx4ocr 0.1.2).
//!
//! - Hermetic: host ops (resize, perspective crops, rec preprocessing,
//!   detector sizing, DB post-process, reading order, `np.argsort`, f32
//!   mean) are compared bit-exactly with OpenCV / numpy output.
//! - Checkpoint (GPU + HF cache): detector probability map, detections,
//!   the first recognition batch and the final lines for each golden image.
//! - E2E (`AIMD_E2E=1`): `recognize_path` on the golden PNGs.

use super::ctc::{np_argsort, np_mean_f32};
use super::db::{DbParams, db_postprocess, reading_order};
use super::image_ops::{BgrImage, resize_linear, rotate_crop};
use super::pipeline::{
    PipelineError, det_padded_size, det_preprocess, det_resize_target, rec_batches, rec_preprocess,
};
use super::test_support::{
    e2e_enabled, golden_image, manifest, quad_of, read_bytes, read_f32, shared_pipeline, usize_of,
};

fn synthetic() -> BgrImage {
    golden_image("synthetic.png")
}

#[test]
fn resize_matches_cv2_inter_linear() {
    let m = manifest();
    let card = synthetic();
    let want = read_bytes("resize.u8");
    let mut off = 0;
    for case in m["resize"].as_array().unwrap() {
        let (w, h) = (usize_of(&case[0]), usize_of(&case[1]));
        let got = resize_linear(&card, w, h);
        let n = w * h * 3;
        assert_eq!(got.data, want[off..off + n], "resize {w}x{h}");
        off += n;
    }
    assert_eq!(off, want.len());
}

#[test]
fn rotate_crop_matches_cv2_warp_perspective() {
    let m = manifest();
    let card = synthetic();
    let want = read_bytes("crops.u8");
    let mut off = 0;
    for case in m["crops"].as_array().unwrap() {
        let quad = quad_of(&case["quad"]);
        let (w, h) = (usize_of(&case["w"]), usize_of(&case["h"]));
        let got = rotate_crop(&card, &quad);
        assert_eq!((got.width, got.height), (w, h), "crop size {quad:?}");
        let n = w * h * 3;
        assert_eq!(got.data, want[off..off + n], "crop pixels {quad:?}");
        off += n;
    }
    assert_eq!(off, want.len());
}

#[test]
fn rec_preprocess_matches_resize_norm_img() {
    let m = manifest();
    let case = &m["rec_in"];
    let crop_case = &m["crops"][usize_of(&case["crop"])];
    let crop = rotate_crop(&synthetic(), &quad_of(&crop_case["quad"]));
    let (got, w) = rec_preprocess(&crop, case["max_wh_ratio"].as_f64().unwrap());
    assert_eq!(w, usize_of(&case["w"]));
    assert_eq!(got, read_f32("rec_in.f32"));
}

#[test]
fn det_sizes_match_resize_det_image() {
    for case in manifest()["det_resize"].as_array().unwrap() {
        let (w, h) = (usize_of(&case["src"][0]), usize_of(&case["src"][1]));
        let (pw, ph) = det_padded_size(w, h);
        let shape = &case["shape"];
        assert_eq!(
            (ph as f64, pw as f64),
            (shape[0].as_f64().unwrap(), shape[1].as_f64().unwrap()),
            "padded source of {w}x{h}"
        );
        let want = (usize_of(&case["resized"][0]), usize_of(&case["resized"][1]));
        assert_eq!(det_resize_target(pw, ph), want, "target of {w}x{h}");
    }
}

#[test]
fn db_postprocess_matches_mlx4ocr() {
    let m = manifest();
    let db = &m["db"];
    let (w, h) = (usize_of(&db["w"]), usize_of(&db["h"]));
    let (src_w, src_h) = (usize_of(&db["src_w"]), usize_of(&db["src_h"]));
    let prob = read_f32("db_prob.f32");
    let dets = db_postprocess(&prob, w, h, src_w, src_h, &DbParams::default());
    let want = db["dets"].as_array().unwrap();
    assert_eq!(dets.len(), want.len(), "detection count");
    for (got, want) in dets.iter().zip(want) {
        assert_eq!(got.quad, quad_of(&want["quad"]), "quad");
        let ws = want["score"].as_f64().unwrap();
        assert!((got.score - ws).abs() < 1e-9, "score {} vs {ws}", got.score);
    }
    let quads: Vec<_> = dets.iter().map(|d| d.quad).collect();
    let order: Vec<usize> = db["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(usize_of)
        .collect();
    assert_eq!(reading_order(&quads), order);
}

#[test]
fn argsort_and_mean_match_numpy() {
    let m = manifest();
    let values: Vec<f64> = m["argsort"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let want: Vec<usize> = m["argsort"]["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(usize_of)
        .collect();
    assert_eq!(np_argsort(&values), want);
    for case in m["mean"].as_array().unwrap() {
        let v: Vec<f32> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            np_mean_f32(&v),
            case["mean"].as_f64().unwrap() as f32,
            "mean of {}",
            v.len()
        );
    }
}

/// Checkpoint-tier bounds.
///
/// The port is bit-exact with mlx4ocr on Metal for these images (identical
/// probability maps, step maxima, scores and quads); the Metal bounds only
/// leave room for MLX kernel changes across versions. mlx's CUDA backend runs
/// f32 convs and matmuls in TF32 by default (`MLX_ENABLE_TF32`), which moves
/// the probability map by ~2e-2, rec step maxima by ~3e-2 (flipping the odd
/// step argmax without changing the collapsed text) and box corners by a
/// pixel, so off Metal these are noise bounds and the final lines' text must
/// still match exactly.
struct Bounds {
    det_prob: f32,
    rec_prob: f32,
    rec_argmax_misses: usize,
    score: f32,
    quad_px: f32,
}

fn bounds() -> Bounds {
    if aimd_mlx::metal_kernels_available().unwrap_or(false) {
        Bounds {
            det_prob: 1e-3,
            rec_prob: 1e-4,
            rec_argmax_misses: 0,
            score: 1e-3,
            quad_px: 0.0,
        }
    } else {
        Bounds {
            det_prob: 5e-2,
            rec_prob: 1e-1,
            rec_argmax_misses: 2,
            score: 1e-2,
            quad_px: 2.0,
        }
    }
}

/// Checkpoint tier: every stage of the pipeline on each golden image.
#[test]
fn checkpoint_pipeline_matches_mlx4ocr() {
    let Some(pipe) = shared_pipeline() else {
        return;
    };
    let bound = bounds();
    let m = manifest();
    for entry in m["images"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let image = golden_image(&format!("images/{name}.png"));
        assert_eq!(
            (image.width, image.height),
            (usize_of(&entry["width"]), usize_of(&entry["height"]))
        );

        // Detector probability map (sampled every `prob_step` pixels).
        let input = det_preprocess(&image);
        let (w, h) = (input.resized.width, input.resized.height);
        assert_eq!(
            (w, h),
            (
                usize_of(&entry["resized"][0]),
                usize_of(&entry["resized"][1])
            ),
            "{name}: det size"
        );
        let prob = pipe.det_prob_map(&input).unwrap();
        let step = usize_of(&entry["prob_step"]);
        let want = read_f32(&format!("{name}_prob.f32"));
        let mut max_abs = 0f32;
        let mut k = 0;
        for y in (0..h).step_by(step) {
            for x in (0..w).step_by(step) {
                max_abs = max_abs.max((prob[y * w + x] - want[k]).abs());
                k += 1;
            }
        }
        assert_eq!(k, want.len(), "{name}: prob sample count");
        eprintln!("{name}: det prob max abs diff {max_abs:.3e} over {k} samples");
        assert!(
            max_abs < bound.det_prob,
            "{name}: det prob max abs diff {max_abs}"
        );

        // Detections and the first recognition batch.
        let dets = pipe.detect(&image).unwrap();
        assert_eq!(
            dets.len(),
            usize_of(&entry["num_dets"]),
            "{name}: detections"
        );
        let crops: Vec<BgrImage> = dets.iter().map(|d| rotate_crop(&image, &d.quad)).collect();
        let (indices, max_wh) = rec_batches(&crops).into_iter().next().unwrap();
        let rec0 = &entry["rec0"];
        let want_idx: Vec<usize> = rec0["crops"]
            .as_array()
            .unwrap()
            .iter()
            .map(usize_of)
            .collect();
        assert_eq!(indices, want_idx, "{name}: first rec batch");
        let mut batch = Vec::new();
        let mut img_w = 0;
        for &i in &indices {
            let (slab, w) = rec_preprocess(&crops[i], max_wh);
            batch.extend_from_slice(&slab);
            img_w = w;
        }
        let (idx, max, steps) = pipe.rec_step_maxima(&batch, indices.len(), img_w).unwrap();
        let (mut argmax_miss, mut max_abs) = (0usize, 0f32);
        for (b, (want_a, want_m)) in rec0["argmax"]
            .as_array()
            .unwrap()
            .iter()
            .zip(rec0["max"].as_array().unwrap())
            .enumerate()
        {
            let (want_a, want_m) = (want_a.as_array().unwrap(), want_m.as_array().unwrap());
            assert_eq!(want_a.len(), steps, "{name}: rec steps");
            for t in 0..steps {
                argmax_miss += (idx[b * steps + t] as u64 != want_a[t].as_u64().unwrap()) as usize;
                max_abs =
                    max_abs.max((max[b * steps + t] - want_m[t].as_f64().unwrap() as f32).abs());
            }
        }
        eprintln!(
            "{name}: rec batch 0 argmax mismatches {argmax_miss}, max-prob abs diff {max_abs:.3e}"
        );
        assert!(
            argmax_miss <= bound.rec_argmax_misses,
            "{name}: rec argmax mismatches {argmax_miss}"
        );
        assert!(
            max_abs < bound.rec_prob,
            "{name}: rec max prob diff {max_abs}"
        );

        // Final lines.
        let out = pipe.recognize_image(&image).unwrap();
        let want_lines = entry["lines"].as_array().unwrap();
        assert_eq!(out.lines.len(), want_lines.len(), "{name}: line count");
        let mut worst_score = 0f32;
        for (got, want) in out.lines.iter().zip(want_lines) {
            assert_eq!(got.text, want["text"].as_str().unwrap(), "{name}: text");
            let want_quad = quad_of(&want["quad"]);
            let quad_diff = got
                .quad
                .iter()
                .flatten()
                .zip(want_quad.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                quad_diff <= bound.quad_px,
                "{name}: quad of {:?}: {:?} vs {want_quad:?}",
                got.text,
                got.quad
            );
            worst_score =
                worst_score.max((got.score - want["score"].as_f64().unwrap() as f32).abs());
        }
        eprintln!(
            "{name}: {} lines, worst score diff {worst_score:.3e}",
            out.lines.len()
        );
        assert!(
            worst_score < bound.score,
            "{name}: score diff {worst_score}"
        );
        let text: Vec<&str> = want_lines
            .iter()
            .map(|l| l["text"].as_str().unwrap())
            .collect();
        assert_eq!(out.markdown, format!("{}\n", text.join("\n")));
    }

    // A blank page has no detections.
    let blank = BgrImage {
        width: 96,
        height: 64,
        data: vec![255; 96 * 64 * 3],
    };
    assert!(matches!(
        pipe.recognize_image(&blank),
        Err(PipelineError::EmptyContent)
    ));
}

/// E2E tier: file decode + full pipeline, with per-page latency.
#[test]
fn e2e_recognize_path_golden_images() {
    if !e2e_enabled() {
        eprintln!("skipping: set AIMD_E2E=1 for the PP-OCRv6 E2E tier");
        return;
    }
    let Some(pipe) = shared_pipeline() else {
        return;
    };
    let m = manifest();
    for entry in m["images"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        let path = super::test_support::golden_dir().join(format!("images/{name}.png"));
        let t = std::time::Instant::now();
        let out = pipe.recognize_path(&path).unwrap();
        let first = t.elapsed();
        let t = std::time::Instant::now();
        pipe.recognize_path(&path).unwrap();
        let warm = t.elapsed();
        let text: Vec<&str> = entry["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["text"].as_str().unwrap())
            .collect();
        assert_eq!(out.markdown, format!("{}\n", text.join("\n")), "{name}");
        eprintln!("{name}: first {first:?}, warm {warm:?}");
    }
}
