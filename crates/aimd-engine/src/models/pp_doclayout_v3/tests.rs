//! PP-DocLayoutV3 parity tests against `tests/golden/pp_doclayout_v3/`.
//!
//! Hermetic: the cv2 `INTER_CUBIC` resize port reproduces the golden
//! resized pages bit for bit, and the SDK post-process run on the golden
//! raw outputs reproduces the golden region lists. Checkpoint tier (Metal +
//! cached snapshot, one shared load): backbone / encoder / mask-feature
//! stage slices, the decoder from the reference's queries and initial
//! boxes, and the final regions end to end.

use aimd_mlx::MlxArray;
use serde_json::Value;

use super::config::{INPUT_SIZE, NUM_QUERIES};
use super::model::Overrides;
use super::postprocess::{LayoutRegion, PX_LABELS, postprocess, reading_order};
use super::preprocess::preprocess;
use super::test_support::{manifest, pages, read_f32, rgb_image, shared_pipeline};

fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn golden_regions(v: &Value) -> Vec<LayoutRegion> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let class_id = usize_of(&r["class_id"]);
            let bbox = r["bbox"].as_array().unwrap();
            LayoutRegion {
                index: usize_of(&r["index"]),
                label: PX_LABELS[class_id],
                class_id,
                score: r["score"].as_f64().unwrap() as f32,
                bbox: std::array::from_fn(|i| bbox[i].as_i64().unwrap() as i32),
            }
        })
        .collect()
}

/// Same regions in the same order with the same labels; boxes within
/// `px` pixels and scores within `score_tol`.
fn assert_regions(got: &[LayoutRegion], want: &[LayoutRegion], px: i32, score_tol: f32, ctx: &str) {
    let labels = |r: &[LayoutRegion]| r.iter().map(|x| x.label).collect::<Vec<_>>();
    assert_eq!(labels(got), labels(want), "{ctx}: labels / order");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.index, w.index, "{ctx}");
        assert_eq!(g.class_id, w.class_id, "{ctx}");
        let d = g
            .bbox
            .iter()
            .zip(&w.bbox)
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap();
        assert!(
            d <= px,
            "{ctx}: region {} bbox {:?} vs {:?}",
            g.index,
            g.bbox,
            w.bbox
        );
        assert!(
            (g.score - w.score).abs() <= score_tol,
            "{ctx}: region {} score {} vs {}",
            g.index,
            g.score,
            w.score
        );
    }
}

#[test]
fn cubic_resize_matches_cv2_bit_for_bit() {
    for page in pages() {
        let name = page["name"].as_str().unwrap();
        let rgb = rgb_image(&format!("images/{name}.png"));
        let (pixels, bgr) = preprocess(&rgb);
        // `cv2.imwrite` stored the BGR array, so the PNG decodes to RGB.
        let want: Vec<u8> = rgb_image(&format!("{name}_resized.png"))
            .as_raw()
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        let diff = bgr.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{name}: {diff} resized bytes differ from cv2");
        // The model input slice is exactly `x * float32(1 / 255)`.
        let spec = &page["slices"]["pixels"];
        let s = usize_of(&spec["s"]);
        let got: Vec<f32> = (0..INPUT_SIZE)
            .step_by(s)
            .flat_map(|y| (0..INPUT_SIZE).step_by(s).map(move |x| (y, x)))
            .flat_map(|(y, x)| (0..3).map(move |c| (y * INPUT_SIZE + x) * 3 + c))
            .map(|i| pixels[i])
            .collect();
        assert_eq!(
            got,
            read_f32(&format!("{name}_pixels.f32")),
            "{name}: pixels"
        );
    }
}

#[test]
fn postprocess_reproduces_sdk_regions() {
    for page in pages() {
        let name = page["name"].as_str().unwrap();
        let got = postprocess(
            &read_f32(&format!("{name}_logits.f32")),
            &read_f32(&format!("{name}_boxes.f32")),
            &read_f32(&format!("{name}_order.f32")),
            page["width"].as_u64().unwrap() as u32,
            page["height"].as_u64().unwrap() as u32,
        );
        let want = golden_regions(&page["regions"]);
        assert!(!want.is_empty());
        assert_regions(&got, &want, 0, 1e-6, name);
    }
    let syn = &manifest()["synthetic_post"];
    let order = read_f32("synthetic_post_order.f32");
    let seq: Vec<usize> = syn["order_seq"]
        .as_array()
        .unwrap()
        .iter()
        .map(usize_of)
        .collect();
    assert_eq!(reading_order(&order, NUM_QUERIES), seq, "synthetic order");
    let got = postprocess(
        &read_f32("synthetic_post_logits.f32"),
        &read_f32("synthetic_post_boxes.f32"),
        &order,
        syn["width"].as_u64().unwrap() as u32,
        syn["height"].as_u64().unwrap() as u32,
    );
    let want = golden_regions(&syn["regions"]);
    assert!(want.len() > 20);
    assert_regions(&got, &want, 0, 1e-6, "synthetic");
}

/// `[1, H, W, C]` -> host `x[::s, ::s, ::cs]`.
fn nhwc_slice(a: &MlxArray, s: usize, cs: usize) -> Vec<f32> {
    let d = a.shape().unwrap();
    let (h, w, c) = (d[1], d[2], d[3]);
    let v = a.to_f32_vec().unwrap();
    let mut out = Vec::new();
    for y in (0..h).step_by(s) {
        for x in (0..w).step_by(s) {
            for ch in (0..c).step_by(cs) {
                out.push(v[(y * w + x) * c + ch]);
            }
        }
    }
    out
}

/// `(max |a - b|, max |b|)`.
fn err(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len());
    let d = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (d, want.iter().fold(0f32, |m, v| m.max(v.abs())))
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32) {
    let (d, scale) = err(got, want);
    eprintln!(
        "{name}: max abs err {d:.3e} (ref max {scale:.3e}, rel {:.3e})",
        d / scale
    );
    assert!(
        d <= rel * scale.max(1.0),
        "{name}: max abs err {d} > {rel} * {scale}"
    );
}

fn page_pixels(page: &Value) -> Vec<f32> {
    let name = page["name"].as_str().unwrap();
    preprocess(&rgb_image(&format!("images/{name}.png"))).0
}

#[test]
fn checkpoint_stages_match_reference() {
    let Some(p) = shared_pipeline() else { return };
    for page in pages() {
        let name = page["name"].as_str().unwrap();
        let (_, t) = p
            .model()
            .forward_impl(&page_pixels(&page), INPUT_SIZE, &Overrides::default())
            .unwrap();
        let slices = &page["slices"];
        let stage_arrays = t
            .stages
            .iter()
            .enumerate()
            .map(|(i, a)| (format!("stage{}", i + 1), a))
            .chain(
                t.pan
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (format!("pan{i}"), a)),
            )
            .chain(std::iter::once(("mask_feat".to_string(), &t.mask_feat)));
        for (key, a) in stage_arrays {
            let spec = &slices[&key];
            let got = nhwc_slice(a, usize_of(&spec["s"]), usize_of(&spec["cs"]));
            check(
                &format!("{name}/{key}"),
                &got,
                &read_f32(&format!("{name}_{key}.f32")),
                1e-4,
            );
        }
        let want_topk: Vec<usize> = page["topk"]
            .as_array()
            .unwrap()
            .iter()
            .map(usize_of)
            .collect();
        let same = t.topk.iter().filter(|i| want_topk.contains(i)).count();
        eprintln!("{name}: top-300 overlap {same}/300");
        assert!(same >= 295, "{name}: top-300 overlap {same}");
        // Initial boxes come from thresholded 200x200 masks: a pixel whose
        // mask logit sits at ~0 can flip and move one edge by 1/200, so
        // compare per query and require nearly all to match.
        let want_init = read_f32(&format!("{name}_init_ref.f32"));
        let close = (0..t.topk.len())
            .filter(|&i| {
                let Some(j) = want_topk.iter().position(|&w| w == t.topk[i]) else {
                    return false;
                };
                (0..4).all(|k| (t.init_ref[i * 4 + k] - want_init[j * 4 + k]).abs() < 1e-4)
            })
            .count();
        eprintln!("{name}: initial boxes matching {close}/300");
        assert!(close >= 285, "{name}: initial boxes matching {close}/300");
    }
}

#[test]
fn checkpoint_decoder_matches_reference_from_reference_queries() {
    let Some(p) = shared_pipeline() else { return };
    for page in pages() {
        let name = page["name"].as_str().unwrap();
        let ov = Overrides {
            topk: Some(
                page["topk"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(usize_of)
                    .collect(),
            ),
            init_ref: Some(read_f32(&format!("{name}_init_ref.f32"))),
        };
        let (raw, t) = p
            .model()
            .forward_impl(&page_pixels(&page), INPUT_SIZE, &ov)
            .unwrap();
        let hidden = |i: usize| -> Vec<f32> {
            let v = t.hidden[i].to_f32_vec().unwrap();
            v.as_chunks::<256>()
                .0
                .iter()
                .flat_map(|r| r.iter().step_by(4).copied())
                .collect()
        };
        check(
            &format!("{name}/hidden0"),
            &hidden(0),
            &read_f32(&format!("{name}_hidden0.f32")),
            2e-4,
        );
        check(
            &format!("{name}/hidden5"),
            &hidden(5),
            &read_f32(&format!("{name}_hidden5.f32")),
            5e-3,
        );
        let refs: Vec<f32> = t.refs.concat();
        check(
            &format!("{name}/refs"),
            &refs,
            &read_f32(&format!("{name}_refs.f32")),
            1e-3,
        );
        check(
            &format!("{name}/logits"),
            &raw.logits,
            &read_f32(&format!("{name}_logits.f32")),
            2e-3,
        );
        check(
            &format!("{name}/boxes"),
            &raw.boxes,
            &read_f32(&format!("{name}_boxes.f32")),
            1e-3,
        );
        // Compare the order logits above the diagonal (the rest is -1e4).
        let want = read_f32(&format!("{name}_order.f32"));
        let upper = |v: &[f32]| -> Vec<f32> {
            (0..NUM_QUERIES)
                .flat_map(|i| ((i + 1)..NUM_QUERIES).map(move |j| (i, j)))
                .map(|(i, j)| v[i * NUM_QUERIES + j])
                .collect()
        };
        check(
            &format!("{name}/order"),
            &upper(&raw.order),
            &upper(&want),
            5e-3,
        );
    }
}

#[test]
fn checkpoint_detect_matches_sdk_regions() {
    let Some(p) = shared_pipeline() else { return };
    for page in pages() {
        let name = page["name"].as_str().unwrap();
        let got = p.detect(&rgb_image(&format!("images/{name}.png"))).unwrap();
        for r in &got {
            eprintln!(
                "{name}: {} {} {:.4} {:?}",
                r.index, r.label, r.score, r.bbox
            );
        }
        assert_regions(&got, &golden_regions(&page["regions"]), 1, 2e-3, name);
    }
}
