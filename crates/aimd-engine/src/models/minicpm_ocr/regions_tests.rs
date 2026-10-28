//! Hermetic goldens for the region pipeline (`regions`, `markdown`,
//! `latex`, `otsl`) against the MiniCPM-OCR SDK functions
//! (`scripts/export_minicpm_regions_reference.py` ->
//! `tests/golden/minicpm_regions/`). Exact string / pixel-hash equality.

use std::path::{Path, PathBuf};

use image::RgbImage;
use serde_json::Value;

use super::latex::repair_latex_formula;
use super::markdown::{
    IMAGE_PLACEHOLDER, RegionBlock, RegionFormat, assemble_markdown, classify_engine_text,
    has_mixed_inline_formula_content, normalize_and_dedupe_blocks, normalize_crop_text,
    postprocess_content, repair_display_formula, split_text_block, truncate_repetitive_content,
    truncate_repetitive_suffix,
};
use super::otsl::{convert_otsl_to_html, html_unescape, otsl_pad_to_sqr_v2};
use super::regions::{RegionJob, RegionOptions, finish_page, plan_page};
use crate::models::pp_doclayout_v3::postprocess::{LayoutRegion, PX_LABELS};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn load(name: &str) -> Value {
    let path = golden_dir().join("minicpm_regions").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("golden json")
}

fn s(v: &Value) -> &str {
    v.as_str()
        .unwrap_or_else(|| panic!("expected string, got {v}"))
}

fn u(v: &Value) -> usize {
    v.as_u64()
        .unwrap_or_else(|| panic!("expected integer, got {v}")) as usize
}

fn cases<'a>(golden: &'a Value, key: &str) -> &'a Vec<Value> {
    let list = golden[key]
        .as_array()
        .unwrap_or_else(|| panic!("missing {key}"));
    assert!(!list.is_empty(), "{key}: no cases");
    list
}

/// Collects mismatches so one run reports them all.
#[derive(Default)]
struct Mismatches(Vec<String>);

impl Mismatches {
    fn check(&mut self, what: &str, input: &dyn std::fmt::Debug, got: &str, want: &str) {
        if got != want {
            self.0.push(format!(
                "{what}({input:?}):\n   got {got:?}\n  want {want:?}"
            ));
        }
    }

    fn finish(self) {
        if !self.0.is_empty() {
            let shown: Vec<&String> = self.0.iter().take(12).collect();
            panic!(
                "{} mismatches:\n{}",
                self.0.len(),
                shown
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
}

fn block_from(v: &Value) -> RegionBlock {
    RegionBlock {
        order: u(&v["order"]),
        label: s(&v["label"]).to_string(),
        content: s(&v["content"]).to_string(),
        inline_formula_mixed: v["inline_formula_mixed"].as_bool().unwrap_or(false),
    }
}

#[test]
fn text_functions_match_sdk() {
    let g = load("text_cases.json");
    let mut m = Mismatches::default();
    for c in cases(&g, "truncate_repetitive_suffix") {
        let t = s(&c["text"]);
        m.check(
            "truncate_repetitive_suffix",
            &t,
            &truncate_repetitive_suffix(t),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "truncate_repetitive_content") {
        let (t, n) = (s(&c["content"]), u(&c["min_count"]));
        m.check(
            "truncate_repetitive_content",
            &(t, n),
            &truncate_repetitive_content(t, n),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "repair_display_formula") {
        let t = s(&c["content"]);
        m.check(
            "repair_display_formula",
            &t,
            &repair_display_formula(t),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "has_mixed_inline_formula_content") {
        let t = s(&c["content"]);
        let want = c["out"].as_bool().expect("bool").to_string();
        m.check(
            "has_mixed",
            &t,
            &has_mixed_inline_formula_content(t).to_string(),
            &want,
        );
    }
    for c in cases(&g, "split_text_block") {
        let (t, p) = (
            s(&c["content"]),
            c["protect_inline_math"].as_bool().expect("bool"),
        );
        m.check(
            "split_text_block",
            &(t, p),
            &split_text_block(t, p),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "repair_latex_formula") {
        let (t, l) = (s(&c["s"]), s(&c["label"]));
        m.check(
            "repair_latex_formula",
            &(t, l),
            &repair_latex_formula(t, l),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "postprocess_content") {
        let (t, f, l) = (s(&c["content"]), s(&c["output_format"]), s(&c["label"]));
        let format = RegionFormat::parse(f).expect("format");
        m.check(
            "postprocess_content",
            &(t, f, l),
            &postprocess_content(t, format, l),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "normalize_crop_text") {
        let (t, l) = (s(&c["text"]), s(&c["label"]));
        m.check(
            "normalize_crop_text",
            &(t, l),
            &normalize_crop_text(t, l),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "classify_engine_text") {
        let r = s(&c["raw"]);
        let (text, error) = classify_engine_text(r);
        m.check("classify_engine_text", &r, &text, s(&c["text"]));
        let want = c["error"].as_str().map(str::to_string);
        m.check(
            "classify_engine_text.error",
            &r,
            &format!("{error:?}"),
            &format!("{want:?}"),
        );
    }
    m.finish();
}

#[test]
fn otsl_and_unescape_match_sdk() {
    let g = load("text_cases.json");
    let mut m = Mismatches::default();
    for c in cases(&g, "otsl_pad_to_sqr_v2") {
        let t = s(&c["s"]);
        m.check(
            "otsl_pad_to_sqr_v2",
            &t,
            &otsl_pad_to_sqr_v2(t),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "convert_otsl_to_html") {
        let t = s(&c["s"]);
        m.check(
            "convert_otsl_to_html",
            &t,
            &convert_otsl_to_html(t),
            s(&c["out"]),
        );
    }
    for c in cases(&g, "html_unescape") {
        let t = s(&c["s"]);
        m.check("html_unescape", &t, &html_unescape(t), s(&c["out"]));
    }
    m.finish();
}

#[test]
fn block_assembly_matches_sdk() {
    let g = load("text_cases.json");
    let mut m = Mismatches::default();
    for (k, c) in cases(&g, "blocks").iter().enumerate() {
        let blocks: Vec<RegionBlock> = c["blocks"]
            .as_array()
            .expect("blocks")
            .iter()
            .map(block_from)
            .collect();
        let finals = normalize_and_dedupe_blocks(&blocks);
        let got: Vec<String> = finals
            .iter()
            .map(|b| format!("{}|{}|{}", b.order, b.label, b.content))
            .collect();
        let want: Vec<String> = c["final"]
            .as_array()
            .expect("final")
            .iter()
            .map(|b| format!("{}|{}|{}", u(&b["order"]), s(&b["label"]), s(&b["content"])))
            .collect();
        m.check(
            "normalize_and_dedupe_blocks",
            &k,
            &got.join("\n"),
            &want.join("\n"),
        );
        m.check(
            "assemble_markdown",
            &k,
            &assemble_markdown(&finals, true),
            s(&c["markdown"]),
        );
    }
    m.finish();
}

// --- pages ------------------------------------------------------------------------

fn fnv1a64(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The exporter's synthetic page: `(x * 7 + y * 13 + c * 31) % 251`.
fn synthetic_page(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        let v = |c: u32| ((x * 7 + y * 13 + c * 31) % 251) as u8;
        image::Rgb([v(0), v(1), v(2)])
    })
}

fn page_image(page: &Value) -> RgbImage {
    let (w, h) = (u(&page["width"]) as u32, u(&page["height"]) as u32);
    let source = s(&page["source"]);
    let img = if source == "synthetic" {
        synthetic_page(w, h)
    } else {
        image::open(golden_dir().join(source))
            .expect("page png")
            .to_rgb8()
    };
    assert_eq!(img.dimensions(), (w, h), "{source}");
    img
}

fn layout_regions(page: &Value) -> Vec<LayoutRegion> {
    page["regions"]
        .as_array()
        .expect("regions")
        .iter()
        .enumerate()
        .map(|(index, r)| {
            let label = s(&r["label"]);
            let class_id = PX_LABELS
                .iter()
                .position(|&l| l == label)
                .expect("PX label");
            let b: Vec<i32> = r["bbox"]
                .as_array()
                .expect("bbox")
                .iter()
                .map(|v| v.as_i64().expect("int") as i32)
                .collect();
            LayoutRegion {
                index,
                label: PX_LABELS[class_id],
                class_id,
                score: 0.9,
                bbox: [b[0], b[1], b[2], b[3]],
            }
        })
        .collect()
}

fn boxes(v: &Value) -> Vec<[i32; 4]> {
    v.as_array()
        .expect("boxes")
        .iter()
        .map(|b| {
            let b: Vec<i32> = b
                .as_array()
                .expect("box")
                .iter()
                .map(|x| x.as_i64().expect("int") as i32)
                .collect();
            [b[0], b[1], b[2], b[3]]
        })
        .collect()
}

fn plan(page: &Value) -> Vec<RegionJob> {
    plan_page(&layout_regions(page), &page_image(page))
}

#[test]
fn crop_plan_matches_sdk() {
    let g = load("crop_cases.json");
    let pages = g["pages"].as_array().expect("pages");
    assert!(pages.len() >= 4);
    for page in pages {
        let name = s(&page["name"]);
        let jobs = plan(page);
        let want = page["blocks"].as_array().expect("blocks");
        assert_eq!(jobs.len(), want.len(), "{name}: block count");
        for (job, b) in jobs.iter().zip(want) {
            let ctx = format!("{name} order {}", job.order);
            assert_eq!(job.order, u(&b["order"]), "{ctx}");
            assert_eq!(job.label, s(&b["label"]), "{ctx}");
            assert_eq!(
                vec![job.bbox],
                boxes(&serde_json::json!([b["bbox"]])),
                "{ctx}: bbox"
            );
            assert_eq!(job.regions, boxes(&b["regions"]), "{ctx}: regions");
            assert_eq!(
                job.image.width() as usize,
                u(&b["crop_width"]),
                "{ctx}: crop width"
            );
            assert_eq!(
                job.image.height() as usize,
                u(&b["crop_height"]),
                "{ctx}: crop height"
            );
            assert_eq!(
                fnv1a64(job.image.as_raw()),
                s(&b["crop_fnv"]),
                "{ctx}: crop pixels"
            );
            assert_eq!(
                job.recipe.is_some(),
                b["recognised"].as_bool().expect("bool"),
                "{ctx}"
            );
            assert_eq!(
                job.recipe.is_none(),
                b["skip_vlm"].as_bool().expect("bool"),
                "{ctx}"
            );
            if let Some(r) = job.recipe {
                assert_eq!(r.prompt, s(&b["prompt"]), "{ctx}");
                assert_eq!(r.max_new_tokens, u(&b["max_new_tokens"]), "{ctx}");
                assert_eq!(r.slice_nums, u(&b["slice_nums"]), "{ctx}");
                assert_eq!(r.output_format.as_str(), s(&b["output_format"]), "{ctx}");
                assert!(b["resize_min_side"].is_null(), "{ctx}: resize_min_side");
            }
        }
    }
}

#[test]
fn page_assembly_matches_sdk() {
    let crops = load("crop_cases.json");
    let e2e = load("e2e_cases.json");
    let crop_pages = crops["pages"].as_array().expect("pages");
    for page in e2e["pages"].as_array().expect("pages") {
        let name = s(&page["name"]);
        let crop_page = crop_pages
            .iter()
            .find(|p| s(&p["name"]) == name)
            .expect("crop page");
        let jobs = plan(crop_page);
        let raw_of = |order: usize| {
            page["raw"]
                .as_array()
                .expect("raw")
                .iter()
                .find(|r| u(&r["order"]) == order)
                .map(|r| s(&r["text"]).to_string())
        };
        // The SDK engine saw exactly the recognised crops, in order.
        let calls = page["calls"].as_array().expect("calls");
        let recognised: Vec<&RegionJob> = jobs.iter().filter(|j| j.recipe.is_some()).collect();
        assert_eq!(recognised.len(), calls.len(), "{name}: engine calls");
        for (job, call) in recognised.iter().zip(calls) {
            let r = job.recipe.expect("recipe");
            assert_eq!(
                fnv1a64(job.image.as_raw()),
                s(&call["fnv"]),
                "{name}: model input"
            );
            assert_eq!(r.prompt, s(&call["prompt"]), "{name}");
            assert_eq!(r.max_new_tokens, u(&call["max_new_tokens"]), "{name}");
            assert_eq!(r.slice_nums, u(&call["slice_nums"]), "{name}");
        }
        let raw: Vec<Option<String>> = jobs
            .iter()
            .map(|j| j.recipe.and_then(|_| raw_of(j.order)))
            .collect();
        let out = finish_page(
            &jobs,
            &raw,
            &RegionOptions {
                emit_image_placeholder: true,
            },
        );
        assert_eq!(out.markdown, s(&page["markdown"]), "{name}: markdown");
        let finals = page["final"].as_array().expect("final");
        assert_eq!(out.blocks.len(), finals.len(), "{name}");
        for (block, want) in out.blocks.iter().zip(finals) {
            assert_eq!(block.order, u(&want["order"]), "{name}");
            assert_eq!(block.label, s(&want["label"]), "{name}");
            assert_eq!(
                block.content,
                s(&want["content"]),
                "{name} order {}",
                block.order
            );
        }
        let pieces: Vec<&str> = out
            .blocks
            .iter()
            .filter_map(|b| b.markdown.as_deref())
            .collect();
        assert_eq!(pieces.join("\n\n"), out.markdown, "{name}: pieces");

        // aimd default: no image placeholder, otherwise identical pieces.
        let aimd = finish_page(&jobs, &raw, &RegionOptions::default());
        assert!(!aimd.markdown.contains(IMAGE_PLACEHOLDER), "{name}");
        let kept: Vec<&str> = pieces
            .iter()
            .copied()
            .filter(|p| *p != IMAGE_PLACEHOLDER)
            .collect();
        assert_eq!(aimd.markdown, kept.join("\n\n"), "{name}: aimd markdown");
    }
}

#[test]
fn recognize_page_drives_the_closure_in_order() {
    let g = load("crop_cases.json");
    let page = g["pages"]
        .as_array()
        .expect("pages")
        .iter()
        .find(|p| s(&p["name"]) == "landscape")
        .expect("landscape");
    let mut seen = Vec::new();
    let out = super::regions::recognize_page::<String>(
        &layout_regions(page),
        &page_image(page),
        &RegionOptions::default(),
        &mut |job, recipe| {
            seen.push((job.order, recipe.prompt));
            Ok(format!("block {}", job.order))
        },
    )
    .expect("page");
    let orders: Vec<usize> = seen.iter().map(|(o, _)| *o).collect();
    assert_eq!(orders, vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]);
    assert!(
        out.markdown
            .starts_with("## block 0\n\n## block 1\n\nblock 2")
    );
    // A failing recogniser aborts the page.
    let err = super::regions::recognize_page::<&str>(
        &layout_regions(page),
        &page_image(page),
        &RegionOptions::default(),
        &mut |_, _| Err("cancelled"),
    );
    assert_eq!(err, Err("cancelled"));
}
