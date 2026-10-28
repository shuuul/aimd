//! The MiniCPM-OCR SDK region pipeline between layout detection and page
//! Markdown (spec 009 WS-05): crop planning, the per-label prompt map, the
//! block executor's post-processing, and Markdown assembly.
//!
//! Ported from the official SDK (`sdk/minicpmvocr`, default config):
//!
//! - crop chain of the `ppdoclayout3_torch` provider
//!   (`PPDocLayoutV3TorchDetector.crop`): `_filter_overlap_boxes` (drop
//!   `reference`, boxes under 6 px, inline formulas overlapping > 0.5, the
//!   smaller of two boxes overlapping > 0.7), rectangle crops (no polygons,
//!   so no masking), `_merge_blocks` (side-by-side or aligned stacked text
//!   blocks are pasted onto one white canvas and recognised once);
//! - `_layout_record._build_layout_from_blocks`: folded blocks disappear,
//!   their boxes stay in the leader's `regions`, `order` is the index in
//!   the crop list;
//! - `prompts.DEFAULT_PROMPT_MAP` with `apply_skip_vlm_labels` for the
//!   default `skip_vlm_labels` (`image`, `header_image`, `footer_image`;
//!   chart and seal recognition are on), legacy override mode, no
//!   `resize_min_side`, `slice_nums` 9 everywhere;
//! - `_blocks._execute_block` (`classify_engine_text`,
//!   `postprocess_content`, `inline_formula_mixed`) and `_finalize_page`
//!   (`_normalize_and_dedupe_blocks`, `assemble_markdown`).
//!
//! The recogniser is a closure, so the whole driver runs without weights;
//! `tests/golden/minicpm_regions/` pins it to the SDK
//! (`scripts/export_minicpm_regions_reference.py`).

use image::{Rgb, RgbImage};

use super::markdown::{
    IMAGE_LABELS, RegionBlock, RegionFormat, assemble_parts, classify_engine_text,
    has_mixed_inline_formula_content, normalize_and_dedupe_blocks, postprocess_content,
};
use crate::models::pp_doclayout_v3::postprocess::LayoutRegion;

/// `constants.NON_MERGE_LABELS`.
pub const NON_MERGE_LABELS: [&str; 6] = [
    "image",
    "header_image",
    "footer_image",
    "chart",
    "seal",
    "table",
];

/// One label's recipe (`prompts.get_element_cfg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementRecipe {
    /// Task prompt after the image.
    pub prompt: &'static str,
    /// Generation budget (`max_new_tokens`).
    pub max_new_tokens: usize,
    /// Slice cap (`slice_nums`, MiniCPM-V `max_slice_nums`).
    pub slice_nums: usize,
    pub output_format: RegionFormat,
}

const TEXT_PROMPT: &str = "Text Recognition:";
/// `DEFAULT_PROMPT_MAP["default"]["slice_nums"]` (no element overrides it).
pub const DEFAULT_SLICE_NUMS: usize = 9;

/// `get_element_cfg(load_prompt_map(DEFAULT_CONFIG), label)`: `None` for the
/// labels that never reach the VLM (the image labels).
pub fn element_recipe(label: &str) -> Option<ElementRecipe> {
    use RegionFormat::{Html, Latex, Text};
    if IMAGE_LABELS.contains(&label) {
        return None;
    }
    let (prompt, max_new_tokens, output_format) = match label {
        "text" | "content" | "abstract" | "algorithm" => (TEXT_PROMPT, 4096, Text),
        "reference" | "reference_content" | "vertical_text" => (TEXT_PROMPT, 2048, Text),
        "vision_footnote" | "footnote" | "aside_text" => (TEXT_PROMPT, 1024, Text),
        "doc_title" | "paragraph_title" | "figure_title" | "header" | "footer" => {
            (TEXT_PROMPT, 512, Text)
        }
        "table" => ("Table Recognition:", 8192, Html),
        "display_formula" => ("Formula Recognition:", 2048, Latex),
        "inline_formula" => (TEXT_PROMPT, 1024, Latex),
        "formula_number" | "number" => (TEXT_PROMPT, 256, Text),
        "chart" => ("Chart to Table:", 4096, Html),
        "seal" => ("Seal Recognition:", 256, Text),
        // `DEFAULT_PROMPT_MAP["default"]`.
        _ => (TEXT_PROMPT, 8192, Text),
    };
    Some(ElementRecipe {
        prompt,
        max_new_tokens,
        slice_nums: DEFAULT_SLICE_NUMS,
        output_format,
    })
}

/// One block of a page, ready for the recogniser.
#[derive(Clone, Debug)]
pub struct RegionJob {
    /// Reading-order index (the SDK `order`; folded blocks leave gaps).
    pub order: usize,
    pub label: &'static str,
    /// Leader box `[x0, y0, x1, y1]` in page pixels.
    pub bbox: [i32; 4],
    /// `bbox` followed by the boxes folded into this block's crop.
    pub regions: Vec<[i32; 4]>,
    /// The crop sent to the model (a merged canvas for folded blocks).
    pub image: RgbImage,
    /// `None` when the block never reaches the VLM.
    pub recipe: Option<ElementRecipe>,
}

/// Output knobs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionOptions {
    /// Emit the SDK's `![image]()` for image regions and empty chart / seal
    /// regions. aimd leaves it off (spec 009: image regions emit nothing).
    pub emit_image_placeholder: bool,
}

/// One block of the assembled page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageBlock {
    pub order: usize,
    pub label: &'static str,
    pub bbox: [i32; 4],
    pub regions: Vec<[i32; 4]>,
    /// Whether the recogniser ran on it.
    pub recognised: bool,
    /// Normalized block content (`crop_blocks[].content`).
    pub content: String,
    /// This block's piece of [`PageOutput::markdown`], if it contributes one.
    pub markdown: Option<String>,
}

/// A recognised page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageOutput {
    /// The SDK page Markdown: block pieces joined with `"\n\n"`.
    pub markdown: String,
    /// Final blocks in reading order (anchors: label, boxes, piece).
    pub blocks: Vec<PageBlock>,
}

// --- crop planning ------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Det {
    label: &'static str,
    coord: [f64; 4],
}

/// `_filter_overlap_boxes(boxes, "auto")` without polygons.
fn filter_overlap_boxes(boxes: Vec<Det>) -> Vec<Det> {
    let boxes: Vec<Det> = boxes
        .into_iter()
        .filter(|b| b.label != "reference")
        .collect();
    let n = boxes.len();
    let area = |b: &Det| ((b.coord[2] - b.coord[0]) * (b.coord[3] - b.coord[1])).abs();
    let overlap = |a: &Det, b: &Det| {
        let iw = (a.coord[2].min(b.coord[2]) - a.coord[0].max(b.coord[0])).max(0.0);
        let ih = (a.coord[3].min(b.coord[3]) - a.coord[1].max(b.coord[1])).max(0.0);
        let small = area(a).min(area(b));
        if small > 0.0 { iw * ih / small } else { 0.0 }
    };
    let mut dropped = vec![false; n];
    for i in 0..n {
        let (w, h) = (
            boxes[i].coord[2] - boxes[i].coord[0],
            boxes[i].coord[3] - boxes[i].coord[1],
        );
        if w < 6.0 || h < 6.0 {
            dropped[i] = true;
            continue;
        }
        for j in i + 1..n {
            if dropped[i] || dropped[j] {
                continue;
            }
            let ratio = overlap(&boxes[i], &boxes[j]);
            let (li, lj) = (boxes[i].label, boxes[j].label);
            if (li == "inline_formula" || lj == "inline_formula") && ratio > 0.5 {
                if li == "inline_formula" {
                    dropped[i] = true;
                }
                if lj == "inline_formula" {
                    dropped[j] = true;
                }
                continue;
            }
            if ratio > 0.7 {
                if area(&boxes[i]) >= area(&boxes[j]) {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                }
            }
        }
    }
    boxes
        .into_iter()
        .zip(dropped)
        .filter(|(_, d)| !d)
        .map(|(b, _)| b)
        .collect()
}

/// `img_rgb[ymin:ymax, xmin:xmax]` with `int()` coordinates (clamped like
/// numpy slicing; layout boxes are already inside the page).
fn crop(page: &RgbImage, coord: [f64; 4]) -> RgbImage {
    let (w, h) = (page.width() as i64, page.height() as i64);
    let c = |v: f64, hi: i64| (v as i64).clamp(0, hi);
    let (x0, y0) = (c(coord[0], w), c(coord[1], h));
    let (x1, y1) = (c(coord[2], w).max(x0), c(coord[3], h).max(y0));
    image::imageops::crop_imm(
        page,
        x0 as u32,
        y0 as u32,
        (x1 - x0) as u32,
        (y1 - y0) as u32,
    )
    .to_image()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Align {
    Left,
    Right,
    Center,
}

/// `_calculate_projection_overlap_ratio(b1, b2, "horizontal", "union")`.
fn horizontal_overlap(b1: &[f64; 4], b2: &[f64; 4]) -> f64 {
    let overlap = b1[2].min(b2[2]) - b1[0].max(b2[0]);
    if overlap <= 0.0 {
        return 0.0;
    }
    let reference = b1[2].max(b2[2]) - b1[0].min(b2[0]);
    if reference > 0.0 {
        overlap / reference
    } else {
        0.0
    }
}

/// `_calculate_overlap_ratio(b1, b2, "union")`.
fn union_overlap(b1: &[f64; 4], b2: &[f64; 4]) -> f64 {
    let iw = (b1[2].min(b2[2]) - b1[0].max(b2[0])).max(0.0);
    let ih = (b1[3].min(b2[3]) - b1[1].max(b2[1])).max(0.0);
    let ia = iw * ih;
    let a = |b: &[f64; 4]| ((b[2] - b[0]) * (b[3] - b[1])).max(0.0);
    let reference = a(b1) + a(b2) - ia;
    if reference > 0.0 { ia / reference } else { 0.0 }
}

/// `_merge_images`: stack crops on a white canvas with the given
/// horizontal alignments.
fn merge_images(images: &[&RgbImage], aligns: &[Align]) -> RgbImage {
    if images.len() == 1 {
        return images[0].clone();
    }
    let mut offsets = vec![0u32; images.len()];
    let mut merged_w = images[0].width();
    for i in 1..images.len() {
        let w2 = images[i].width();
        let step = merged_w.max(w2);
        let (x1, x2) = match aligns.get(i - 1).copied().unwrap_or(Align::Left) {
            Align::Center => ((step - merged_w) / 2, (step - w2) / 2),
            Align::Right => (step - merged_w, step - w2),
            Align::Left => (0, 0),
        };
        for off in offsets.iter_mut().take(i) {
            *off += x1;
        }
        offsets[i] = x2;
        merged_w = step;
    }
    let total_h = images.iter().map(|im| im.height()).sum();
    let mut canvas = RgbImage::from_pixel(merged_w, total_h, Rgb([255, 255, 255]));
    let mut y = 0;
    for (im, &x) in images.iter().zip(&offsets) {
        image::imageops::replace(&mut canvas, *im, x as i64, y as i64);
        y += im.height();
    }
    canvas
}

/// A block out of `crop()`: `image = None` means folded into its group's
/// leader.
struct CropBlock {
    label: &'static str,
    coord: [f64; 4],
    image: Option<RgbImage>,
    group_id: Option<usize>,
}

/// `_merge_blocks(blocks, NON_MERGE_LABELS, "auto")`.
fn merge_blocks(blocks: Vec<(Det, RgbImage)>) -> Vec<CropBlock> {
    let is_non_merge = |label: &str| NON_MERGE_LABELS.contains(&label);
    let mergeable: Vec<usize> = (0..blocks.len())
        .filter(|&i| !is_non_merge(blocks[i].0.label))
        .collect();
    let aligned = |a: f64, b: f64| (a - b).abs() <= 5.0;
    let overlaps_other = |bi: usize, pi: usize| {
        let (p, b) = (&blocks[pi].0.coord, &blocks[bi].0.coord);
        let min_box = [
            p[0].min(b[0]),
            p[1].min(b[1]),
            p[2].max(b[2]),
            p[3].max(b[3]),
        ];
        blocks.iter().enumerate().any(|(oi, (ob, _))| {
            oi != bi
                && oi != pi
                && is_non_merge(ob.label)
                && union_overlap(&min_box, &ob.coord) > 0.0
        })
    };

    let mut groups: Vec<(Vec<usize>, Vec<Align>)> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut aligns: Vec<Align> = Vec::new();
    for (k, &idx) in mergeable.iter().enumerate() {
        if current.is_empty() {
            current = vec![idx];
            aligns = Vec::new();
            continue;
        }
        let prev_idx = mergeable[k - 1];
        let (prev, block) = (&blocks[prev_idx].0, &blocks[idx].0);
        let (pb, bb) = (&prev.coord, &block.coord);
        let iou_h = horizontal_overlap(bb, pb);
        let is_cross = iou_h == 0.0
            && block.label == "text"
            && block.label == prev.label
            && bb[0] > pb[2]
            && bb[1] < pb[3]
            && (bb[0] - pb[2]) < (pb[2] - pb[0]).max(bb[2] - bb[0]) * 0.3;
        let is_updown = iou_h > 0.0
            && (block.label == "text" || block.label == "vertical_text")
            && block.label == prev.label
            && bb[3] >= pb[1]
            && (bb[1] - pb[3]).abs() < (pb[3] - pb[1]).max(bb[3] - bb[1]) * 0.5
            && (aligned(bb[0], pb[0]) ^ aligned(bb[2], pb[2]))
            && overlaps_other(idx, prev_idx);
        if is_cross || is_updown {
            let align = if is_cross || !(aligned(bb[0], pb[0]) || aligned(bb[2], pb[2])) {
                Align::Center
            } else if aligned(bb[0], pb[0]) {
                Align::Left
            } else {
                Align::Right
            };
            current.push(idx);
            aligns.push(align);
        } else {
            groups.push((std::mem::take(&mut current), std::mem::take(&mut aligns)));
            current = vec![idx];
        }
    }
    if !current.is_empty() {
        groups.push((current, aligns));
    }

    let mut slots: Vec<Option<(Det, RgbImage)>> = blocks.into_iter().map(Some).collect();
    let non_merge: Vec<bool> = slots
        .iter()
        .map(|s| s.as_ref().is_some_and(|(d, _)| is_non_merge(d.label)))
        .collect();
    let take = |slots: &mut Vec<Option<(Det, RgbImage)>>, i: usize| slots[i].take();
    let mut result = Vec::new();
    let mut used = vec![false; slots.len()];
    let mut idx = 0;
    while idx < slots.len() {
        let group = groups
            .iter()
            .find(|(gidx, _)| gidx.iter().min() == Some(&idx) && gidx.iter().all(|&i| !used[i]));
        if let Some((gidx, galigns)) = group {
            let start = idx;
            let end = *gidx.iter().max().unwrap_or(&idx);
            let parts: Vec<(Det, RgbImage)> = gidx
                .iter()
                .map(|&i| take(&mut slots, i).expect("group block used once"))
                .collect();
            let w = parts.iter().map(|(_, im)| im.width()).max().unwrap_or(0);
            let h: u32 = parts.iter().map(|(_, im)| im.height()).sum();
            let tall = w == 0 || h as f64 / w as f64 >= 3.0;
            if tall {
                for (det, im) in parts {
                    result.push(CropBlock {
                        label: det.label,
                        coord: det.coord,
                        image: Some(im),
                        group_id: None,
                    });
                }
            } else {
                let images: Vec<&RgbImage> = parts.iter().map(|(_, im)| im).collect();
                let merged = merge_images(&images, galigns);
                let mut merged = Some(merged);
                for (det, _) in parts {
                    result.push(CropBlock {
                        label: det.label,
                        coord: det.coord,
                        image: merged.take(),
                        group_id: Some(gidx[0]),
                    });
                }
            }
            for &i in gidx {
                used[i] = true;
            }
            for ni in start + 1..end {
                if non_merge[ni] && !used[ni] {
                    if let Some((det, im)) = take(&mut slots, ni) {
                        result.push(CropBlock {
                            label: det.label,
                            coord: det.coord,
                            image: Some(im),
                            group_id: None,
                        });
                    }
                    used[ni] = true;
                }
            }
            idx = end + 1;
            continue;
        }
        if non_merge[idx] && !used[idx] {
            if let Some((det, im)) = take(&mut slots, idx) {
                result.push(CropBlock {
                    label: det.label,
                    coord: det.coord,
                    image: Some(im),
                    group_id: None,
                });
            }
            used[idx] = true;
        }
        idx += 1;
    }
    result
}

/// Python `round()` of a whole-pixel float (banker's rounding is moot for
/// the integer layout boxes, but kept for exactness).
fn py_round(v: f64) -> i32 {
    v.round_ties_even() as i32
}

/// Crop plan of one page: the SDK's `crop()` + `_build_layout_from_blocks`
/// on PP-DocLayoutV3 regions (reading order, page pixels).
pub fn plan_page(layout: &[LayoutRegion], page: &RgbImage) -> Vec<RegionJob> {
    let dets: Vec<Det> = layout
        .iter()
        .map(|r| Det {
            label: r.label,
            coord: r.bbox.map(f64::from),
        })
        .collect();
    let dets = filter_overlap_boxes(dets);
    if dets.is_empty() {
        return Vec::new();
    }
    let cropped: Vec<(Det, RgbImage)> =
        dets.into_iter().map(|d| (d, crop(page, d.coord))).collect();
    let blocks = merge_blocks(cropped);
    let bbox_of = |b: &CropBlock| b.coord.map(py_round);

    // Fold leaderless blocks into their group's leader (or the last leader).
    let mut folded: Vec<Vec<[i32; 4]>> = vec![Vec::new(); blocks.len()];
    let mut is_leader = vec![false; blocks.len()];
    let mut last_leader = None;
    for (i, block) in blocks.iter().enumerate() {
        if block.image.is_some() {
            last_leader = Some(i);
            is_leader[i] = true;
            continue;
        }
        let leader = block
            .group_id
            .filter(|&g| g < blocks.len() && is_leader[g])
            .or(last_leader);
        if let Some(l) = leader {
            folded[l].push(bbox_of(block));
        }
    }
    let mut jobs = Vec::new();
    for (order, block) in blocks.into_iter().enumerate() {
        let bbox = bbox_of(&block);
        let Some(image) = block.image else { continue };
        let mut regions = vec![bbox];
        regions.append(&mut folded[order]);
        jobs.push(RegionJob {
            order,
            label: block.label,
            bbox,
            regions,
            image,
            recipe: element_recipe(block.label),
        });
    }
    jobs
}

/// Finish a page from the recogniser's raw outputs (`raw[i]` for
/// `jobs[i]`, `None` for skipped jobs): `_execute_block`'s
/// post-processing, `_normalize_and_dedupe_blocks`, `assemble_markdown`.
pub fn finish_page(
    jobs: &[RegionJob],
    raw: &[Option<String>],
    options: &RegionOptions,
) -> PageOutput {
    let blocks: Vec<RegionBlock> = jobs
        .iter()
        .zip(raw)
        .map(|(job, raw)| match (job.recipe, raw) {
            (Some(recipe), Some(raw)) => {
                let (text, _error) = classify_engine_text(raw);
                RegionBlock {
                    order: job.order,
                    label: job.label.to_string(),
                    content: postprocess_content(&text, recipe.output_format, job.label),
                    inline_formula_mixed: job.label == "inline_formula"
                        && has_mixed_inline_formula_content(&text),
                }
            }
            _ => RegionBlock {
                order: job.order,
                label: job.label.to_string(),
                content: String::new(),
                inline_formula_mixed: false,
            },
        })
        .collect();
    let finals = normalize_and_dedupe_blocks(&blocks);
    let parts = assemble_parts(&finals, options.emit_image_placeholder);
    let markdown = parts
        .iter()
        .map(|(_, p)| p.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut pieces: Vec<Option<String>> = vec![None; finals.len()];
    for (i, part) in parts {
        pieces[i] = Some(part);
    }
    let out_blocks = finals
        .into_iter()
        .zip(pieces)
        .filter_map(|(block, piece)| {
            let job = jobs.iter().find(|j| j.order == block.order)?;
            Some(PageBlock {
                order: block.order,
                label: job.label,
                bbox: job.bbox,
                regions: job.regions.clone(),
                recognised: job.recipe.is_some(),
                content: block.content,
                markdown: piece,
            })
        })
        .collect();
    PageOutput {
        markdown,
        blocks: out_blocks,
    }
}

/// Recognise one page: plan the crops, call `recognise` once per block that
/// reaches the VLM (in reading order), and assemble the Markdown.
///
/// The caller supplies the page, its PP-DocLayoutV3 regions and a
/// recogniser, typically
/// `MiniCpmOcrPipeline::recognize_rgb8(job.image.as_raw(), w, h,
/// OcrRequest { task: recipe.prompt, max_tokens: Some(recipe.max_new_tokens),
/// max_slices: Some(recipe.slice_nums), no_repeat_ngram: true })`. An error
/// from the closure (failure, cancellation) aborts the page; the SDK would
/// instead record an empty block.
pub fn recognize_page<E>(
    layout: &[LayoutRegion],
    page: &RgbImage,
    options: &RegionOptions,
    recognise: &mut dyn FnMut(&RegionJob, &ElementRecipe) -> Result<String, E>,
) -> Result<PageOutput, E> {
    let jobs = plan_page(layout, page);
    let mut raw = Vec::with_capacity(jobs.len());
    for job in &jobs {
        raw.push(match &job.recipe {
            Some(recipe) => Some(recognise(job, recipe)?),
            None => None,
        });
    }
    Ok(finish_page(&jobs, &raw, options))
}
