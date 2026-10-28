//! The MiniCPM-OCR SDK's PP-DocLayoutV3 post-process
//! (`layouts/ppdoclayout3_torch.py`: `_postprocess_7col` +
//! `_layout_analysis_postprocess`), ported with numpy's dtypes: f32 math
//! wherever numpy keeps float32, f64 only where the SDK converts with
//! `float()`, and numpy's introsort `argsort` wherever ties could reorder
//! rows.
//!
//! Steps: sigmoid scores, top 300 (query, class) pairs, `cxcywh -> xyxy`
//! scaled to the source image, reading order from `order_logits` (vote
//! count per query), `np.round` of the boxes, score threshold 0.3, NMS
//! (IoU 0.6 within a class, 0.98 across classes, `+1` pixel areas), the
//! page-sized `image` filter, containment removal for the
//! `MERGE_BBOXES_MODE` "large" classes, sort by reading order, clamp to
//! the image and drop degenerate boxes.

use super::config::{NUM_LABELS, NUM_QUERIES};
use crate::models::pp_ocrv6::ctc::np_argsort;

/// The SDK's `PX_LABELS` (PaddleX names; index = class id).
pub const PX_LABELS: [&str; NUM_LABELS] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

/// `MERGE_BBOXES_MODE`: classes whose boxes swallow contained boxes
/// (`chart`, `display_formula`, `doc_title`, `inline_formula`,
/// `paragraph_title`), applied in this order.
pub const MERGE_LARGE: [usize; 5] = [3, 5, 6, 15, 17];

/// Score threshold of `_layout_analysis_postprocess`.
pub const THRESHOLD: f32 = 0.3;
/// NMS IoU thresholds (same class, different classes).
pub const IOU_SAME: f64 = 0.6;
pub const IOU_DIFF: f64 = 0.98;

/// One detected layout region, in reading order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutRegion {
    /// Position in reading order (0-based).
    pub index: usize,
    /// `PX_LABELS[class_id]`.
    pub label: &'static str,
    pub class_id: usize,
    pub score: f32,
    /// `[xmin, ymin, xmax, ymax]` in source pixels, clamped to the image.
    pub bbox: [i32; 4],
}

/// numpy's float32 `1 / (1 + exp(-clip(x, -50, 50)))`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-50.0, 50.0)).exp())
}

/// `_get_order_np`: reading position of every query from the `[n, n]`
/// pointer logits (column vote sums, then `argsort`).
pub fn reading_order(order_logits: &[f32], n: usize) -> Vec<usize> {
    assert_eq!(order_logits.len(), n * n, "order_logits must be [n, n]");
    // upper = triu(os, 1); lower = tril(1 - os.T, -1); both summed over
    // axis 0 row by row in f32.
    let mut upper = vec![0f32; n];
    let mut lower = vec![0f32; n];
    for i in 0..n {
        for j in 0..n {
            let u = if j > i {
                sigmoid(order_logits[i * n + j])
            } else {
                0.0
            };
            upper[j] += u;
            let l = if i > j {
                1.0 - sigmoid(order_logits[j * n + i])
            } else {
                0.0
            };
            lower[j] += l;
        }
    }
    let votes: Vec<f64> = upper
        .iter()
        .zip(&lower)
        .map(|(u, l)| f64::from(u + l))
        .collect();
    let mut seq = vec![0usize; n];
    for (rank, p) in np_argsort(&votes).into_iter().enumerate() {
        seq[p] = rank;
    }
    seq
}

/// A 7-column row: label, score, x1, y1, x2, y2, order (all f32).
type Row = [f32; 7];

/// `_postprocess_7col` for one image.
fn rows_7col(
    logits: &[f32],
    boxes: &[f32],
    order_logits: &[f32],
    img_w: u32,
    img_h: u32,
) -> Vec<Row> {
    let q = boxes.len() / 4;
    let c = logits.len() / q;
    let scores: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();
    // argpartition + argsort(-top_scores): the top `NUM_QUERIES` flat
    // indices by descending score. Which tied low scores make the cut and
    // their order cannot change the output (they fall under the 0.3
    // threshold), so a full descending sort stands in for argpartition,
    // followed by numpy's argsort of the negated top scores.
    let mut flat: Vec<usize> = (0..scores.len()).collect();
    flat.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    flat.truncate(NUM_QUERIES.min(scores.len()));
    let neg: Vec<f64> = flat.iter().map(|&i| f64::from(-scores[i])).collect();
    let top: Vec<usize> = np_argsort(&neg).into_iter().map(|k| flat[k]).collect();
    let seq = reading_order(order_logits, q);
    let scale = [img_w as f32, img_h as f32, img_w as f32, img_h as f32];
    top.into_iter()
        .map(|i| {
            let qi = i / c;
            let b = &boxes[qi * 4..qi * 4 + 4];
            let xyxy = [
                b[0] - 0.5 * b[2],
                b[1] - 0.5 * b[3],
                b[0] + 0.5 * b[2],
                b[1] + 0.5 * b[3],
            ];
            [
                (i % c) as f32,
                scores[i],
                xyxy[0] * scale[0],
                xyxy[1] * scale[1],
                xyxy[2] * scale[2],
                xyxy[3] * scale[3],
                seq[qi] as f32,
            ]
        })
        .collect()
}

/// `_iou_np` (inclusive `+1` pixel extents, f32 math, f64 result).
fn iou(b1: &Row, b2: &Row) -> f64 {
    let xi1 = b1[2].max(b2[2]);
    let yi1 = b1[3].max(b2[3]);
    let xi2 = b1[4].min(b2[4]);
    let yi2 = b1[5].min(b2[5]);
    let iw = xi2 - xi1 + 1.0;
    let ih = yi2 - yi1 + 1.0;
    if iw <= 0.0 || ih <= 0.0 {
        return 0.0;
    }
    let ia = iw * ih;
    let a1 = (b1[4] - b1[2] + 1.0) * (b1[5] - b1[3] + 1.0);
    let a2 = (b2[4] - b2[2] + 1.0) * (b2[5] - b2[3] + 1.0);
    f64::from(ia / (a1 + a2 - ia))
}

/// `_nms_np`: indices kept, in descending score order.
fn nms(rows: &[Row]) -> Vec<usize> {
    let neg: Vec<f64> = rows.iter().map(|r| f64::from(-r[1])).collect();
    let mut rem = np_argsort(&neg);
    let mut sel = Vec::new();
    while !rem.is_empty() {
        let c = rem[0];
        sel.push(c);
        let cb = &rows[c];
        rem = rem[1..]
            .iter()
            .copied()
            .filter(|&r| {
                let rb = &rows[r];
                let t = if cb[0] != rb[0] { IOU_DIFF } else { IOU_SAME };
                iou(cb, rb) < t
            })
            .collect();
    }
    sel
}

/// `_is_contained_np`: at least 90% of `b1`'s area lies inside `b2` (f32
/// ratio compared with `float32(0.9)`, as numpy does under NEP 50).
fn is_contained(b1: &Row, b2: &Row) -> bool {
    let ix1 = b1[2].max(b2[2]);
    let iy1 = b1[3].max(b2[3]);
    let ix2 = b1[4].min(b2[4]);
    let iy2 = b1[5].min(b2[5]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let ia = iw * ih;
    let a1 = (b1[4] - b1[2]) * (b1[5] - b1[3]);
    a1 > 0.0 && ia / a1 >= 0.9f32
}

/// `_postprocess_7col` + `_layout_analysis_postprocess` for one image:
/// `logits` `[Q, 25]`, `boxes` `[Q, 4]` (normalized cxcywh), `order_logits`
/// `[Q, Q]`, the source size, and the regions in reading order.
pub fn postprocess(
    logits: &[f32],
    boxes: &[f32],
    order_logits: &[f32],
    img_w: u32,
    img_h: u32,
) -> Vec<LayoutRegion> {
    let mut rows = rows_7col(logits, boxes, order_logits, img_w, img_h);
    for r in &mut rows {
        for v in &mut r[2..6] {
            *v = v.round_ties_even();
        }
    }
    rows.retain(|r| r[1] > THRESHOLD && r[0] > -1.0);
    if rows.is_empty() {
        return Vec::new();
    }
    rows = nms(&rows).into_iter().map(|i| rows[i]).collect();
    if rows.len() > 1 {
        let img_idx = PX_LABELS
            .iter()
            .position(|&l| l == "image")
            .map(|i| i as f32);
        let img_area = f64::from(img_w) * f64::from(img_h);
        let area_thres = if img_w > img_h { 0.82 } else { 0.93 };
        let filtered: Vec<Row> = rows
            .iter()
            .copied()
            .filter(|b| {
                if Some(b[0]) != img_idx {
                    return true;
                }
                let x1 = b[2].max(0.0) as i64;
                let y1 = b[3].max(0.0) as i64;
                let x2 = b[4].min(img_w as f32) as i64;
                let y2 = b[5].min(img_h as f32) as i64;
                ((x2 - x1) * (y2 - y1)) as f64 <= area_thres * img_area
            })
            .collect();
        if !filtered.is_empty() {
            rows = filtered;
        }
    }
    if rows.len() >= 2 {
        // `formula_idx` is None: PX_LABELS has no plain "formula".
        let mut keep = vec![true; rows.len()];
        for cat in MERGE_LARGE {
            let cat = cat as f32;
            for i in 0..rows.len() {
                if (0..rows.len())
                    .any(|j| i != j && rows[j][0] == cat && is_contained(&rows[i], &rows[j]))
                {
                    keep[i] = false;
                }
            }
        }
        rows = rows
            .into_iter()
            .zip(keep)
            .filter_map(|(r, k)| k.then_some(r))
            .collect();
        if rows.is_empty() {
            return Vec::new();
        }
    }
    let order: Vec<f64> = rows.iter().map(|r| f64::from(r[6])).collect();
    let mut out = Vec::new();
    for i in np_argsort(&order) {
        let b = &rows[i];
        let xmin = b[2].max(0.0) as i32;
        let ymin = b[3].max(0.0) as i32;
        let xmax = b[4].min(img_w as f32) as i32;
        let ymax = b[5].min(img_h as f32) as i32;
        if xmax <= xmin || ymax <= ymin {
            continue;
        }
        let class_id = b[0] as usize;
        out.push(LayoutRegion {
            index: out.len(),
            label: PX_LABELS[class_id],
            class_id,
            score: b[1],
            bbox: [xmin, ymin, xmax, ymax],
        });
    }
    out
}
