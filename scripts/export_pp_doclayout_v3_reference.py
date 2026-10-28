#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch==2.14.1",
#     "transformers==5.21.0",
#     "numpy==2.4.6",
#     "opencv-python-headless==4.13.0.92",
#     "pillow==12.2.0",
#     "huggingface_hub",
# ]
# ///
"""Export PP-DocLayoutV3 golden fixtures (spec 009 WS-04).

Reference: the MiniCPM-OCR SDK's `ppdoclayout3_torch` layout provider, i.e.
transformers' `PPDocLayoutV3ForObjectDetection` in f32 on the CPU, fed by
the SDK preprocessing (RGB -> BGR, `cv2.resize(800x800, INTER_CUBIC)`,
`x * (1 / 255)` in f32, channels stay BGR) and finished by the SDK's numpy
post-process (`_postprocess_7col` + `_layout_analysis_postprocess`: score
threshold 0.3, NMS same 0.6 / diff 0.98, image-area filter, containment
filter with `MERGE_BBOXES_MODE`, reading order from `order_logits`). The
post-process functions and label tables below are copied verbatim from
OpenBMB/MiniCPM-OCR `sdk/minicpmvocr/layouts/ppdoclayout3_torch.py` and
`sdk/minicpmvocr/constants.py` (main, 2026-10-28) so this script needs no
SDK install. mlx-vlm's port is not used: its `detect` resizes RGB to
1024x1024 and post-processes differently from the SDK.

Writes `crates/aimd-engine/tests/golden/pp_doclayout_v3/`:

- `images/<page>.png`: input pages (lossless). `report` is
  `target/corpus/ocr/report-p1.png` (`scripts/build_corpus.py`), `landscape`
  is drawn here with PIL (title, paragraphs, table, formula, picture).
  Existing PNGs are reused unless `--refresh-images` is given;
- `<page>_resized.png`: the 800x800 `cv2.resize` output (BGR written with
  `cv2.imwrite`, so it decodes to the RGB view of the resized pixels);
- `<page>_<stage>.f32`: little-endian f32 slices. Feature maps are stored
  NHWC as `x[::s, ::s, ::cs]` (`s`, `cs` in the manifest): `pixels`, the
  four backbone stages, the three PAN levels, `mask_feat`; `init_ref` is
  the decoder's initial (pre-sigmoid) reference boxes `[300, 4]`,
  `refs` the six per-layer reference boxes `[6, 300, 4]`, `hidden0` /
  `hidden5` decoder layer outputs `[300, 256 / 4]` (every 4th channel),
  `logits` `[300, 25]`, `boxes` `[300, 4]`, `order` `[300, 300]`;
- `synthetic_post_*.f32`: a crafted raw-output case (overlapping, nested and
  page-sized boxes) that exercises NMS, the image-area filter and the
  containment filter;
- `manifest.json`: versions, per-page sizes, slice specs, the encoder top-300
  query indices and the SDK region list (`label`, `class_id`, `score`,
  `bbox`, reading-order `index`).

Run from the repo root: `uv run scripts/export_pp_doclayout_v3_reference.py`.
Needs the cached `PaddlePaddle/PP-DocLayoutV3_safetensors` snapshot (never
downloads).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "pp_doclayout_v3"
CORPUS_REPORT = REPO / "target" / "corpus" / "ocr" / "report-p1.png"
MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
MAX_FILE = 1 << 20

# --- SDK constants (sdk/minicpmvocr/constants.py) ---------------------------

PX_LABELS: List[str] = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]  # fmt: skip

MERGE_BBOXES_MODE: Dict[int, str] = {
    3: "large",
    5: "large",
    6: "large",
    15: "large",
    17: "large",
}

# --- SDK post-process (sdk/minicpmvocr/layouts/ppdoclayout3_torch.py) -------


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _get_order_np(order_logits: np.ndarray) -> np.ndarray:
    if order_logits.ndim != 2 or order_logits.shape[0] != order_logits.shape[1]:
        raise ValueError(f"order_logits must be (N, N), got {order_logits.shape}")
    os_np = _sigmoid_np(order_logits.astype(np.float32)).astype(np.float32)
    N = order_logits.shape[0]
    upper = np.triu(os_np, 1)
    lower = np.tril(np.float32(1.0) - os_np.T, -1)
    votes = upper.sum(axis=0, dtype=np.float32) + lower.sum(axis=0, dtype=np.float32)
    ptrs = np.argsort(votes)
    seq = np.zeros(N, dtype=int)
    seq[ptrs] = np.arange(N)
    return seq


def _postprocess_7col(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    order_logits: np.ndarray,
    img_w: int,
    img_h: int,
    topk: int = 300,
) -> np.ndarray:
    num_queries, num_classes = logits.shape
    scores_flat = _sigmoid_np(logits).flatten()
    top_idx = np.argpartition(-scores_flat, min(topk, len(scores_flat)))[:topk]
    top_scores = scores_flat[top_idx]
    sort_idx = np.argsort(-top_scores)
    top_idx = top_idx[sort_idx]
    top_scores = top_scores[sort_idx]
    labels = (top_idx % num_classes).astype(np.float32)
    qidx = top_idx // num_classes
    cxcy = pred_boxes[:, :2]
    wh = pred_boxes[:, 2:]
    xyxy = np.concatenate([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)
    out_shape = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    xyxy_scaled = xyxy * out_shape
    top_boxes = xyxy_scaled[qidx]
    seq = _get_order_np(order_logits)
    order_col = seq[qidx].astype(np.float32)
    return np.column_stack([
        labels, top_scores,
        top_boxes[:, 0], top_boxes[:, 1], top_boxes[:, 2], top_boxes[:, 3],
        order_col,
    ])  # fmt: skip


def _iou_np(b1: np.ndarray, b2: np.ndarray) -> float:
    xi1 = max(b1[2], b2[2])
    yi1 = max(b1[3], b2[3])
    xi2 = min(b1[4], b2[4])
    yi2 = min(b1[5], b2[5])
    iw = xi2 - xi1 + 1
    ih = yi2 - yi1 + 1
    if iw <= 0 or ih <= 0:
        return 0.0
    ia = iw * ih
    a1 = (b1[4] - b1[2] + 1) * (b1[5] - b1[3] + 1)
    a2 = (b2[4] - b2[2] + 1) * (b2[5] - b2[3] + 1)
    return float(ia / (a1 + a2 - ia))


def _nms_np(
    boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.95
) -> List[int]:
    idx = np.argsort(-boxes[:, 1])
    sel: List[int] = []
    rem = list(idx)
    while rem:
        c = rem[0]
        sel.append(c)
        cb = boxes[c]
        rem = rem[1:]
        nr = []
        for r in rem:
            rb = boxes[r]
            t = iou_diff if cb[0] != rb[0] else iou_same
            if _iou_np(cb, rb) < t:
                nr.append(r)
        rem = nr
    return sel


def _is_contained_np(b1: np.ndarray, b2: np.ndarray) -> bool:
    ix1 = max(b1[2], b2[2])
    iy1 = max(b1[3], b2[3])
    ix2 = min(b1[4], b2[4])
    iy2 = min(b1[5], b2[5])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    ia = iw * ih
    a1 = (b1[4] - b1[2]) * (b1[5] - b1[3])
    return bool(ia / a1 >= 0.9) if a1 > 0 else False


def _check_containment_np(
    boxes: np.ndarray,
    formula_idx: Optional[int] = None,
    cat_idx: Optional[int] = None,
    mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(boxes)
    co = np.zeros(n, dtype=int)
    cb_ = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                formula_idx is not None
                and boxes[i][0] == formula_idx
                and boxes[j][0] != formula_idx
            ):
                continue
            if cat_idx is not None and mode:
                if mode == "large" and boxes[j][0] == cat_idx:
                    if _is_contained_np(boxes[i], boxes[j]):
                        cb_[i] = 1
                        co[j] = 1
                elif mode == "small" and boxes[i][0] == cat_idx:
                    if _is_contained_np(boxes[i], boxes[j]):
                        cb_[i] = 1
                        co[j] = 1
            else:
                if _is_contained_np(boxes[i], boxes[j]):
                    cb_[i] = 1
                    co[j] = 1
    return co, cb_


def _layout_analysis_postprocess(
    boxes_7col: np.ndarray,
    img_w: int,
    img_h: int,
    labels: List[str],
    threshold: float = 0.3,
    do_nms: bool = True,
    merge_bboxes_mode: Optional[dict] = None,
) -> List[Dict]:
    boxes = boxes_7col.copy()
    boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(int)
    keep = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
    boxes = boxes[keep]
    if len(boxes) == 0:
        return []
    if do_nms:
        sel = _nms_np(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
        boxes = boxes[sel]
        if len(boxes) == 0:
            return []
    if len(boxes) > 1:
        img_idx = labels.index("image") if "image" in labels else None
        img_area = img_w * img_h
        area_thres = 0.82 if img_w > img_h else 0.93
        filtered = []
        for b in boxes:
            if b[0] == img_idx:
                x1 = int(max(0, b[2]))
                y1 = int(max(0, b[3]))
                x2 = int(min(img_w, b[4]))
                y2 = int(min(img_h, b[5]))
                ba = (x2 - x1) * (y2 - y1)
                if ba <= area_thres * img_area:
                    filtered.append(b)
            else:
                filtered.append(b)
        if filtered:
            boxes = np.array(filtered, dtype=boxes.dtype)
        if len(boxes) == 0:
            return []
    if merge_bboxes_mode and len(boxes) >= 2:
        formula_idx = labels.index("formula") if "formula" in labels else None
        if isinstance(merge_bboxes_mode, str):
            if merge_bboxes_mode != "union":
                co, cb = _check_containment_np(boxes[:, :6], formula_idx)
                if merge_bboxes_mode == "large":
                    boxes = boxes[cb == 0]
                elif merge_bboxes_mode == "small":
                    boxes = boxes[(co == 0) | (cb == 1)]
        elif isinstance(merge_bboxes_mode, dict):
            keep_mask = np.ones(len(boxes), dtype=bool)
            for cat_idx, layout_mode in merge_bboxes_mode.items():
                if layout_mode == "union":
                    continue
                co, cb = _check_containment_np(
                    boxes[:, :6], formula_idx, cat_idx, layout_mode
                )
                if layout_mode == "large":
                    keep_mask &= cb == 0
                elif layout_mode == "small":
                    keep_mask &= (co == 0) | (cb == 1)
            boxes = boxes[keep_mask]
        if len(boxes) == 0:
            return []
    sorted_idx = np.argsort(boxes[:, 6])
    boxes = boxes[sorted_idx, :6]
    result: List[Dict] = []
    for b in boxes:
        xmin = int(max(0, b[2]))
        ymin = int(max(0, b[3]))
        xmax = int(min(img_w, b[4]))
        ymax = int(min(img_h, b[5]))
        if xmax <= xmin or ymax <= ymin:
            continue
        result.append({
            "cls_id": int(b[0]),
            "label": labels[int(b[0])],
            "score": float(b[1]),
            "coordinate": [xmin, ymin, xmax, ymax],
        })  # fmt: skip
    return result


def sdk_regions(logits, boxes, order, w, h) -> list[dict]:
    """`PPDocLayoutV3TorchDetector._detect_one` after the forward pass."""
    boxes_7col = _postprocess_7col(logits, boxes, order, w, h, topk=300)
    dicts = _layout_analysis_postprocess(
        boxes_7col, w, h, PX_LABELS, threshold=0.3, do_nms=True,
        merge_bboxes_mode=MERGE_BBOXES_MODE,
    )  # fmt: skip
    return [
        {
            "index": i,
            "label": b["label"],
            "class_id": b["cls_id"],
            "score": b["score"],
            "bbox": b["coordinate"],
        }
        for i, b in enumerate(dicts)
    ]


# --- fixtures ----------------------------------------------------------------


def write_f32(name: str, arr) -> None:
    data = np.ascontiguousarray(np.asarray(arr, dtype="<f4")).tobytes()
    assert len(data) <= MAX_FILE, f"{name}: {len(data)} bytes"
    (OUT / name).write_bytes(data)


def nhwc_slice(t: torch.Tensor, s: int, cs: int) -> np.ndarray:
    """`[1, C, H, W]` -> `x[::s, ::s, ::cs]` in NHWC."""
    return t[0].permute(1, 2, 0)[::s, ::s, ::cs].contiguous().numpy()


def landscape_page() -> Image.Image:
    """A deterministic 1400x1000 page: title, text, table, formula, picture."""
    w, h = 1400, 1000
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = lambda size: ImageFont.load_default(size=size)  # noqa: E731
    d.text((70, 50), "Layout Analysis of Printed Pages", fill=(0, 0, 0), font=font(46))
    d.text((70, 140), "1. Introduction", fill=(0, 0, 0), font=font(30))
    body = [
        "Document layout detection finds titles, paragraphs, tables and",
        "formulas on a page and returns them in reading order. This page",
        "was drawn for a regression fixture: every block below is placed",
        "so that the detector sees a typical two-column report layout.",
    ]
    for i, line in enumerate(body):
        d.text((70, 195 + i * 32), line, fill=(20, 20, 20), font=font(22))
    d.text((70, 345), "2. Results", fill=(0, 0, 0), font=font(30))
    x0, y0, cw, rh = 70, 400, 150, 44
    rows = [
        ("Model", "Pages", "Time", "Score"),
        ("Base", "120", "3.1 s", "0.91"),
        ("Large", "120", "5.4 s", "0.94"),
        ("Tiny", "120", "1.2 s", "0.85"),
    ]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            box = (x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            d.rectangle(box, outline=(0, 0, 0), width=2)
            d.text((box[0] + 12, box[1] + 10), cell, fill=(0, 0, 0), font=font(22))
    d.text(
        (70, 640),
        "The score follows from the formula",
        fill=(20, 20, 20),
        font=font(22),
    )
    d.text(
        (200, 700), "S = (1 / N) * sum_i ( p_i - q_i )^2", fill=(0, 0, 0), font=font(30)
    )
    d.text((1180, 700), "(1)", fill=(0, 0, 0), font=font(28))
    # A picture: sky, sun, hills.
    px0, py0, px1, py1 = 820, 140, 1330, 560
    for y in range(py0, py1):
        t = (y - py0) / (py1 - py0)
        d.line([(px0, y), (px1, y)], fill=(int(90 + 120 * t), int(150 + 70 * t), 235))
    d.ellipse((1180, 180, 1270, 270), fill=(250, 210, 60))
    d.polygon([(px0, py1), (960, 380), (1100, py1)], fill=(60, 130, 70))
    d.polygon([(1000, py1), (1180, 330), (px1, py1)], fill=(40, 110, 60))
    d.text((900, 580), "Figure 1. A landscape picture.", fill=(0, 0, 0), font=font(22))
    for i in range(4):
        d.text(
            (70, 790 + i * 32),
            "Further text continues below the formula and closes the page body. " * 1,
            fill=(20, 20, 20),
            font=font(22),
        )
    d.text((680, 950), "- 3 -", fill=(60, 60, 60), font=font(20))
    return img


def page_images(refresh: bool) -> dict[str, Path]:
    images = OUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    pages = {"report": images / "report.png", "landscape": images / "landscape.png"}
    if refresh or not pages["report"].exists():
        Image.open(CORPUS_REPORT).convert("RGB").save(pages["report"], optimize=True)
    if refresh or not pages["landscape"].exists():
        landscape_page().save(pages["landscape"], optimize=True)
    for p in pages.values():
        assert p.stat().st_size <= MAX_FILE, p
    return pages


SLICES = {
    # name: (stride, channel stride)
    "pixels": (16, 1),
    "stage1": (8, 4),
    "stage2": (4, 16),
    "stage3": (2, 32),
    "stage4": (1, 64),
    "pan0": (4, 8),
    "pan1": (2, 8),
    "pan2": (1, 8),
    "mask_feat": (8, 1),
}


def run_page(model, name: str, path: Path) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h0, w0 = rgb.shape[:2]
    # SDK `_detect_one` preprocessing.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (800, 800), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f"{name}_resized.png"), resized)
    sp = list(cv2.split(resized))
    for c in range(3):
        sp[c] = sp[c].astype(np.float32) * (1.0 / 255.0)
    img_norm = cv2.merge(sp).transpose(2, 0, 1)
    pixel_values = torch.from_numpy(img_norm).unsqueeze(0).to(dtype=torch.float32)

    captured: dict = {}
    hooks = [
        model.model.backbone.register_forward_hook(
            lambda m, i, o: captured.__setitem__("backbone", [f for f, _ in o])
        ),
        model.model.encoder.register_forward_hook(
            lambda m, i, o: captured.__setitem__("encoder", o)
        ),
    ]
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    for hk in hooks:
        hk.remove()

    stages = {
        "pixels": pixel_values,
        **{f"stage{i + 1}": f for i, f in enumerate(captured["backbone"])},
        **{f"pan{i}": f for i, f in enumerate(captured["encoder"].last_hidden_state)},
        "mask_feat": captured["encoder"].mask_feat,
    }
    shapes = {}
    for key, (s, cs) in SLICES.items():
        t = stages[key]
        shapes[key] = {"shape": list(t.shape), "s": s, "cs": cs}
        write_f32(f"{name}_{key}.f32", nhwc_slice(t, s, cs))

    enc_class = out.enc_outputs_class[0]
    topk = torch.topk(enc_class.max(-1).values, 300).indices
    write_f32(f"{name}_init_ref.f32", out.init_reference_points[0].numpy())
    write_f32(f"{name}_refs.f32", out.intermediate_reference_points[0].numpy())
    hidden = out.intermediate_hidden_states[0]
    write_f32(f"{name}_hidden0.f32", hidden[0, :, ::4].numpy())
    write_f32(f"{name}_hidden5.f32", hidden[5, :, ::4].numpy())

    logits = out.logits[0].numpy()
    boxes = out.pred_boxes[0].numpy()
    order = out.order_logits[0].numpy()
    write_f32(f"{name}_logits.f32", logits)
    write_f32(f"{name}_boxes.f32", boxes)
    write_f32(f"{name}_order.f32", order)

    regions = sdk_regions(logits, boxes, order, w0, h0)
    print(f"{name}: {w0}x{h0}, {len(regions)} regions")
    for r in regions:
        print("   ", r["index"], r["label"], f"{r['score']:.3f}", r["bbox"])
    return {
        "name": name,
        "width": w0,
        "height": h0,
        "slices": shapes,
        "topk": topk.tolist(),
        "regions": regions,
    }


def synthetic_post() -> dict:
    """Crafted raw outputs: duplicates, nested boxes and page-sized images."""
    rng = np.random.default_rng(9)
    q, c = 300, len(PX_LABELS)
    logits = rng.normal(-7.0, 1.0, (q, c)).astype(np.float32)
    boxes = np.stack(
        [
            rng.uniform(0.1, 0.9, q),
            rng.uniform(0.1, 0.9, q),
            rng.uniform(0.02, 0.4, q),
            rng.uniform(0.01, 0.2, q),
        ],
        axis=1,
    ).astype(np.float32)
    live = rng.choice(q, 60, replace=False)
    for k, qi in enumerate(live):
        label = int(rng.integers(0, c))
        logits[qi, label] = np.float32(rng.uniform(-1.0, 4.0))
        if k % 5 == 1:
            # second label on the same box (cross-class NMS at 0.98)
            logits[qi, (label + 3) % c] = np.float32(rng.uniform(-0.8, 3.0))
    for k in range(0, 50, 2):
        a, b = live[k], live[k + 1]
        if k % 6 == 0:
            # b nested inside a (containment)
            boxes[b] = boxes[a] * np.array([1, 1, 0.6, 0.6], np.float32)
            logits[a, [3, 5, 6, 15, 17][k % 5]] = np.float32(2.5)
        elif k % 6 == 2:
            # near-duplicate of a (same-class NMS)
            boxes[b] = boxes[a] + rng.normal(0, 0.004, 4).astype(np.float32)
            logits[b] = logits[a] - np.float32(0.3)
    # page-sized images: one above the area threshold, one below it
    boxes[live[50]] = np.array([0.5, 0.5, 0.99, 0.99], np.float32)
    logits[live[50], PX_LABELS.index("image")] = np.float32(3.0)
    boxes[live[51]] = np.array([0.5, 0.5, 0.9, 0.9], np.float32)
    logits[live[51], PX_LABELS.index("image")] = np.float32(2.0)
    # degenerate box clipped away after rounding
    boxes[live[52]] = np.array([1.2, 0.5, 0.1, 0.1], np.float32)
    logits[live[52], 22] = np.float32(3.0)
    order = rng.normal(0.0, 3.0, (q, q)).astype(np.float32)
    order[np.tril_indices(q)] = np.float32(-1e4)
    w, h = 1000, 1400
    write_f32("synthetic_post_logits.f32", logits)
    write_f32("synthetic_post_boxes.f32", boxes)
    write_f32("synthetic_post_order.f32", order)
    regions = sdk_regions(logits, boxes, order, w, h)
    print(f"synthetic_post: {len(regions)} regions")
    return {
        "width": w,
        "height": h,
        "order_seq": _get_order_np(order).tolist(),
        "regions": regions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh-images", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    snapshot = Path(snapshot_download(MODEL, local_files_only=True))
    model = AutoModelForObjectDetection.from_pretrained(
        str(snapshot), dtype=torch.float32, local_files_only=True
    ).eval()
    pages = page_images(args.refresh_images)
    manifest = {
        "model": MODEL,
        "snapshot": snapshot.name,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "labels": PX_LABELS,
        "pages": [run_page(model, n, p) for n, p in pages.items()],
        "synthetic_post": synthetic_post(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
