#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx4ocr==0.1.2",
#     "mlx==0.32.4",
#     "numpy==2.4.6",
#     "opencv-python-headless==4.13.0.92",
#     "pyclipper==1.4.0",
#     "shapely==2.1.2",
#     "pillow==12.2.0",
#     "pyyaml==6.0.3",
# ]
# ///
"""Export PP-OCRv6 medium golden fixtures from mlx4ocr (spec 004 WS-02).

Writes `crates/aimd-engine/tests/golden/ppocrv6/`:

Hermetic host-op fixtures (no checkpoint needed to *use* them):

- `synthetic.png`: a deterministic 98x62 BGR test card (stored RGB);
- `resize.u8`: `cv2.resize(INTER_LINEAR)` of it to several sizes;
- `crops.u8`: `get_rotate_crop_image` (perspective warp, cubic,
  replicate border, rot90 for tall crops) for several quads;
- `rec_in.f32`: `resize_norm_img` of one crop;
- `db_prob.f32`: a synthetic 160x200 DB probability map (filled rotated
  rectangles, blurred) and the `db_postprocess` + `sorted_detections`
  output for it;
- `manifest.json`: the case tables above plus `resize_det_image` target
  sizes, `np.argsort` and `np.float32` mean cases.

Checkpoint fixtures (mlx4ocr medium det + rec, `compile_models=False`):

- `images/<name>.png`: the PIL-decoded pixels of mlx4ocr's example images,
  re-encoded losslessly, so the Rust side sees the exact decoder input
  mlx4ocr saw (JPEG decoders differ by a few levels: zune-jpeg vs
  libjpeg-turbo);
- `<name>_prob.f32`: the detector probability map sampled every 8th row
  and column;
- `<name>_rec0.json` / `manifest.json`: per-step argmax/max of the first
  recognition batch and the final lines (text, score, quad).

Recognition weights: mlx4ocr 0.1.2 patches the HF rec checkpoint from the
official Paddle `.pdparams` because the safetensors export swapped
`head.encoder.conv_block.0` and `.1`. This exporter does not download or
unpickle anything: it swaps the two tensors in place instead, which is
what the Rust loader does.

Run from the repo root: `uv run scripts/export_ppocrv6_reference.py`
(add `--source DIR` with mlx4ocr's `examples/images` to refresh the PNGs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import mlx4ocr.models.rec.model as recmodel
import numpy as np
from mlx4ocr.hub.download import HubArtifacts
from mlx4ocr.pipeline.crop import get_rotate_crop_image, sorted_detections
from mlx4ocr.pipeline.ocr import PP_OCRv6
from mlx4ocr.postprocess.db import db_postprocess
from mlx4ocr.preprocess.det import det_preprocess, nhwc_prob_to_nchw, resize_det_image
from mlx4ocr.preprocess.rec import resize_norm_img
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "ppocrv6"
HF = Path.home() / ".cache" / "huggingface" / "hub"
DET_REPO = "models--PaddlePaddle--PP-OCRv6_medium_det_safetensors"
REC_REPO = "models--PaddlePaddle--PP-OCRv6_medium_rec_safetensors"
IMAGES = ("sample_doc.jpg", "general_ocr_002.png", "img_10.jpg")
# img_10 is a 1280x720 photo; its lossless re-encoding is cropped to the
# text region so the fixture stays under 1 MB.
IMAGE_CROPS = {"img_10.jpg": (0, 0, 400, 200)}
MAX_FILE = 1 << 20
PROB_STEP = 8


def artifacts(repo: str, task: str) -> HubArtifacts:
    root = sorted((HF / repo / "snapshots").iterdir())[0]
    return HubArtifacts(
        root=root,
        config=root / "config.json",
        inference=root / "inference.yml",
        weights=root / "model.safetensors",
        preprocessor=root / "preprocessor_config.json",
        variant="medium",
        task=task,
    )


def swap_head_convs(variant, tensors):
    out = dict(tensors)
    for key in list(tensors):
        if key.startswith("head.encoder.conv_block.0."):
            other = key.replace("conv_block.0.", "conv_block.1.")
            out[key], out[other] = tensors[other], tensors[key]
    return out


def quad_list(points) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def synthetic_card() -> np.ndarray:
    """Deterministic BGR test card: gradients, stripes and a dark glyph-ish block."""
    h, w = 62, 98
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (x * 255 // (w - 1)).astype(np.uint8)
    img[..., 1] = (y * 255 // (h - 1)).astype(np.uint8)
    img[..., 2] = (((x // 3 + y // 5) % 2) * 200 + 30).astype(np.uint8)
    img[20:40, 30:70] = (17, 99, 230)
    img[25:35, 35:65] = (240, 12, 64)
    return img


def host_fixtures(manifest: dict) -> None:
    card = synthetic_card()
    Image.fromarray(card[:, :, ::-1]).save(OUT / "synthetic.png", optimize=True)

    resize_cases = [(49, 31), (300, 200), (40, 25), (98, 48), (13, 90), (97, 61)]
    blobs = []
    for w, h in resize_cases:
        blobs.append(cv2.resize(card, (w, h)).tobytes())
    (OUT / "resize.u8").write_bytes(b"".join(blobs))
    manifest["resize"] = [[w, h] for w, h in resize_cases]

    quads = [
        [[10, 8], [60, 8], [60, 30], [10, 30]],
        [[12.5, 20.25], [80.75, 12.5], [84.0, 40.0], [15.0, 48.5]],
        [[30, 2], [44, 2], [44, 60], [30, 60]],
        [[-3, -2], [101, 1], [99, 66], [-1, 63]],
        [[40, 10], [41, 10], [41, 11], [40, 11]],
    ]
    crops = []
    crop_meta = []
    for q in quads:
        c = np.ascontiguousarray(get_rotate_crop_image(card, np.asarray(q, np.float32)))
        crops.append(c.tobytes())
        crop_meta.append({"quad": q, "w": int(c.shape[1]), "h": int(c.shape[0])})
    (OUT / "crops.u8").write_bytes(b"".join(crops))
    manifest["crops"] = crop_meta

    crop1 = np.ascontiguousarray(
        get_rotate_crop_image(card, np.asarray(quads[1], np.float32))
    )
    max_wh = max(320 / 48, crop1.shape[1] / float(crop1.shape[0]))
    rec_in = resize_norm_img(crop1, max_wh).transpose(1, 2, 0)
    (OUT / "rec_in.f32").write_bytes(np.ascontiguousarray(rec_in, "<f4").tobytes())
    manifest["rec_in"] = {
        "crop": 1,
        "max_wh_ratio": max_wh,
        "w": int(rec_in.shape[1]),
    }

    det_sizes = [(600, 800), (896, 528), (362, 78), (20, 10), (5000, 300), (1920, 1080)]
    det_rows = []
    for w, h in det_sizes:
        resized, shape = resize_det_image(
            np.zeros((h, w, 3), np.uint8), limit_side_len=960, limit_type="min"
        )
        det_rows.append(
            {
                "src": [w, h],
                "resized": [int(resized.shape[1]), int(resized.shape[0])],
                "shape": [float(v) for v in shape],
            }
        )
    manifest["det_resize"] = det_rows

    # Synthetic DB map: rotated rectangles of varying size and strength.
    ph, pw = 160, 200
    prob = np.zeros((ph, pw), np.float32)
    rects = [
        ((40, 20), (60, 14), 0, 0.95),
        ((140, 25), (80, 12), -8, 0.9),
        ((60, 70), (90, 20), 15, 0.8),
        ((160, 80), (20, 50), 0, 0.7),
        ((100, 130), (150, 10), 3, 0.6),
        ((20, 140), (4, 4), 0, 0.9),
        ((180, 140), (12, 8), 30, 0.4),
    ]
    for center, size, angle, value in rects:
        box = cv2.boxPoints((center, size, angle)).astype(np.int32)
        cv2.fillPoly(prob, [box], value)
    prob = cv2.GaussianBlur(prob, (5, 5), 1.2)
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    (OUT / "db_prob.f32").write_bytes(prob.astype("<f4").tobytes())
    src_h, src_w = 300.0, 380.0
    dets = db_postprocess(
        prob[None, None], (src_h, src_w, ph / src_h, pw / src_w), score_mode="fast"
    )
    ordered = sorted_detections(dets)
    manifest["db"] = {
        "w": pw,
        "h": ph,
        "src_w": int(src_w),
        "src_h": int(src_h),
        "dets": [{"quad": quad_list(d.box.points), "score": d.score} for d in dets],
        "order": [dets.index(d) for d in ordered],
    }

    ratios = np.round(np.linspace(0.3, 9.0, 40) * 7 % 5, 1).astype(np.float64)
    manifest["argsort"] = {
        "values": ratios.tolist(),
        "indices": np.argsort(ratios).tolist(),
    }
    means = []
    for n in (5, 20, 300):
        v = (np.sin(np.arange(n, dtype=np.float64) * 1.7) * 0.25 + 0.7).astype(
            np.float32
        )
        means.append({"values": v.tolist(), "mean": float(np.mean(v))})
    manifest["mean"] = means


def image_fixtures(manifest: dict, source: Path | None) -> None:
    recmodel.load_patched_recognition_tensors = swap_head_convs
    det = artifacts(DET_REPO, "det")
    rec = artifacts(REC_REPO, "rec")
    ocr = PP_OCRv6.from_artifacts("medium", det, rec, compile_models=False)
    images_dir = OUT / "images"
    images_dir.mkdir(exist_ok=True)
    entries = []
    for name in IMAGES:
        stem = Path(name).stem
        png = images_dir / f"{stem}.png"
        if source is not None:
            with Image.open(source / name) as im:
                rgb = im.convert("RGB")
            if name in IMAGE_CROPS:
                rgb = rgb.crop(IMAGE_CROPS[name])
            rgb.save(png, optimize=True)
        with Image.open(png) as im:
            bgr = np.asarray(im.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()

        pre = det_preprocess(bgr, limit_side_len=960, limit_type="min")
        prob = nhwc_prob_to_nchw(ocr.detector(pre.image))[0, 0]
        sampled = np.ascontiguousarray(prob[::PROB_STEP, ::PROB_STEP], "<f4")
        (OUT / f"{stem}_prob.f32").write_bytes(sampled.tobytes())

        result = ocr.predict(bgr)
        lines = [
            {
                "text": b.text,
                "score": float(b.recognition_score),
                "quad": quad_list(b.box.points),
            }
            for b in result.result.blocks
        ]

        # First recognition batch, rebuilt the way `PP_OCRv6.predict` does.
        shape = pre.shape
        dets = sorted_detections(
            db_postprocess(prob[None, None], shape, score_mode="fast")
        )
        crops = [
            get_rotate_crop_image(bgr, np.asarray(d.box.points, np.float32))
            for d in dets
        ]
        ratios = np.asarray([c.shape[1] / float(c.shape[0]) for c in crops])
        first = np.argsort(ratios)[:6]
        max_wh = max(320 / 48, *[ratios[i] for i in first])
        batch = np.stack(
            [resize_norm_img(crops[int(i)], max_wh).transpose(1, 2, 0) for i in first]
        )
        probs = ocr.recognizer(mx.array(batch))
        rec0 = {
            "crops": [int(i) for i in first],
            "argmax": np.asarray(mx.argmax(probs, axis=-1)).tolist(),
            "max": np.asarray(mx.max(probs, axis=-1)).astype(np.float64).tolist(),
        }
        entries.append(
            {
                "name": stem,
                "width": int(bgr.shape[1]),
                "height": int(bgr.shape[0]),
                "resized": [int(pre.image.shape[2]), int(pre.image.shape[1])],
                "prob_step": PROB_STEP,
                "num_dets": len(dets),
                "rec0": rec0,
                "lines": lines,
            }
        )
        print(f"{stem}: {len(lines)} lines, {len(dets)} dets")
    manifest["images"] = entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="mlx4ocr examples/images dir")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"mlx4ocr": "0.1.2", "mlx": mx.__version__}
    host_fixtures(manifest)
    image_fixtures(manifest, args.source)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    )
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            assert size <= MAX_FILE, f"{path} is {size} bytes (> 1 MB)"
            print(f"{size:>9} {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
