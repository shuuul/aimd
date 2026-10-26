#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm==0.7.10",
#     "mlx4ocr==0.1.2",
#     "opencv-python-headless==4.13.0.92",
#     "pillow==12.2.0",
#     "pyclipper==1.4.0",
#     "pyyaml==6.0.3",
#     "shapely==2.1.2",
#     "transformers==5.20.2",
# ]
# ///
"""Corpus baseline for the spec 004 OCR models (PaddleOCR-VL-1.6, PP-OCRv6).

Usage: uv run scripts/corpus_baseline_paddle.py [CORPUS_DIR] [--only vl|pp]

Separate from ``corpus_baseline.py`` because these references need newer
mlx / mlx-vlm pins than the Unlimited-OCR baseline. Adds to
``CORPUS_DIR/baseline.json`` (other keys are kept):

- ``ocr_paddleocr_vl``: official bf16 ``PaddlePaddle/PaddleOCR-VL-1.6``.
- ``ocr_paddleocr_vl_4bit``: ``mlx-community/PaddleOCR-VL-1.6-4bit`` (the
  native ``--ocr-bits 4`` quantizes the bf16 checkpoint bit-identically).
- ``ocr_pp_ocrv6``: mlx4ocr medium det + rec (``compile_models=False``), with
  the HF rec head convs swapped back exactly as the Rust loader does.

mlx-vlm runs ``OCR:`` greedily with 4096 max tokens, the native settings.
Models load from the local HF cache only. Score with
``cargo run --release -p aimd-engine --example corpus_gate -- --ocr-model ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

VL_REPOS = {
    "ocr_paddleocr_vl": "PaddlePaddle/PaddleOCR-VL-1.6",
    "ocr_paddleocr_vl_4bit": "mlx-community/PaddleOCR-VL-1.6-4bit",
}
VL_MAX_TOKENS = 4096


def run_vl(corpus: Path, items: list[dict], repo: str) -> dict[str, dict]:
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(repo)
    prompt = apply_chat_template(processor, model.config, "OCR:", num_images=1)
    out = {}
    for item in items:
        pages = []
        for page in item["pages"]:
            model.language_model._position_ids = None
            model.language_model._rope_deltas = None
            text, last = "", None
            for last in stream_generate(
                model,
                processor,
                prompt,
                image=[str(corpus / page)],
                max_tokens=VL_MAX_TOKENS,
                temperature=0.0,
            ):
                text += last.text
            pages.append(
                {
                    "markdown": text.strip(),
                    "finish": last.finish_reason if last else "error",
                }
            )
        out[item["id"]] = {"pages": pages}
    return out


def swap_head_convs(variant, tensors):
    out = dict(tensors)
    for key in list(tensors):
        if key.startswith("head.encoder.conv_block.0."):
            other = key.replace("conv_block.0.", "conv_block.1.")
            out[key], out[other] = tensors[other], tensors[key]
    return out


def pp_artifacts(task: str):
    from huggingface_hub import snapshot_download
    from mlx4ocr.hub.download import HubArtifacts

    root = Path(
        snapshot_download(
            f"PaddlePaddle/PP-OCRv6_medium_{task}_safetensors", local_files_only=True
        )
    )
    return HubArtifacts(
        root=root,
        config=root / "config.json",
        inference=root / "inference.yml",
        weights=root / "model.safetensors",
        preprocessor=root / "preprocessor_config.json",
        variant="medium",
        task=task,
    )


def run_pp(corpus: Path, items: list[dict]) -> dict[str, dict]:
    import mlx4ocr.models.rec.model as recmodel
    import numpy as np
    from mlx4ocr.pipeline.ocr import PP_OCRv6
    from PIL import Image

    recmodel.load_patched_recognition_tensors = swap_head_convs
    ocr = PP_OCRv6.from_artifacts(
        "medium", pp_artifacts("det"), pp_artifacts("rec"), compile_models=False
    )
    out = {}
    for item in items:
        pages = []
        for page in item["pages"]:
            with Image.open(corpus / page) as im:
                bgr = np.asarray(im.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()
            blocks = ocr.predict(bgr).result.blocks
            text = "\n".join(b.text for b in blocks)
            pages.append({"markdown": text, "finish": "stop" if text else "empty"})
        out[item["id"]] = {"pages": pages}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="target/corpus")
    ap.add_argument("--only", choices=["vl", "pp"])
    args = ap.parse_args()
    corpus = Path(args.corpus).resolve()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    path = corpus / "baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.only in (None, "vl"):
        for key, repo in VL_REPOS.items():
            baseline[key] = run_vl(corpus, manifest["ocr"], repo)
    if args.only in (None, "pp"):
        baseline["ocr_pp_ocrv6"] = run_pp(corpus, manifest["ocr"])
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
