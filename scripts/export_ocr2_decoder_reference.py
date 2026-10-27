#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm @ git+https://github.com/shuuul/mlx-vlm@223c2980",
#     "numpy==2.4.6",
# ]
# ///
"""Export golden decoder fixtures for the Unlimited-OCR-2 (DASH) native port.

Runs the mlx-vlm Unlimited-OCR-2 port (checkpoint
``mlx-community/Unlimited-OCR-2-4bit``) on a two-page base-mode request
(``ocr_e2e/page_text.png`` then ``ocr_e2e/small_text.png``) and writes into
``crates/aimd-engine/tests/golden/ocr2_decoder/``:

- ``page{0,1}_features.bf16`` the projected image features of each page
                      (``[273, 1280]`` raw little-endian bf16 bits), the
                      rows ``get_input_embeddings`` splices at the image
                      tokens. The Rust test splices them itself, so the
                      decoder goldens do not depend on vision drift.
- ``logits.bf16``     full-prompt (unchunked) prefill logits rows
                      ``logits_kept_positions``, bf16 bits.
- ``decode_logits.bf16`` logits of the decode steps ``decode_kept_steps``
                      of the forced stream (step ``k`` consumes
                      ``greedy_tokens[k]``), bf16 bits.
- ``manifest.json``   prompt ids, the greedy stream (up to EOS or
                      ``MAX_STEPS``) with per-step top-2 logit gaps, the
                      prefill routing probe (``topi``/``bias``/``valid`` of
                      layers ``ROUTE_LAYERS`` at query rows ``ROUTE_ROWS``),
                      versions and sha256s.

Usage (from the repo root, with the shuuul/mlx-vlm port installed):

    uv run scripts/export_ocr2_decoder_reference.py [--checkpoint DIR]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "crates/aimd-engine/tests/golden/ocr2_decoder"
PAGES = [
    ROOT / "crates/aimd-engine/tests/golden/ocr_e2e/page_text.png",
    ROOT / "crates/aimd-engine/tests/golden/ocr_e2e/small_text.png",
]
PROMPT = "Multi page parsing."
REPO = "models--mlx-community--Unlimited-OCR-2-4bit"
MAX_STEPS = 400
ROUTE_LAYERS = [0, 5, 11]
DECODE_KEPT = 2


def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    snaps = sorted(glob.glob(os.path.join(hub, REPO, "snapshots", "*")))
    if not snaps:
        raise SystemExit(f"{REPO} not in the HF cache")
    return Path(snaps[0])


def bf16_bits(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.bfloat16).view(mx.uint16))


def write(name: str, data: np.ndarray, shas: dict) -> None:
    path = OUT / name
    path.write_bytes(data.astype("<u2").tobytes())
    if path.stat().st_size > 1 << 20:
        raise SystemExit(f"{name} exceeds 1 MB")
    shas[name] = hashlib.sha256(path.read_bytes()).hexdigest()


def top2_gap(row: np.ndarray) -> float:
    a, b = np.partition(row, -2)[-2:]
    return float(abs(a - b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    args = ap.parse_args()

    import mlx_vlm
    from mlx_vlm import load
    from mlx_vlm.models.unlimited_ocr import dash
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    routes = []
    route_select = dash.route_select

    def probe(*a, **k):
        out = route_select(*a, **k)
        routes.append(out)
        return out

    dash.route_select = probe

    snapshot = resolve_checkpoint(args.checkpoint)
    model, processor = load(str(snapshot))
    lm = model.language_model
    prompt = apply_chat_template(processor, model.config, PROMPT, num_images=len(PAGES))
    inputs = prepare_inputs(processor, images=[str(p) for p in PAGES], prompts=prompt)
    ids = inputs["input_ids"]
    emb = model.get_input_embeddings(
        ids,
        inputs["pixel_values"],
        images_spatial_crop=inputs.get("images_spatial_crop"),
        images_seq_mask=inputs.get("images_seq_mask"),
    ).inputs_embeds
    prompt_ids = np.array(ids[0]).astype(np.int64)
    image_token = int(lm.dash_params.image_token_id)
    image_pos = np.where(prompt_ids == image_token)[0]
    n_pages = len(PAGES)
    per_page = len(image_pos) // n_pages
    assert per_page * n_pages == len(image_pos), "pages differ in image tokens"

    OUT.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}
    feats = emb[0][mx.array(image_pos)]
    for p in range(n_pages):
        write(f"page{p}_features.bf16", bf16_bits(feats[p * per_page : (p + 1) * per_page]), shas)

    cache = lm.make_cache()
    lm._prompt_ids = prompt_ids
    logits = lm(ids, inputs_embeds=emb, cache=cache).logits[0].astype(mx.float32)
    mx.eval(logits)
    L, vocab = logits.shape
    kept = [int(image_pos[per_page // 2]), int(image_pos[-1]) + 1, L - 1]
    write("logits.bf16", bf16_bits(logits[mx.array(kept)]), shas)

    assert len(routes) == lm.config.num_hidden_layers, "one route per layer"
    rows = [int(image_pos[0]) + 200, int(image_pos[per_page]) + 250, L - 1]
    probe_out = []
    for layer in ROUTE_LAYERS:
        topi, bias, valid = (np.array(a) for a in routes[layer])
        probe_out.append(
            {
                "layer": layer,
                "topi": topi[rows].tolist(),
                "bias": np.round(bias[rows].astype(np.float64), 6).tolist(),
                "valid": valid[rows].astype(bool).tolist(),
            }
        )
    dash.route_select = route_select

    eos = int(processor.tokenizer.eos_token_id)
    row = np.array(logits[-1])
    tokens, gaps, decode_rows = [], [], []
    page_token = int(lm.dash_params.page_token_id)
    for step in range(MAX_STEPS):
        tok = int(row.argmax())
        tokens.append(tok)
        gaps.append(round(top2_gap(row), 4))
        if tok == eos:
            break
        out = lm(mx.array([[tok]]), cache=cache).logits[0, -1].astype(mx.float32)
        row = np.array(out)
        if tok == page_token and len(decode_rows) < DECODE_KEPT:
            decode_rows.append((step, out))
    if len(decode_rows) < DECODE_KEPT:
        decode_rows.append((len(tokens) - 2, out))
    write("decode_logits.bf16", bf16_bits(mx.stack([r for _, r in decode_rows])), shas)

    manifest = {
        "checkpoint": REPO,
        "snapshot": snapshot.name,
        "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__,
        "pages": [p.name for p in PAGES],
        "prompt_ids": prompt_ids.tolist(),
        "image_token_id": image_token,
        "page_token_id": page_token,
        "per_page_image_tokens": per_page,
        "hidden": int(emb.shape[-1]),
        "vocab": int(vocab),
        "logits_kept_positions": kept,
        "route_rows": rows,
        "routes": probe_out,
        "greedy_tokens": tokens,
        "top2_gaps": gaps,
        "eos_id": eos,
        "finished_on_eos": tokens[-1] == eos,
        "decode_kept_steps": [s for s, _ in decode_rows],
        "sha256": shas,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"prompt {L} tokens, {len(tokens)} greedy tokens, eos={tokens[-1] == eos}")


if __name__ == "__main__":
    main()
