#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm==0.7.10",
#     "numpy==2.5.4",
#     "pillow==13.0.0",
#     "transformers==5.20.2",
# ]
# ///
"""Export golden references for the native PaddleOCR-VL-1.6 port (spec 004 WS-03).

Loads the official dense bf16 `PaddlePaddle/PaddleOCR-VL-1.6` and the
`mlx-community/PaddleOCR-VL-1.6-4bit` repo from the local HF cache
(`HF_HUB_OFFLINE=1`, never downloads) through the real mlx-vlm 0.7.10 graph
(`mlx_vlm.models.paddleocr_vl`) and writes, into
`crates/aimd-engine/tests/golden/paddleocr_vl/`:

- `images/{small_text,page_text}.png`: rendered fixtures (only when absent;
  the committed PNGs are the inputs from then on). `small_text` exercises the
  smart_resize `min_pixels` upscale path, `page_text` the `max_pixels`
  downscale path.
- `preprocess.json`: smart_resize cases, the u8 -> bf16 pixel LUT
  (`(v / 255.0 - 0.5) / 0.5` through the exact numpy -> mlx -> bf16 chain),
  and per fixture the resized size, `image_grid_thw`, FNV-1a 64 hashes of the
  resized RGB8 buffer and of the `[N, 3, 14, 14]` bf16 pixel bytes, the
  formatted prompt, prompt ids, mRoPE position ids and rope delta.
- `<prec>_vision.bf16u16` + manifest rows: sampled rows of every vision
  stage on `small_text` (patch embed, + position embedding, encoder layers
  0/13/26, post layernorm), and `<prec>_projector.bf16u16`: the full
  projector output (the features the decoder golden splices), for `prec` in
  {bf16, 4bit}. Every manifest dump names its file.
- `<prec>_page_*.bf16u16`: sampled rows of the page embeddings and projector
  output (the large-grid position-embedding interpolation).
- `<prec>_logprobs_*.bin`: the first decode steps of `generate_step` on
  `small_text` (top-16 ids/values and a strided slice of the log-probs).
- `manifest.json`: everything above plus the greedy token streams / texts of
  both fixtures in both precisions (`stream_generate`, temperature 0,
  max_tokens 4096), mlx-vlm timings, and the 4-bit load-time-quantization
  bit-exactness check (dense bf16 weights quantized with `mx.quantize`
  (affine, group 64) vs the mlx-community tensors).
- Spotting (spec 005 WS-03), following the checkpoint README: PIL LANCZOS
  hashes on LCG noise (`preprocess.json` `lanczos`), per fixture the x2
  LANCZOS upscale hash, grid, pixel hash and prompt ids
  (`preprocess.json` `spotting`), and per precision the `Spotting:` greedy
  stream with its parsed spots (`manifest.json` `<prec>.fixtures.*.spotting`,
  max_pixels 2048*28*28).

Usage: uv run scripts/export_paddleocr_vl_reference.py
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from mlx_vlm import load, stream_generate  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl import smart_resize  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import prepare_inputs  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "paddleocr_vl"
IMAGES = OUT / "images"

DENSE = "PaddlePaddle/PaddleOCR-VL-1.6"
Q4 = "mlx-community/PaddleOCR-VL-1.6-4bit"
PROMPT = "OCR:"
SPOTTING_PROMPT = "Spotting:"
SPOTTING_UPSCALE = 1500
SPOTTING_MAX_PIXELS = 2048 * 28 * 28
MAX_TOKENS = 4096
LOGPROB_STEPS = 6
LOGPROB_STRIDE = 97
TOPK = 16
VISION_ROWS = 24
PAGE_ROWS = 32
GROUP_SIZE = 64

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"

PAGE_LINES = [
    ("bold", 34, "Quarterly Engineering Report"),
    ("", 0, ""),
    ("regular", 26, "The native OCR engine converts scanned pages into"),
    ("regular", 26, "Markdown for language model workflows. It runs a"),
    ("regular", 26, "vision transformer and a compact decoder on MLX."),
    ("", 0, ""),
    ("bold", 28, "Key results"),
    ("regular", 26, "1. Latency is measured per page at 144 dpi."),
    ("regular", 26, "2. Output text matches the reference decoder."),
    ("regular", 26, "3. Weights load once per process."),
    ("", 0, ""),
    ("cjk", 28, "文档解析将扫描页面转换为结构化文本。"),
    ("", 0, ""),
    ("regular", 26, "Contact: platform team, building 7, room 210."),
]


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def render_fixtures() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    small = IMAGES / "small_text.png"
    if not small.exists():
        img = Image.new("RGB", (480, 120), (255, 255, 255))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(ARIAL, 28)
        d.text((12, 20), "Hello PaddleOCR-VL 2026", fill=(0, 0, 0), font=font)
        d.text((12, 64), "native MLX in Rust.", fill=(20, 20, 20), font=font)
        img.save(small)
    page = IMAGES / "page_text.png"
    if not page.exists():
        img = Image.new("RGB", (1000, 1240), (255, 255, 255))
        d = ImageDraw.Draw(img)
        y = 60
        for kind, px, text in PAGE_LINES:
            if not text:
                y += 26
                continue
            path = {"bold": ARIAL_BOLD, "regular": ARIAL, "cjk": CJK}[kind]
            font = ImageFont.truetype(path, px)
            d.text((70, y), text, fill=(0, 0, 0), font=font)
            y += int(px * 1.7)
        img.save(page)


def bf16_bytes(a: mx.array) -> bytes:
    a = a.astype(mx.bfloat16)
    return bytes(np.array(a.view(mx.uint16)).astype("<u2").tobytes())


def rows_of(a: mx.array, n: int) -> list[int]:
    total = a.shape[0]
    if total <= n:
        return list(range(total))
    return sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})


def write_dumps(file: str, stages: dict[str, tuple[mx.array, list[int]]]) -> list[dict]:
    """Write the selected rows of each stage into one golden file; every
    manifest entry names its file (each file stays under the 500 KB hook)."""
    dumps = []
    blob = bytearray()
    for stage, (a, rows) in stages.items():
        data = bf16_bytes(a[mx.array(rows)])
        dumps.append(
            {
                "stage": stage,
                "file": file,
                "shape": list(a.shape),
                "rows": rows,
                "offset": len(blob),
                "bytes": len(data),
            }
        )
        blob += data
    (OUT / file).write_bytes(bytes(blob))
    return dumps


def pixel_lut() -> list[int]:
    """u8 -> bf16 bits through the processor's numpy f64 math and the
    BatchFeature (`return_tensors="mlx"`) + `mx.array(pv, dtype=bf16)` chain."""
    v = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
    v = np.repeat(v, 3, axis=2)
    x = v / 255.0
    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)
    x = (x - mean) / std
    a = mx.array(np.array([x]))  # BatchFeature mlx conversion
    a = mx.array(a, dtype=mx.bfloat16)
    bits = np.array(a.view(mx.uint16)).reshape(256, 3)
    assert (bits[:, 0] == bits[:, 1]).all() and (bits[:, 0] == bits[:, 2]).all()
    return [int(b) for b in bits[:, 0]]


def smart_resize_cases() -> list[list[int]]:
    sizes = [
        (120, 480),
        (1240, 1000),
        (28, 28),
        (10, 300),
        (300, 10),
        (27, 5000),
        (100, 100),
        (336, 336),
        (2000, 1500),
        (1754, 1240),
        (3508, 2480),
        (842, 595),
        (1191, 842),
        (14, 2000),
        (50, 9999),
        (999, 1001),
        (448, 252),
        (1080, 1920),
        (4000, 3000),
        (200, 39999),
    ]
    out = []
    for h, w in sizes:
        try:
            rh, rw = smart_resize(
                h, w, factor=28, min_pixels=112896, max_pixels=1003520
            )
            out.append([h, w, rh, rw])
        except ValueError:
            out.append([h, w, -1, -1])
    return out


def make_noise_image(width: int, height: int, seed: int) -> Image.Image:
    """LCG-noise RGB image (same generator as export_ocr_vision_reference)."""
    state = seed & 0x7FFFFFFF
    data = bytearray(width * height * 3)
    for i in range(len(data)):
        state = (state * 1103515245 + 12345) % (1 << 31)
        data[i] = (state >> 16) & 0xFF
    return Image.frombytes("RGB", (width, height), bytes(data))


def lanczos_cases() -> list[dict]:
    """PIL LANCZOS resizes of LCG noise (x2 upscale, odd ratios, and the
    downscale branch with filterscale > 1), as FNV-1a 64 of the RGB8 bytes."""
    out = []
    for (w, h), (dw, dh), seed in (
        ((37, 23), (74, 46), 7),
        ((50, 50), (83, 31), 11),
        ((97, 61), (40, 29), 13),
        ((64, 9), (128, 18), 17),
    ):
        dst = make_noise_image(w, h, seed).resize((dw, dh), Image.Resampling.LANCZOS)
        out.append(
            {
                "source": [w, h],
                "target": [dw, dh],
                "lcg_seed": seed,
                "rgb8_fnv1a64": fnv1a64(np.asarray(dst, dtype=np.uint8).tobytes()),
            }
        )
    return out


def spotting_image(png: Path, tmp: Path) -> tuple[Path, Image.Image]:
    """README spotting preprocessing: x2 LANCZOS when both sides < 1500.
    The result is saved as a lossless PNG for the mlx-vlm processor."""
    img = Image.open(png).convert("RGB")
    w, h = img.size
    if w < SPOTTING_UPSCALE and h < SPOTTING_UPSCALE:
        img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    out = tmp / f"{png.stem}_spotting.png"
    img.save(out)
    return out, img


LOC_ITEM = re.compile(r"(.*?)((?:<\|LOC_\d+\|>){8})", re.S)
LOC_VALUE = re.compile(r"<\|LOC_(\d+)\|>")


def parse_spots(text: str) -> list[dict]:
    """PaddleX `post_process_for_spotting` grammar: text followed by eight
    LOC tokens (TL, TR, BR, BL corners on a 0..1000 grid), normalized."""
    spots = []
    for body, locs in LOC_ITEM.findall(text):
        vals = [int(v) for v in LOC_VALUE.findall(locs)]
        quad = [[vals[i] / 1000.0, vals[i + 1] / 1000.0] for i in range(0, 8, 2)]
        if body.strip():
            spots.append({"text": body.strip(), "quad": quad})
    return spots


def processor_inputs(processor, prompt: str, png: Path):
    inputs = prepare_inputs(processor, images=[png.as_posix()], prompts=[prompt])
    return inputs


def vision_stages(model, pv: mx.array, grid: mx.array) -> dict[str, mx.array]:
    vis = model.visual
    hs = vis._pack_batch_as_sequence(pv, grid)
    stages = {}
    stages["patch_embed"] = vis.embeddings._patch_embed(hs).squeeze(0)
    x = vis.embeddings(hs, grid)
    stages["embeddings"] = x
    rot = vis.rot_pos_emb(grid)
    cu = vis._build_cu_seqlens(grid)
    for i, layer in enumerate(vis.layers):
        x = layer(x, cu_seqlens=cu, rotary_pos_emb=rot)
        if i in (0, 13, len(vis.layers) - 1):
            stages[f"layer_{i}"] = x
    x = vis.post_layernorm(x)
    stages["post_layernorm"] = x
    stages["projector"] = vis.projector(x, grid)
    full = vis(pv, grid, output_hidden_states=False)
    mx.eval(full, *stages.values())
    assert mx.array_equal(full, stages["projector"]).item(), "stage replay != visual()"
    return stages


def logprob_steps(model, inputs, n_steps: int):
    ar = importlib.import_module("mlx_vlm.generate.ar")
    kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }
    model.language_model._position_ids = None
    model.language_model._rope_deltas = None
    gen = ar.generate_step(
        inputs["input_ids"],
        model,
        inputs["pixel_values"],
        inputs.get("attention_mask"),
        max_tokens=n_steps,
        temperature=0.0,
        prompt_cache=make_prompt_cache(model.language_model),
        **kwargs,
    )
    steps = []
    for token, logprobs in gen:
        lp = logprobs.astype(mx.float32)
        mx.eval(lp)
        arr = np.array(lp).reshape(-1)
        top = np.argsort(-arr, kind="stable")[:TOPK]
        steps.append(
            {
                "token": int(token),
                "top_ids": [int(i) for i in top],
                "top_vals": [float(arr[i]) for i in top],
                "strided": arr[::LOGPROB_STRIDE].astype("<f4").tobytes(),
                "dtype": str(logprobs.dtype),
            }
        )
    return steps


def greedy(model, processor, prompt: str, png: Path):
    model.language_model._position_ids = None
    model.language_model._rope_deltas = None
    t0 = time.perf_counter()
    text, last = "", None
    for r in stream_generate(
        model,
        processor,
        prompt,
        image=[png.as_posix()],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    ):
        text += r.text
        last = r
    elapsed = time.perf_counter() - t0
    return {
        "text": text,
        "tokens": [int(t) for t in (last.token_ids or [])],
        "finish_reason": last.finish_reason,
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "elapsed_s": elapsed,
        "markdown": processor.tokenizer.decode(
            last.token_ids or [], skip_special_tokens=True
        ).strip(),
    }


def quant_check(dense_model, q4_dir: Path) -> dict:
    dense = dict(tree_flatten(dense_model.parameters()))
    repo = {}
    for f in sorted(q4_dir.glob("*.safetensors")):
        repo.update(mx.load(f.as_posix()))
    quantized = sorted(k[: -len(".scales")] for k in repo if k.endswith(".scales"))
    qset = set(quantized)
    exact, mismatched = 0, []
    for base in quantized:
        w = dense[base + ".weight"]
        wq, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=4)
        ok = (
            mx.array_equal(wq, repo[base + ".weight"]).item()
            and mx.array_equal(s, repo[base + ".scales"]).item()
            and mx.array_equal(b, repo[base + ".biases"]).item()
        )
        if ok:
            exact += 1
        else:
            mismatched.append(base)
    dense_same, dense_diff = 0, []
    for k, v in repo.items():
        stem = k.rsplit(".", 1)[0]
        if stem in qset and k.rsplit(".", 1)[1] in ("weight", "scales", "biases"):
            continue
        ref = dense.get(k)
        if ref is not None and ref.dtype == v.dtype and mx.array_equal(ref, v).item():
            dense_same += 1
        else:
            dense_diff.append(k)
    unquantized_dense = sorted(
        k[: -len(".weight")]
        for k, v in dense.items()
        if k.endswith(".weight") and v.ndim == 2 and k[: -len(".weight")] not in qset
    )
    return {
        "quantized_modules": quantized,
        "quantized_exact": exact,
        "quantized_mismatched": mismatched,
        "other_tensors_identical": dense_same,
        "other_tensors_different": dense_diff,
        "unquantized_rank2_modules": unquantized_dense,
    }


def spotting(
    model, processor, prec: str, prec_entry: dict, manifest: dict, preprocess: dict
):
    """`Spotting:` per the checkpoint README (x2 LANCZOS below 1500 px,
    max_pixels 2048*28*28): preprocessing facts into `preprocess.json`
    (bf16 pass) and the greedy generation + parsed spots per fixture."""
    prompt = apply_chat_template(processor, model.config, SPOTTING_PROMPT, num_images=1)
    manifest["spotting_formatted_prompt"] = prompt
    ip = processor.image_processor
    default_max = ip.max_pixels
    ip.max_pixels = SPOTTING_MAX_PIXELS
    with tempfile.TemporaryDirectory() as tmp:
        for name in manifest["fixtures"]:
            spng, simg = spotting_image(IMAGES / f"{name}.png", Path(tmp))
            inputs = processor_inputs(processor, prompt, spng)
            ids = [int(x) for x in np.array(inputs["input_ids"]).reshape(-1)]
            grid = [int(x) for x in np.array(inputs["image_grid_thw"]).reshape(-1)]
            if prec == "bf16":
                pv = mx.array(inputs["pixel_values"], dtype=mx.bfloat16)
                pv_bits = np.array(pv.view(mx.uint16)).astype("<u2")
                preprocess.setdefault("spotting", {})[name] = {
                    "upscaled_wh": list(simg.size),
                    "upscaled_rgb8_fnv1a64": fnv1a64(
                        np.asarray(simg, dtype=np.uint8).tobytes()
                    ),
                    "grid_thw": grid,
                    "pixel_bf16_fnv1a64": fnv1a64(pv_bits.tobytes()),
                    "prompt_ids": ids,
                }
            else:
                assert preprocess["spotting"][name]["prompt_ids"] == ids
            g = greedy(model, processor, prompt, spng)
            g["plain"] = LOC_VALUE.sub("", g["markdown"]).strip()
            g["spots"] = parse_spots(g["markdown"])
            prec_entry["fixtures"][name]["spotting"] = g
            print(
                f"[{prec}] {name} spotting: {len(g['tokens'])} tokens finish={g['finish_reason']} "
                f"prompt {g['prompt_tokens']}, {g['elapsed_s']:.2f} s, {len(g['spots'])} spots",
                file=sys.stderr,
            )
    ip.max_pixels = default_max


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_fixtures()
    manifest: dict = {
        "versions": {
            "mlx": mx.__version__,
            "mlx_vlm": importlib.import_module("mlx_vlm").__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "logprob_stride": LOGPROB_STRIDE,
        "topk": TOPK,
        "fixtures": ["small_text", "page_text"],
    }
    preprocess: dict = {
        "smart_resize": smart_resize_cases(),
        "pixel_lut_bf16": pixel_lut(),
        "lanczos": lanczos_cases(),
        "fixtures": {},
    }

    from huggingface_hub import snapshot_download

    dense_dir = Path(snapshot_download(DENSE, local_files_only=True))
    q4_dir = Path(snapshot_download(Q4, local_files_only=True))

    for prec, repo in (("bf16", DENSE), ("4bit", Q4)):
        model, processor = load(repo)
        prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
        manifest["formatted_prompt"] = prompt
        prec_entry = {"repo": repo, "fixtures": {}}
        for name in manifest["fixtures"]:
            png = IMAGES / f"{name}.png"
            inputs = processor_inputs(processor, prompt, png)
            ids = [int(x) for x in np.array(inputs["input_ids"]).reshape(-1)]
            grid = inputs["image_grid_thw"]
            grid_l = [int(x) for x in np.array(grid).reshape(-1)]
            pv = mx.array(inputs["pixel_values"], dtype=mx.bfloat16)
            mx.eval(pv)
            if prec == "bf16":
                pil = Image.open(png)
                pil = ImageOps.exif_transpose(pil).convert("RGB")
                rh, rw = grid_l[1] * 14, grid_l[2] * 14
                resized = pil.resize((rw, rh), Image.Resampling.BICUBIC)
                u8 = np.asarray(resized, dtype=np.uint8)
                lut = np.array(preprocess["pixel_lut_bf16"], dtype=np.uint16)
                chw = lut[u8].transpose(2, 0, 1)  # [3, H, W]
                gh, gw = grid_l[1], grid_l[2]
                patches = (
                    chw.reshape(3, gh, 14, gw, 14)
                    .transpose(1, 3, 0, 2, 4)
                    .reshape(-1, 3, 14, 14)
                )
                pv_bits = np.array(pv.view(mx.uint16)).reshape(-1, 3, 14, 14)
                assert (patches == pv_bits).all(), f"{name}: LUT/patchify mismatch"
                lm = model.language_model
                pos, delta = lm.get_rope_index(inputs["input_ids"], grid, None, None)
                preprocess["fixtures"][name] = {
                    "image": f"images/{name}.png",
                    "size_wh": list(pil.size),
                    "resized_hw": [rh, rw],
                    "grid_thw": grid_l,
                    "resized_rgb8_fnv1a64": fnv1a64(u8.tobytes()),
                    "pixel_bf16_fnv1a64": fnv1a64(pv_bits.astype("<u2").tobytes()),
                    "n_patches": int(pv_bits.shape[0]),
                    "prompt_ids": ids,
                    "position_ids": np.array(pos).reshape(3, -1).astype(int).tolist(),
                    "rope_delta": int(np.array(delta).reshape(-1)[0]),
                }
            else:
                assert preprocess["fixtures"][name]["prompt_ids"] == ids
            fx: dict = {}
            if name == "small_text":
                stages = vision_stages(model, pv, grid)
                proj = stages.pop("projector")
                dumps = write_dumps(
                    f"{prec}_vision.bf16u16",
                    {s: (a, rows_of(a, VISION_ROWS)) for s, a in stages.items()},
                )
                dumps += write_dumps(
                    f"{prec}_projector.bf16u16",
                    {"projector": (proj, list(range(proj.shape[0])))},
                )
                fx["vision"] = {"dumps": dumps}
                steps = logprob_steps(model, inputs, LOGPROB_STEPS)
                lp_blob = b"".join(s.pop("strided") for s in steps)
                (OUT / f"{prec}_logprobs_small.f32").write_bytes(lp_blob)
                fx["logprobs"] = {"file": f"{prec}_logprobs_small.f32", "steps": steps}
            else:
                stages = {}
                vis = model.visual
                hs = vis._pack_batch_as_sequence(pv, grid)
                stages["embeddings"] = vis.embeddings(hs, grid)
                stages["projector"] = vis(pv, grid, output_hidden_states=False)
                mx.eval(*stages.values())
                dumps = write_dumps(
                    f"{prec}_page_vision.bf16u16",
                    {s: (a, rows_of(a, PAGE_ROWS)) for s, a in stages.items()},
                )
                fx["vision"] = {"dumps": dumps}
            fx["greedy"] = greedy(model, processor, prompt, png)
            g = fx["greedy"]
            print(
                f"[{prec}] {name}: {len(g['tokens'])} tokens finish={g['finish_reason']} "
                f"prompt {g['prompt_tokens']} @ {g['prompt_tps']:.0f} tok/s, gen {g['generation_tps']:.1f} tok/s, "
                f"{g['elapsed_s']:.2f} s\n{g['text']!r}",
                file=sys.stderr,
            )
            prec_entry["fixtures"][name] = fx
        spotting(model, processor, prec, prec_entry, manifest, preprocess)
        if prec == "bf16":
            dense_model = model
        else:
            manifest["quant_check_4bit"] = quant_check(dense_model, q4_dir)
            del dense_model
        manifest[prec] = prec_entry
        del model

    manifest["checkpoints"] = {"bf16": dense_dir.name, "4bit": q4_dir.name}
    (OUT / "preprocess.json").write_text(json.dumps(preprocess, indent=1) + "\n")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    )
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            assert f.stat().st_size <= 1 << 20, f"{f} exceeds 1 MB"
    qc = manifest["quant_check_4bit"]
    print(
        f"4-bit quant check: {qc['quantized_exact']}/{len(qc['quantized_modules'])} exact, "
        f"mismatched {qc['quantized_mismatched'][:5]}, other identical {qc['other_tensors_identical']}, "
        f"other different {qc['other_tensors_different'][:10]}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
