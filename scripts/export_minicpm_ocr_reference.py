#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@230fc308efb577d87446306bfb8a3e9f1a129558",
#     "numpy==2.5.4",
#     "pillow==13.0.0",
#     "torch==2.14.1",
#     "torchvision==0.29.1",
#     "transformers==5.21.0",
# ]
# ///
"""Export golden references for the native MiniCPM-OCR 1.0 port (spec 009).

The numerical source of truth is the original implementation: transformers'
`minicpm_ocr` (`MiniCPMOcrForConditionalGeneration` + `MiniCPMOcrProcessor`,
the code vLLM serving and the official SDK build on), run in torch bf16 on
the CPU from the official `openbmb/MiniCPM-OCR-1.0` snapshot in the local HF
cache (`HF_HUB_OFFLINE=1`, never downloads). It writes, into
`crates/aimd-engine/tests/golden/minicpm_ocr/`:

- `images/{small_text,page_text}.png`: rendered fixtures (only when absent).
  `small_text` is unsliced (upscaled to the 448 budget), `page_text` slices
  into a 2 x 3 grid plus the overview.
- `preprocess.json`: the u8 -> bf16 pixel LUT (the processor's
  `rescale_and_normalize`, then bf16), slicing cases (grid, source size,
  refine size), the 70-bucket position boundaries, and per fixture each
  crop's size, target grid, row/column buckets, FNV-1a 64 hashes of its
  resized RGB8 bytes (torchvision's uint8 antialiased bicubic) and of its
  `[L, 14, 14, 3]` bf16 patch bytes, plus the prompt ids and
  image-placeholder spans.
- `<prec>_vision.bf16u16`: sampled rows of every vision stage on
  `small_text` (patch embed, + position embedding, encoder layers 0/13/26,
  post layernorm, merger); `<prec>_page.bf16u16`: sampled merger rows of
  every `page_text` crop.
- `<prec>_decoder.bf16u16`: sampled rows of the prefill hidden states after
  layers 0 (Gated DeltaNet), 3 (full attention), 11 and 23 and the final
  norm on `small_text`.
- `<prec>_logprobs_small.f32`: the first greedy steps' f32 log-softmax of
  the logits (top-16 ids/values in the manifest, a strided slice in the
  file).
- `manifest.json`: all of the above plus greedy generations (the SDK's
  `no_repeat_ngram_size=100`) of both fixtures in both precisions, and the
  load-time 4-bit quantization facts.

4-bit has no official checkpoint. Its reference is the same transformers
graph with mlx-vlm's convert set (`mlx_vlm.convert`: full-attention
q/k/v/o and every MLP) replaced by the affine group-64 4-bit weights of
`mx.quantize`, dequantized in f32 and applied as an f32 matmul rounded once
to bf16 (what `quantized_matmul` computes). The exporter also checks that
`mx.quantize` of the dense weights is bit-exact with `mlx_vlm.convert`.

Usage: uv run scripts/export_minicpm_ocr_reference.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "minicpm_ocr"
IMAGES = OUT / "images"

DENSE = "openbmb/MiniCPM-OCR-1.0"
PROMPT = "Text Recognition:"
MAX_TOKENS = 4096
NO_REPEAT_NGRAM = 100
LOGPROB_STEPS = 6
LOGPROB_STRIDE = 97
TOPK = 16
VISION_ROWS = 24
PAGE_ROWS = 12
MERGER_ROWS = 64
DECODER_ROWS = 24
GROUP_SIZE = 64
PATCH = 14
SCALE = 448
MAX_SLICES = 9
POS_SIDE = 70
IMAGE_PAD = "<|image_pad|>"

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


def render_fixtures() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    small = IMAGES / "small_text.png"
    if not small.exists():
        img = Image.new("RGB", (480, 120), (255, 255, 255))
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(ARIAL, 28)
        d.text((12, 20), "Hello MiniCPM-OCR 2026", fill=(0, 0, 0), font=font)
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
            d.text((70, y), text, fill=(0, 0, 0), font=ImageFont.truetype(path, px))
            y += int(px * 1.7)
        img.save(page)


# --- Slicing, transformers `MiniCPMOcrImageProcessor` semantics (4x mode) ---


def ensure_divide(length: float, d: int) -> int:
    return max(round(length / d) * d, d)


def find_best_resize(w: float, h: float, allow_upscale: bool) -> tuple[int, int]:
    if w * h > SCALE * SCALE or allow_upscale:
        ratio = w / h
        h = int(SCALE / math.sqrt(ratio))
        w = int(h * ratio)
    return ensure_divide(w, PATCH * 2), ensure_divide(h, PATCH * 2)


def sliced_grid(w: int, h: int) -> tuple[tuple[int, int] | None, bool]:
    """`(cols, rows)` or None, and whether the choice broke a tie."""
    log_ratio = math.log(w / h)
    multiple = min(math.ceil(w * h / (SCALE * SCALE)), MAX_SLICES)
    if multiple <= 1:
        return None, False
    best, best_err, tie = (1, 1), float("inf"), False
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > MAX_SLICES:
            continue
        for rows in range(1, n + 1):
            if n % rows:
                continue
            cols = n // rows
            err = abs(log_ratio - math.log(cols / rows))
            if err < best_err:
                best, best_err = (cols, rows), err
            elif err == best_err:
                tie = True
                if rows > best[1]:
                    best = (cols, rows)
    return best, tie


def slice_plan(w: int, h: int) -> dict:
    grid, tie = sliced_grid(w, h)
    if grid is None:
        return {"grid": None, "tie": tie, "source": find_best_resize(w, h, True)}
    cols, rows = grid
    rw, rh = ensure_divide(w, cols), ensure_divide(h, rows)
    cw, ch = find_best_resize(rw / cols, rh / rows, True)
    return {
        "grid": [cols, rows],
        "tie": tie,
        "source": find_best_resize(w, h, False),
        "refine": [cw * cols, ch * rows],
    }


SLICING_SIZES = [
    (480, 120),
    (1000, 1240),
    (28, 28),
    (300, 10),
    (10, 300),
    (448, 448),
    (449, 448),
    (896, 448),
    (1240, 1754),
    (2480, 3508),
    (1920, 1080),
    (5000, 27),
    (640, 64),
    (1200, 600),
    (900, 900),
]


def slicing_cases() -> list[dict]:
    out = []
    for w, h in SLICING_SIZES:
        plan = slice_plan(w, h)
        out.append(
            {
                "size_wh": [w, h],
                **{k: list(v) if isinstance(v, tuple) else v for k, v in plan.items()},
            }
        )
    return out


def boundaries() -> np.ndarray:
    return np.arange(1 / POS_SIDE, 1.0, 1 / POS_SIDE).astype(np.float32)


def buckets(n: int) -> list[int]:
    frac = np.clip(
        np.arange(n, dtype=np.float32) / np.float32(n), 0.0, np.float32(1.0 - 1e-6)
    )
    return [int(x) for x in (frac[:, None] >= boundaries()[None, :]).sum(axis=1)]


def rows_of(n_total: int, n: int, extra: tuple[int, ...] = ()) -> list[int]:
    if n_total <= n:
        return list(range(n_total))
    rows = {round(i * (n_total - 1) / (n - 1)) for i in range(n)}
    rows.update(e for e in extra if 0 <= e < n_total)
    return sorted(rows)


def image_spans(ids: list[int], pad_id: int) -> list[list[int]]:
    spans, start = [], None
    for i, t in enumerate(ids + [-1]):
        if t == pad_id and start is None:
            start = i
        elif t != pad_id and start is not None:
            spans.append([start, i])
            start = None
    return spans


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return (
        t.detach()
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )


def write_dumps(
    file: str, stages: dict[str, tuple[torch.Tensor, list[int]]]
) -> list[dict]:
    """Selected rows of each stage in one golden file; every manifest entry
    names its file."""
    dumps, blob = [], bytearray()
    for stage, (a, rows) in stages.items():
        data = bf16_bits(a[rows]).astype("<u2").tobytes()
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


def pixel_lut(processor) -> list[int]:
    """u8 -> bf16 bits: the processor's `rescale_and_normalize` in f32, then
    the `pixel_values.to(bf16)` cast of `get_image_features`."""
    ip = processor.image_processor
    v = (
        torch.arange(256, dtype=torch.float32)
        .view(1, 1, 1, 256)
        .expand(1, 3, 1, 256)
        .contiguous()
    )
    out = ip.rescale_and_normalize(
        v, True, ip.rescale_factor, True, ip.image_mean, ip.image_std
    )
    bits = bf16_bits(out[0, :, 0, :])
    assert (bits[0] == bits[1]).all() and (bits[0] == bits[2]).all()
    return [int(b) for b in bits[0]]


def processor_inputs(processor, png: Path):
    """Processor inputs plus the uint8 crops it resized (overview, then the
    refine image cut row-major into the grid)."""
    ip = processor.image_processor
    resized: list[np.ndarray] = []
    orig = ip.resize

    def capture(image, size, **kw):
        out = orig(image, size=size, **kw)
        assert out.dtype == torch.uint8
        resized.append(out.permute(1, 2, 0).numpy().copy())
        return out

    ip.resize = capture
    try:
        img = Image.open(png).convert("RGB")
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        prompt = processor.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=prompt, images=[img], return_tensors="pt")
    finally:
        ip.resize = orig
    plan = slice_plan(*img.size)
    crops = [resized[0]]
    if plan["grid"] is not None:
        cols, rows = plan["grid"]
        refine = resized[1]
        ch, cw = refine.shape[0] // rows, refine.shape[1] // cols
        crops += [
            refine[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            for r in range(rows)
            for c in range(cols)
        ]
    return prompt, inputs, img, plan, crops


def preprocess_entry(
    inputs, img, plan, crops, png: Path, lut: list[int], pad_id: int
) -> dict:
    """The `preprocess.json` fixture row, cross-checked against the
    processor's pixel values (`[1, 3, 14, sum(L) * 14]`, NaViT-packed)."""
    pv = bf16_bits(inputs["pixel_values"][0])
    tgts = inputs["target_sizes"].tolist()
    assert len(tgts) == len(crops), f"{png.name}: {len(tgts)} crops vs {len(crops)}"
    lut_arr = np.array(lut, dtype=np.uint16)
    rows, at = [], 0
    for i, u8 in enumerate(crops):
        h, w = u8.shape[:2]
        gh, gw = h // PATCH, w // PATCH
        assert [gh, gw] == tgts[i], f"{png.name} crop {i} tgt"
        # [L, 14, 14, 3] in (py, px, c) order: the patch-embedding matmul rows.
        bits = lut_arr[u8].reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 1, 3, 4)
        n = gh * gw
        ref = (
            pv[:, :, at * PATCH : (at + n) * PATCH]
            .reshape(3, PATCH, n, PATCH)
            .transpose(2, 1, 3, 0)
        )
        at += n
        assert (bits.reshape(-1, PATCH, PATCH, 3) == ref).all(), (
            f"{png.name} crop {i} pixels"
        )
        rows.append(
            {
                "size_wh": [w, h],
                "tgt_hw": [gh, gw],
                "bucket_h": buckets(gh),
                "bucket_w": buckets(gw),
                "rgb8_fnv1a64": fnv1a64(np.ascontiguousarray(u8).tobytes()),
                "patch_bf16_fnv1a64": fnv1a64(bits.astype("<u2").tobytes()),
            }
        )
    ids = inputs["input_ids"][0].tolist()
    spans = image_spans(ids, pad_id)
    assert len(spans) == len(crops)
    for span, row in zip(spans, rows):
        assert span[1] - span[0] == row["tgt_hw"][0] * row["tgt_hw"][1] // 4
    return {
        "image": f"images/{png.name}",
        "size_wh": list(img.size),
        "plan": {k: list(v) if isinstance(v, tuple) else v for k, v in plan.items()},
        "crops": rows,
        "prompt_ids": ids,
        "image_spans": spans,
    }


def first(out):
    return out[0] if isinstance(out, tuple) else out


def vision_stages(model, inputs) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Vision stage outputs for the first crop and the merged features of
    every crop, from hooks on one `get_image_features` call (4x mode)."""
    inner = model.model
    vt = inner.vision_tower
    n0 = int(inputs["target_sizes"][0].prod())
    got: dict[str, torch.Tensor] = {}
    hooks = []

    def keep(name, fn):
        hooks.append(
            fn.register_forward_hook(lambda _m, _i, o: got.__setitem__(name, first(o)))
        )

    keep("patch_embed", vt.embeddings.patch_embedding)
    keep("embeddings", vt.embeddings)
    last = len(vt.encoder.layers) - 1
    for i in (0, 13, last):
        keep(f"layer_{i}", vt.encoder.layers[i])
    keep("post_layernorm", vt.post_layernorm)
    with torch.inference_mode():
        feats = inner.get_image_features(
            inputs["pixel_values"], inputs["target_sizes"], downsample_mode="4x"
        ).pooler_output
    for h in hooks:
        h.remove()
    stages = {"patch_embed": got["patch_embed"].flatten(2).transpose(1, 2)[0, :n0]}
    for k in ["embeddings", *(f"layer_{i}" for i in (0, 13, last)), "post_layernorm"]:
        stages[k] = got[k][0, :n0]
    stages["merger"] = feats[0]
    return stages, list(feats)


def decoder_taps(model, inputs) -> dict[str, torch.Tensor]:
    """Prefill hidden states of the language model (one uncached forward)."""
    lm = model.model.language_model
    got: dict[str, torch.Tensor] = {}
    hooks = [
        lm.register_forward_pre_hook(
            lambda _m, _a, kw: got.__setitem__("inputs_embeds", kw["inputs_embeds"]),
            with_kwargs=True,
        )
    ]
    last = len(lm.layers) - 1
    for i in (0, 3, 11, last):
        hooks.append(
            lm.layers[i].register_forward_hook(
                lambda _m, _i, o, i=i: got.__setitem__(f"layer_{i}", first(o))
            )
        )
    hooks.append(
        lm.norm.register_forward_hook(
            lambda _m, _i, o: got.__setitem__("final_norm", first(o))
        )
    )
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    for h in hooks:
        h.remove()
    return {
        k: got[k][0]
        for k in [
            "inputs_embeds",
            "layer_0",
            "layer_3",
            "layer_11",
            f"layer_{last}",
            "final_norm",
        ]
    }


def logprob_steps(model, inputs, n_steps: int) -> list[dict]:
    """The first greedy steps' f32 log-softmax (generate's f32 logits)."""
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=n_steps,
            output_logits=True,
            return_dict_in_generate=True,
        )
    p = inputs["input_ids"].shape[1]
    steps = []
    for token, logits in zip(out.sequences[0, p:].tolist(), out.logits):
        arr = torch.log_softmax(logits[0].float(), dim=-1).numpy()
        top = np.argsort(-arr, kind="stable")[:TOPK]
        steps.append(
            {
                "token": int(token),
                "top_ids": [int(i) for i in top],
                "top_vals": [float(arr[i]) for i in top],
                "strided": arr[::LOGPROB_STRIDE].astype("<f4").tobytes(),
            }
        )
    return steps


def greedy(model, processor, inputs) -> dict:
    t0 = time.perf_counter()
    with torch.inference_mode():
        seq = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
    p = inputs["input_ids"].shape[1]
    tokens = seq[0, p:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    return {
        "tokens": tokens,
        "finish_reason": "stop" if tokens and tokens[-1] in eos else "length",
        "prompt_tokens": p,
        "elapsed_s": time.perf_counter() - t0,
        "markdown": processor.tokenizer.decode(
            tokens, skip_special_tokens=True
        ).strip(),
    }


def mlx_vlm_quant_set() -> tuple[
    list[str], dict[str, tuple[mx.array, mx.array, mx.array]]
]:
    """Module names `mlx_vlm.convert` quantizes (4-bit, group 64) and their
    converted weights."""
    from mlx_vlm.convert import convert

    with tempfile.TemporaryDirectory() as tmp:
        q4 = Path(tmp) / "q4"
        convert(
            hf_path=DENSE,
            mlx_path=q4.as_posix(),
            quantize=True,
            q_bits=4,
            q_group_size=GROUP_SIZE,
        )
        q = {}
        for f in sorted(q4.glob("*.safetensors")):
            q.update(mx.load(f.as_posix()))
    names = sorted(k[: -len(".scales")] for k in q if k.endswith(".scales"))
    return names, {
        n: (q[n + ".weight"], q[n + ".scales"], q[n + ".biases"]) for n in names
    }


def hf_module(model, mlx_name: str) -> torch.nn.Linear:
    """`language_model.model.layers.L.x.y` -> transformers'
    `model.language_model.layers.L.x.y`."""
    assert mlx_name.startswith("language_model.model."), mlx_name
    return model.get_submodule(
        "model.language_model." + mlx_name[len("language_model.model.") :]
    )


class DequantLinear(torch.nn.Module):
    """`quantized_matmul`: the f32 dequantized weight, f32 accumulation, one
    rounding to bf16."""

    def __init__(self, w: np.ndarray):
        super().__init__()
        self.w = torch.from_numpy(w)

    def forward(self, x):
        return F.linear(x.float(), self.w).to(x.dtype)


def quantize_4bit(model) -> dict:
    names, converted = mlx_vlm_quant_set()
    mismatched = []
    for n in names:
        lin = hf_module(model, n)
        assert lin.bias is None, n
        dense = mx.array(lin.weight.detach().float().numpy()).astype(mx.bfloat16)
        wq, s, b = mx.quantize(dense, group_size=GROUP_SIZE, bits=4)
        cw, cs, cb = converted[n]
        if not (
            mx.array_equal(wq, cw).item()
            and mx.array_equal(s, cs).item()
            and mx.array_equal(b, cb).item()
        ):
            mismatched.append(n)
        w = mx.dequantize(
            wq,
            s.astype(mx.float32),
            b.astype(mx.float32),
            group_size=GROUP_SIZE,
            bits=4,
        )
        parent, leaf = n.rsplit(".", 1)
        hf_module(model, parent).register_module(
            leaf, DequantLinear(np.array(w, dtype=np.float32))
        )
    return {"quantized_modules": names, "quantized_mismatched": mismatched}


def run_precision(prec, model, processor, manifest, preprocess, lut) -> None:
    pad_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    entry: dict = {"fixtures": {}}
    for name in manifest["fixtures"]:
        png = IMAGES / f"{name}.png"
        prompt, inputs, img, plan, crops = processor_inputs(processor, png)
        manifest["formatted_prompt"] = prompt
        row = preprocess_entry(inputs, img, plan, crops, png, lut, pad_id)
        if prec == "bf16":
            preprocess["fixtures"][name] = row
        else:
            assert preprocess["fixtures"][name]["prompt_ids"] == row["prompt_ids"]
        fx: dict = {}
        stages, feats = vision_stages(model, inputs)
        if name == "small_text":
            dumps = {
                k: (v, rows_of(v.shape[0], VISION_ROWS))
                for k, v in stages.items()
                if k != "merger"
            }
            dumps["merger"] = (
                stages["merger"],
                rows_of(stages["merger"].shape[0], MERGER_ROWS),
            )
            fx["vision"] = write_dumps(f"{prec}_vision.bf16u16", dumps)
            taps = decoder_taps(model, inputs)
            n = taps["final_norm"].shape[0]
            fx["decoder"] = write_dumps(
                f"{prec}_decoder.bf16u16",
                {k: (v, rows_of(n, DECODER_ROWS, (n - 1,))) for k, v in taps.items()},
            )
            steps = logprob_steps(model, inputs, LOGPROB_STEPS)
            blob = b"".join(s.pop("strided") for s in steps)
            (OUT / f"{prec}_logprobs_small.f32").write_bytes(blob)
            fx["logprobs"] = {"file": f"{prec}_logprobs_small.f32", "steps": steps}
        else:
            fx["vision"] = write_dumps(
                f"{prec}_page.bf16u16",
                {
                    f"merger_crop_{i}": (f, rows_of(f.shape[0], PAGE_ROWS))
                    for i, f in enumerate(feats)
                },
            )
        g = greedy(model, processor, inputs)
        fx["greedy"] = g
        print(
            f"[{prec}] {name}: {len(g['tokens'])} tokens finish={g['finish_reason']} "
            f"prompt {g['prompt_tokens']}, {g['elapsed_s']:.2f} s\n{g['markdown']}",
            file=sys.stderr,
        )
        entry["fixtures"][name] = fx
    manifest[prec] = entry


def main() -> None:
    import importlib
    import transformers

    OUT.mkdir(parents=True, exist_ok=True)
    render_fixtures()
    torch.manual_seed(0)
    manifest: dict = {
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "mlx": mx.__version__,
            "mlx_vlm": importlib.import_module("mlx_vlm").__version__,
            "mlx_vlm_git": "230fc308efb577d87446306bfb8a3e9f1a129558",
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM,
        "logprob_stride": LOGPROB_STRIDE,
        "topk": TOPK,
        "fixtures": ["small_text", "page_text"],
    }
    processor = AutoProcessor.from_pretrained(DENSE)
    model = AutoModelForImageTextToText.from_pretrained(
        DENSE, dtype=torch.bfloat16
    ).eval()
    lut = pixel_lut(processor)
    preprocess: dict = {
        "pixel_lut_bf16": lut,
        "pos_boundaries_f32": [float(b) for b in boundaries()],
        "slicing": slicing_cases(),
        "fixtures": {},
    }
    run_precision("bf16", model, processor, manifest, preprocess, lut)
    manifest["quant_4bit"] = quantize_4bit(model)
    run_precision("4bit", model, processor, manifest, preprocess, lut)
    (OUT / "preprocess.json").write_text(
        json.dumps(preprocess, indent=1, ensure_ascii=False) + "\n"
    )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    )
    q = manifest["quant_4bit"]
    print(
        f"4bit: {len(q['quantized_modules'])} quantized, {len(q['quantized_mismatched'])} mismatched",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
