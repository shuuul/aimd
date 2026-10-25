#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.2",
#     "numpy==2.4.6",
# ]
# ///
"""Export MLX affine-quantization golden reference fixtures (WS-02).

Deterministic exporter for `crates/aimd-weights/tests/golden/quant/`. Every case stores:

- ``payload.bin``  raw packed-quant bytes (MLX stores them as little-endian
  ``uint32`` words; saved as the exact file bytes, i.e. little-endian).
- ``scales.bin``   per-group scale bytes (f16 or bf16 bit patterns, LE).
- ``biases.bin``   per-group bias bytes (same dtype as scales).
- ``dequant.bin``  ``mx.dequantize`` output bytes (same dtype as scales),
  row-major, the bit-exact ground truth for ``aimd-weights``.

Cases:

(a) synthetic random matrices of several shapes quantized with
    ``mx.quantize(group_size=64, bits=4|8, mode="affine")`` for both f16 and
    bf16 inputs (the scale/bias/output dtype follows the input dtype);
(b) real checkpoint slices: the first rows of small quantized tensors from
    the cached ``mlx-community`` checkpoints (Qwen3-ASR-1.7B 4-bit and 8-bit,
    Unlimited-OCR 4-bit; bf16 scales).

All metadata lands in ``manifest.json`` (bits, group_size, mode, shapes,
dtypes, filenames, byte layouts, sha256). Run:

    uv run scripts/export_quant_reference.py
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "crates" / "aimd-weights" / "tests" / "golden" / "quant"
GROUP_SIZE = 64

HF_SNAPSHOTS = {
    "asr-4bit": Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-4bit/snapshots",
    "asr-8bit": Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots",
    "ocr-4bit": Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Unlimited-OCR-4bit/snapshots",
}

# Real checkpoint tensors to slice (base names; payload lives at
# "<base>.weight", scales at "<base>.scales", biases at "<base>.biases").
CHECKPOINT_SLICES = [
    ("asr-4bit", "model.layers.0.self_attn.k_proj", 8),
    ("asr-8bit", "model.layers.0.self_attn.k_proj", 8),
    ("ocr-4bit", "language_model.lm_head", 8),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantized_tensor_bits(arr: mx.array) -> np.ndarray:
    """Return the raw 16-bit patterns of an f16/bf16 MLX array, little-endian."""
    if arr.dtype == mx.float16:
        return np.array(arr).view(np.uint16)
    if arr.dtype == mx.bfloat16:
        # bf16 -> f32 is exact; the top 16 bits are the bf16 bit pattern.
        f32 = np.array(arr.astype(mx.float32))
        return (f32.view(np.uint32) >> 16).astype(np.uint16)
    raise TypeError(f"unsupported dtype {arr.dtype}")


def mx_array_from_bits(bits: np.ndarray, dtype) -> mx.array:
    """Rebuild an f16/bf16 MLX array from 16-bit patterns without casting."""
    bits = np.ascontiguousarray(bits, dtype=np.uint16)
    if dtype == mx.float16:
        return mx.array(bits.view(np.float16))
    if dtype == mx.bfloat16:
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        return mx.array(f32).astype(mx.bfloat16)
    raise TypeError(f"unsupported dtype {dtype}")


def synthesize(shape: tuple[int, int], dtype, bits: int, seed: int, magnitude: float):
    """Deterministically build a random f16/bf16 matrix and quantize it."""
    rng = np.random.default_rng(seed)
    w_np = (rng.standard_normal(shape) * magnitude).astype(np.float16)
    if dtype == mx.bfloat16:
        # Truncate through bf16 so the input values are exactly representable.
        f32 = w_np.astype(np.float32)
        w = mx.array(f32).astype(mx.bfloat16)
    else:
        w = mx.array(w_np)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits, mode="affine")
    dequant = mx.dequantize(
        wq, scales, biases, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    assert wq.dtype == mx.uint32, wq.dtype
    assert scales.dtype == dtype and biases.dtype == dtype
    assert dequant.dtype == dtype
    return wq, scales, biases, dequant


def snapshot_dir(model_key: str) -> Path | None:
    root = HF_SNAPSHOTS[model_key]
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[-1] if snaps else None


def load_safetensors_header(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        data = fh.read()
    return header, data


def tensor_bytes(header: dict, data: bytes, name: str) -> bytes:
    start, end = header[name]["data_offsets"]
    return data[start:end]


def checkpoint_slice(model_key: str, base: str, rows: int):
    """Quantized slice (first `rows`) of a real checkpoint tensor."""
    snap = snapshot_dir(model_key)
    if snap is None:
        return None
    path = snap / "model.safetensors"
    if not path.is_file():
        return None
    header, data = load_safetensors_header(path)
    scale_info = header.get(f"{base}.scales")
    if scale_info is None:
        raise SystemExit(f"{model_key}: tensor {base}.scales not found in {path}")
    scale_rows, groups = scale_info["shape"]
    cols = groups * GROUP_SIZE
    rows = min(rows, scale_rows)
    quant_cfg = json.loads((snap / "config.json").read_text())["quantization"]
    bits = quant_cfg["bits"]
    assert quant_cfg["group_size"] == GROUP_SIZE and quant_cfg["mode"] == "affine"

    payload = np.frombuffer(
        tensor_bytes(header, data, f"{base}.weight"), dtype=np.uint32
    ).reshape(scale_info_rows(header, base), -1)[:rows]
    scales_bits = np.frombuffer(
        tensor_bytes(header, data, f"{base}.scales"), dtype=np.uint16
    ).reshape(scale_info_rows(header, base), groups)[:rows]
    biases_bits = np.frombuffer(
        tensor_bytes(header, data, f"{base}.biases"), dtype=np.uint16
    ).reshape(scale_info_rows(header, base), groups)[:rows]

    # Rebuild MLX arrays from the raw bits so no value is cast.
    scale_dtype = mx.bfloat16  # mlx-community checkpoints store bf16 scales
    scales = mx_array_from_bits(scales_bits, scale_dtype)
    biases = mx_array_from_bits(biases_bits, scale_dtype)
    wq = mx.array(np.ascontiguousarray(payload), dtype=mx.uint32)
    dequant = mx.dequantize(
        wq, scales, biases, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    assert dequant.dtype == scale_dtype
    return wq, scales, biases, dequant, bits, (rows, cols)


def scale_info_rows(header: dict, base: str) -> int:
    return header[f"{base}.scales"]["shape"][0]


def write_case(case_id: str, meta: dict, wq, scales, biases, dequant) -> dict:
    case_dir = OUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    payload_bytes = np.array(wq).tobytes()  # uint32, little-endian on disk
    scales_bytes = quantized_tensor_bits(scales).tobytes()
    biases_bytes = quantized_tensor_bits(biases).tobytes()
    dequant_bytes = quantized_tensor_bits(dequant).tobytes()
    files = {}
    for key, blob in (
        ("payload", payload_bytes),
        ("scales", scales_bytes),
        ("biases", biases_bytes),
        ("dequant", dequant_bytes),
    ):
        (case_dir / f"{key}.bin").write_bytes(blob)
        files[key] = f"{case_id}/{key}.bin"
    scale_dtype = "F16" if scales.dtype == mx.float16 else "BF16"
    entry = {
        "id": case_id,
        "mode": "affine",
        "bits": int(meta["bits"]),
        "group_size": int(meta["group_size"]),
        "shape": list(meta["shape"]),  # logical dequantized shape
        "payload_dtype": "U32",
        "payload_shape": [meta["shape"][0], meta["shape"][1] * meta["bits"] // 32],
        "scale_dtype": scale_dtype,
        "scales_shape": [meta["shape"][0], meta["shape"][1] // meta["group_size"]],
        "biases_shape": [meta["shape"][0], meta["shape"][1] // meta["group_size"]],
        "dequant_dtype": scale_dtype,
        "byte_layout": (
            "payload: little-endian u32 words, row-major; element j of a row "
            "lives in byte (j*bits)//8 at bit shift (j*bits)%8 (LSB-first). "
            "scales/biases/dequant: f16 or bf16 bit patterns, little-endian, "
            "row-major; group g of row r covers columns [g*group_size, "
            "(g+1)*group_size)."
        ),
        "files": files,
        "sha256": {
            "payload": sha256(payload_bytes),
            "scales": sha256(scales_bytes),
            "biases": sha256(biases_bytes),
            "dequant": sha256(dequant_bytes),
        },
    }
    return entry


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    # (a) synthetic random matrices, f16 and bf16, bits 4 and 8.
    shapes = [(1, 64), (2, 128), (3, 256), (4, 192), (8, 64)]
    seed = 0
    for dtype_name, dtype in (("f16", mx.float16), ("bf16", mx.bfloat16)):
        for bits in (4, 8):
            for shape in shapes:
                seed += 1
                wq, scales, biases, dequant = synthesize(shape, dtype, bits, seed, 1.0)
                case_id = f"synthetic-{dtype_name}-b{bits}-{'x'.join(map(str, shape))}"
                meta = {"bits": bits, "group_size": GROUP_SIZE, "shape": shape}
                cases.append(write_case(case_id, meta, wq, scales, biases, dequant))
    # one large-magnitude case to exercise wider exponents
    for dtype_name, dtype in (("f16", mx.float16), ("bf16", mx.bfloat16)):
        seed += 1
        wq, scales, biases, dequant = synthesize((3, 128), dtype, 4, seed, 50.0)
        case_id = f"synthetic-{dtype_name}-b4-3x128-large"
        meta = {"bits": 4, "group_size": GROUP_SIZE, "shape": (3, 128)}
        cases.append(write_case(case_id, meta, wq, scales, biases, dequant))

    # (b) real checkpoint slices (skipped when the HF cache is absent).
    for model_key, base, rows in CHECKPOINT_SLICES:
        case_id = f"checkpoint-{model_key}-{base.replace('.', '-')}-r{rows}"
        try:
            sliced = checkpoint_slice(model_key, base, rows)
        except FileNotFoundError:
            sliced = None
        if sliced is None:
            print(f"skip (no cache): {model_key} {base}", file=sys.stderr)
            continue
        wq, scales, biases, dequant, bits, shape = sliced
        meta = {"bits": bits, "group_size": GROUP_SIZE, "shape": shape}
        cases.append(write_case(case_id, meta, wq, scales, biases, dequant))
        print(f"exported checkpoint slice: {case_id} shape={shape} bits={bits}")

    manifest = {
        "generator": "scripts/export_quant_reference.py",
        "quantization": {
            "mode": "affine",
            "group_size": GROUP_SIZE,
            "bits": [4, 8],
        },
        "layout": {
            "payload": (
                "Packed quantized weights as little-endian uint32 words, "
                "row-major (rows, cols*bits/32). Within a row, element j is "
                "stored in byte (j*bits)//8, at bit offset (j*bits)%8 from the "
                "LSB of that byte: bits=4 packs two 4-bit elements per byte "
                "(low nibble = even element, high nibble = odd element); "
                "bits=8 stores one element per byte in order."
            ),
            "scales_biases": (
                "f16 or bf16 bit patterns (little-endian u16), shape "
                "(rows, cols/group_size), row-major; group g of row r covers "
                "columns [g*group_size, (g+1)*group_size)."
            ),
            "dequant": (
                "out[r,c] = dequant_dtype(scales_f32[r, c//group_size] * q[r,c] "
                "+ biases_f32[r, c//group_size]); arithmetic in f32, single "
                "round-to-nearest-even cast to the scale dtype (verified "
                "bit-exact against mlx.core dequantize for f16 and bf16)."
            ),
        },
        "cases": cases,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(
        (OUT_DIR / f).stat().st_size for c in cases for f in c["files"].values()
    )
    print(f"manifest: {OUT_DIR / 'manifest.json'}")
    print(f"cases: {len(cases)}, fixture bytes: {total} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
