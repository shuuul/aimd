#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "easydict==1.13",
#     "einops==0.8.1",
#     "mlx==0.32.2",
#     "mlx-vlm==0.7.0",
#     "numpy==2.4.6",
#     "pillow==12.3.0",
#     "safetensors==0.8.0",
#     "torch==2.9.0",
# ]
# ///
"""Export golden reference tensors for the Unlimited-OCR vision path (WS-07a).

The towers run Baidu's native reference code: `deepencoder.py` from the
`baidu/Unlimited-OCR` HF snapshot (SAM ViT-B + CLIP-L, QuickGELU, CLIP
LayerNorm eps 1e-5, bicubic-antialias position embeddings, half-pixel
rel-pos interpolation), in torch f32, with the weights aimd loads: the bf16
vision tensors of `mlx-community/Unlimited-OCR-4bit` (identical to Baidu's)
and its 4-bit affine projector dequantized. mlx-vlm 0.7.0 only supplies the
gundam preprocessing (`dynamic_preprocess`, checked against its processor);
its vision graph diverged from the reference (Blaizzy/mlx-vlm#2603).

Runs the vision path on deterministic synthetic images in gundam mode (1024
global view + 640 local crops) and dumps per-stage tensors as raw
little-endian f32 (plus bf16-bit final features and u8 resized pixels) with a
`manifest.json` into `crates/aimd-engine/tests/golden/ocr_vision/`.

Stages dumped per image (mirroring `Model.get_input_embeddings`):

- resized u8 pixels (PIL bicubic resize + gray pad) — the pixel-exact
  preprocessing contract for the Rust port;
- SAM tower (`sam_model`): post patch-embed, post pos-embed add, post blocks
  0/1/6/11, post neck+net_2+net_3;
- CLIP tower (`vision_model`, fed the SAM output as patch embeddings): post
  pre_layernorm, post layers 0/1/6/11, final;
- quantized linear projector output (global + crops);
- final shaped [n_tokens, 1280] features (image_newline columns +
  view_separator row, local-then-global order) as bf16 bits.

Also dumps small resize-parity fixtures (u8, full) and the crop-path
rel-pos/pos-embed interpolation tables. Total size is asserted < 10 MB.

Deterministic: fixed synthetic images (no RNG, no fonts), fixed prompt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from safetensors import safe_open

from mlx_vlm import load
from mlx_vlm.models.deepseekocr.processing_deepseekocr import dynamic_preprocess

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "ocr_vision"
IMAGES_DIR = OUT / "images"

CHECKPOINT = "mlx-community/Unlimited-OCR-4bit"
REFERENCE = "baidu/Unlimited-OCR"
BASE_SIZE = 1024
IMAGE_SIZE = 640
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
# Mirrors aimd's gundam-mode prompt (src/aimd/plugins/ocr/models/mlx.py,
# UNLIMITED_OCR_SINGLE_PROMPT) with the single <image> slot first.
PROMPT = "<image>\ndocument parsing."


# ---------------------------------------------------------------------------
# Deterministic synthetic images (no RNG, no fonts).
# ---------------------------------------------------------------------------


def make_document_image(width: int, height: int) -> Image.Image:
    """Text-like page: paper gradient, ruled text bars, shapes, borders."""
    img = Image.new("RGB", (width, height), (250, 250, 247))
    draw = ImageDraw.Draw(img)
    # Vertical luminance gradient on the paper.
    for y in range(height):
        v = 250 - (60 * y) // max(1, height - 1)
        draw.line([(0, y), (width, y)], fill=(v, v, min(255, v + 3)))
    margin_x = width // 10
    # Text-like bars: mixed widths/greys to exercise spatial frequency content.
    y = height // 12
    seed = 7
    while y < height - height // 12:
        seed = (seed * 1103515245 + 12345) % (1 << 31)
        bar_w = margin_x + (seed % (width - 2 * margin_x))
        grey = 30 + (seed >> 5) % 60
        draw.rectangle(
            [margin_x, y, margin_x + bar_w, y + max(3, height // 60)],
            fill=(grey, grey, grey),
        )
        y += height // 30
    # Shapes with hard edges and smooth interiors.
    draw.ellipse(
        [
            width // 2 - width // 8,
            height // 2 - height // 10,
            width // 2 + width // 8,
            height // 2 + height // 10,
        ],
        outline=(20, 20, 160),
        width=max(2, width // 200),
    )
    draw.line(
        [(margin_x, height - 10), (width - margin_x, height // 2)],
        fill=(180, 30, 30),
        width=max(2, width // 300),
    )
    draw.rectangle(
        [width - margin_x - width // 6, 10, width - margin_x, 10 + height // 8],
        outline=(30, 130, 30),
        width=max(2, width // 250),
    )
    # Diagonal checker patch (high-frequency content).
    cell = max(4, width // 64)
    for iy in range(6):
        for ix in range(10):
            if (ix + iy) % 2 == 0:
                x0 = margin_x + ix * cell * 2
                y0 = height // 3 + iy * cell * 2
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(10, 10, 10))
    # Border.
    draw.rectangle([0, 0, width - 1, height - 1], outline=(60, 60, 60), width=2)
    return img


def make_noise_image(width: int, height: int, seed: int) -> Image.Image:
    """LCG-noise RGB image for resize-parity fixtures (deterministic)."""
    state = seed & 0x7FFFFFFF
    data = bytearray(width * height * 3)
    for i in range(len(data)):
        state = (state * 1103515245 + 12345) % (1 << 31)
        data[i] = (state >> 16) & 0xFF
    return Image.frombytes("RGB", (width, height), bytes(data))


# ---------------------------------------------------------------------------
# PIL gundam preprocessing (mirrors UnlimitedOCRProcessor.tokenize_with_images).
# ---------------------------------------------------------------------------


def gundam_preprocess(img: Image.Image):
    """Returns (crops_u8, crop_ratio, global_u8) mirroring the processor."""
    w, h = img.size
    if img.size[0] <= 640 and img.size[1] <= 640:
        crop_ratio = [1, 1]
        crops_raw = []
    else:
        crops_raw, crop_ratio = dynamic_preprocess(
            img, min_num=2, max_num=32, image_size=IMAGE_SIZE
        )
    global_view = ImageOps.pad(
        img,
        (BASE_SIZE, BASE_SIZE),
        color=tuple(int(x * 255) for x in IMAGE_MEAN),
    )
    crops_u8 = [np.asarray(c, dtype=np.uint8) for c in crops_raw]
    return crops_u8, crop_ratio, np.asarray(global_view, dtype=np.uint8)


def image_transform_u8_to_bf16(u8: np.ndarray) -> mx.array:
    """ImageTransform: HWC u8 -> CHW normalized bf16 ((x-mean)/std)."""
    img = mx.array(u8) / 255.0
    img = mx.transpose(img, [2, 0, 1])
    mean = mx.array(IMAGE_MEAN).reshape(-1, 1, 1)
    std = mx.array(IMAGE_STD).reshape(-1, 1, 1)
    img = (img - mean) / std
    return img.astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Dump helpers.
# ---------------------------------------------------------------------------


def f32_bf16_bits(a: mx.array) -> np.ndarray:
    """bf16 (RNE) bit patterns of an MLX array as uint16 numpy array."""
    f32 = np.asarray(a.astype(mx.float32))
    u32 = f32.view(np.uint32)
    # Round-to-nearest-even truncation to bf16 (same rounding MLX uses).
    rounded = (
        u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    ) >> np.uint32(16)
    return rounded.astype(np.uint16)


def dump_bytes(name: str, data: bytes) -> None:
    (OUT / name).write_bytes(data)


def dump_f32(name: str, arr: np.ndarray, entry: dict) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    dump_bytes(name, arr.tobytes())
    entry["file"] = name
    entry["dtype"] = "f32"
    entry["shape"] = list(arr.shape)


def dump_f32_rows(name: str, arr2d: np.ndarray, rows: list[int], entry: dict) -> None:
    """Dump selected rows of a 2-D [L, C] f32 array (channels complete)."""
    sub = np.ascontiguousarray(arr2d[rows], dtype="<f4")
    dump_bytes(name, sub.tobytes())
    entry["file"] = name
    entry["dtype"] = "f32"
    entry["shape"] = list(sub.shape)
    entry["source_shape"] = list(arr2d.shape)
    entry["rows"] = rows


def dump_bf16_rows_full(name: str, arr2d_mx: mx.array, entry: dict) -> None:
    """Dump a full 2-D [L, C] bf16 tensor as bf16 bits (uint16 LE)."""
    bits = f32_bf16_bits(arr2d_mx)
    bits = np.ascontiguousarray(bits)
    dump_bytes(name, bits.tobytes())
    entry["file"] = name
    entry["dtype"] = "bf16_bits_u16"
    entry["shape"] = [bits.shape[0], bits.shape[1]]


def dump_bf16_rows(name: str, arr2d_mx: mx.array, rows: list[int], entry: dict) -> None:
    """Dump selected rows of a 2-D [L, C] bf16 tensor as bf16 bits (uint16 LE)."""
    bits = np.ascontiguousarray(f32_bf16_bits(arr2d_mx)[rows])
    dump_bytes(name, bits.tobytes())
    entry["file"] = name
    entry["dtype"] = "bf16_bits_u16"
    entry["shape"] = [bits.shape[0], bits.shape[1]]
    entry["source_shape"] = list(arr2d_mx.shape)
    entry["rows"] = rows


# Committed goldens stay <= 1 MB: larger final-feature tensors keep every
# other row plus the last one.
MAX_FULL_FEATURE_BYTES = 1 << 20


def dump_u8_rows(name: str, arr: np.ndarray, rows: list[int], entry: dict) -> None:
    """Dump selected leading-dim rows of an HWC u8 array (W and C complete)."""
    sub = np.ascontiguousarray(arr[rows], dtype=np.uint8)
    dump_bytes(name, sub.tobytes())
    entry["file"] = name
    entry["dtype"] = "u8"
    entry["shape"] = list(sub.shape)
    entry["source_shape"] = list(arr.shape)
    entry["rows"] = rows


def stage_rows(length: int, denom: int = 24) -> list[int]:
    """Deterministic row sample covering head, stride, and tail."""
    s = max(1, length // denom)
    return sorted(set(range(0, length, s)) | set(range(min(16, length))) | {length - 1})


def pixel_rows(length: int) -> list[int]:
    """Every 5th leading-dim row (each axis' coefficient table is shared by
    every output pixel of that axis, so row sampling loses no coverage)."""
    return list(range(0, length, 5))


# ---------------------------------------------------------------------------
# Baidu reference towers (torch f32) with aimd's checkpoint weights.
# ---------------------------------------------------------------------------


def find_reference_snapshot() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    root = base / "models--baidu--Unlimited-OCR" / "snapshots"
    for entry in sorted(root.iterdir()):
        if (entry / "deepencoder.py").is_file():
            return entry
    raise SystemExit(f"deepencoder.py not found under {root}")


def load_deepencoder(snapshot: Path):
    spec = importlib.util.spec_from_file_location(
        "baidu_deepencoder", snapshot / "deepencoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_torch(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def dequantize_affine(w: np.ndarray, scales, biases, group: int = 64, bits: int = 4):
    """MLX affine dequantization: uint32 words packing 32/bits values, low bits first."""
    per = 32 // bits
    shifts = (np.arange(per, dtype=np.uint32) * bits)[None, None, :]
    q = (w[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    q = q.reshape(w.shape[0], -1).astype(np.float32)
    q = q.reshape(w.shape[0], -1, group)
    out = q * scales[..., None] + biases[..., None]
    return out.reshape(w.shape[0], -1)


def load_reference_towers(deepencoder, checkpoint: Path):
    """SAM / CLIP / projector modules from Baidu's code with the MLX checkpoint.

    MLX stores convolutions OHWI; torch expects OIHW."""
    sam = deepencoder.build_sam_vit_b()
    clip = deepencoder.build_clip_l()
    projector = torch.nn.Linear(2048, 1280)
    with safe_open((checkpoint / "model.safetensors").as_posix(), "pt") as ft:

        def tensor(key: str) -> torch.Tensor:
            return ft.get_tensor(key).to(torch.float32)

        for prefix, module in (("sam_model.", sam), ("vision_model.", clip)):
            state = {}
            for name, ref in module.state_dict().items():
                if name == "embeddings.position_ids":
                    state[name] = ref
                    continue
                t = tensor(prefix + name)
                if t.ndim == 4 and t.shape != ref.shape:
                    t = t.permute(0, 3, 1, 2)
                assert t.shape == ref.shape, (prefix + name, t.shape, ref.shape)
                state[name] = t
            module.load_state_dict(state, strict=True)
        w = dequantize_affine(
            ft.get_tensor("projector.layers.weight").numpy(),
            tensor("projector.layers.scales").numpy(),
            tensor("projector.layers.biases").numpy(),
        )
        projector.load_state_dict(
            {"weight": to_torch(w), "bias": tensor("projector.layers.bias")}
        )
        image_newline = tensor("image_newline")
        view_separator = tensor("view_separator")
    for m in (sam, clip, projector):
        m.eval().to(torch.float32)
    return sam, clip, projector, image_newline, view_separator


def flat_np(t: torch.Tensor) -> np.ndarray:
    return t.reshape(-1, t.shape[-1]).detach().numpy()


@torch.no_grad()
def run_sam_staged(
    deepencoder,
    sam,
    pixels: torch.Tensor,
    manifest_dumps: list[dict],
    prefix: str,
    dump: bool = True,
):
    """pixels: [N, 3, H, W] f32. Returns the SAM output [N, 1024, h, w] (NCHW).

    Mirrors ``ImageEncoderViT.forward`` with stage taps; dumps are NHWC
    token-flat like the Rust port."""
    pe = sam.patch_embed(pixels)  # NHWC
    x1 = pe + deepencoder.get_abs_pos_sam(sam.pos_embed, pe.size(1))
    if dump:
        for stage, t in (("patch_embed", pe), ("pos_added", x1)):
            entry = {"stage": f"{prefix}.sam.{stage}", "layout": "token_flat"}
            dump_f32_rows(
                f"{prefix}_sam_{stage}.f32",
                flat_np(t),
                stage_rows(t.shape[1] * t.shape[2]),
                entry,
            )
            manifest_dumps.append(entry)
    h = x1
    for i, blk in enumerate(sam.blocks):
        h = blk(h)
        if dump and i in (0, 1, 6, 11):
            entry = {"stage": f"{prefix}.sam.block{i}", "layout": "token_flat"}
            dump_f32_rows(
                f"{prefix}_sam_block{i}.f32",
                flat_np(h),
                stage_rows(h.shape[1] * h.shape[2]),
                entry,
            )
            manifest_dumps.append(entry)
    h = sam.neck(h.permute(0, 3, 1, 2))
    h = sam.net_3(sam.net_2(h))
    if dump:
        nhwc = h.permute(0, 2, 3, 1)
        entry = {"stage": f"{prefix}.sam.final", "layout": "token_flat"}
        dump_f32_rows(
            f"{prefix}_sam_final.f32",
            flat_np(nhwc),
            stage_rows(nhwc.shape[1] * nhwc.shape[2]),
            entry,
        )
        manifest_dumps.append(entry)
    return h


@torch.no_grad()
def run_clip_staged(
    clip,
    pixels: torch.Tensor,
    sam_out: torch.Tensor,
    manifest_dumps: list[dict],
    prefix: str,
    dump: bool = True,
):
    """Returns the CLIP output [N, 1+L, 1024] (``VitModel.forward`` with taps)."""

    def tap(stage: str, h: torch.Tensor):
        if not dump:
            return
        entry = {"stage": f"{prefix}.clip.{stage}", "layout": "token_flat"}
        dump_f32_rows(
            f"{prefix}_clip_{stage}.f32",
            flat_np(h),
            stage_rows(h.shape[0] * h.shape[1]),
            entry,
        )
        manifest_dumps.append(entry)

    h = clip.pre_layrnorm(clip.embeddings(pixels, sam_out))
    tap("pre_layernorm", h)
    for i, layer in enumerate(clip.transformer.layers):
        h = layer(h)
        if i in (0, 1, 6, 11):
            tap(f"layer{i}", h)
    tap("final", h)
    return h


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def find_snapshot() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    root = base / "models--mlx-community--Unlimited-OCR-4bit" / "snapshots"
    for entry in sorted(root.iterdir()):
        if (entry / "config.json").is_file() and (
            entry / "model.safetensors"
        ).is_file():
            return entry
    raise SystemExit(f"checkpoint snapshot not found under {root}")


def main() -> None:
    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)  # stale dumps from previous runs must not survive
    OUT.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = find_snapshot()

    # mlx-vlm: gundam preprocessing only (checked against its processor).
    _, processor = load(snapshot.as_posix())
    reference = find_reference_snapshot()
    deepencoder = load_deepencoder(reference)
    sam, clip, projector, image_newline, view_separator = load_reference_towers(
        deepencoder, snapshot
    )

    manifest: dict = {
        "kind": "ocr_vision",
        "checkpoint": CHECKPOINT,
        "checkpoint_snapshot": snapshot.as_posix(),
        "reference": f"{REFERENCE} deepencoder.py ({reference.name}), torch f32",
        "gundam": {
            "base_size": BASE_SIZE,
            "image_size": IMAGE_SIZE,
            "mean": list(IMAGE_MEAN),
            "prompt": PROMPT,
            "cropping": True,
        },
        "images": {},
        "resize_fixtures": [],
        "aux": {},
        "dumps": [],
        "notes": [
            "resize_fixtures sources are LCG-noise RGB images: "
            "state = (state * 1103515245 + 12345) % 2**31 starting from "
            "lcg_seed, each source byte = (state >> 16) & 0xFF in HWC order.",
            "Vision tensors (vision_model.*, sam_model.*, image_newline, "
            "view_separator) are stored RAW BF16 (unquantized); only "
            "projector.layers is 4-bit affine (group 64) — see "
            "safetensors header survey in the WS-07a report.",
            "Stage tensors are token-flattened [L, C] f32 row samples "
            "(channels complete); final features are full bf16 bits.",
            "Towers: Baidu deepencoder.py in torch f32 (pixels rounded to "
            "bf16 first, like the reference pipeline); the 4-bit projector is "
            "dequantized to f32.",
        ],
    }

    # --- deterministic source images (committed under images/) -------------
    doc_small = make_document_image(500, 400)  # <= 640 both dims: no crops
    doc_crop = make_document_image(1000, 500)  # aspect 2.0 -> crop_ratio (2,1)
    doc_small.save(IMAGES_DIR / "source_small_500x400.png")
    doc_crop.save(IMAGES_DIR / "source_crop_1000x500.png")

    # --- resize parity fixtures (small, full dumps) ------------------------
    fixtures = [
        # (w, h, dst_w, dst_h, seed) — mixed up/downscale, non-integer ratios
        (37, 23, 61, 38, 11),
        (37, 23, 18, 12, 12),
        (64, 49, 103, 79, 13),
        (91, 17, 31, 6, 14),
        (110, 109, 263, 260, 15),
        (48, 120, 121, 303, 16),
        (75, 150, 150, 75, 17),
        (33, 33, 47, 47, 18),
        (200, 53, 53, 200, 19),
        (128, 96, 96, 128, 20),
        (321, 167, 167, 321, 21),
        (61, 31, 213, 108, 22),
    ]
    for w, h, dw, dh, seed in fixtures:
        src = make_noise_image(w, h, seed)
        dst = src.resize((dw, dh))  # PIL default: bicubic
        name = f"resize_{w}x{h}_to_{dw}x{dh}.u8"
        arr = np.asarray(dst, dtype=np.uint8)
        dump_bytes(name, arr.tobytes())
        manifest["resize_fixtures"].append(
            {
                "source": [w, h],
                "target": [dw, dh],
                "file": name,
                "shape": [h, dw, 3],
                "lcg_seed": seed,
                "note": "PIL BICUBIC resize of an LCG-noise RGB image, full u8 dump",
            }
        )

    # --- per-image golden exports ------------------------------------------
    total_bytes = 0

    for tag, img in (("small", doc_small), ("crop", doc_crop)):
        crops_u8, crop_ratio, global_u8 = gundam_preprocess(img)
        prep = processor.process_one(
            prompt=PROMPT,
            images=[img],
            cropping=True,
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
        )
        crops_ref, ori_ref = prep["images"]
        spatial_crop = prep["images_spatial_crop"].tolist()

        # Cross-check the manual preprocessing against the real processor.
        my_global = image_transform_u8_to_bf16(global_u8)
        assert mx.array_equal(my_global, ori_ref[0]), f"{tag}: global pixels differ"
        if crops_u8:
            my_crops = mx.stack(
                [image_transform_u8_to_bf16(c) for c in crops_u8], axis=0
            )
            assert mx.array_equal(my_crops, crops_ref), f"{tag}: crop pixels differ"
        else:
            assert int(crops_ref.sum().item()) == 0, f"{tag}: unexpected crops"

        # Dump resized u8 pixels (pixel-exact preprocessing contract).
        img_meta = {
            "source_size": list(img.size),
            "crop_ratio": list(crop_ratio),
            "spatial_crop": spatial_crop[0] if spatial_crop else None,
            "global_shape": [BASE_SIZE, BASE_SIZE, 3],
            "pad_color": 127,
            "n_crops": len(crops_u8),
        }
        dumps = manifest["dumps"]
        entry = {"stage": f"{tag}.pixels.global_u8", "layout": "hwc_rows"}
        dump_u8_rows(f"{tag}_global_u8.u8", global_u8, pixel_rows(BASE_SIZE), entry)
        dumps.append(entry)
        img_meta["global_dump_rows"] = pixel_rows(BASE_SIZE)
        if crops_u8:
            crop_rows = pixel_rows(IMAGE_SIZE)
            crop_files = []
            for ci, cu8 in enumerate(crops_u8):
                fname = f"{tag}_crop{ci}_u8.u8"
                centry = {"stage": f"{tag}.pixels.crop{ci}_u8", "layout": "hwc_rows"}
                dump_u8_rows(fname, cu8, crop_rows, centry)
                dumps.append(centry)
                crop_files.append(fname)
            img_meta["crop_dump_files"] = crop_files
            img_meta["crop_dump_rows"] = crop_rows
        # Normalized bf16 pixel spot check (rows 0..2 of channel 0).
        norm_slice = my_global[0, :3, :]
        nentry = {"stage": f"{tag}.pixels.global_norm_ch0_rows0_2"}
        dump_f32(
            f"{tag}_global_norm_ch0_rows0_2.f32",
            np.asarray(norm_slice.astype(mx.float32)),
            nentry,
        )
        dumps.append(nentry)

        # --- towers (Baidu reference, torch f32) ------------------------------
        def pixels(arr: mx.array) -> torch.Tensor:
            return to_torch(np.asarray(arr.astype(mx.float32)))

        image_ori = pixels(ori_ref)  # [1, 3, 1024, 1024], bf16-rounded values
        # The global 1024 pass exercises the 64x64 grid (window pad 64->70);
        # the crop pass exercises the 40x40 grid (rel-pos interp) and the
        # 16->10 CLIP pos-embed resample. The CLIP tower stage dumps are taken
        # once (crop pass). The small image keeps only pixels/projector/final.
        tower_dump = tag == "crop"
        sam_global = run_sam_staged(
            deepencoder, sam, image_ori, dumps, f"{tag}_global", dump=tower_dump
        )
        clip_global = run_clip_staged(
            clip, image_ori, sam_global, dumps, f"{tag}_global", dump=False
        )

        @torch.no_grad()
        def projector_out(sam_out: torch.Tensor, clip_out: torch.Tensor):
            feats = torch.cat(
                (clip_out[:, 1:], sam_out.flatten(2).permute(0, 2, 1)), dim=-1
            )
            return projector(feats)

        proj_global = projector_out(sam_global, clip_global)
        entry = {"stage": f"{tag}.global.projector", "layout": "token_flat"}
        dump_f32_rows(
            f"{tag}_global_projector.f32",
            flat_np(proj_global),
            stage_rows(proj_global.shape[0] * proj_global.shape[1]),
            entry,
        )
        dumps.append(entry)

        # Global shaping: [h, w] -> append image_newline column -> flatten.
        gf = proj_global[0]
        hw, n_dim = gf.shape
        h = w = int(hw**0.5)
        gf = torch.cat(
            [gf.reshape(h, w, n_dim), image_newline[None, None, :].expand(h, 1, n_dim)],
            dim=1,
        ).reshape(-1, n_dim)

        if crops_u8:
            crops = pixels(crops_ref)
            sam_local = run_sam_staged(
                deepencoder, sam, crops, dumps, f"{tag}_crops", dump=True
            )
            clip_local = run_clip_staged(
                clip, crops, sam_local, dumps, f"{tag}_crops", dump=True
            )
            proj_local = projector_out(sam_local, clip_local)
            entry = {"stage": f"{tag}.crops.projector", "layout": "token_flat"}
            dump_f32_rows(
                f"{tag}_crops_projector.f32",
                flat_np(proj_local),
                stage_rows(proj_local.shape[0] * proj_local.shape[1]),
                entry,
            )
            dumps.append(entry)

            width_crop_num, height_crop_num = crop_ratio
            _, hw2, n_dim2 = proj_local.shape
            h2 = w2 = int(hw2**0.5)
            lf = (
                proj_local.reshape(height_crop_num, width_crop_num, h2, w2, n_dim2)
                .permute(0, 2, 1, 3, 4)
                .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
            )
            lf = torch.cat(
                [
                    lf,
                    image_newline[None, None, :].expand(
                        height_crop_num * h2, 1, n_dim2
                    ),
                ],
                dim=1,
            ).reshape(-1, n_dim2)
            features = torch.cat([lf, gf, view_separator[None, :]], dim=0)
        else:
            features = torch.cat([gf, view_separator[None, :]], dim=0)
        features = mx.array(features.numpy())

        n_tokens = features.shape[0]
        fentry = {"stage": f"{tag}.final_features", "layout": "full_bf16"}
        if features.size * 2 <= MAX_FULL_FEATURE_BYTES:
            dump_bf16_rows_full(f"{tag}_final_features.bf16u16", features, fentry)
        else:
            fentry["layout"] = "rows_bf16"
            rows = sorted(set(range(0, n_tokens, 2)) | {n_tokens - 1})
            dump_bf16_rows(f"{tag}_final_features.bf16u16", features, rows, fentry)
        dumps.append(fentry)
        img_meta["n_tokens"] = int(n_tokens)
        img_meta["feature_dim"] = int(features.shape[1])
        manifest["images"][tag] = img_meta

        # --- aux tables (crop path only) ------------------------------------
        if tag == "crop" and crops_u8:
            # SAM pos-embed bicubic-antialias interpolation (64 -> 40).
            grid = IMAGE_SIZE // 16
            pe_grid = deepencoder.get_abs_pos_sam(sam.pos_embed, grid).detach()
            flat = flat_np(pe_grid)
            entry = {"stage": "aux.sam_pos_embed_interp_40", "layout": "token_flat"}
            dump_f32_rows(
                "aux_sam_pos_embed_interp_40.f32",
                flat,
                stage_rows(flat.shape[0]),
                entry,
            )
            dumps.append(entry)
            # rel_pos half-pixel linear interpolation for global-attn blocks at 40x40.
            rel = sam.blocks[2].attn.rel_pos_h  # global attention block
            rp = deepencoder.get_rel_pos(grid, grid, rel).detach().numpy()
            entry = {"stage": "aux.rel_pos_h_interp_40"}
            dump_f32("aux_rel_pos_h_interp_40.f32", rp, entry)
            dumps.append(entry)
            # CLIP pos-embed bicubic-antialias interpolation (16 -> 10, + cls).
            pos = clip.embeddings.position_embedding(clip.embeddings.position_ids)
            cp = deepencoder.get_abs_pos(pos, 1 + (IMAGE_SIZE // 64) ** 2).detach()
            flat = flat_np(cp)
            entry = {"stage": "aux.clip_pos_embed_interp_10", "layout": "token_flat"}
            dump_f32_rows(
                "aux_clip_pos_embed_interp_10.f32",
                flat,
                stage_rows(flat.shape[0]),
                entry,
            )
            dumps.append(entry)

    total_bytes += sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())

    # Deterministic ratio candidate order (CPython set order is pinned here so
    # the Rust port can hardcode the exact iteration order of
    # `sorted(target_ratios, key=lambda x: x[0] * x[1])` from dynamic_preprocess).
    target_ratios = set(
        (i, j)
        for n in range(2, 33)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= 32 and i * j >= 2
    )
    ratio_order = sorted(target_ratios, key=lambda x: x[0] * x[1])
    manifest["aux"]["dynamic_preprocess_ratio_order"] = [list(r) for r in ratio_order]

    manifest["total_bytes"] = total_bytes
    assert total_bytes < 10_000_000, f"golden set too large: {total_bytes} bytes"

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"exported {total_bytes} bytes under {OUT}")
    for tag, entry in manifest["images"].items():
        print(
            f"  {tag}: source={entry['source_size']} crop_ratio={entry['crop_ratio']} "
            f"n_tokens={entry['n_tokens']}"
        )


if __name__ == "__main__":
    main()
