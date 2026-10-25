"""Dump the reference vision features spliced into the <image> positions.

Usage: uv run python scripts/ocr_e2e_features.py crates/aimd-engine/tests/golden/ocr_e2e <out_dir> name [name...]
(small_text_features.bf16 is committed; the 6 MB page dump is not.)
Writes <out_dir>/<name>_features.bf16 (raw little-endian bf16 [n_image, 1280]).
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template


def main():
    fixtures, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model, processor = load("mlx-community/Unlimited-OCR-4bit")
    prompt = apply_chat_template(
        processor, model.config, "document parsing.", num_images=1
    )
    for name in sys.argv[3:]:
        image = Image.open(fixtures / f"{name}.png").convert("RGB")
        inp = processor.process_one(
            prompt, [image], base_size=1024, image_size=640, cropping=True
        )
        emb = model.get_input_embeddings(
            inp["input_ids"],
            pixel_values=inp["images"],
            images_spatial_crop=inp["images_spatial_crop"],
            images_seq_mask=inp["images_seq_mask"],
        ).inputs_embeds
        mask = np.array(inp["images_seq_mask"][0].tolist(), dtype=bool)
        feats = emb[0][mx.array(np.where(mask)[0])].astype(mx.bfloat16)
        mx.eval(feats)
        # bf16 bits via a uint16 view.
        bits = np.array(feats.view(mx.uint16), copy=False).astype("<u2")
        bits.tofile(out / f"{name}_features.bf16")
        print(
            name, feats.shape, feats.dtype, float(feats.astype(mx.float32).abs().mean())
        )


if __name__ == "__main__":
    main()
