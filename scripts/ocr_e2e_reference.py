# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.2",
#     "mlx-vlm==0.7.0",
#     "pillow==12.3.0",
# ]
# ///
"""Render OCR E2E fixtures and dump the mlx_vlm reference.

Usage: uv run scripts/ocr_e2e_reference.py crates/aimd-engine/tests/golden/ocr_e2e [--no-generate]
Writes <name>.png plus <name>.json (input_ids, image token positions,
crop ratio, raw text, tokens, finish_reason, normalized markdown, timing).
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"

FIXTURES = {
    # <= 640 on both sides: global view only (273 image tokens).
    "small_text": (
        (560, 180),
        28,
        [
            "Native engines make context",
            "preparation fast and portable.",
            "Invoice 2026-10-25: total 42.",
        ],
    ),
    # Page-like: > 640 triggers gundam crops.
    "page_text": (
        (900, 1160),
        26,
        [
            "Quarterly Report",
            "",
            "The native OCR engine converts scanned pages",
            "into Markdown for language model workflows.",
            "It runs a SAM vision encoder, a CLIP tower,",
            "and a mixture-of-experts decoder on the CPU.",
            "",
            "Key results",
            "1. Latency is measured per page.",
            "2. Output is normalized Markdown.",
            "3. Repetition loops are blocked by an",
            "   n-gram guard with a window of 128.",
            "",
            "Contact: engineering team, building 7.",
        ],
    ),
}


def render(path: Path, size, font_px, lines):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, font_px)
    y = 30
    for line in lines:
        draw.text((36, y), line, fill=(0, 0, 0), font=font)
        y += int(font_px * 1.6)
    img.save(path)


def main():
    out = Path(sys.argv[1])
    generate = "--no-generate" not in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    from reference_unlimited_ocr import (
        UNLIMITED_OCR_MAX_TOKENS,
        UNLIMITED_OCR_NGRAM_SIZE,
        UNLIMITED_OCR_NGRAM_WINDOW,
        UNLIMITED_OCR_SINGLE_PROMPT,
        SlidingWindowNoRepeatNgramProcessor,
        normalize_unlimited_ocr_markdown,
    )

    model, processor = load("mlx-community/Unlimited-OCR-4bit")
    prompt = apply_chat_template(
        processor, model.config, UNLIMITED_OCR_SINGLE_PROMPT, num_images=1
    )
    for name, (size, font_px, lines) in FIXTURES.items():
        png = out / f"{name}.png"
        render(png, size, font_px, lines)
        image = Image.open(png).convert("RGB")
        inputs = processor.process_one(
            prompt, [image], base_size=1024, image_size=640, cropping=True
        )
        ids = [int(x) for x in inputs["input_ids"][0].tolist()]
        mask = [bool(x) for x in inputs["images_seq_mask"][0].tolist()]
        pos = [i for i, m in enumerate(mask) if m]
        rec = {
            "image": png.name,
            "size": list(size),
            "formatted_prompt": prompt,
            "prompt_ids_len": len(ids),
            "prompt_ids_head": ids[:3],
            "prompt_ids_tail": ids[pos[-1] + 1 :] if pos else ids,
            "image_token_count": len(pos),
            "image_token_first": pos[0],
            "image_token_last": pos[-1],
            "image_tokens_contiguous": pos == list(range(pos[0], pos[-1] + 1)),
            "images_spatial_crop": [
                int(v) for v in inputs["images_spatial_crop"].tolist()[0]
            ],
            "num_crops": int(inputs["images"][0].shape[0])
            if inputs["images_spatial_crop"].tolist()[0] != [1, 1]
            else 0,
        }
        if generate:
            guard = SlidingWindowNoRepeatNgramProcessor(
                UNLIMITED_OCR_NGRAM_SIZE, UNLIMITED_OCR_NGRAM_WINDOW
            )
            t0 = time.perf_counter()
            text, tokens, finish = "", [], None
            for r in stream_generate(
                model,
                processor,
                prompt,
                image=[png.as_posix()],
                max_tokens=UNLIMITED_OCR_MAX_TOKENS,
                temperature=0.0,
                cropping=True,
                image_size=640,
                base_size=1024,
                logits_processors=[guard],
            ):
                text += r.text
                if r.token is not None:
                    tokens.append(int(r.token))
                finish = r.finish_reason
                rec["prompt_tokens"] = r.prompt_tokens
            rec["seconds"] = time.perf_counter() - t0
            rec["raw_text"] = text
            rec["tokens"] = tokens
            rec["finish_reason"] = finish
            rec["markdown"] = normalize_unlimited_ocr_markdown(text.strip())
        (out / f"{name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1) + "\n"
        )
        print(name, {k: v for k, v in rec.items() if k != "tokens"})
        mx.clear_cache()


if __name__ == "__main__":
    main()
