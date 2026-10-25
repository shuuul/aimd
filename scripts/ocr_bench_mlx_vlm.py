# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.2",
#     "mlx-vlm==0.7.0",
#     "pillow==12.3.0",
# ]
# ///
"""Benchmark the mlx-vlm Unlimited-OCR reference (the native MLX port's target).

Usage: uv run scripts/ocr_bench_mlx_vlm.py [--rounds N] image.png [...]

Prints per round: total wall time, vision time (``get_input_embeddings``,
evaluated), and mlx-vlm's own prompt / generation tok/s from
``stream_generate`` with the production settings (temperature 0, 8192 max
tokens, sliding-window n-gram guard). Compare with
``cargo run --release -p aimd-engine --example bench_ocr``.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

from reference_unlimited_ocr import (
    UNLIMITED_OCR_MAX_TOKENS,
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_NGRAM_WINDOW,
    UNLIMITED_OCR_SINGLE_PROMPT,
    SlidingWindowNoRepeatNgramProcessor,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, processor = load("mlx-community/Unlimited-OCR-4bit")
    mx.eval(model.parameters())
    print(f"load: {time.perf_counter() - t0:.2f} s")
    prompt = apply_chat_template(
        processor, model.config, UNLIMITED_OCR_SINGLE_PROMPT, num_images=1
    )
    for path in args.images:
        image = Image.open(path).convert("RGB")
        for rnd in range(args.rounds):
            inp = processor.process_one(
                prompt, [image], base_size=1024, image_size=640, cropping=True
            )
            tv = time.perf_counter()
            emb = model.get_input_embeddings(
                inp["input_ids"],
                pixel_values=inp["images"],
                images_spatial_crop=inp["images_spatial_crop"],
                images_seq_mask=inp["images_seq_mask"],
            ).inputs_embeds
            mx.eval(emb)
            vision = time.perf_counter() - tv
            mx.clear_cache()

            guard = SlidingWindowNoRepeatNgramProcessor(
                UNLIMITED_OCR_NGRAM_SIZE, UNLIMITED_OCR_NGRAM_WINDOW
            )
            t = time.perf_counter()
            last = None
            for last in stream_generate(
                model,
                processor,
                prompt,
                image=[path],
                max_tokens=UNLIMITED_OCR_MAX_TOKENS,
                temperature=0.0,
                cropping=True,
                image_size=640,
                base_size=1024,
                logits_processors=[guard],
            ):
                pass
            total = time.perf_counter() - t
            print(
                f"{path} round{rnd + 1}: total {total:.2f} s | vision {vision:.3f} s | "
                f"prompt {last.prompt_tokens} tok {last.prompt_tps:.0f} tok/s | "
                f"generation {last.generation_tokens} tok {last.generation_tps:.1f} tok/s"
                f" | finish={last.finish_reason}"
            )
            mx.clear_cache()


if __name__ == "__main__":
    main()
