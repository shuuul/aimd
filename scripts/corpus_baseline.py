# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.2",
#     "mlx-audio==0.5.3",
#     "mlx-vlm==0.7.0",
#     "pillow==12.3.0",
# ]
# ///
"""Run the mlx-audio / mlx-vlm reference over the golden corpus (the baseline).

Usage: uv run scripts/corpus_baseline.py [CORPUS_DIR] [--only asr|ocr]
                                         [--asr-repo ID] [--ocr-repo ID]

Reads ``CORPUS_DIR/manifest.json`` from ``scripts/build_corpus.py`` and writes
``CORPUS_DIR/baseline.json`` in the schema ``corpus_gate`` reads: ASR clip id
-> transcript, OCR document id -> per-page normalized markdown and finish
reason. Settings mirror production: greedy decoding, the clip's language
hint, OCR gundam mode with the sliding-window n-gram guard and 8192 max
tokens. Scoring happens in ``cargo run --release -p aimd-engine --example
corpus_gate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_unlimited_ocr import (
    UNLIMITED_OCR_MAX_TOKENS,
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_NGRAM_WINDOW,
    UNLIMITED_OCR_SINGLE_PROMPT,
    SlidingWindowNoRepeatNgramProcessor,
    normalize_unlimited_ocr_markdown,
)


def run_asr(corpus: Path, items: list[dict], repo: str) -> dict[str, str]:
    from mlx_audio.stt import load

    model = load(repo)
    out = {}
    for item in items:
        result = model.generate(
            str(corpus / item["audio"]), language=item["language_name"], temperature=0.0
        )
        out[item["id"]] = result.text.strip()
    return out


def run_ocr(corpus: Path, items: list[dict], repo: str) -> dict[str, dict]:
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(repo)
    prompt = apply_chat_template(
        processor, model.config, UNLIMITED_OCR_SINGLE_PROMPT, num_images=1
    )
    out = {}
    for item in items:
        pages = []
        for page in item["pages"]:
            guard = SlidingWindowNoRepeatNgramProcessor(
                UNLIMITED_OCR_NGRAM_SIZE, UNLIMITED_OCR_NGRAM_WINDOW
            )
            text, last = [], None
            for last in stream_generate(
                model,
                processor,
                prompt,
                image=[str(corpus / page)],
                max_tokens=UNLIMITED_OCR_MAX_TOKENS,
                temperature=0.0,
                cropping=True,
                image_size=640,
                base_size=1024,
                logits_processors=[guard],
            ):
                text.append(last.text)
            pages.append(
                {
                    "markdown": normalize_unlimited_ocr_markdown("".join(text)),
                    "finish": last.finish_reason if last else "error",
                }
            )
        out[item["id"]] = {"pages": pages}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="target/corpus")
    ap.add_argument("--only", choices=["asr", "ocr"])
    ap.add_argument("--asr-repo", default="mlx-community/Qwen3-ASR-1.7B-4bit")
    ap.add_argument("--ocr-repo", default="mlx-community/Unlimited-OCR-4bit")
    args = ap.parse_args()
    corpus = Path(args.corpus).resolve()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    path = corpus / "baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.only in (None, "asr"):
        baseline["asr"] = run_asr(corpus, manifest["asr"], args.asr_repo)
    if args.only in (None, "ocr"):
        baseline["ocr"] = run_ocr(corpus, manifest["ocr"], args.ocr_repo)
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
