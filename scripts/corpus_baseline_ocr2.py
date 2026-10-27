#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm @ git+https://github.com/shuuul/mlx-vlm@223c2980",
#     "pillow==12.3.0",
# ]
# ///
"""Run the mlx-vlm Unlimited-OCR-2 port over the golden corpus OCR set.

Usage: uv run scripts/corpus_baseline_ocr2.py [CORPUS_DIR] [--ocr-repo ID]

Writes the ``ocr_unlimited_ocr2`` key of ``CORPUS_DIR/baseline.json``:
document id -> ``{"raw", "finish"}``. Each document is one request, as aimd
sends a PDF window: every page in one prompt (``<PAGE><image>`` per page,
``Multi page parsing.``), the processor's auto layout (gundam for one page,
base mode for several), greedy decoding, the n-gram guard (35, window 128 for
one page, 1024 for several), the upstream page-count guard and 8192 max
tokens per page. ``raw`` keeps the special tokens; ``corpus_gate
--ocr-model unlimited-ocr-2`` converts it with the same Markdown port as the
native output, so the gate compares model output only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_unlimited_ocr import (
    UNLIMITED_OCR_MAX_TOKENS,
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_NGRAM_WINDOW,
    SlidingWindowNoRepeatNgramProcessor,
    patch_mlx_vlm_to_baidu,
)

PROMPT = "Multi page parsing."
MULTI_PAGE_NGRAM_WINDOW = 1024
MAX_PAGE_TOKENS = 20000


class PageCountGuard:
    """Upstream page-count guard (mirrors ``generate::PageCountGuard``).

    EOS is banned until every page marker is out, forced once there are more
    markers than pages; a page that reaches ``max_page_tokens`` is closed with
    the next ``<PAGE>`` (or EOS on the last page).
    """

    def __init__(self, pages: int, page_id: int, eos_id: int) -> None:
        self.pages, self.page_id, self.eos_id = pages, page_id, eos_id

    def __call__(self, tokens, logits):
        import mlx.core as mx

        seq = [
            int(t) for t in (tokens.tolist() if hasattr(tokens, "tolist") else tokens)
        ]
        markers = seq.count(self.page_id)
        last = max((i for i, t in enumerate(seq) if t == self.page_id), default=-1)
        page_len = len(seq) - last - 1 if last >= 0 else len(seq)
        vocab = int(logits.shape[-1])
        if markers > self.pages:
            force = self.eos_id
        elif page_len >= MAX_PAGE_TOKENS:
            force = self.page_id if markers < self.pages else self.eos_id
        else:
            force = None
        if force is not None:
            mask = mx.arange(vocab) == force
            return mx.where(mask, logits, -mx.inf)
        if markers < self.pages:
            mask = mx.arange(vocab) == self.eos_id
            return mx.where(mask, -mx.inf, logits)
        return logits


class Generated:
    """Feeds ``processor`` the generated stream (plus ``keep`` prompt tokens).

    This mlx-vlm passes the whole sequence, prompt included, to
    ``logits_processors``; the native guards see only the last prompt token
    (n-gram) or nothing of the prompt (page count). The first call fixes the
    prompt length.
    """

    def __init__(self, processor, keep: int) -> None:
        self.processor, self.keep, self.prompt = processor, keep, None

    def __call__(self, tokens, logits):
        if self.prompt is None:
            self.prompt = len(tokens)
        return self.processor(tokens[self.prompt - self.keep :], logits)


def run_ocr(corpus: Path, items: list[dict], repo: str) -> dict[str, dict]:
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(repo)
    patch_mlx_vlm_to_baidu(model)
    tokenizer = processor.tokenizer
    page_id = int(model.language_model.dash_params.page_token_id)
    eos_id = int(tokenizer.eos_token_id)
    out = {}
    for item in items:
        n = len(item["pages"])
        prompt = apply_chat_template(processor, model.config, PROMPT, num_images=n)
        window = UNLIMITED_OCR_NGRAM_WINDOW if n == 1 else MULTI_PAGE_NGRAM_WINDOW
        guards = [
            Generated(
                SlidingWindowNoRepeatNgramProcessor(UNLIMITED_OCR_NGRAM_SIZE, window), 1
            ),
            Generated(PageCountGuard(n, page_id, eos_id), 0),
        ]
        tokens, last = [], None
        for last in stream_generate(
            model,
            processor,
            prompt,
            image=[str(corpus / page) for page in item["pages"]],
            max_tokens=UNLIMITED_OCR_MAX_TOKENS * n,
            temperature=0.0,
            logits_processors=guards,
        ):
            tokens.append(int(last.token))
        if tokens and tokens[-1] == eos_id:
            tokens.pop()
        raw = tokenizer.decode(tokens, skip_special_tokens=False).strip()
        out[item["id"]] = {
            "raw": raw,
            "finish": last.finish_reason if last else "error",
        }
        print(
            f"{item['id']}: {n} pages, {len(tokens)} tokens, {out[item['id']]['finish']}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", default="target/corpus")
    ap.add_argument("--ocr-repo", default="mlx-community/Unlimited-OCR-2-4bit")
    args = ap.parse_args()
    corpus = Path(args.corpus).resolve()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    path = corpus / "baseline.json"
    baseline = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    baseline["ocr_unlimited_ocr2"] = run_ocr(corpus, manifest["ocr"], args.ocr_repo)
    path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
