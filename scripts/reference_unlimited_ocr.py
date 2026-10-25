"""Frozen copy of the Unlimited-OCR reference helpers used by the scripts.

Copied verbatim (no behavioural changes) from
``src/aimd/plugins/ocr/models/mlx.py`` (constants, ``SlidingWindowNoRepeatNgramProcessor``,
``_as_int_list``) and ``src/aimd/plugins/ocr/models/unlimited.py``
(``_DETECTION_LINE``, ``normalize_unlimited_ocr_markdown``) at commit 3a6cc6d,
so the scripts keep working after the Python package is removed.
"""

from __future__ import annotations

from importlib import import_module
import re
from typing import Any

UNLIMITED_OCR_SINGLE_PROMPT = "document parsing."
# Practical per-page cap. Upstream docs show 32768, but without an n-gram guard
# Unlimited-OCR can loop ("R. R. R. ...") until max_tokens and burn minutes/page.
UNLIMITED_OCR_MAX_TOKENS = 8192
# Baidu single-image example values (SlidingWindowNoRepeatNgramProcessor).
UNLIMITED_OCR_NGRAM_SIZE = 35
UNLIMITED_OCR_NGRAM_WINDOW = 128

# Detection lines normally open with <|det|>, but vLLM-served checkpoints may
# emit the category and box without the opening token (header [x, y]<|/det|>).
_DETECTION_LINE = re.compile(
    r"(?:<\|det\|>)?([^<\s|]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)",
    re.DOTALL,
)


def normalize_unlimited_ocr_markdown(raw: str) -> str:
    """Remove Unlimited-OCR layout protocol while preserving document content."""
    if "<|det|>" not in raw and "<|/det|>" not in raw:
        return raw.strip()

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        match = _DETECTION_LINE.match(line)
        if match:
            category, content = match.group(1).strip(), match.group(2).strip()
            if category == "image":
                continue
            if current is not None:
                blocks.append(current)
            current = [content] if content else []
            continue
        if current is None:
            current = []
        current.append(line)
    if current is not None:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks).strip()


class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window (Baidu Unlimited-OCR).

    Port of ``SlidingWindowNoRepeatNgramProcessor`` from baidu/Unlimited-OCR for
    mlx-vlm's ``logits_processors`` hook: ``processor(tokens, logits) -> logits``.
    ``tokens`` is the generated-token stream (not the full prompt).
    """

    def __init__(
        self,
        ngram_size: int,
        window: int,
        whitelist_token_ids: set[int] | None = None,
    ) -> None:
        if ngram_size < 1:
            raise ValueError("ngram_size must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids or ())

    def __call__(self, tokens: Any, logits: Any) -> Any:
        mx = import_module("mlx.core")
        sequence = _as_int_list(tokens)
        if len(sequence) < self.ngram_size:
            return logits

        search_start = max(0, len(sequence) - self.window)
        search_end = len(sequence) - self.ngram_size + 1
        if search_end <= search_start:
            return logits

        if self.ngram_size > 1:
            current_prefix = tuple(sequence[-(self.ngram_size - 1) :])
        else:
            current_prefix = ()

        banned: set[int] = set()
        for idx in range(search_start, search_end):
            ngram = sequence[idx : idx + self.ngram_size]
            if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                banned.add(int(ngram[-1]))
        banned.difference_update(self.whitelist)
        if not banned:
            return logits

        # logits: [vocab] or [batch, vocab]
        vocab = int(logits.shape[-1])
        banned_ids = [token_id for token_id in banned if 0 <= token_id < vocab]
        if not banned_ids:
            return logits
        idx = mx.array(banned_ids)
        updates = mx.full((len(banned_ids),), float("-inf"), dtype=logits.dtype)
        if getattr(logits, "ndim", 1) == 1:
            return mx.put_along_axis(
                logits[None, :], idx[None, :], updates[None, :], axis=1
            )[0]
        batch = int(logits.shape[0])
        batch_idx = mx.broadcast_to(idx[None, :], (batch, len(banned_ids)))
        batch_updates = mx.broadcast_to(updates[None, :], (batch, len(banned_ids)))
        return mx.put_along_axis(logits, batch_idx, batch_updates, axis=1)


def _as_int_list(tokens: Any) -> list[int]:
    """Normalize mlx/numpy/list token streams to a flat int list."""
    if tokens is None:
        return []
    if hasattr(tokens, "reshape"):
        flat = tokens.reshape(-1)
        if hasattr(flat, "tolist"):
            return [int(x) for x in flat.tolist()]
    if isinstance(tokens, (list, tuple)):
        out: list[int] = []
        for item in tokens:
            if isinstance(item, (list, tuple)):
                out.extend(int(x) for x in item)
            else:
                out.append(int(item))
        return out
    try:
        return [int(tokens)]
    except (TypeError, ValueError):
        return []
