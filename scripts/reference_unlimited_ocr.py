"""Frozen copy of the Unlimited-OCR reference helpers used by the scripts.

Copied verbatim (no behavioural changes) from
``src/aimd/plugins/ocr/models/mlx.py`` (constants, ``SlidingWindowNoRepeatNgramProcessor``,
``_as_int_list``) and ``src/aimd/plugins/ocr/models/unlimited.py``
(``_DETECTION_LINE``, ``normalize_unlimited_ocr_markdown``) at commit 3a6cc6d,
so the scripts keep working after the Python package is removed.
"""

from __future__ import annotations

from importlib import import_module
import math
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


# ---------------------------------------------------------------------------
# mlx-vlm -> Baidu reference patches (added after the frozen copy above).
# ---------------------------------------------------------------------------


def patch_mlx_vlm_to_baidu(model: Any) -> None:
    """Make a loaded mlx-vlm Unlimited-OCR (v1/v2) follow Baidu's native code.

    mlx-vlm (0.7.x, deepseekocr vision + unlimited_ocr R-SWA) diverges from
    ``deepencoder.py`` / ``modeling_deepseekv2.py`` in five places; the
    golden exporters patch them so their references match the native model:

    1. CLIP MLP: ``quick_gelu`` (x * sigmoid(1.702 x)), not exact GELU.
    2. CLIP layer LayerNorm eps 1e-5 (``layernorm_epsilon``), not 1e-6.
    3. CLIP pos-embed resample over the two spatial axes (bicubic,
       antialias), not over (dim, src) through a channels-last Upsample.
    4. SAM ``get_rel_pos``: ``F.interpolate(mode="linear")`` half-pixel
       source index, not ``m * scale``.
    5. v1 R-SWA keeps the whole prompt: the reference prefills it in one call,
       so ``prefill_length`` is the prompt length even though mlx-vlm feeds the
       last prompt token as the first decode step.
    """
    import mlx.core as mx

    vision_mod = import_module("mlx_vlm.models.deepseekocr.vision")
    sam_mod = import_module("mlx_vlm.models.deepseekocr.sam")
    kernels = import_module("mlx_vlm.models.kernels")

    def quick_gelu(x):
        return x * mx.sigmoid(1.702 * x)

    for _, module in model.vision_model.named_modules():
        if isinstance(module, vision_mod.EncoderLayer):
            module.mlp.activation_fn = quick_gelu
            module.layer_norm1.eps = 1e-5
            module.layer_norm2.eps = 1e-5

    def get_abs_pos(self, abs_pos, tgt_size):
        dim = abs_pos.shape[-1]
        abs_pos_new = mx.squeeze(abs_pos, axis=0)
        cls_token, old = abs_pos_new[:1], abs_pos_new[1:]
        src = int(math.sqrt(abs_pos_new.shape[0] - 1))
        tgt = int(math.sqrt(tgt_size))
        if src == tgt:
            return abs_pos
        grid = old.reshape(1, src, src, dim).transpose(0, 3, 1, 2)
        new = kernels.bicubic_interpolate(
            grid.astype(mx.float32), size=(tgt, tgt), antialias=True
        ).astype(abs_pos.dtype)
        new = new.transpose(0, 2, 3, 1).reshape(tgt * tgt, dim)
        return mx.concatenate([cls_token, new], axis=0).reshape(1, tgt * tgt + 1, dim)

    vision_mod.VisionEmbeddings._get_abs_pos = get_abs_pos

    def get_rel_pos(q_size, k_size, rel_pos):
        max_rel = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel:
            dtype = rel_pos.dtype
            src = rel_pos.astype(mx.float32)
            n = src.shape[0]
            real = mx.maximum(
                (mx.arange(max_rel, dtype=mx.float32) + 0.5) * (n / max_rel) - 0.5, 0.0
            )
            lo = real.astype(mx.int32)
            hi = mx.minimum(lo + 1, n - 1)
            w = (real - lo.astype(mx.float32))[:, None]
            rel_pos = ((1 - w) * src[lo] + w * src[hi]).astype(dtype)
        q = mx.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k = mx.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        rel = (q - k) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos[rel.astype(mx.int32)]

    sam_mod.get_rel_pos = get_rel_pos

    try:
        lang = import_module("mlx_vlm.models.unlimited_ocr.language")
    except ImportError:
        return
    ring = getattr(lang, "RingSlidingKVCache", None)
    if ring is None or getattr(ring, "_aimd_patched", False):
        return
    base_update = lang.KVCache.update_and_fetch
    orig_update = ring.update_and_fetch

    def update_and_fetch(self, keys, values):
        if self.prefill_length is None:
            if getattr(self, "_aimd_chunk", False):
                return base_update(self, keys, values)
            # First non-chunk call ends the prompt.
            self.prefill_length = self.offset + int(keys.shape[2])
        return orig_update(self, keys, values)

    ring.update_and_fetch = update_and_fetch
    ring._aimd_patched = True
    lm_cls = type(model.language_model)
    orig_call = lm_cls.__call__

    def call(self, *args, cache=None, **kwargs):
        chunk = "n_to_process" in kwargs
        for c in cache or []:
            if isinstance(c, ring):
                c._aimd_chunk = chunk
        return orig_call(self, *args, cache=cache, **kwargs)

    lm_cls.__call__ = call
