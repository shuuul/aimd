"""Shared model precision parsing helpers."""

from .errors import ProcessingFailedError

SUPPORTED_PRECISIONS = ("4bit", "6bit", "8bit", "bf16")
QUANTIZED_PRECISIONS = ("4bit", "6bit", "8bit")


def normalize_precision(precision: str | None) -> str | None:
    """Normalize a user-supplied precision value or reject it.

    Accepts ``4bit``, ``6bit``, ``8bit``, ``bf16`` and dash/space/case variants
    such as ``4-bit`` or ``BF16``. Returns ``None`` when no precision is given.
    """
    if precision is None:
        return None
    normalized = precision.strip().lower().replace("-", "").replace(" ", "")
    if normalized in SUPPORTED_PRECISIONS:
        return normalized
    raise ProcessingFailedError(
        f"Unsupported precision {precision!r}. "
        f"Supported precisions: {', '.join(SUPPORTED_PRECISIONS)}."
    )


def normalize_transformers_precision(precision: str | None) -> str | None:
    """Normalize precision for Transformers backends, rejecting quantized values."""
    normalized = normalize_precision(precision)
    if normalized in QUANTIZED_PRECISIONS:
        raise ProcessingFailedError(
            f"Transformers backends do not support quantized precision "
            f"{normalized!r}. Use bf16 or omit precision for automatic dtype "
            "selection."
        )
    return normalized
