"""ASR constants."""

from aimd.core.errors import ProcessingFailedError
from aimd.core.precision import normalize_precision

AUDIO_FILE_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
    ".wma",
    ".webm",
    ".mp4a",
}

VIDEO_FILE_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".ts",
    ".m4v",
}

AUDIO_EXTENSIONS = AUDIO_FILE_EXTENSIONS | VIDEO_FILE_EXTENSIONS

MLX_AUDIO_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-4bit"
MLX_AUDIO_MODELS = {
    "mlx-community/Qwen3-ASR-1.7B-4bit": "Qwen3-ASR 1.7B (4-bit quantized, default)",
    "mlx-community/Qwen3-ASR-1.7B-6bit": "Qwen3-ASR 1.7B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-8bit": "Qwen3-ASR 1.7B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-1.7B-bf16": "Qwen3-ASR 1.7B (bf16)",
    "mlx-community/Qwen3-ASR-0.6B-4bit": "Qwen3-ASR 0.6B (4-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-6bit": "Qwen3-ASR 0.6B (6-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-8bit": "Qwen3-ASR 0.6B (8-bit quantized)",
    "mlx-community/Qwen3-ASR-0.6B-bf16": "Qwen3-ASR 0.6B (bf16)",
}

# Native Transformers Qwen3-ASR checkpoints (transformers>=5.14.1).
TRANSFORMERS_ASR_DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
TRANSFORMERS_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B-hf": "Qwen3-ASR 1.7B native HF (default)",
    "Qwen/Qwen3-ASR-0.6B-hf": "Qwen3-ASR 0.6B native HF",
    # Legacy non-hf IDs remain accepted and resolve to the native -hf checkpoints.
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR 1.7B (legacy alias → 1.7B-hf)",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR 0.6B (legacy alias → 0.6B-hf)",
}

# Backwards-compatible names retained for callers importing the older Qwen constants.
QWEN_ASR_MODELS = TRANSFORMERS_ASR_MODELS
QWEN_ASR_DEFAULT_MODEL = TRANSFORMERS_ASR_DEFAULT_MODEL

_LEGACY_TRANSFORMERS_ASR_MODEL_MAP = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen/Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B-hf",
}

# User-facing kebab-case aliases (plus legacy underscore variants) mapped to the
# Qwen3-ASR model size token used by both MLX and Transformers checkpoints.
QWEN3_ASR_MODEL_ALIASES = {
    "qwen3-asr-1.7b": "1.7B",
    "qwen3-asr-0.6b": "0.6B",
    # Legacy aliases retained for backwards compatibility.
    "qwen3_asr_1_7b": "1.7B",
    "qwen3_asr_0_6b": "0.6B",
}


def resolve_transformers_asr_model(model: str | None) -> str:
    """Resolve a Transformers ASR model ID, mapping legacy names to native -hf."""
    if not model:
        return TRANSFORMERS_ASR_DEFAULT_MODEL
    requested = model.strip()
    if requested in _LEGACY_TRANSFORMERS_ASR_MODEL_MAP:
        return _LEGACY_TRANSFORMERS_ASR_MODEL_MAP[requested]
    alias_size = QWEN3_ASR_MODEL_ALIASES.get(requested.lower())
    if alias_size is not None:
        return f"Qwen/Qwen3-ASR-{alias_size}-hf"
    return requested


def resolve_mlx_asr_model(model: str | None, precision: str | None = None) -> str:
    """Resolve an MLX ASR alias plus precision to an mlx-community model ID.

    Accepts the kebab-case aliases ``qwen3-asr-1.7b``/``qwen3-asr-0.6b`` (and
    legacy underscore variants) combined with a precision, or an explicit
    ``mlx-community/Qwen3-ASR-{size}-{precision}`` ID. When no precision is
    given, 4bit is used. A precision that conflicts with an explicit ID suffix
    raises ProcessingFailedError.
    """
    normalized_precision = normalize_precision(precision)
    requested = (model or "").strip()

    if not requested:
        size = "1.7B"
    elif requested in MLX_AUDIO_MODELS:
        embedded_precision = requested.rsplit("-", 1)[-1]
        if (
            normalized_precision is not None
            and normalized_precision != embedded_precision
        ):
            raise ProcessingFailedError(
                f"Precision {normalized_precision!r} conflicts with explicit MLX ASR "
                f"model ID {requested!r}. Drop --precision or use the alias form."
            )
        return requested
    else:
        size = QWEN3_ASR_MODEL_ALIASES.get(requested.lower())
        if size is None:
            supported_aliases = "qwen3-asr-1.7b, qwen3-asr-0.6b"
            supported_models = ", ".join(MLX_AUDIO_MODELS)
            raise ProcessingFailedError(
                f"Unsupported MLX ASR model {requested!r}. "
                f"Supported aliases: {supported_aliases}. "
                f"Supported model IDs: {supported_models}"
            )

    final_precision = normalized_precision or "4bit"
    return f"mlx-community/Qwen3-ASR-{size}-{final_precision}"
