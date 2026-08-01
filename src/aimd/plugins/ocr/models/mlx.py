"""mlx-vlm OCR model adapter."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.core.precision import SUPPORTED_PRECISIONS, normalize_precision
from aimd.core.version import parse_version_tuple

DEFAULT_OCR_MODEL = "unlimited-ocr"
MLX_VLM_DEFAULT_MODEL_ID = "mlx-community/Unlimited-OCR-4bit"
GLM_OCR_DEFAULT_MODEL_ID = "mlx-community/GLM-OCR-4bit"

# OCR model families supported on the MLX backend, mapped to their
# mlx-community repository name.
_MLX_OCR_FAMILY_REPOS = {
    "unlimited-ocr": "Unlimited-OCR",
    "glm-ocr": "GLM-OCR",
}


MLX_VLM_DEFAULT_PROMPT = "Text Recognition:"
# mlx-vlm added Unlimited-OCR in PR #1427, first released in v0.6.4.
MLX_VLM_UNLIMITED_OCR_MIN_VERSION = (0, 6, 4)
UNLIMITED_OCR_SINGLE_PROMPT = "document parsing."
# Practical per-page cap. Upstream docs show 32768, but without an n-gram guard
# Unlimited-OCR can loop ("R. R. R. ...") until max_tokens and burn minutes/page.
UNLIMITED_OCR_MAX_TOKENS = 8192
# Baidu single-image example values (SlidingWindowNoRepeatNgramProcessor).
UNLIMITED_OCR_NGRAM_SIZE = 35
UNLIMITED_OCR_NGRAM_WINDOW = 128

_cached_mlx_vlm_model = None
_cached_mlx_vlm_processor = None
_cached_mlx_vlm_model_id: str | None = None


def resolve_mlx_vlm_model(model: str | None, precision: str | None = None) -> str:
    """Resolve an aimd OCR model alias plus precision to an mlx-vlm model ID.

    Accepts the kebab-case aliases ``unlimited-ocr``/``glm-ocr`` (and legacy
    underscore variants, optionally with an embedded precision suffix) or an
    explicit ``mlx-community/{Unlimited-OCR,GLM-OCR}-{precision}`` ID. When no
    precision is given, 4bit is used. A precision that conflicts with an
    explicit ID suffix raises ProcessingFailedError.
    """
    normalized_precision = normalize_precision(precision)
    requested = (model or DEFAULT_OCR_MODEL).strip().lower()
    requested = requested.replace(" ", "_").replace("_", "-")

    candidate = requested.removeprefix("mlx-community/")
    embedded_precision: str | None = None
    for known_precision in SUPPORTED_PRECISIONS:
        if candidate.endswith(f"-{known_precision}"):
            embedded_precision = known_precision
            candidate = candidate[: -len(known_precision) - 1]
            break

    repo = _MLX_OCR_FAMILY_REPOS.get(candidate)
    if repo is None:
        raise ProcessingFailedError(
            f"Unsupported MLX VLM OCR model {model!r}. Supported models: "
            "unlimited-ocr (default), glm-ocr, or an explicit "
            "mlx-community/Unlimited-OCR-{4bit,6bit,8bit,bf16} or "
            "mlx-community/GLM-OCR-{4bit,6bit,8bit,bf16} Hugging Face model ID."
        )

    if (
        normalized_precision is not None
        and embedded_precision is not None
        and normalized_precision != embedded_precision
    ):
        raise ProcessingFailedError(
            f"Precision {normalized_precision!r} conflicts with explicit MLX OCR "
            f"model {model!r}. Drop the precision argument or use the alias form."
        )

    final_precision = normalized_precision or embedded_precision or "4bit"
    return f"mlx-community/{repo}-{final_precision}"


def _is_unlimited_ocr_model(model_id: str) -> bool:
    """Return True when the resolved model is an Unlimited-OCR checkpoint."""
    normalized = model_id.strip().lower()
    return normalized.endswith("/unlimited-ocr") or "/unlimited-ocr-" in normalized


class MLXVLMOCRModel:
    """Run OCR through mlx-vlm on image paths."""

    def __init__(
        self, model_id: str | None = None, precision: str | None = None
    ) -> None:
        self.model_id = resolve_mlx_vlm_model(model_id, precision)

    def recognize_image(self, image_path: Path) -> str:
        model, processor = _load_mlx_vlm_model(self.model_id)
        if _is_unlimited_ocr_model(self.model_id):
            return _generate_unlimited_ocr_text(model, processor, image_path)
        return _generate_mlx_vlm_text(model, processor, image_path)

    def recognize_images(self, image_paths: list[Path]) -> list[str]:
        model, processor = _load_mlx_vlm_model(self.model_id)
        if _is_unlimited_ocr_model(self.model_id):
            # Page-by-page single-image gundam mode. One-shot multipage generation
            # drops <PAGE> segments on long PDFs (observed 30/41), so do not batch.
            return [
                _generate_unlimited_ocr_text(model, processor, image_path)
                for image_path in image_paths
            ]
        return [
            _generate_mlx_vlm_text(model, processor, image_path)
            for image_path in image_paths
        ]


def _load_mlx_vlm_modules():
    try:
        return import_module("mlx_vlm"), import_module("mlx_vlm.prompt_utils")
    except ImportError as exc:
        raise BackendUnavailableError(
            "MLX VLM OCR requires mlx-vlm. Install OCR dependencies with `uv sync` "
            "on macOS/Python 3.12+, then retry."
        ) from exc


def _require_mlx_vlm_unlimited_support() -> None:
    """Fail fast when installed mlx-vlm cannot run Unlimited-OCR."""
    mlx_vlm, _prompt_utils = _load_mlx_vlm_modules()
    version = getattr(mlx_vlm, "__version__", "0")
    if parse_version_tuple(version) < MLX_VLM_UNLIMITED_OCR_MIN_VERSION:
        min_version = ".".join(str(part) for part in MLX_VLM_UNLIMITED_OCR_MIN_VERSION)
        raise BackendUnavailableError(
            f"Unlimited-OCR on macOS requires mlx-vlm>={min_version} "
            f"(installed {version}). Upgrade with `uv sync`, then retry."
        )
    try:
        import_module("mlx_vlm.models.unlimited_ocr")
    except ImportError as exc:
        min_version = ".".join(str(part) for part in MLX_VLM_UNLIMITED_OCR_MIN_VERSION)
        raise BackendUnavailableError(
            "Installed mlx-vlm build lacks Unlimited-OCR support "
            f"(requires mlx-vlm>={min_version}). Upgrade with `uv sync`, then retry."
        ) from exc


def _load_mlx_vlm_model(model_id: str) -> tuple[object, object]:
    """Load or return cached mlx-vlm model state."""
    global _cached_mlx_vlm_model, _cached_mlx_vlm_processor, _cached_mlx_vlm_model_id  # noqa: PLW0603
    if _cached_mlx_vlm_model is not None and _cached_mlx_vlm_model_id == model_id:
        return _cached_mlx_vlm_model, _cached_mlx_vlm_processor

    if _is_unlimited_ocr_model(model_id):
        _require_mlx_vlm_unlimited_support()

    mlx_vlm, _prompt_utils = _load_mlx_vlm_modules()
    try:
        _cached_mlx_vlm_model, _cached_mlx_vlm_processor = mlx_vlm.load(model_id)
    except Exception as exc:  # noqa: BLE001 - upstream model load errors vary
        _cached_mlx_vlm_model = None
        _cached_mlx_vlm_processor = None
        _cached_mlx_vlm_model_id = None
        raise BackendUnavailableError(
            f"Unable to load MLX VLM OCR model {model_id!r}: {exc}"
        ) from exc
    _cached_mlx_vlm_model_id = model_id
    return _cached_mlx_vlm_model, _cached_mlx_vlm_processor


def _generate_mlx_vlm_text(
    model: object,
    processor: object,
    image_path: Path,
) -> str:
    """Generate OCR text for one image using an already-loaded mlx-vlm model."""
    mlx_vlm, prompt_utils = _load_mlx_vlm_modules()
    config = getattr(model, "config", None)
    formatted_prompt = prompt_utils.apply_chat_template(
        processor,
        config,
        MLX_VLM_DEFAULT_PROMPT,
        num_images=1,
    )
    result = mlx_vlm.generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path.as_posix()],
        max_tokens=4096,
    )
    text = str(getattr(result, "text", result)).strip()
    if not text:
        raise ProcessingFailedError("MLX VLM OCR produced empty content")
    return text


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


def _generate_unlimited_ocr_text(
    model: object,
    processor: object,
    image_path: Path,
) -> str:
    """Run Unlimited-OCR via mlx-vlm in upstream single-image gundam mode."""
    mlx_vlm, prompt_utils = _load_mlx_vlm_modules()
    config = getattr(model, "config", None)
    formatted_prompt = prompt_utils.apply_chat_template(
        processor,
        config,
        UNLIMITED_OCR_SINGLE_PROMPT,
        num_images=1,
    )
    # Upstream gundam mode: 1024 global view + 640 local crops.
    # Enable Baidu's sliding-window no-repeat n-gram guard (examples use 35/128).
    ngram_processor = SlidingWindowNoRepeatNgramProcessor(
        UNLIMITED_OCR_NGRAM_SIZE,
        UNLIMITED_OCR_NGRAM_WINDOW,
    )
    try:
        result = mlx_vlm.generate(
            model,
            processor,
            formatted_prompt,
            image=[image_path.as_posix()],
            max_tokens=UNLIMITED_OCR_MAX_TOKENS,
            temperature=0.0,
            cropping=True,
            image_size=640,
            base_size=1024,
            logits_processors=[ngram_processor],
        )
    except Exception as exc:  # noqa: BLE001 - upstream model errors vary
        raise ProcessingFailedError(f"Unlimited-OCR inference failed: {exc}") from exc

    text = str(getattr(result, "text", result)).strip()
    if not text:
        raise ProcessingFailedError("Unlimited-OCR produced empty content")
    return text
