"""Base contracts and shared helpers for Transformers OCR model adapters."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.core.precision import normalize_transformers_precision


class TransformersOCRModel(Protocol):
    """Common interface for Transformers OCR model adapters."""

    model_id: str

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> str:
        """Recognize text from one image."""

    def recognize_images(
        self,
        image_paths: list[Path],
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> list[str]:
        """Recognize text from ordered images."""


_cached_model = None
_cached_processor = None
_cached_model_name: str | None = None
_cached_precision: str | None = None


def clear_model_cache() -> None:
    """Clear the in-process Transformers OCR model cache."""
    global _cached_model, _cached_processor, _cached_model_name, _cached_precision  # noqa: PLW0603
    _cached_model = None
    _cached_processor = None
    _cached_model_name = None
    _cached_precision = None


def get_cached_model_and_processor(
    model_name: str,
    loader: Callable[[str], tuple[object, object]],
    *,
    precision: str | None = None,
) -> tuple[object, object]:
    """Load or return a cached OCR-capable Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name, _cached_precision  # noqa: PLW0603
    if (
        _cached_model is not None
        and _cached_model_name == model_name
        and _cached_precision == precision
    ):
        return _cached_model, _cached_processor

    _ensure_cuda_runtime()
    try:
        _cached_model, _cached_processor = loader(model_name)
    except Exception:
        clear_model_cache()
        raise

    _cached_model_name = model_name
    _cached_precision = precision
    return _cached_model, _cached_processor


def validate_transformers_precision(precision: str | None) -> str | None:
    """Normalize an OCR Transformers precision, rejecting quantized values early."""
    return normalize_transformers_precision(precision)


def get_cuda_dtype(precision: str | None = None):
    """Return the CUDA dtype for OCR inference, honoring an explicit precision."""
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if precision is None:
        return torch.bfloat16 if bf16_supported else torch.float16
    if precision == "bf16":
        if not bf16_supported:
            raise BackendUnavailableError(
                "precision=bf16 requires a CUDA device with bf16 support. "
                "Omit precision to use automatic dtype selection."
            )
        return torch.bfloat16
    raise ProcessingFailedError(
        f"Transformers OCR backend does not support precision {precision!r}. "
        "Use bf16 or omit precision for automatic dtype selection."
    )


def inputs_to_model_device(inputs, model: object):
    """Move processor tensors to the model device."""
    import torch

    device = next(model.parameters()).device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _ensure_cuda_runtime() -> None:
    try:
        import torch
        import transformers  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    if not torch.cuda.is_available():
        raise BackendUnavailableError(
            "CUDA is not available. The Transformers OCR backend requires a "
            "CUDA-capable GPU for practical inference."
        )
