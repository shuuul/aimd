"""Base contracts and shared helpers for Transformers OCR model adapters."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aimd.core.errors import EngineUnavailableError

_cached_model = None
_cached_processor = None
_cached_model_name: str | None = None


class TransformersOCRModel(Protocol):
    """Model-specific adapter used by the Linux/CUDA Transformers OCR engine."""

    model_id: str

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> str:
        """Recognize one image."""
        ...

    def recognize_images(
        self,
        image_paths: list[Path],
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> list[str]:
        """Recognize multiple images, preserving input order."""
        ...


def clear_model_cache() -> None:
    """Clear the in-process Transformers OCR model cache."""
    global _cached_model, _cached_processor, _cached_model_name  # noqa: PLW0603
    _cached_model = None
    _cached_processor = None
    _cached_model_name = None


def get_cached_model_and_processor(
    model_name: str,
    loader: Callable[[str], tuple[object, object]],
) -> tuple[object, object]:
    """Load or return a cached OCR-capable Transformers model and processor."""
    global _cached_model, _cached_processor, _cached_model_name  # noqa: PLW0603
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model, _cached_processor

    _ensure_cuda_runtime()
    try:
        _cached_model, _cached_processor = loader(model_name)
    except Exception:
        clear_model_cache()
        raise

    _cached_model_name = model_name
    return _cached_model, _cached_processor


def get_cuda_dtype():
    """Return the preferred CUDA dtype for OCR inference."""
    try:
        import torch
    except ImportError as exc:
        raise EngineUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    return (
        torch.bfloat16
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        else torch.float16
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
        raise EngineUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    if not torch.cuda.is_available():
        raise EngineUnavailableError(
            "CUDA is not available. The Transformers OCR backend requires a "
            "CUDA-capable GPU for practical inference."
        )
