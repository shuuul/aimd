"""Simple platform backend selection for ASR (no heavy preflight dicts)."""

import platform

from aimd.core.errors import BackendUnavailableError

from .platform_utils import is_apple_silicon


def _module_available(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_transcription_backend() -> str:
    """Select transcription backend or raise clear error. Let actual libs fail for details."""
    system = platform.system()
    if system == "Darwin":
        if is_apple_silicon() and _module_available("mlx_audio"):
            return "mlx"
        raise BackendUnavailableError(
            "mlx-audio backend requires macOS on Apple Silicon with mlx-audio installed."
        )
    else:
        has_torch = _module_available("torch")
        has_torchaudio = _module_available("torchaudio")
        has_transformers = _module_available("transformers")
        if has_torch and has_torchaudio and has_transformers and _cuda_available():
            return "transformers"
        raise BackendUnavailableError(
            "Transformers ASR backend requires torch, torchaudio, transformers and CUDA."
        )
