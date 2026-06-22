"""Environment capability detection for transcription backends."""

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import platform

from .errors import BackendUnavailableError
from .platform_utils import is_apple_silicon


@dataclass
class BackendCapability:
    name: str
    available: bool
    reason: str | None = None
    fix_hint: str | None = None
    deprecated: bool = False


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


@lru_cache(maxsize=1)
def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def get_backend_capabilities() -> dict[str, BackendCapability]:
    """Return availability details for each supported transcription backend."""
    is_macos = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"
    apple_silicon = is_apple_silicon() if is_macos else False
    has_mlx_audio = _module_available("mlx_audio")
    has_torch = _module_available("torch")
    has_torchaudio = _module_available("torchaudio")
    has_transformers = _module_available("transformers")
    torch_cuda_available = _torch_cuda_available() if has_torch else False

    capabilities: dict[str, BackendCapability] = {}

    mlx_available = is_macos and apple_silicon and has_mlx_audio
    mlx_reason = None
    if not is_macos:
        mlx_reason = "mlx is only supported on macOS."
    elif not apple_silicon:
        mlx_reason = "mlx requires Apple Silicon (M1/M2/M3/M4)."
    elif not has_mlx_audio:
        mlx_reason = "mlx_audio module is not installed."

    capabilities["mlx"] = BackendCapability(
        name="mlx",
        available=mlx_available,
        reason=mlx_reason,
        fix_hint=None if mlx_available else "Install dependency: mlx-audio",
    )

    qwen_available = (
        is_linux
        and has_torch
        and has_torchaudio
        and has_transformers
        and torch_cuda_available
    )
    qwen_reason = None
    if not is_linux:
        qwen_reason = "qwen backend is only supported on Linux."
    elif not has_torch:
        qwen_reason = "torch module is not installed."
    elif not has_torchaudio:
        qwen_reason = "torchaudio module is not installed."
    elif not has_transformers:
        qwen_reason = "transformers module is not installed."
    elif not torch_cuda_available:
        qwen_reason = "CUDA is not available in the current PyTorch runtime."

    capabilities["qwen"] = BackendCapability(
        name="qwen",
        available=qwen_available,
        reason=qwen_reason,
        fix_hint=(
            None
            if qwen_available
            else "Install dependencies: torch, torchaudio, transformers (Linux + CUDA required)"
        ),
    )

    return capabilities


def select_transcription_backend() -> str:
    """Select the platform transcription backend with fail-fast capability checks."""
    capabilities = get_backend_capabilities()
    if platform.system() == "Darwin":
        priority = ("mlx",)
    else:
        priority = ("qwen",)

    for candidate in priority:
        if capabilities[candidate].available:
            return candidate

    reasons = "; ".join(
        f"{name}: {capabilities[name].reason or 'unavailable'}" for name in priority
    )
    raise BackendUnavailableError(
        f"No available transcription backend for this environment. {reasons}"
    )
