"""ASR model adapters."""

from .mlx import transcribe_audio_mlx
from .qwen import transcribe_audio_qwen

__all__ = ["transcribe_audio_mlx", "transcribe_audio_qwen"]
