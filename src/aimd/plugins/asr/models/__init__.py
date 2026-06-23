"""ASR model adapters."""

from .mlx import transcribe_audio_mlx
from .transformers import transcribe_audio_transformers

__all__ = ["transcribe_audio_mlx", "transcribe_audio_transformers"]
