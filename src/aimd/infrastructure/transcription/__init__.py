"""Transcription infrastructure."""

from ..capabilities.detector import resolve_engine_with_preflight
from .processor import get_text_from_audio

__all__ = ["get_text_from_audio", "resolve_engine_with_preflight"]
