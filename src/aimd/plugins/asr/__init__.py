"""ASR package for local audio/video transcription."""

from ._plugin import __plugin_interface_version__, register_converters, transcribe_file

__all__ = [
    "__plugin_interface_version__",
    "register_converters",
    "transcribe_file",
]
