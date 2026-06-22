"""ASR exceptions."""


class AsrError(Exception):
    """Base ASR error."""


class UnsupportedInputError(AsrError):
    """Raised when an input cannot be transcribed."""


class ProcessingFailedError(AsrError):
    """Raised when transcription fails."""
