"""ASR exceptions."""


class AsrError(Exception):
    """Base ASR error."""


class UnsupportedInputError(AsrError):
    """Raised when an input cannot be transcribed."""


class BackendUnavailableError(AsrError):
    """Raised when no transcription backend can run in this environment."""


class ProcessingFailedError(AsrError):
    """Raised when transcription fails."""
