"""ASR exceptions."""


class AsrError(Exception):
    """Base ASR error."""


class UnsupportedInputError(AsrError):
    """Raised when an input cannot be transcribed."""


class UnsupportedEngineError(AsrError):
    """Raised when an unknown engine is requested."""


class EngineUnavailableError(AsrError):
    """Raised when a known engine cannot run in this environment."""


class ProcessingFailedError(AsrError):
    """Raised when transcription fails."""
