"""Domain-level exceptions used across CLI, service, and API layers."""


class AimdError(Exception):
    """Base exception for predictable aimd errors."""

    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedInputError(AimdError):
    """Raised when input source is unsupported."""

    status_code = 400


class BackendUnavailableError(AimdError):
    """Raised when no backend can run on the current environment."""

    status_code = 422


class InputNotFoundError(AimdError):
    """Raised when input path does not exist."""

    status_code = 404


class ProcessingFailedError(AimdError):
    """Raised when processing fails in a known way."""

    status_code = 500
