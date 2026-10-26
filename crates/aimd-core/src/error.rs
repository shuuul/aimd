//! Domain errors (ports `aimd.core.errors`).

use std::fmt;

/// Error category; mirrors the Python `AimdError` subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// `UnsupportedInputError`
    UnsupportedInput,
    /// `BackendUnavailableError` (missing pandoc/yt-dlp/ffmpeg/model/device)
    BackendUnavailable,
    /// `InputNotFoundError`
    InputNotFound,
    /// `ProcessingFailedError`
    ProcessingFailed,
    /// `ProcessingCancelledError`
    Cancelled,
}

impl ErrorKind {
    /// Stable snake_case name for JSON error bodies.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::UnsupportedInput => "unsupported_input",
            ErrorKind::BackendUnavailable => "backend_unavailable",
            ErrorKind::InputNotFound => "input_not_found",
            ErrorKind::ProcessingFailed => "processing_failed",
            ErrorKind::Cancelled => "processing_cancelled",
        }
    }
}

/// A predictable, user-facing aimd error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AimdError {
    pub kind: ErrorKind,
    pub message: String,
}

impl AimdError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedInput, message)
    }
    pub fn backend_unavailable(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::BackendUnavailable, message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InputNotFound, message)
    }
    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::ProcessingFailed, message)
    }
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Cancelled, message)
    }
}

impl fmt::Display for AimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AimdError {}

pub type Result<T, E = AimdError> = std::result::Result<T, E>;
