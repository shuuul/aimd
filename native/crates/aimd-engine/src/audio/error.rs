//! Typed errors for the audio front-end.

use std::path::PathBuf;

use thiserror::Error;

/// Errors raised by wav loading and mel spectrogram computation.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to read wav file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("not a RIFF container: bad magic in {path}")]
    NotRiff { path: PathBuf },

    #[error("RIFF container in {path} is not in WAVE form")]
    NotWave { path: PathBuf },

    #[error(
        "wav file {path} truncated in chunk '{chunk}': need {needed} bytes, file has {available}"
    )]
    Truncated {
        path: PathBuf,
        chunk: String,
        needed: u64,
        available: u64,
    },

    #[error("wav file {path} has no 'fmt ' chunk")]
    MissingFmtChunk { path: PathBuf },

    #[error("wav file {path} has no 'data' chunk")]
    MissingDataChunk { path: PathBuf },

    #[error("wav file {path}: 'fmt ' chunk size {size} is invalid")]
    InvalidFmtChunk { path: PathBuf, size: u32 },

    #[error("wav file {path}: unsupported compression format code {code}")]
    UnsupportedFormat { path: PathBuf, code: u16 },

    #[error("wav file {path}: unsupported bits-per-sample {bits} for format code {code}")]
    UnsupportedBitsPerSample { path: PathBuf, bits: u16, code: u16 },

    #[error("wav file {path}: expected 16 kHz, got {actual} Hz")]
    UnsupportedSampleRate { path: PathBuf, actual: u32 },

    #[error("wav file {path}: expected mono, got {actual} channels")]
    UnsupportedChannels { path: PathBuf, actual: u16 },

    #[error(
        "wav file {path}: data chunk size {size} is not a whole number of {block_align}-byte frames"
    )]
    PartialDataFrame {
        path: PathBuf,
        size: u32,
        block_align: u16,
    },

    #[error("mel input too short: {samples} samples, need at least {required}")]
    InputTooShort { samples: usize, required: usize },

    #[error("invalid mel config: {0}")]
    InvalidConfig(String),
}
