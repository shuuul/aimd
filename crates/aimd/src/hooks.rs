//! Cooperative cancellation and progress reporting (ports the Python
//! `cancellation_check` / `progress_reporter` callables).

use crate::error::{AimdError, Result};

/// Callbacks threaded through every pipeline. Pipelines call
/// [`ProcessHooks::check_cancelled`] only at safe boundaries (between
/// segments/pages/chapters), never inside uninterruptible work.
pub trait ProcessHooks: Send + Sync {
    /// `true` once the caller asked to stop.
    fn is_cancelled(&self) -> bool {
        false
    }

    /// Stage names used by the Python code: `extracting`, `downloading`,
    /// `transcribing`, `converting`, `recognizing`. `current`/`total` are
    /// unit counts (segments, pages, chapters) when known.
    fn progress(
        &self,
        _stage: &str,
        _current: Option<u64>,
        _total: Option<u64>,
        _message: Option<&str>,
    ) {
    }

    /// Returns `Err(Cancelled)` when cancellation was requested.
    fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(AimdError::cancelled("Processing cancelled"))
        } else {
            Ok(())
        }
    }
}

/// Hooks that never cancel and drop progress.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoHooks;

impl ProcessHooks for NoHooks {}
