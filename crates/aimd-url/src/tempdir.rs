//! Self-removing scratch directory (`tempfile.TemporaryDirectory(dir=...)`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use aimd_core::{AimdError, Result};

#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// Create a fresh directory under `base` (default: the system temp dir).
    pub fn new(base: Option<&Path>, prefix: &str) -> Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let base = base.map_or_else(std::env::temp_dir, Path::to_path_buf);
        std::fs::create_dir_all(&base).map_err(|err| {
            AimdError::failed(format!("Cannot create temp dir {}: {err}", base.display()))
        })?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        for _ in 0..64 {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("{prefix}-{}-{nanos:x}-{n}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(AimdError::failed(format!(
                        "Cannot create temp dir under {}: {err}",
                        base.display()
                    )));
                }
            }
        }
        Err(AimdError::failed(format!(
            "Cannot create a unique temp dir under {}",
            base.display()
        )))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
