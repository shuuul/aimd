//! Minimal self-deleting temporary directories and file names.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use aimd_core::{AimdError, Result};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique `{prefix}{pid}-{n}-{nanos}` name.
pub fn unique_name(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{prefix}{}-{n}-{nanos:x}", std::process::id())
}

/// Base directory: `temp_dir` when given, else the system temp dir.
pub fn base_dir(temp_dir: Option<&Path>) -> PathBuf {
    temp_dir.map_or_else(std::env::temp_dir, Path::to_path_buf)
}

/// A directory removed (recursively) on drop.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(temp_dir: Option<&Path>, prefix: &str) -> Result<Self> {
        let base = base_dir(temp_dir);
        std::fs::create_dir_all(&base).map_err(|e| {
            AimdError::failed(format!(
                "Cannot create temp directory {}: {e}",
                base.display()
            ))
        })?;
        let path = base.join(unique_name(prefix));
        std::fs::create_dir(&path).map_err(|e| {
            AimdError::failed(format!(
                "Cannot create temp directory {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self { path })
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

/// A file path removed on drop (when it exists).
#[derive(Debug)]
pub struct TempFile {
    path: PathBuf,
}

impl TempFile {
    /// Reserve `{base}/{prefix}...{suffix}` (the file is not created).
    pub fn reserve(temp_dir: Option<&Path>, prefix: &str, suffix: &str) -> Result<Self> {
        let base = base_dir(temp_dir);
        std::fs::create_dir_all(&base).map_err(|e| {
            AimdError::failed(format!(
                "Cannot create temp directory {}: {e}",
                base.display()
            ))
        })?;
        Ok(Self {
            path: base.join(format!("{}{suffix}", unique_name(prefix))),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
