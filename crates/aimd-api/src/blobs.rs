//! Sidecar-owned blob storage for desktop workers (ports
//! `aimd.interfaces.api.blobs`). Layout: `<root>/<uuid>/<safe name>` plus
//! `<root>/<uuid>/meta.json` (`{"filename": "<safe name>"}`).

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use crate::util::{parse_uuid, uuid4};

#[derive(Debug)]
pub enum BlobError {
    /// `blob_id must be a UUID` (HTTP 422).
    InvalidId,
    /// Unknown blob (HTTP 404).
    NotFound(String),
    Io(io::Error),
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlobError::InvalidId => f.write_str("blob_id must be a UUID"),
            BlobError::NotFound(id) => f.write_str(id),
            BlobError::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BlobError {}

impl From<io::Error> for BlobError {
    fn from(e: io::Error) -> Self {
        BlobError::Io(e)
    }
}

/// Canonical UUID string or [`BlobError::InvalidId`].
pub fn parse_blob_id(value: &str) -> Result<String, BlobError> {
    parse_uuid(value).ok_or(BlobError::InvalidId)
}

/// `_safe_filename`: basename, `[^A-Za-z0-9._-]+` -> `_`, trimmed of
/// `.`/`_`; `input.bin` when nothing usable remains.
pub fn safe_filename(filename: Option<&str>) -> String {
    let raw = filename
        .unwrap_or("")
        .split('/')
        .rfind(|part| !part.is_empty() && *part != ".")
        .unwrap_or("")
        .trim();
    if raw.is_empty() || raw == "." || raw == ".." {
        return "input.bin".into();
    }
    let mut cleaned = String::with_capacity(raw.len());
    let mut in_run = false;
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
            cleaned.push(c);
            in_run = false;
        } else if !in_run {
            cleaned.push('_');
            in_run = true;
        }
    }
    let cleaned = cleaned.trim_matches(|c| c == '.' || c == '_');
    if cleaned.is_empty() {
        "input.bin".into()
    } else {
        cleaned.to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredBlob {
    pub blob_id: String,
    pub path: PathBuf,
    pub filename: String,
    pub size: u64,
}

/// Store uploaded bytes under a sidecar-owned directory.
#[derive(Debug, Clone)]
pub struct BlobStore {
    pub root: PathBuf,
}

impl BlobStore {
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// `AIMD_BLOB_DIR`, else a fresh private `aimd-blobs-*` temp dir.
    pub fn from_env_value(configured: Option<&str>) -> io::Result<Self> {
        if let Some(dir) = configured.filter(|d| !d.is_empty()) {
            return Self::new(dir);
        }
        use std::os::unix::fs::DirBuilderExt;
        let base = std::env::temp_dir();
        loop {
            let candidate = base.join(format!("aimd-blobs-{}", &uuid4()[..8]));
            match std::fs::DirBuilder::new().mode(0o700).create(&candidate) {
                Ok(()) => return Ok(Self { root: candidate }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Path for staging an upload before its blob id is known.
    pub fn staging_path(&self) -> PathBuf {
        self.root.join(format!(".upload-{}", uuid4()))
    }

    fn target(
        &self,
        filename: Option<&str>,
        blob_id: Option<&str>,
    ) -> Result<(String, String, PathBuf), BlobError> {
        let stored_id = match blob_id.filter(|b| !b.is_empty()) {
            Some(b) => parse_blob_id(b)?,
            None => uuid4(),
        };
        let safe_name = safe_filename(filename);
        let blob_dir = self.root.join(&stored_id);
        std::fs::create_dir_all(&blob_dir)?;
        Ok((stored_id, safe_name, blob_dir))
    }

    fn finish(
        blob_dir: &Path,
        stored_id: String,
        safe_name: String,
        size: u64,
    ) -> Result<StoredBlob, BlobError> {
        let meta = serde_json::json!({ "filename": safe_name }).to_string();
        std::fs::write(blob_dir.join("meta.json"), meta)?;
        Ok(StoredBlob {
            blob_id: stored_id,
            path: blob_dir.join(&safe_name),
            filename: safe_name,
            size,
        })
    }

    /// Store `data` (`BlobStore.put`).
    pub fn put(
        &self,
        data: &[u8],
        filename: Option<&str>,
        blob_id: Option<&str>,
    ) -> Result<StoredBlob, BlobError> {
        let (stored_id, safe_name, blob_dir) = self.target(filename, blob_id)?;
        std::fs::write(blob_dir.join(&safe_name), data)?;
        Self::finish(&blob_dir, stored_id, safe_name, data.len() as u64)
    }

    /// Move an already-written staging file into place.
    pub fn put_staged(
        &self,
        staged: &Path,
        size: u64,
        filename: Option<&str>,
        blob_id: Option<&str>,
    ) -> Result<StoredBlob, BlobError> {
        let (stored_id, safe_name, blob_dir) = self.target(filename, blob_id)?;
        std::fs::rename(staged, blob_dir.join(&safe_name))?;
        Self::finish(&blob_dir, stored_id, safe_name, size)
    }

    pub fn resolve(&self, blob_id: &str) -> Result<StoredBlob, BlobError> {
        let stored_id = parse_blob_id(blob_id)?;
        let blob_dir = self.root.join(&stored_id);
        if !blob_dir.is_dir() {
            return Err(BlobError::NotFound(stored_id));
        }
        let filename = filename_for(&blob_dir);
        let path = blob_dir.join(&filename);
        let Ok(meta) = std::fs::metadata(&path).map(|m| (m.is_file(), m.len())) else {
            return Err(BlobError::NotFound(stored_id));
        };
        if !meta.0 {
            return Err(BlobError::NotFound(stored_id));
        }
        Ok(StoredBlob {
            blob_id: stored_id,
            path,
            filename,
            size: meta.1,
        })
    }
}

fn filename_for(blob_dir: &Path) -> String {
    let meta_path = blob_dir.join("meta.json");
    if let Ok(text) = std::fs::read_to_string(&meta_path)
        && let Ok(serde_json::Value::Object(payload)) = serde_json::from_str(&text)
        && let Some(serde_json::Value::String(name)) = payload.get("filename")
        && !name.is_empty()
        && blob_dir.join(name).is_file()
    {
        return name.clone();
    }
    let mut children: Vec<String> = std::fs::read_dir(blob_dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n != "meta.json")
                .collect()
        })
        .unwrap_or_default();
    children.sort();
    children
        .into_iter()
        .next()
        .unwrap_or_else(|| "input.bin".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_filename_matches_python() {
        assert_eq!(safe_filename(Some("notes.md")), "notes.md");
        assert_eq!(safe_filename(Some("../a b?.md")), "a_b_.md");
        assert_eq!(safe_filename(Some("dir/sub/")), "sub");
        assert_eq!(safe_filename(Some("..")), "input.bin");
        assert_eq!(safe_filename(Some("")), "input.bin");
        assert_eq!(safe_filename(None), "input.bin");
        assert_eq!(safe_filename(Some("._x_.")), "x");
        assert_eq!(safe_filename(Some("讲座 录音.m4a")), "m4a");
        assert_eq!(safe_filename(Some("talk 讲座.m4a")), "talk_.m4a");
    }
}
