//! Small shared helpers: temp dirs, Python-compatible line splitting, titles.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use aimd_core::AimdError;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temp directory removed on drop (`tempfile.TemporaryDirectory` equivalent).
pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn new_in(parent: Option<&Path>) -> aimd_core::Result<Self> {
        let base = parent
            .map(Path::to_path_buf)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&base).map_err(io_err)?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        for _ in 0..16 {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("aimd-doc-{}-{nanos:08x}-{n}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_err(e)),
            }
        }
        Err(AimdError::failed("Could not create a temporary directory"))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn io_err(e: std::io::Error) -> AimdError {
    AimdError::failed(format!("I/O error: {e}"))
}

/// Python `str.splitlines()` (no line endings kept).
pub(crate) fn splitlines(text: &str) -> Vec<&str> {
    let is_sep = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut out = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        if is_sep(c) {
            out.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && matches!(iter.peek(), Some((_, '\n'))) {
                iter.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Python `Path.read_text(encoding="utf-8", errors="ignore")` with universal
/// newlines: invalid bytes are dropped, `\r\n`/`\r` become `\n`.
pub(crate) fn read_text_ignore(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes())
            .chars()
            .filter(|c| *c != '\u{FFFD}')
            .collect(),
    };
    Ok(universal_newlines(&text))
}

pub(crate) fn universal_newlines(text: &str) -> String {
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text.to_string()
    }
}

/// Port of `_extract_title_from_markdown`.
pub(crate) fn extract_title(content: &str, fallback: &str) -> String {
    for line in splitlines(content) {
        let stripped = line.trim();
        if let Some(rest) = stripped.strip_prefix("# ") {
            let t = rest.trim();
            return if t.is_empty() { fallback } else { t }.to_string();
        }
        if !stripped.is_empty()
            && !(stripped.starts_with("![")
                || stripped.starts_with('<')
                || stripped.starts_with(":::"))
        {
            return stripped.chars().take(100).collect();
        }
    }
    fallback.to_string()
}

/// Recursive sorted walk; `f` receives every entry path (dirs and files).
pub(crate) fn walk_sorted(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        out.push(p.clone());
        if p.is_dir() {
            walk_sorted(&p, out);
        }
    }
}
