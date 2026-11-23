//! Document I/O and title helpers.

use std::path::{Path, PathBuf};

use crate::error::AimdError;
use crate::util::pytext::splitlines;

pub(crate) fn io_err(e: std::io::Error) -> AimdError {
    AimdError::failed(format!("I/O error: {e}"))
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
