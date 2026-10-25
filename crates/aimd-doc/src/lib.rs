//! Document conversion pipeline (WS-17). Ports `aimd.plugins.doc`.
//!
//! - Pandoc CLI for every Pandoc-readable format (docx/odt/pptx/xlsx/html/...).
//! - EPUB: hand-written zip + OPF spine walk, per-chapter Pandoc, Markdown
//!   cleanup, flat image extraction.
//! - Text-layer PDFs via `pdf-inspector`.
//! - `.txt` / `.md` pass through.

mod clean;
mod epub;
mod pandoc;
mod pdf;
mod util;
mod zip;

use std::path::Path;

use aimd_core::{AimdError, ProcessHooks};

use crate::pandoc::{PandocCli, Runner};
use crate::util::{TempDir, extract_title, io_err, read_text_ignore};

/// Converted document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocOutput {
    pub markdown: String,
    pub title: Option<String>,
}

/// Every extension routed here (lowercase, with the leading dot).
pub const DOCUMENT_EXTENSIONS: &[&str] = &[
    ".adoc",
    ".asciidoc",
    ".bib",
    ".biblatex",
    ".bits",
    ".commonmark",
    ".creole",
    ".csv",
    ".dbk",
    ".docbook",
    ".docx",
    ".dokuwiki",
    ".endnote",
    ".enw",
    ".epub",
    ".fb2",
    ".gfm",
    ".haddock",
    ".htm",
    ".html",
    ".ipynb",
    ".jats",
    ".json",
    ".latex",
    ".ltx",
    ".man",
    ".markdown",
    ".md",
    ".mdoc",
    ".mediawiki",
    ".muse",
    ".native",
    ".odt",
    ".opml",
    ".org",
    ".pdf",
    ".pptx",
    ".ris",
    ".rst",
    ".rtf",
    ".t2t",
    ".tex",
    ".textile",
    ".tsv",
    ".txt",
    ".typ",
    ".typst",
    ".vimwiki",
    ".wiki",
    ".xlsx",
];

/// `ext` is an extension with its leading dot, e.g. `.docx` (case-insensitive).
pub fn is_document_extension(ext: &str) -> bool {
    DOCUMENT_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Convert a local document to Markdown.
///
/// `output_dir`: where assets and the combined `<stem>.md` are written (EPUB
/// defaults to `<parent>/<stem>`; other formats write nothing when `None`).
/// `temp_dir` redirects scratch space.
pub fn convert_document(
    path: &Path,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
    hooks: &dyn ProcessHooks,
) -> aimd_core::Result<DocOutput> {
    convert_with(&PandocCli::default(), path, output_dir, temp_dir, hooks)
}

pub(crate) fn convert_with(
    runner: &dyn Runner,
    path: &Path,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
    hooks: &dyn ProcessHooks,
) -> aimd_core::Result<DocOutput> {
    if !path.exists() {
        return Err(AimdError::not_found(format!(
            "Document file not found: {}",
            path.display()
        )));
    }
    let suffix = path
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let ext = suffix.to_lowercase();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    match ext.as_str() {
        ".pdf" => {
            let (title, markdown) = pdf::process_pdf_text(path)?;
            Ok(DocOutput { markdown, title })
        }
        ".txt" | ".md" => passthrough(path, &stem, output_dir),
        _ => {
            let Some(format) = pandoc::input_format(&ext) else {
                return Err(AimdError::unsupported(format!(
                    "Unsupported Pandoc document format: {suffix}"
                )));
            };
            if format == "epub" {
                let out = epub::process_epub(runner, path, output_dir, temp_dir, hooks)?;
                return Ok(DocOutput {
                    markdown: out.markdown,
                    title: Some(out.title),
                });
            }
            process_pandoc_document(runner, path, format, &stem, output_dir, temp_dir)
        }
    }
}

fn passthrough(path: &Path, stem: &str, output_dir: Option<&Path>) -> aimd_core::Result<DocOutput> {
    let markdown = read_text_ignore(path).map_err(io_err)?;
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir).map_err(io_err)?;
        let target = dir.join(format!("{stem}.md"));
        if target.canonicalize().ok() != path.canonicalize().ok() {
            std::fs::write(&target, &markdown).map_err(io_err)?;
        }
    }
    Ok(DocOutput {
        title: Some(extract_title(&markdown, stem)),
        markdown,
    })
}

fn process_pandoc_document(
    runner: &dyn Runner,
    path: &Path,
    format: &str,
    stem: &str,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
) -> aimd_core::Result<DocOutput> {
    let (output_file, media, _guard);
    match output_dir {
        None => {
            let tmp = TempDir::new_in(temp_dir)?;
            output_file = tmp.path().join(format!("{stem}.md"));
            media = None;
            _guard = Some(tmp);
        }
        Some(dir) => {
            std::fs::create_dir_all(dir).map_err(io_err)?;
            output_file = dir.join(format!("{stem}.md"));
            media = Some(dir.join("media"));
            _guard = None;
        }
    }
    runner.run(path, format, &output_file, media.as_deref())?;
    let markdown = read_text_ignore(&output_file).map_err(io_err)?;
    Ok(DocOutput {
        title: Some(extract_title(&markdown, stem)),
        markdown,
    })
}

#[cfg(test)]
mod tests;

/// Exposed only for the tier-3 parity test (`tests/parity.rs`).
#[doc(hidden)]
pub fn __clean_markdown(text: &str) -> String {
    clean::clean_markdown(text)
}
