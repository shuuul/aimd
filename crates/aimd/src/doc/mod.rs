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

use crate::error::AimdError;
use crate::hooks::ProcessHooks;

use crate::doc::util::{extract_title, io_err, read_text_ignore};
use crate::sys::System;
use crate::util::tempdir::TempDir;

/// Converted document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocOutput {
    pub markdown: String,
    pub title: Option<String>,
}

/// Every extension routed here (lowercase, with the leading dot).
#[cfg(test)]
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
#[cfg(test)]
pub fn is_document_extension(ext: &str) -> bool {
    DOCUMENT_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Convert a local document to Markdown.
///
/// `output_dir`: where assets and the combined `<stem>.md` are written (EPUB
/// defaults to `<parent>/<stem>`; other formats write nothing when `None`).
/// `temp_dir` redirects scratch space.
pub(crate) fn convert_document(
    system: &(impl System + ?Sized),
    path: &Path,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
    hooks: &dyn ProcessHooks,
) -> crate::Result<DocOutput> {
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
                let out = epub::process_epub(system, path, output_dir, temp_dir, hooks)?;
                return Ok(DocOutput {
                    markdown: out.markdown,
                    title: Some(out.title),
                });
            }
            process_pandoc_document(system, path, format, &stem, output_dir, temp_dir)
        }
    }
}

fn passthrough(path: &Path, stem: &str, output_dir: Option<&Path>) -> crate::Result<DocOutput> {
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
    system: &(impl System + ?Sized),
    path: &Path,
    format: &str,
    stem: &str,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
) -> crate::Result<DocOutput> {
    let (output_file, media, _guard);
    match output_dir {
        None => {
            let tmp = TempDir::new(temp_dir, "aimd-doc-")?;
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
    pandoc::convert(system, path, format, &output_file, media.as_deref())?;
    let markdown = read_text_ignore(&output_file).map_err(io_err)?;
    Ok(DocOutput {
        title: Some(extract_title(&markdown, stem)),
        markdown,
    })
}

#[cfg(test)]
mod tests;
