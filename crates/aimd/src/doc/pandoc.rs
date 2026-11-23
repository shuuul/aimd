//! Pandoc CLI runner and the extension -> Pandoc reader map.

use crate::sys::System;
use std::ffi::OsString;
use std::path::Path;

use crate::error::AimdError;

pub(crate) const MARKDOWN_FORMAT: &str = "markdown_mmd-raw_html";

/// Filename extension -> Pandoc reader. Mirrors Python's
/// `PANDOC_INPUT_FORMAT_BY_EXTENSION`, plus `.pptx`/`.xlsx` (readable since
/// Pandoc 3.8). `.md` is not here: `.txt`/`.md` pass through unchanged.
pub(crate) const PANDOC_INPUT_FORMAT_BY_EXTENSION: &[(&str, &str)] = &[
    (".adoc", "asciidoc"),
    (".asciidoc", "asciidoc"),
    (".bib", "bibtex"),
    (".biblatex", "biblatex"),
    (".bits", "bits"),
    (".commonmark", "commonmark"),
    (".creole", "creole"),
    (".csv", "csv"),
    (".dbk", "docbook"),
    (".docbook", "docbook"),
    (".docx", "docx"),
    (".dokuwiki", "dokuwiki"),
    (".endnote", "endnotexml"),
    (".enw", "endnotexml"),
    (".epub", "epub"),
    (".fb2", "fb2"),
    (".gfm", "gfm"),
    (".haddock", "haddock"),
    (".htm", "html"),
    (".html", "html"),
    (".ipynb", "ipynb"),
    (".jats", "jats"),
    (".json", "json"),
    (".latex", "latex"),
    (".ltx", "latex"),
    (".man", "man"),
    (".markdown", "markdown"),
    (".mdoc", "mdoc"),
    (".mediawiki", "mediawiki"),
    (".muse", "muse"),
    (".native", "native"),
    (".odt", "odt"),
    (".opml", "opml"),
    (".org", "org"),
    (".pptx", "pptx"),
    (".ris", "ris"),
    (".rst", "rst"),
    (".rtf", "rtf"),
    (".t2t", "t2t"),
    (".tex", "latex"),
    (".textile", "textile"),
    (".tsv", "tsv"),
    (".typ", "typst"),
    (".typst", "typst"),
    (".vimwiki", "vimwiki"),
    (".wiki", "mediawiki"),
    (".xlsx", "xlsx"),
];

pub(crate) fn input_format(ext_lower: &str) -> Option<&'static str> {
    PANDOC_INPUT_FORMAT_BY_EXTENSION
        .iter()
        .find(|(e, _)| *e == ext_lower)
        .map(|(_, f)| *f)
}

pub(crate) fn convert(
    system: &(impl System + ?Sized),
    input: &Path,
    input_format: &str,
    output: &Path,
    extract_media: Option<&Path>,
) -> crate::Result<()> {
    let mut args: Vec<OsString> = vec![
        input.into(),
        "-f".into(),
        input_format.into(),
        "-t".into(),
        MARKDOWN_FORMAT.into(),
        "--wrap=none".into(),
        "-o".into(),
        output.into(),
    ];
    if let Some(dir) = extract_media {
        args.extend([OsString::from("--extract-media"), dir.into()]);
    }
    let result = match system.run(Path::new("pandoc"), &args, None) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(AimdError::backend_unavailable(
                "Pandoc CLI is required for this document format",
            ));
        }
        Err(e) => {
            return Err(AimdError::failed(format!(
                "Pandoc conversion failed for {}: {e}",
                file_name(input)
            )));
        }
    };
    if !result.success() {
        return Err(AimdError::failed(format!(
            "Pandoc conversion failed for {}: {}",
            file_name(input),
            String::from_utf8_lossy(&result.stderr).trim()
        )));
    }
    Ok(())
}

pub(crate) fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}
