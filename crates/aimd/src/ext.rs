//! File-extension tables (ports `aimd.plugins.asr.const`,
//! `aimd.plugins.ocr.const`, and the Pandoc map in
//! `aimd.plugins.doc.conversion`). Every extension is lowercase with its
//! leading dot.

use std::path::Path;

/// Audio inputs routed to transcription.
pub const AUDIO_FILE_EXTENSIONS: &[&str] = &[
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma", ".webm", ".mp4a",
];

/// Video inputs routed to transcription (audio extracted by ffmpeg).
pub const VIDEO_FILE_EXTENSIONS: &[&str] = &[
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m4v",
];

/// Formats the Python ASR backends read without an ffmpeg transcode
/// (`audio_utils.SUPPORTED_AUDIO_FORMATS`); remote ASR uploads these as-is.
pub const SUPPORTED_AUDIO_FORMATS: &[&str] = &[".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"];

/// OCR image inputs.
pub const IMAGE_FILE_EXTENSIONS: &[&str] = &[".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"];

/// OCR document inputs.
pub const OCR_DOCUMENT_EXTENSIONS: &[&str] = &[".pdf"];

/// Document conversions that write an asset directory `<parent>/<stem>/`.
pub const DOCUMENT_ASSET_EXTENSIONS: &[&str] = &[".docx", ".epub", ".odt"];

/// Pandoc input reader per extension (verbatim Python table).
pub const PANDOC_INPUT_FORMAT_BY_EXTENSION: &[(&str, &str)] = &[
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
    (".md", "markdown"),
    (".mdoc", "mdoc"),
    (".mediawiki", "mediawiki"),
    (".muse", "muse"),
    (".native", "native"),
    (".odt", "odt"),
    (".opml", "opml"),
    (".org", "org"),
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
];

/// Non-Pandoc-table extensions the Rust document pipeline converts
/// (decision 2026-10-25: Pandoc >= 3.8 reads pptx/xlsx, `.txt` passes
/// through, text-layer PDFs use pdf-inspector). Legacy binary
/// `.doc`/`.ppt`/`.xls` are intentionally absent.
pub const EXTRA_CONVERT_EXTENSIONS: &[&str] = &[".pdf", ".pptx", ".txt", ".xlsx"];

/// Lowercase `.ext` of a path (`""` when none), like `Path.suffix.lower()`.
pub fn suffix_lower(path: &Path) -> String {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return String::new();
    };
    // Python's suffix ignores a leading dot (".bashrc" has no suffix) and a
    // trailing dot.
    match name.rfind('.') {
        Some(0) | None => String::new(),
        Some(i) if i + 1 == name.len() => String::new(),
        Some(i) => name[i..].to_lowercase(),
    }
}

/// Whether `ext` is a key of [`PANDOC_INPUT_FORMAT_BY_EXTENSION`].
pub fn is_pandoc_extension(ext: &str) -> bool {
    PANDOC_INPUT_FORMAT_BY_EXTENSION
        .iter()
        .any(|(e, _)| *e == ext)
}

/// Whether a lowercase extension is offered to document conversion
/// (Pandoc table + [`EXTRA_CONVERT_EXTENSIONS`]).
pub fn is_convert_extension(ext: &str) -> bool {
    is_pandoc_extension(ext) || EXTRA_CONVERT_EXTENSIONS.contains(&ext)
}

pub fn is_audio_extension(ext: &str) -> bool {
    AUDIO_FILE_EXTENSIONS.contains(&ext) || VIDEO_FILE_EXTENSIONS.contains(&ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_matches_pathlib() {
        assert_eq!(suffix_lower(Path::new("/a/B.WAV")), ".wav");
        assert_eq!(suffix_lower(Path::new("archive.tar.gz")), ".gz");
        assert_eq!(suffix_lower(Path::new(".bashrc")), "");
        assert_eq!(suffix_lower(Path::new("noext")), "");
        assert_eq!(suffix_lower(Path::new("trail.")), "");
    }

    #[test]
    fn extension_tables() {
        // test_audio_transcoding.py::test_mp4a_*; test_ocr.py::test_ocr_extension_constants_are_centralized
        assert!(AUDIO_FILE_EXTENSIONS.contains(&".mp4a"));
        assert!(!SUPPORTED_AUDIO_FORMATS.contains(&".mp4a"));
        assert!(is_audio_extension(".mkv"));
        assert!(OCR_DOCUMENT_EXTENSIONS.contains(&".pdf"));
        assert!(is_convert_extension(".epub"));
        assert!(is_convert_extension(".xlsx"));
        assert!(!is_convert_extension(".doc"));
        assert!(!is_convert_extension(".mobi"));
    }
}
