//! Text-layer PDF conversion through the `pdf-inspector` crate (the library
//! behind the Python `pdf_inspector` binding; default options, no page filter).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use aimd_core::AimdError;

/// Extract `(title, markdown)` from a text-layer PDF. Empty extraction is a
/// `ProcessingFailed` error (scanned PDFs are routed to OCR upstream).
pub(crate) fn process_pdf_text(path: &Path) -> aimd_core::Result<(Option<String>, String)> {
    if !path.exists() {
        return Err(AimdError::not_found(format!(
            "Document file not found: {}",
            path.display()
        )));
    }
    let name = crate::pandoc::file_name(path);
    let result = catch_unwind(AssertUnwindSafe(|| pdf_inspector::process_pdf(path)))
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| AimdError::failed(format!("pdf-inspector failed to process PDF: {name}")))?;
    let markdown = result.markdown.unwrap_or_default().trim().to_string();
    if markdown.is_empty() {
        return Err(AimdError::failed(format!(
            "pdf-inspector extracted no text from PDF: {name}"
        )));
    }
    Ok((result.title, markdown))
}
