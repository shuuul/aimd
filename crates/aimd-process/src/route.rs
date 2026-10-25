//! Input routing (ports `get_input_route` / `ensure_supported_input`).

use std::path::Path;

use aimd_core::ext::{
    AUDIO_FILE_EXTENSIONS, IMAGE_FILE_EXTENSIONS, OCR_DOCUMENT_EXTENSIONS, VIDEO_FILE_EXTENSIONS,
    is_convert_extension, suffix_lower,
};
use aimd_core::{AimdError, InputRoute, Result, SourceKind, TaskType, is_url};

use crate::Runtime;

const fn route(source_kind: SourceKind, task_type: Option<TaskType>) -> InputRoute {
    InputRoute {
        source_kind,
        task_type,
    }
}

/// Classify a source and select its task (`None` task = unsupported).
pub fn get_input_route(
    input_source: &str,
    requested: Option<TaskType>,
    rt: &dyn Runtime,
) -> InputRoute {
    let other_than = |task: TaskType| requested.is_some_and(|r| r != task);
    if is_url(input_source) {
        if other_than(TaskType::Transcript) {
            return route(SourceKind::Url, None);
        }
        return route(SourceKind::Url, Some(TaskType::Transcript));
    }
    let path = Path::new(input_source);
    if input_source.is_empty() || !path.exists() {
        return route(SourceKind::Unknown, None);
    }
    let suffix = suffix_lower(path);
    let suffix = suffix.as_str();
    if requested == Some(TaskType::Ocr) {
        if IMAGE_FILE_EXTENSIONS.contains(&suffix) {
            return route(SourceKind::ImageFile, Some(TaskType::Ocr));
        }
        if OCR_DOCUMENT_EXTENSIONS.contains(&suffix) {
            return route(SourceKind::DocumentFile, Some(TaskType::Ocr));
        }
        return route(SourceKind::Unknown, None);
    }
    if IMAGE_FILE_EXTENSIONS.contains(&suffix) {
        return route(SourceKind::ImageFile, Some(TaskType::Ocr));
    }
    if OCR_DOCUMENT_EXTENSIONS.contains(&suffix) && rt.pdf_requires_ocr(path) {
        return route(SourceKind::DocumentFile, Some(TaskType::Ocr));
    }
    for (exts, kind) in [
        (AUDIO_FILE_EXTENSIONS, SourceKind::AudioFile),
        (VIDEO_FILE_EXTENSIONS, SourceKind::VideoFile),
    ] {
        if exts.contains(&suffix) {
            if other_than(TaskType::Transcript) {
                return route(kind, None);
            }
            return route(kind, Some(TaskType::Transcript));
        }
    }
    if is_convert_extension(suffix) {
        if other_than(TaskType::Convert) {
            return route(SourceKind::DocumentFile, None);
        }
        return route(SourceKind::DocumentFile, Some(TaskType::Convert));
    }
    route(SourceKind::Unknown, None)
}

/// Validate and return a supported route, else the Python domain error.
pub fn ensure_supported_input(
    input_source: &str,
    requested: Option<TaskType>,
    rt: &dyn Runtime,
) -> Result<InputRoute> {
    let route = get_input_route(input_source, requested, rt);
    if route.task_type.is_some() {
        return Ok(route);
    }
    let path = Path::new(input_source);
    if !is_url(input_source) && !suffix_lower(path).is_empty() && !path.exists() {
        return Err(AimdError::not_found(format!(
            "Input file not found: {input_source}"
        )));
    }
    if requested == Some(TaskType::Ocr) {
        return Err(AimdError::unsupported(
            "Unsupported OCR input. OCR supports image files (.png, .jpg, .jpeg, .webp, .tif, \
             .tiff) and PDF files.",
        ));
    }
    Err(AimdError::unsupported(
        "Unsupported input source. Supported inputs: audio/video files, video URLs, supported \
         document files, and OCR-capable images/PDFs.",
    ))
}

/// Validate a raw `--task` / `task_type` string.
pub fn parse_task(value: Option<&str>) -> Result<Option<TaskType>> {
    match value {
        None => Ok(None),
        Some(v) => TaskType::parse(v).map(Some).ok_or_else(|| {
            AimdError::unsupported("Unsupported task. Supported tasks: transcript, convert, ocr.")
        }),
    }
}
