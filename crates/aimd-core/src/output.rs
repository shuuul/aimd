//! Interface-owned Markdown persistence (ports `aimd.interfaces.output`).
//! Shared by the CLI and MCP server.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::anchors;
use crate::error::AimdError;
use crate::model::{ProcessResult, TaskType};

/// Failure while persisting output: a domain error (empty transcript) or an
/// I/O error writing the file.
#[derive(Debug)]
pub enum PersistError {
    Aimd(AimdError),
    Io(std::io::Error),
}

impl fmt::Display for PersistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistError::Aimd(e) => e.fmt(f),
            PersistError::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PersistError {}

impl From<AimdError> for PersistError {
    fn from(e: AimdError) -> Self {
        PersistError::Aimd(e)
    }
}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

/// Persisted Markdown for a task output (empty transcripts are an error).
pub fn build_output_text(task_type: TaskType, markdown: &str) -> Result<&str, AimdError> {
    if task_type == TaskType::Transcript && markdown.is_empty() {
        return Err(AimdError::failed("Transcription returned empty content"));
    }
    Ok(markdown)
}

/// Output file format (`aimd --format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    #[default]
    Markdown,
    /// The MCP `process_input` response object, with `segments`.
    Json,
    Srt,
    Vtt,
}

impl OutputFormat {
    pub const NAMES: &'static str = "markdown, json, srt, vtt";

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "markdown" | "md" => Some(Self::Markdown),
            "json" => Some(Self::Json),
            "srt" => Some(Self::Srt),
            "vtt" | "webvtt" => Some(Self::Vtt),
            _ => None,
        }
    }

    /// File extension (without the dot).
    pub fn extension(self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Json => "json",
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    /// Subtitle formats need a transcript with timed segments.
    pub fn is_subtitle(self) -> bool {
        matches!(self, Self::Srt | Self::Vtt)
    }
}

/// SRT/VTT text for a transcript result's timed segments.
pub fn subtitle_text(result: &ProcessResult, format: OutputFormat) -> Result<String, AimdError> {
    if result.task_type != TaskType::Transcript {
        return Err(AimdError::unsupported(format!(
            "--format {} is only available for transcripts",
            format.extension()
        )));
    }
    let segments = result
        .segments
        .as_deref()
        .filter(|s| s.iter().any(|s| s.start.is_some()))
        .ok_or_else(|| {
            AimdError::unsupported(format!(
                "--format {} needs timed segments (drop --raw-transcript)",
                format.extension()
            ))
        })?;
    let cues = anchors::subtitle_cues(segments);
    Ok(match format {
        OutputFormat::Vtt => anchors::to_vtt(&cues),
        _ => anchors::to_srt(&cues),
    })
}

/// The `process_input` response object: the Python `_process_result_payload`
/// keys, plus `segments` when `include_segments` (spec 005).
pub fn result_payload(
    result: &ProcessResult,
    output_file: Option<String>,
    output_dir: Option<String>,
    include_segments: bool,
) -> Value {
    // serde_json maps are key-sorted; `payload_text` restores Python order.
    let mut value = json!({
        "task_type": result.task_type.as_str(),
        "title": result.text_context.title,
        "markdown": result.markdown,
        "asset_base_uri": result.asset_base_uri,
        "chunk_list": result.text_context.chunk_list,
        "split_header_level": result.text_context.split_header_level,
        "platform": result.platform,
        "output_file": output_file,
        "output_dir": output_dir,
    });
    if include_segments {
        value["segments"] = json!(result.segments.as_deref().unwrap_or_default());
    }
    value
}

/// Ordered keys of the `process_input` payload.
pub const PAYLOAD_KEY_ORDER: [&str; 10] = [
    "task_type",
    "title",
    "markdown",
    "asset_base_uri",
    "chunk_list",
    "split_header_level",
    "platform",
    "output_file",
    "output_dir",
    "segments",
];

/// Indented (2 spaces) JSON text with the payload keys in Python order, as
/// FastMCP renders the text content block.
pub fn payload_text(value: &Value) -> String {
    let Value::Object(map) = value else {
        return serde_json::to_string_pretty(value).unwrap_or_default();
    };
    let mut keys: Vec<&str> = PAYLOAD_KEY_ORDER
        .iter()
        .copied()
        .filter(|k| map.contains_key(*k))
        .collect();
    keys.extend(
        map.keys()
            .map(String::as_str)
            .filter(|k| !PAYLOAD_KEY_ORDER.contains(k)),
    );
    if keys.is_empty() {
        return "{}".to_string();
    }
    let mut out = String::from("{\n");
    for (i, key) in keys.iter().enumerate() {
        let rendered = serde_json::to_string_pretty(&map[*key]).unwrap_or_default();
        let indented = rendered.replace('\n', "\n  ");
        out.push_str(&format!(
            "  {}: {}{}\n",
            serde_json::to_string(key).unwrap_or_default(),
            indented,
            if i + 1 < keys.len() { "," } else { "" }
        ));
    }
    out.push('}');
    out
}

/// `Path.resolve()`: absolute, symlinks resolved (falls back to joining
/// the cwd when the path cannot be canonicalized).
pub fn resolve_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        }
    })
}

/// Write the exact task Markdown to `output_file` (creating parents) and
/// return the resolved path.
pub fn persist_output(
    output_file: &Path,
    task_type: TaskType,
    markdown: &str,
) -> Result<PathBuf, PersistError> {
    let text = build_output_text(task_type, markdown)?;
    if let Some(parent) = output_file.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output_file, text)?;
    Ok(resolve_path(output_file))
}

/// Interface-facing output locations after optional persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedOutput {
    pub output_file: Option<String>,
    pub output_dir: Option<String>,
    pub ignored_output_file: bool,
}

/// Persist a result when an interface requested a file output. Asset
/// conversions (`output_dir` set) ignore the request.
pub fn persist_result_output_if_requested(
    result: &ProcessResult,
    requested_output_file: Option<&Path>,
) -> Result<PersistedOutput, PersistError> {
    let output_dir = result
        .output_dir
        .as_deref()
        .map(|d| resolve_path(d).to_string_lossy().into_owned());
    let Some(requested) = requested_output_file else {
        return Ok(PersistedOutput {
            output_file: None,
            output_dir,
            ignored_output_file: false,
        });
    };
    if result.output_dir.is_some() {
        return Ok(PersistedOutput {
            output_file: None,
            output_dir,
            ignored_output_file: true,
        });
    }
    let resolved = persist_output(requested, result.task_type, &result.markdown)?;
    Ok(PersistedOutput {
        output_file: Some(resolved.to_string_lossy().into_owned()),
        output_dir: None,
        ignored_output_file: false,
    })
}

/// `AIMD_TEMP_DIR` (created when set) for MCP requests.
pub fn get_request_temp_dir() -> std::io::Result<Option<PathBuf>> {
    match std::env::var_os("AIMD_TEMP_DIR") {
        Some(dir) if !dir.is_empty() => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            Ok(Some(dir))
        }
        _ => Ok(None),
    }
}

/// `--model` help (native-engine model matrix).
pub const MODEL_HELP_TEXT: &str = "Model for transcription or OCR. OCR: unlimited-ocr-2 (default; \
Unlimited-OCR-2, PDFs in 10-page requests), unlimited-ocr (Unlimited-OCR v1, 5-page requests), \
pp-ocrv6 (PP-OCRv6 medium det + rec; fast line-level text), or paddleocr-vl (PaddleOCR-VL-1.6), \
or an explicit mlx-community/Unlimited-OCR{,-2}-{4bit,6bit,8bit} model ID. ASR: qwen3-asr-1.7b (default) or qwen3-asr-0.6b combined with --precision, or explicit \
mlx-community/Qwen3-ASR-* model IDs (legacy underscore aliases still work).";

/// `--precision` help.
pub const PRECISION_HELP_TEXT: &str = "Model precision/quantization: 4bit, 6bit, 8bit, or bf16 \
(dash variants like 4-bit are accepted). Qwen3-ASR and Unlimited-OCR (1 and 2) select the matching \
mlx-community checkpoint (default 4bit; bf16 is not supported there yet). paddleocr-vl defaults \
to bf16 and quantizes at load for 4bit/6bit/8bit. pp-ocrv6 ignores precision.";

/// `--context` help.
pub const CONTEXT_HELP_TEXT: &str = "ASR context/biasing text (proper nouns, names, domain \
vocabulary) that helps transcription accuracy. For URL inputs, page metadata (title, \
description, tags) is appended automatically unless --no-context is set.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TextContext;

    fn result(task_type: TaskType, markdown: &str, output_dir: Option<PathBuf>) -> ProcessResult {
        ProcessResult {
            task_type,
            text_context: TextContext {
                title: "t".into(),
                chunk_list: vec![],
                split_header_level: None,
            },
            markdown: markdown.into(),
            asset_base_uri: None,
            output_dir,
            platform: None,
            segments: None,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("aimd-core-output-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn segment_payload_rounds_float_noise() {
        let mut r = result(TaskType::Ocr, "a", None);
        r.segments = Some(vec![crate::model::Segment {
            text: "a".into(),
            bbox: Some([1.0 / 27.0, 0.026_026, 0.255_255, 0.053_053]),
            quad: Some([
                [1.0 / 27.0, 0.5],
                [0.25, 0.5],
                [0.25, 0.75],
                [1.0 / 27.0, 0.75],
            ]),
            start: Some(2.880_000_114),
            score: Some(0.999_742_2),
            ..Default::default()
        }]);
        let value = result_payload(&r, None, None, true);
        assert_eq!(
            value["segments"][0],
            serde_json::json!({"text": "a", "bbox": [0.037, 0.026, 0.255, 0.053],
                "quad": [[0.037, 0.5], [0.25, 0.5], [0.25, 0.75], [0.037, 0.75]],
                "start": 2.88, "score": 0.9997})
        );
    }

    #[test]
    fn empty_transcript_is_rejected() {
        // test_cli.py::test_cli_rejects_empty_transcript_output
        let err = build_output_text(TaskType::Transcript, "").unwrap_err();
        assert_eq!(err.message, "Transcription returned empty content");
        assert_eq!(build_output_text(TaskType::Ocr, "").unwrap(), "");
    }

    #[test]
    fn persist_writes_exact_markdown_and_creates_parents() {
        let dir = scratch("persist");
        let target = dir.join("nested/out.md");
        let resolved = persist_output(&target, TaskType::Convert, "# A\n\nB\n").unwrap();
        assert_eq!(std::fs::read_to_string(&resolved).unwrap(), "# A\n\nB\n");
        assert!(resolved.is_absolute());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn asset_results_ignore_requested_output_file() {
        let dir = scratch("assets");
        let r = result(TaskType::Convert, "x", Some(dir.clone()));
        let out = persist_result_output_if_requested(&r, Some(&dir.join("o.md"))).unwrap();
        assert!(out.ignored_output_file);
        assert_eq!(out.output_file, None);
        assert!(!dir.join("o.md").exists());

        let r = result(TaskType::Transcript, "hello", None);
        let none = persist_result_output_if_requested(&r, None).unwrap();
        assert_eq!(none.output_file, None);
        let written = persist_result_output_if_requested(&r, Some(&dir.join("t.md"))).unwrap();
        assert_eq!(
            std::fs::read_to_string(written.output_file.unwrap()).unwrap(),
            "hello"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
