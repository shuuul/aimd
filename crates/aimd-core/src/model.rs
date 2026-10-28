//! Request/result models (ports `aimd.core.models`). Field names match the
//! Python models so MCP JSON stays wire-compatible.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Url,
    AudioFile,
    VideoFile,
    DocumentFile,
    ImageFile,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Transcript,
    Convert,
    Ocr,
}

impl TaskType {
    /// Parse `transcript` / `convert` / `ocr` (exact, lowercase).
    pub fn parse(value: &str) -> Option<TaskType> {
        match value {
            "transcript" => Some(TaskType::Transcript),
            "convert" => Some(TaskType::Convert),
            "ocr" => Some(TaskType::Ocr),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TaskType::Transcript => "transcript",
            TaskType::Convert => "convert",
            TaskType::Ocr => "ocr",
        }
    }
}

/// Classified input source and selected processing task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputRoute {
    pub source_kind: SourceKind,
    pub task_type: Option<TaskType>,
}

/// Title plus context-window chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContext {
    pub title: String,
    pub chunk_list: Vec<String>,
    /// Header level used for splitting (1-6); `None` when not split by headers.
    pub split_header_level: Option<u8>,
}

/// Canonical process request. Output files are interface-owned and are
/// intentionally absent here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessInput {
    pub input_source: String,
    pub task_type: Option<TaskType>,
    pub model: Option<String>,
    pub language: Option<String>,
    /// 0-based inclusive start page for OCR PDF inputs (signed so negative
    /// values reach the Python-compatible validation message).
    pub start: Option<i64>,
    /// 0-based inclusive end page for OCR PDF inputs.
    pub end: Option<i64>,
    pub save_original: Option<PathBuf>,
    pub cookies: Option<PathBuf>,
    pub cookies_from_browser: Option<String>,
    pub temp_dir: Option<PathBuf>,
    pub raw_transcript: bool,
    pub precision: Option<String>,
    /// Explicit ASR biasing text.
    pub context: Option<String>,
    /// Build ASR context from URL page metadata (default `true`).
    pub metadata_context: bool,
    pub asr_base_url: Option<String>,
    pub asr_model: Option<String>,
    pub asr_api_key: Option<String>,
    pub ocr_base_url: Option<String>,
    pub ocr_model: Option<String>,
    pub ocr_api_key: Option<String>,
    /// Source anchors in `markdown`: OCR position comments and `[hh:mm:ss]`
    /// transcript paragraphs (default `false`).
    pub anchors: bool,
    /// The caller returns `segments` (MCP `include_segments`, CLI
    /// `--format json`; default `false`).
    pub include_segments: bool,
}

impl Default for ProcessInput {
    fn default() -> Self {
        Self {
            input_source: String::new(),
            task_type: None,
            model: None,
            language: None,
            start: None,
            end: None,
            save_original: None,
            cookies: None,
            cookies_from_browser: None,
            temp_dir: None,
            raw_transcript: false,
            precision: None,
            context: None,
            metadata_context: true,
            asr_base_url: None,
            asr_model: None,
            asr_api_key: None,
            ocr_base_url: None,
            ocr_model: None,
            ocr_api_key: None,
            anchors: false,
            include_segments: false,
        }
    }
}

impl ProcessInput {
    /// Whether positions/timestamps are computed (forced alignment, OCR
    /// locations): for anchored Markdown or returned `segments`.
    pub fn wants_positions(&self) -> bool {
        self.anchors || self.include_segments
    }
}

impl ProcessInput {
    pub fn new(input_source: impl Into<String>) -> Self {
        Self {
            input_source: input_source.into(),
            ..Self::default()
        }
    }
}

/// One located piece of the source: an OCR block or line (page + box) or
/// a transcript sentence (start/end seconds). `None` fields are omitted on
/// the wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    /// OCR category (`title`, `text`, `table`, `line`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// 1-based page (1 for single images).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// `[x0, y0, x1, y1]` normalized to the page image (0..1).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "wire::coords"
    )]
    pub bbox: Option<[f32; 4]>,
    /// Rotated line corners, normalized like `bbox`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "wire::quad"
    )]
    pub quad: Option<[[f32; 2]; 4]>,
    /// Seconds from the start of the media.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "wire::seconds"
    )]
    pub start: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "wire::seconds"
    )]
    pub end: Option<f64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "wire::score"
    )]
    pub score: Option<f32>,
}

/// Wire rounding for `Segment`: coordinates to 3 decimals (as in the anchor
/// comments), seconds to milliseconds, scores to 4 decimals, so f32 values
/// do not leak float noise into JSON.
mod wire {
    use serde::Serializer;
    use serde::ser::SerializeSeq;

    fn round(value: f64, places: i32) -> f64 {
        let scale = 10f64.powi(places);
        (value * scale).round() / scale
    }

    pub fn coords<S: Serializer>(v: &Option<[f32; 4]>, s: S) -> Result<S::Ok, S::Error> {
        let v = v.expect("skipped when None");
        s.collect_seq(v.iter().map(|&x| round(x.into(), 3)))
    }

    pub fn quad<S: Serializer>(v: &Option<[[f32; 2]; 4]>, s: S) -> Result<S::Ok, S::Error> {
        let v = v.expect("skipped when None");
        let mut seq = s.serialize_seq(Some(4))?;
        for [x, y] in v {
            seq.serialize_element(&[round(x.into(), 3), round(y.into(), 3)])?;
        }
        seq.end()
    }

    pub fn seconds<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(round(v.expect("skipped when None"), 3))
    }

    pub fn score<S: Serializer>(v: &Option<f32>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(round(v.expect("skipped when None").into(), 4))
    }
}

/// Canonical process result. `markdown` is lossless; `text_context` holds
/// the chunked view; `segments` locates the content in the source when the
/// route can (OCR blocks/lines, timed transcripts).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessResult {
    pub task_type: TaskType,
    pub text_context: TextContext,
    pub markdown: String,
    pub asset_base_uri: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub platform: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}
