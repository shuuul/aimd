//! Versioned HTTP schemas shared by the synchronous and job APIs (ports
//! `aimd.interfaces.api.schemas`). Field order and names match the
//! pydantic models so the JSON is wire-compatible with the Python API.

use aimd_core::TaskType;
use serde::Serialize;

use crate::util::Timestamp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Completed | JobState::Failed | JobState::Cancelled
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStage {
    Downloading,
    Extracting,
    Transcribing,
    Ocr,
    Converting,
    Saving,
}

impl JobStage {
    /// Map a pipeline progress stage (`ProcessHooks::progress`) onto the
    /// frozen job stage vocabulary; `recognizing` is the OCR stage.
    pub fn from_hook(stage: &str) -> Option<JobStage> {
        match stage {
            "downloading" => Some(JobStage::Downloading),
            "extracting" => Some(JobStage::Extracting),
            "transcribing" => Some(JobStage::Transcribing),
            "ocr" | "recognizing" => Some(JobStage::Ocr),
            "converting" => Some(JobStage::Converting),
            "saving" => Some(JobStage::Saving),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationStatus {
    None,
    Requested,
    Cancelled,
    CompletedAfterRequest,
}

/// Sidecar health response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

impl Default for HealthResponse {
    fn default() -> Self {
        Self { status: "ok" }
    }
}

/// Response returned after storing sidecar-owned blob bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BlobUploadResponse {
    pub blob_id: String,
    pub bytes: u64,
    pub filename: Option<String>,
}

/// AIMD processing request accepted by the synchronous and job APIs.
/// Parsed by [`crate::validate::parse_process_request`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessRequest {
    pub input_source: Option<String>,
    pub blob_id: Option<String>,
    pub task_type: Option<TaskType>,
    pub output_file: Option<String>,
    pub model: Option<String>,
    pub precision: Option<String>,
    pub asr_base_url: Option<String>,
    pub asr_model: Option<String>,
    pub asr_api_key: Option<String>,
    pub ocr_base_url: Option<String>,
    pub ocr_model: Option<String>,
    pub ocr_api_key: Option<String>,
    pub context: Option<String>,
    pub metadata_context: bool,
    pub language: Option<String>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub save_original: Option<String>,
    pub cookies: Option<String>,
    pub cookies_from_browser: Option<String>,
    pub raw_transcript: bool,
}

impl Default for ProcessRequest {
    fn default() -> Self {
        Self {
            input_source: None,
            blob_id: None,
            task_type: None,
            output_file: None,
            model: None,
            precision: None,
            asr_base_url: None,
            asr_model: None,
            asr_api_key: None,
            ocr_base_url: None,
            ocr_model: None,
            ocr_api_key: None,
            context: None,
            metadata_context: true,
            language: None,
            start: None,
            end: None,
            save_original: None,
            cookies: None,
            cookies_from_browser: None,
            raw_transcript: false,
        }
    }
}

impl ProcessRequest {
    /// Request for a local path or URL.
    pub fn input(source: impl Into<String>) -> Self {
        Self {
            input_source: Some(source.into()),
            ..Self::default()
        }
    }

    /// Request for an uploaded blob.
    pub fn blob(blob_id: impl Into<String>) -> Self {
        Self {
            blob_id: Some(blob_id.into()),
            ..Self::default()
        }
    }

    /// Truthy `blob_id` (pydantic `if request.blob_id`).
    pub fn has_blob(&self) -> bool {
        self.blob_id.as_deref().is_some_and(|b| !b.is_empty())
    }
}

/// Backward-compatible synchronous processing response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessResponse {
    pub task_type: TaskType,
    pub title: String,
    pub markdown: String,
    pub asset_base_uri: Option<String>,
    pub chunk_list: Vec<String>,
    pub split_header_level: Option<u8>,
    pub platform: Option<String>,
    pub output_file: Option<String>,
    pub output_dir: Option<String>,
}

/// Lossless editor artifact returned by a completed job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProcessArtifact {
    pub task_type: TaskType,
    pub title: String,
    pub markdown: String,
    pub source_uri: String,
    pub asset_base_uri: Option<String>,
    pub platform: Option<String>,
    pub output_file: Option<String>,
    pub output_dir: Option<String>,
}

/// Monotonic event emitted by a processing job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobEvent {
    pub job_id: String,
    pub state: JobState,
    pub stage: Option<JobStage>,
    pub current: Option<u64>,
    pub total: Option<u64>,
    pub message: Option<String>,
    pub cancellation_status: CancellationStatus,
    pub sequence: u64,
    pub created_at: Timestamp,
}

/// Current state and terminal result of a job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobSnapshot {
    pub job_id: String,
    pub state: JobState,
    pub request: ProcessRequest,
    pub artifact: Option<ProcessArtifact>,
    pub error: Option<String>,
    pub stage: Option<JobStage>,
    pub current: Option<u64>,
    pub total: Option<u64>,
    pub cancellation_status: CancellationStatus,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

/// Links returned when a job is accepted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobCreated {
    pub job_id: String,
    pub state: JobState,
    pub status_url: String,
    pub events_url: String,
}
