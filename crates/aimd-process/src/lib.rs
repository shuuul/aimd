//! Routing and orchestration over the doc/url/engine crates (ports
//! `aimd.core.process` plus the ASR/OCR plugins).
//!
//! Entry point: [`process_input`]. Every side effect (environment, `PATH`
//! lookups, subprocesses, model loading, PDF probing, document and URL
//! conversion) goes through the [`Runtime`] trait so tests can replace any
//! of them; [`NativeRuntime`] is the real implementation.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aimd_core::ext::{DOCUMENT_ASSET_EXTENSIONS, suffix_lower};
use aimd_core::url::path_to_file_uri;
use aimd_core::{
    AimdError, ProcessHooks, ProcessInput, ProcessResult, Result, SourceKind, TaskType,
    text_context_from_markdown,
};

pub mod asr;
pub mod audio;
pub mod hf;
pub mod http;
pub mod models;
mod native;
pub mod ocr;
pub mod remote;
pub mod route;
pub mod tempdir;

pub use native::NativeRuntime;
pub use route::{ensure_supported_input, get_input_route};

/// Captured subprocess result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandOutput {
    /// Exit code (`None` when killed by a signal).
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl CommandOutput {
    pub fn success(&self) -> bool {
        self.status == Some(0)
    }

    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// Model-level ASR failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsrFailure {
    /// The model produced no text (Python "empty transcription").
    Empty,
    /// Anything else (message only).
    Other(String),
}

/// A loaded local ASR model.
pub trait AsrModel: Send + Sync {
    /// Transcribe a 16 kHz mono wav with a resolved language name and
    /// optional biasing context; returns the parsed transcript.
    fn transcribe(
        &self,
        wav_path: &Path,
        language: &str,
        context: Option<&str>,
    ) -> std::result::Result<String, AsrFailure>;
}

/// Model-level OCR failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrFailure {
    /// The model produced no content.
    Empty,
    Other(String),
}

/// A loaded local OCR model.
pub trait OcrModel: Send + Sync {
    /// Recognize one image file; returns normalized Markdown.
    fn recognize(&self, image_path: &Path) -> std::result::Result<String, OcrFailure>;
}

/// Converted document (re-export of the doc crate output).
pub use aimd_doc::DocOutput;

/// URL transcript output and the audio-fallback seam (re-exports of the URL
/// crate).
pub use aimd_url::{AudioTranscriber, TranscribeRequest, UrlOutput, markitdown_normalize};

/// [`AudioTranscriber`] for the URL audio fallback over [`asr::transcribe_with_options`]
/// with a runtime's model loading and tools.
pub struct UrlAudioTranscriber<'a>(pub &'a dyn Runtime);

impl AudioTranscriber for UrlAudioTranscriber<'_> {
    fn transcribe(
        &self,
        audio_path: &Path,
        request: &TranscribeRequest,
        hooks: &dyn ProcessHooks,
    ) -> Result<String> {
        let options = asr::AsrOptions {
            language: request.language.clone(),
            model: request.model.clone(),
            precision: request.precision.clone(),
            temp_dir: request.temp_dir.clone(),
            asr_base_url: request.asr_base_url.clone(),
            asr_model: request.asr_model.clone(),
            asr_api_key: request.asr_api_key.clone(),
        };
        asr::transcribe_with_options(
            self.0,
            audio_path,
            &options,
            request.context.as_deref(),
            hooks,
        )
    }
}

/// The real URL pipeline (`aimd_url::convert_url`) with `rt` transcribing
/// the audio fallback. [`NativeRuntime::url_transcript`] calls this.
pub fn convert_url(
    rt: &dyn Runtime,
    request: &ProcessInput,
    hooks: &dyn ProcessHooks,
) -> Result<UrlOutput> {
    aimd_url::convert_url(request, &UrlAudioTranscriber(rt), hooks)
}

/// Local resource kind for [`Runtime::preflight_local`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalModality {
    Asr,
    Ocr,
}

/// System seam. Defaults hit the real system; tests override pieces.
pub trait Runtime: Send + Sync {
    /// Environment lookup.
    fn env(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }

    /// `shutil.which`.
    fn which(&self, program: &str) -> Option<PathBuf> {
        native::which_in_path(program, self.env("PATH").as_deref())
    }

    /// Run a program to completion, capturing output.
    fn run(
        &self,
        program: &Path,
        args: &[String],
        cwd: Option<&Path>,
    ) -> std::io::Result<CommandOutput> {
        native::run_command(program, args, cwd)
    }

    /// Fail fast when local MLX inference cannot run (no Metal device).
    fn preflight_local(&self, modality: LocalModality) -> Result<()>;

    /// Load (or reuse) a local ASR checkpoint by HF id.
    fn load_asr(&self, model_id: &str) -> Result<Arc<dyn AsrModel>>;

    /// Load (or reuse) a local OCR checkpoint by HF id.
    fn load_ocr(&self, model_id: &str) -> Result<Arc<dyn OcrModel>>;

    /// pdf-inspector probe: scanned or broken-encoding PDFs need OCR.
    fn pdf_requires_ocr(&self, path: &Path) -> bool {
        native::pdf_requires_ocr(path)
    }

    /// PDF page count (for OCR page ranges).
    fn pdf_page_count(&self, path: &Path) -> Result<u32> {
        native::pdf_page_count(path)
    }

    /// Document conversion (aimd-doc).
    fn convert_document(
        &self,
        path: &Path,
        output_dir: Option<&Path>,
        temp_dir: Option<&Path>,
        hooks: &dyn ProcessHooks,
    ) -> Result<DocOutput> {
        aimd_doc::convert_document(path, output_dir, temp_dir, hooks)
    }

    /// URL transcript extraction. [`NativeRuntime`] runs [`convert_url`]
    /// (yt-dlp subtitles, then the audio fallback through this runtime);
    /// other runtimes delegate to [`convert_url`] or return a fake.
    fn url_transcript(
        &self,
        request: &ProcessInput,
        hooks: &dyn ProcessHooks,
    ) -> Result<UrlOutput> {
        let _ = (request, hooks);
        Err(AimdError::backend_unavailable(
            "URL transcripts need a runtime that implements url_transcript (NativeRuntime \
             delegates to aimd_process::convert_url).",
        ))
    }

    /// `TextContext` chunk size.
    fn max_chunk_size(&self) -> usize {
        aimd_core::MAX_CHUNK_SIZE
    }
}

/// Process a request with the native runtime.
pub fn process_input(request: &ProcessInput, hooks: &dyn ProcessHooks) -> Result<ProcessResult> {
    process_input_with(request, hooks, &NativeRuntime)
}

/// Process a request over an explicit runtime.
pub fn process_input_with(
    request: &ProcessInput,
    hooks: &dyn ProcessHooks,
    rt: &dyn Runtime,
) -> Result<ProcessResult> {
    let route = ensure_supported_input(&request.input_source, request.task_type, rt)?;
    let task_type = route
        .task_type
        .expect("ensure_supported_input returns a task");
    if route.source_kind == SourceKind::Url {
        let output = rt.url_transcript(request, hooks)?;
        return Ok(ProcessResult {
            task_type: TaskType::Transcript,
            text_context: text_context_from_markdown(
                &output.markdown,
                &output.title,
                Some(&output.title),
                rt.max_chunk_size(),
            ),
            markdown: output.markdown,
            asset_base_uri: Some(request.input_source.clone()),
            output_dir: None,
            platform: Some(output.platform),
        });
    }
    process_local_file(request, task_type, hooks, rt)
}

/// `Path(x).stem` (lossy).
pub(crate) fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `Path.parent` with Python's `.` for bare names.
pub(crate) fn parent_dir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// `convert_file_with_markitdown` + `_process_local_file`.
fn process_local_file(
    request: &ProcessInput,
    task_type: TaskType,
    hooks: &dyn ProcessHooks,
    rt: &dyn Runtime,
) -> Result<ProcessResult> {
    let input_path = Path::new(&request.input_source);
    if !input_path.exists() {
        return Err(AimdError::not_found(format!(
            "Input file not found: {}",
            input_path.display()
        )));
    }
    if !input_path.is_file() {
        return Err(AimdError::unsupported(format!(
            "Path is not a file: {}",
            input_path.display()
        )));
    }
    let stem = file_stem(input_path);
    let suffix = suffix_lower(input_path);
    let output_dir = DOCUMENT_ASSET_EXTENSIONS
        .contains(&suffix.as_str())
        .then(|| parent_dir(input_path).join(&stem));
    let temp_dir = request.temp_dir.as_deref();

    let (raw_markdown, title) = match task_type {
        TaskType::Transcript => {
            let text =
                asr::transcribe_file(rt, input_path, request, request.context.as_deref(), hooks)?;
            (text, Some(stem.clone()))
        }
        TaskType::Ocr => {
            let out = ocr::recognize_file(rt, input_path, request, hooks)?;
            (out.markdown, Some(out.title))
        }
        TaskType::Convert => {
            let doc = rt.convert_document(input_path, output_dir.as_deref(), temp_dir, hooks)?;
            (doc.markdown, doc.title.filter(|t| !t.is_empty()))
        }
    };
    // MarkItDown normalizes every converter result (trailing whitespace per
    // line, 3+ newlines collapsed); files a converter wrote stay as written.
    let markdown = markitdown_normalize(&raw_markdown);
    let text_context =
        text_context_from_markdown(&markdown, &stem, title.as_deref(), rt.max_chunk_size());
    let asset_base_uri = (task_type != TaskType::Transcript).then(|| {
        let dir = output_dir.clone().unwrap_or_else(|| parent_dir(input_path));
        directory_uri(&dir)
    });
    Ok(ProcessResult {
        task_type,
        text_context,
        markdown,
        asset_base_uri,
        output_dir,
        platform: None,
    })
}

/// `_directory_uri`: absolute file URI with a trailing slash.
pub fn directory_uri(directory: &Path) -> String {
    let uri = path_to_file_uri(&aimd_core::output::resolve_path(directory));
    if uri.ends_with('/') {
        uri
    } else {
        format!("{uri}/")
    }
}

#[cfg(test)]
mod tests;
