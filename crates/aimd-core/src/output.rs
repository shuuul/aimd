//! Interface-owned Markdown persistence (ports `aimd.interfaces.output`).
//! Shared by the CLI, HTTP API, and MCP server.

use std::fmt;
use std::path::{Path, PathBuf};

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

/// `AIMD_TEMP_DIR` (created when set) for API/MCP requests.
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
pub const MODEL_HELP_TEXT: &str = "Model for transcription or OCR. Default OCR is unlimited-ocr \
(mlx-community/Unlimited-OCR-4bit), or an explicit mlx-community/Unlimited-OCR-{4bit,6bit,8bit} \
model ID. ASR: qwen3-asr-1.7b (default) or qwen3-asr-0.6b combined with --precision, or explicit \
mlx-community/Qwen3-ASR-* model IDs (legacy underscore aliases still work).";

/// `--precision` help.
pub const PRECISION_HELP_TEXT: &str = "Model precision/quantization: 4bit, 6bit, 8bit, or bf16 \
(dash variants like 4-bit are accepted). The MLX engine selects the matching mlx-community \
checkpoint (default 4bit when omitted); bf16 (dense) checkpoints are not supported by the \
native engine yet.";

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
