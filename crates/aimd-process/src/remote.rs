//! OpenAI-compatible remote ASR/OCR clients (ports
//! `aimd.plugins.asr.models.remote` and `aimd.plugins.ocr.remote`).

use std::path::Path;
use std::time::Duration;

use aimd_core::{AimdError, RemoteBackendConfig, Result};
use aimd_engine::models::qwen3_asr::generate::parse_qwen_output;
use aimd_engine::models::unlimited_ocr::generate::normalize_unlimited_ocr_markdown;
use serde_json::{Value, json};

use crate::http::{self, HttpError, base64_encode, guess_mime, multipart_body};

/// Python `urlopen(..., timeout=600)`.
const TIMEOUT: Duration = Duration::from_secs(600);
/// Baidu sliding-window guard values (`UNLIMITED_OCR_NGRAM_*`).
const NGRAM_SIZE: u64 = 35;
const NGRAM_WINDOW: u64 = 128;
const MULTI_PAGE_WINDOW: u64 = 1024;

fn map_http_error(label: &str, base_url: &str, err: HttpError) -> AimdError {
    match err {
        HttpError::Status(code, detail) if code >= 500 => AimdError::backend_unavailable(format!(
            "Remote {label} backend returned HTTP {code}: {detail}"
        )),
        HttpError::Status(code, detail) => AimdError::failed(format!(
            "Remote {label} request failed with HTTP {code}: {detail}"
        )),
        HttpError::Transport(e) => AimdError::backend_unavailable(format!(
            "Remote {label} backend is unavailable at {base_url}: {e}"
        )),
    }
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| AimdError::failed(format!("Cannot read {}: {e}", path.display())))
}

/// POST `/audio/transcriptions` (multipart) and parse the Qwen output.
pub fn transcribe(
    config: &RemoteBackendConfig,
    file_path: &Path,
    language: Option<&str>,
    context: Option<&str>,
) -> Result<String> {
    let mut fields: Vec<(&str, &str)> = vec![("model", config.model.as_str())];
    if let Some(language) = language.filter(|l| !l.is_empty()) {
        fields.push(("language", language));
    }
    if let Some(context) = context.filter(|c| !c.is_empty()) {
        fields.push(("prompt", context));
    }
    let bytes = read_file(file_path)?;
    let name = file_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mime = guess_mime(file_path).unwrap_or("application/octet-stream");
    let (body, content_type) = multipart_body(&fields, &name, mime, &bytes);
    let headers = [
        ("Authorization", format!("Bearer {}", config.api_key)),
        ("Content-Type", content_type),
    ];
    let url = format!("{}/audio/transcriptions", config.base_url);
    let response = http::post(&url, &headers, &body, TIMEOUT)
        .map_err(|e| map_http_error("ASR", &config.base_url, e))?;
    let payload: Value = serde_json::from_slice(&response)
        .map_err(|_| AimdError::failed("Remote ASR backend returned an invalid JSON response"))?;
    let text = payload.get("text").and_then(Value::as_str).unwrap_or("");
    if text.trim().is_empty() {
        return Err(AimdError::failed("Remote ASR produced empty transcription"));
    }
    let transcription = parse_qwen_output(text);
    if transcription.is_empty() {
        return Err(AimdError::failed("Remote ASR produced empty transcription"));
    }
    Ok(transcription)
}

/// POST `/chat/completions` with one image; returns normalized Markdown.
pub fn recognize_image(
    config: &RemoteBackendConfig,
    image_path: &Path,
    multi_page: bool,
) -> Result<String> {
    let mime = guess_mime(image_path).unwrap_or("image/png");
    let encoded = base64_encode(&read_file(image_path)?);
    let payload = json!({
        "model": config.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\ndocument parsing."},
                {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{encoded}")}},
            ],
        }],
        "skip_special_tokens": false,
        "ngram_size": NGRAM_SIZE,
        "window_size": if multi_page { MULTI_PAGE_WINDOW } else { NGRAM_WINDOW },
    });
    let headers = [
        ("Authorization", format!("Bearer {}", config.api_key)),
        ("Content-Type", "application/json".to_string()),
    ];
    let url = format!("{}/chat/completions", config.base_url);
    let body = serde_json::to_vec(&payload).expect("json payload serializes");
    let response = http::post(&url, &headers, &body, TIMEOUT)
        .map_err(|e| map_http_error("OCR", &config.base_url, e))?;
    let payload: Value = serde_json::from_slice(&response)
        .map_err(|_| AimdError::failed("Remote OCR backend returned an invalid JSON response"))?;
    let content = payload
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .ok_or_else(|| {
            AimdError::failed("Remote OCR backend returned an invalid chat completion")
        })?;
    let text = content.as_str().unwrap_or("");
    if text.trim().is_empty() {
        return Err(AimdError::failed("Remote OCR produced empty content"));
    }
    Ok(normalize_unlimited_ocr_markdown(text).trim().to_string())
}
