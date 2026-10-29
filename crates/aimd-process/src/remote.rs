//! OpenAI-compatible remote ASR/OCR clients (ports
//! `aimd.plugins.asr.models.remote` and `aimd.plugins.ocr.remote`).

use std::path::Path;
use std::time::Duration;

use aimd_core::{AimdError, RemoteBackendConfig, Result, log_info};
use aimd_engine::models::qwen3_asr::generate::parse_qwen_output;
use aimd_engine::models::unlimited_ocr::generate::normalize_unlimited_ocr_markdown;
use aimd_engine::models::unlimited_ocr::markdown2::render_unlimited_ocr2;
use serde_json::{Value, json};

use crate::http::{self, HttpError, base64_encode, guess_mime, multipart_body};

/// Python `urlopen(..., timeout=600)`.
const TIMEOUT: Duration = Duration::from_secs(600);
/// Baidu sliding-window guard values (`UNLIMITED_OCR_NGRAM_*`).
const NGRAM_SIZE: u64 = 35;
const NGRAM_WINDOW: u64 = 128;
const MULTI_PAGE_WINDOW: u64 = 1024;
/// Unlimited-OCR v1 single-image instruction.
const OCR_PROMPT: &str = "<image>\ndocument parsing.";
/// Unlimited-OCR-2 instruction: one `<PAGE>` per image, `Multi page parsing.`
/// even for a single page (v2 does not take the v1 instruction).
const OCR2_PROMPT: &str = "<PAGE><image>\nMulti page parsing.";

/// Whether a served model name is Unlimited-OCR-2 (`Unlimited-OCR-2`,
/// `baidu/Unlimited-OCR-2`, `unlimited_ocr_2-4bit`, ...).
fn is_unlimited_ocr2(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model);
    name.to_ascii_lowercase()
        .replace(['_', ' '], "-")
        .starts_with("unlimited-ocr-2")
}

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
    let attempt = |model: &str| -> std::result::Result<Vec<u8>, HttpError> {
        let ocr2 = is_unlimited_ocr2(model);
        let payload = json!({
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": if ocr2 { OCR2_PROMPT } else { OCR_PROMPT }},
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
        http::post(&url, &headers, &body, TIMEOUT)
    };

    let mut used_model = config.model.clone();
    let response = match attempt(&config.model) {
        Ok(response) => response,
        // The default OCR model name (`Unlimited-OCR`) is a guess about what the
        // server serves. When the server rejects it with a 404, ask what it does
        // serve and retry once if the v2 name is there. A model named explicitly
        // by the user is never substituted.
        Err(HttpError::Status(404, detail)) if config.model_is_default => {
            // One probe decides both branches: either a v2 model to retry with,
            // or the served names to report.
            let served = served_model_names(config);
            match served
                .as_ref()
                .and_then(|names| names.iter().find(|name| is_unlimited_ocr2(name)).cloned())
            {
                Some(model) => {
                    log_info!(
                        "Remote OCR backend does not serve {}; using {model}",
                        config.model
                    );
                    used_model = model.clone();
                    attempt(&model).map_err(|e| map_http_error("OCR", &config.base_url, e))?
                }
                None => {
                    // Nothing usable to fall back to: surface the 404 together
                    // with the models the server does serve, so the fix (name one
                    // with --ocr-model) is obvious.
                    let hint = match served {
                        Some(names) if !names.is_empty() => {
                            format!("; served models: {}", names.join(", "))
                        }
                        _ => String::new(),
                    };
                    return Err(AimdError::failed(format!(
                        "Remote OCR request failed with HTTP 404: {detail}{hint}"
                    )));
                }
            }
        }
        Err(e) => return Err(map_http_error("OCR", &config.base_url, e)),
    };

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
    // v2 renders page-relative boxes; v1 output is normalized as-is. The retry
    // above switches families, so decide on the model that actually answered.
    let markdown = if is_unlimited_ocr2(&used_model) {
        render_unlimited_ocr2(text).markdown
    } else {
        normalize_unlimited_ocr_markdown(text)
    };
    Ok(markdown.trim().to_string())
}

/// Model names served by an OpenAI-compatible backend (`GET /v1/models`).
/// Best-effort: any failure yields `None` and the caller keeps its own error.
fn served_model_names(config: &RemoteBackendConfig) -> Option<Vec<String>> {
    let headers = [("Authorization", format!("Bearer {}", config.api_key))];
    let url = format!("{}/models", config.base_url);
    let body = http::get(&url, &headers, TIMEOUT).ok()?;
    let payload: Value = serde_json::from_slice(&body).ok()?;
    Some(
        payload
            .get("data")?
            .as_array()?
            .iter()
            .filter_map(|m| m.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// One recorded request: request line plus body.
    type Recorded = (String, String);

    /// A scripted OpenAI-compatible backend.
    ///
    /// Each entry answers one request; the script is consumed in order. Requests
    /// are recorded so a test can assert exactly which model names were asked
    /// for and how many round trips happened.
    fn scripted_backend(
        responses: Vec<(u16, String)>,
    ) -> (
        String,
        Arc<Mutex<Vec<Recorded>>>,
        std::thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&recorded);
        let handle = std::thread::spawn(move || {
            for (status, body) in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    return;
                }
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = line.split_once(':')
                        && k.eq_ignore_ascii_case("content-length")
                    {
                        len = v.trim().parse().unwrap_or(0);
                    }
                }
                let mut req_body = vec![0u8; len];
                let _ = reader.read_exact(&mut req_body);
                sink.lock().unwrap().push((
                    request_line.trim_end().to_string(),
                    String::from_utf8_lossy(&req_body).into_owned(),
                ));
                let response = format!(
                    "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    if status == 200 { "OK" } else { "Error" },
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        (format!("http://{addr}"), recorded, handle)
    }

    fn config(base_url: &str, model: &str, model_is_default: bool) -> RemoteBackendConfig {
        RemoteBackendConfig {
            base_url: base_url.to_string(),
            model: model.to_string(),
            api_key: "not-needed".into(),
            model_is_default,
        }
    }

    fn ocr_reply(text: &str) -> String {
        json!({"choices": [{"message": {"content": text}}]}).to_string()
    }

    fn temp_image() -> (crate::tempdir::TempDir, std::path::PathBuf) {
        let dir = crate::tempdir::TempDir::new(None, "aimd-remote-ocr-").unwrap();
        let path = dir.path().join("page.png");
        // 1x1 PNG; the client only base64s it.
        std::fs::write(
            &path,
            [
                0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
                b'D', b'R', 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
                0x00, 0x1f, 0x15, 0xc4, 0x89,
            ],
        )
        .unwrap();
        (dir, path)
    }

    /// The reported failure: the server serves `Unlimited-OCR-2`, the default
    /// name is `Unlimited-OCR`, so the first POST 404s. The client must ask what
    /// is served and retry once with the v2 name.
    #[test]
    fn default_ocr_model_falls_back_to_served_unlimited_ocr2() {
        let (base, recorded, handle) = scripted_backend(vec![
            (
                404,
                r#"{"error":{"message":"The model `Unlimited-OCR` does not exist."}}"#.into(),
            ),
            (
                200,
                json!({"data": [{"id": "Qwen3-ASR-1.7B"}, {"id": "Unlimited-OCR-2"}]}).to_string(),
            ),
            (200, ocr_reply("<PAGE>\n# Title\n\nBody text")),
        ]);
        let (_dir, image) = temp_image();
        let markdown =
            recognize_image(&config(&base, "Unlimited-OCR", true), &image, false).unwrap();
        assert!(markdown.contains("Title"), "{markdown}");
        handle.join().unwrap();

        let requests = recorded.lock().unwrap();
        assert_eq!(requests.len(), 3, "expected POST, GET /models, POST retry");
        assert!(requests[0].0.starts_with("POST /chat/completions"));
        assert!(requests[0].1.contains(r#""model":"Unlimited-OCR""#));
        assert!(requests[1].0.starts_with("GET /models"));
        assert!(requests[2].1.contains(r#""model":"Unlimited-OCR-2""#));
        // The retry must switch to the v2 instruction.
        assert!(requests[2].1.contains("Multi page parsing."));
    }

    /// An explicitly chosen model is never substituted, even on a 404: a typo
    /// must fail loudly instead of silently running a different model.
    #[test]
    fn explicit_ocr_model_is_not_substituted() {
        let (base, recorded, handle) =
            scripted_backend(vec![(404, r#"{"error":{"message":"nope"}}"#.into())]);
        let (_dir, image) = temp_image();
        let err = recognize_image(&config(&base, "my-ocr", false), &image, false).unwrap_err();
        assert!(err.message.contains("404"), "{}", err.message);
        handle.join().unwrap();
        // Exactly one POST; no /models probe.
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }

    /// A default name that 404s on a server with no v2 model reports the served
    /// models instead of leaving the user to guess.
    #[test]
    fn no_v2_available_reports_served_models() {
        let (base, recorded, handle) = scripted_backend(vec![
            (404, r#"{"error":{"message":"no such model"}}"#.into()),
            (200, json!({"data": [{"id": "some-other-ocr"}]}).to_string()),
        ]);
        let (_dir, image) = temp_image();
        let err =
            recognize_image(&config(&base, "Unlimited-OCR", true), &image, false).unwrap_err();
        assert!(err.message.contains("404"), "{}", err.message);
        assert!(
            err.message.contains("some-other-ocr"),
            "message should list served models: {}",
            err.message
        );
        handle.join().unwrap();
        assert_eq!(recorded.lock().unwrap().len(), 2, "no retry POST");
    }

    /// A successful request with the default name makes no extra round trip.
    #[test]
    fn successful_default_model_does_not_probe_models() {
        let (base, recorded, handle) = scripted_backend(vec![(200, ocr_reply("plain text"))]);
        let (_dir, image) = temp_image();
        let markdown =
            recognize_image(&config(&base, "Unlimited-OCR", true), &image, false).unwrap();
        assert_eq!(markdown, "plain text");
        handle.join().unwrap();
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }

    #[test]
    fn unlimited_ocr2_detected_from_served_model_name() {
        for name in [
            "Unlimited-OCR-2",
            "baidu/Unlimited-OCR-2",
            "mlx-community/Unlimited-OCR-2-4bit",
            "unlimited_ocr_2",
        ] {
            assert!(is_unlimited_ocr2(name), "{name}");
        }
        for name in [
            "Unlimited-OCR",
            "mlx-community/Unlimited-OCR-8bit",
            "served",
        ] {
            assert!(!is_unlimited_ocr2(name), "{name}");
        }
    }
}
