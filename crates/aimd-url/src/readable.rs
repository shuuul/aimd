//! Readable HTML extraction (the `readable` task).
//!
//! HTTP(S) pages go to Firecrawl Local by default (`firecrawl-local scrape
//! <url>`, program from `AIMD_FIRECRAWL`). `AIMD_READABLE_BACKEND=defuddle`
//! switches to `npx defuddle parse` (npx command from `AIMD_NPX`) for URLs
//! and local files. Firecrawl Local only fetches URLs, so with the default
//! backend local `.html`/`.htm` files go through the `dom_smoothie`
//! readability crate.

use std::ffi::OsString;
use std::io::ErrorKind as IoErrorKind;
use std::process::Command;

use dom_smoothie::{Config, Readability, TextMode};
use serde_json::Value;

use aimd_core::{AimdError, Result};

/// Readable content plus page metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadableHtml {
    pub markdown: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub source: Option<String>,
}

/// Which extractor handles readable HTML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadableBackend {
    /// `firecrawl-local scrape <url> --format markdown` for URLs;
    /// `dom_smoothie` for local files.
    Firecrawl { program: OsString },
    /// `npx defuddle parse <source> --json --markdown`.
    Defuddle { npx: OsString },
}

fn env_program(key: &str, default: &str) -> OsString {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.into())
}

impl ReadableBackend {
    /// `AIMD_READABLE_BACKEND` (`firecrawl` default, `defuddle`), with
    /// `AIMD_FIRECRAWL` / `AIMD_NPX` program overrides.
    pub fn from_env() -> Self {
        match std::env::var("AIMD_READABLE_BACKEND") {
            Ok(v) if v.trim().eq_ignore_ascii_case("defuddle") => ReadableBackend::Defuddle {
                npx: env_program("AIMD_NPX", "npx"),
            },
            _ => ReadableBackend::Firecrawl {
                program: env_program("AIMD_FIRECRAWL", "firecrawl-local"),
            },
        }
    }
}

fn is_http_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

fn spawn_error(tool: &str, program: &OsString, err: &std::io::Error, hint: &str) -> AimdError {
    if err.kind() == IoErrorKind::NotFound {
        AimdError::backend_unavailable(format!(
            "{tool} is required for readable HTML extraction ({} not found){hint}",
            program.to_string_lossy()
        ))
    } else {
        AimdError::backend_unavailable(format!(
            "failed to run {tool} ({}): {err}",
            program.to_string_lossy()
        ))
    }
}

/// Readability over an HTML string; `url` resolves relative links.
pub fn readable_from_html(html: &str, url: Option<&str>) -> Result<ReadableHtml> {
    let cfg = Config {
        text_mode: TextMode::Markdown,
        ..Config::default()
    };
    let mut readability = Readability::new(html, url.filter(|u| is_http_url(u)), Some(cfg))
        .map_err(|err| AimdError::failed(format!("readability failed: {err}")))?;
    let article = readability
        .parse()
        .map_err(|err| AimdError::failed(format!("readability failed: {err}")))?;
    let title = Some(article.title.trim().to_string()).filter(|t| !t.is_empty());
    Ok(ReadableHtml {
        markdown: article.text_content.trim().to_string(),
        title,
        author: article.byline.filter(|b| !b.trim().is_empty()),
        source: url.map(str::to_string),
    })
}

fn firecrawl(url: &str, program: &OsString) -> Result<ReadableHtml> {
    let output = Command::new(program)
        .args(["scrape", url, "--format", "markdown"])
        .output()
        .map_err(|err| {
            spawn_error(
                "Firecrawl Local",
                program,
                &err,
                " (brew install shuuul/tap/firecrawl-local)",
            )
        })?;
    // The JSON envelope is printed on failure too; stderr only matters when
    // stdout is not JSON (e.g. an argument error).
    let Ok(payload) = serde_json::from_slice::<Value>(&output.stdout) else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AimdError::failed(format!(
            "firecrawl-local failed to scrape {url}: {}",
            stderr.trim()
        )));
    };
    parse_firecrawl(url, &payload)
}

/// `firecrawl-local scrape` JSON for one URL:
/// `{"success":true,"data":{"markdown",...,"metadata":{...},"warning"?}}` or
/// `{"success":false,"error":...}`.
fn parse_firecrawl(url: &str, payload: &Value) -> Result<ReadableHtml> {
    if payload.get("success").and_then(Value::as_bool) != Some(true) {
        let error = payload
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(AimdError::failed(format!(
            "firecrawl-local failed to scrape {url}: {error}"
        )));
    }
    let data = &payload["data"];
    let markdown = data
        .get("markdown")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if markdown.is_empty() {
        let reason = data
            .get("warning")
            .and_then(Value::as_str)
            .map(|w| format!(": {w}"))
            .unwrap_or_default();
        return Err(AimdError::failed(format!(
            "firecrawl-local returned no markdown for {url}{reason}"
        )));
    }
    // Metadata values are strings, or arrays when a page repeats a meta tag.
    let meta = |key: &str| {
        let value = data.pointer(&format!("/metadata/{key}"))?;
        let text = match value {
            Value::Array(items) => items.iter().find_map(Value::as_str)?,
            v => v.as_str()?,
        };
        Some(text.trim().to_string()).filter(|t| !t.is_empty())
    };
    Ok(ReadableHtml {
        markdown: markdown.to_string(),
        title: meta("title"),
        author: meta("author"),
        source: meta("url")
            .or_else(|| meta("sourceURL"))
            .or_else(|| Some(url.to_string())),
    })
}

fn defuddle(source: &str, npx: &OsString) -> Result<ReadableHtml> {
    let output = Command::new(npx)
        .args(["defuddle", "parse", source, "--json", "--markdown"])
        .output()
        .map_err(|err| spawn_error("npx", npx, &err, " (Defuddle backend)"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            stderr.trim().to_string()
        };
        return Err(AimdError::failed(format!("defuddle failed: {msg}")));
    }
    let payload: Value = serde_json::from_slice(&output.stdout)
        .map_err(|err| AimdError::failed(format!("defuddle returned invalid JSON: {err}")))?;
    let text = |key: &str| {
        payload
            .get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    Ok(ReadableHtml {
        markdown: text("contentMarkdown")
            .or_else(|| text("content"))
            .unwrap_or_default(),
        title: text("title"),
        author: text("author"),
        source: text("source").or_else(|| text("url")),
    })
}

/// Extract readable Markdown from an HTTP(S) URL or a local HTML file.
pub fn extract_readable_html(source: &str, backend: &ReadableBackend) -> Result<ReadableHtml> {
    match backend {
        ReadableBackend::Defuddle { npx } => defuddle(source, npx),
        ReadableBackend::DomSmoothie => {
            if is_http_url(source) {
                let html = fetch_html(source, curl)?;
                readable_from_html(&html, Some(source))
            } else {
                let bytes = std::fs::read(source).map_err(|err| {
                    if err.kind() == IoErrorKind::NotFound {
                        AimdError::not_found(format!("Input not found: {source}"))
                    } else {
                        AimdError::failed(format!("Cannot read {source}: {err}"))
                    }
                })?;
                let mut out = readable_from_html(&String::from_utf8_lossy(&bytes), None)?;
                out.source = Some(source.to_string());
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn dom_smoothie_extracts_article_markdown() {
        let html = "<html><head><title>Test Page</title></head><body><nav>Home | About</nav><article><h1>Hello World</h1><p>This is a paragraph with enough text to be considered the main content of the page. It keeps going for a while so that readability heuristics accept it as an article.</p><p>Second paragraph with <a href=\"/x\">a link</a> and <strong>bold</strong> text.</p></article><footer>Copyright</footer></body></html>";
        let out = readable_from_html(html, Some("https://example.com/post")).unwrap();
        assert_eq!(out.title.as_deref(), Some("Test Page"));
        assert!(
            out.markdown.contains("This is a paragraph"),
            "{}",
            out.markdown
        );
        assert!(!out.markdown.contains("Copyright"), "{}", out.markdown);
    }

    #[test]
    fn firecrawl_envelope_maps_markdown_and_metadata() {
        let payload = json!({"success": true, "data": {
            "markdown": "# Hi\n\nBody\n",
            "metadata": {
                "title": ["Page Title", "Dup"],
                "author": "Ada",
                "sourceURL": "https://a.example/x",
                "url": "https://a.example/final"
            }
        }});
        let out = parse_firecrawl("https://a.example/x", &payload).unwrap();
        assert_eq!(out.markdown, "# Hi\n\nBody");
        assert_eq!(out.title.as_deref(), Some("Page Title"));
        assert_eq!(out.author.as_deref(), Some("Ada"));
        assert_eq!(out.source.as_deref(), Some("https://a.example/final"));
    }

    #[test]
    fn firecrawl_failures_and_blocked_pages_are_errors() {
        let failed = json!({"success": false, "url": "u", "code": "X", "error": "DNS failure"});
        let err = parse_firecrawl("u", &failed).unwrap_err();
        assert!(err.message.contains("DNS failure"), "{}", err.message);

        let blocked = json!({"success": true, "data": {
            "markdown": "", "blocked": true, "warning": "site blocked the request"
        }});
        let err = parse_firecrawl("u", &blocked).unwrap_err();
        assert!(err.message.contains("site blocked"), "{}", err.message);
    }
}
