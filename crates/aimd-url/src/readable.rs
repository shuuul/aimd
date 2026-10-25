//! Opt-in readable HTML extraction (ports `aimd.plugins.url.defuddle` and the
//! `AimdReadableHtmlConverter` trigger).
//!
//! The default backend is the `dom_smoothie` readability crate. The old
//! `npx defuddle parse` path stays as an escape hatch:
//! `AIMD_READABLE_BACKEND=defuddle` (npx command from `AIMD_NPX`, default
//! `npx`). Remote pages are fetched with `curl` (`AIMD_CURL` override).

use std::ffi::OsString;
use std::io::ErrorKind as IoErrorKind;
use std::path::Path;
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
    DomSmoothie,
    /// `npx defuddle parse <source> --json --markdown`.
    Defuddle {
        npx: OsString,
    },
}

impl ReadableBackend {
    pub fn from_env() -> Self {
        match std::env::var("AIMD_READABLE_BACKEND") {
            Ok(v) if v.trim().eq_ignore_ascii_case("defuddle") => ReadableBackend::Defuddle {
                npx: std::env::var_os("AIMD_NPX")
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| "npx".into()),
            },
            _ => ReadableBackend::DomSmoothie,
        }
    }
}

fn is_http_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

/// The opt-in trigger of `AimdReadableHtmlConverter.accepts`: requested via
/// `task_type == "readable_html"` or the `defuddle` flag, and the source is
/// an HTTP(S) URL or a local `.html`/`.htm` file.
pub fn readable_html_requested(task_type: Option<&str>, defuddle: bool, source: &str) -> bool {
    if task_type != Some("readable_html") && !defuddle {
        return false;
    }
    if is_http_url(source) {
        return true;
    }
    Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm"))
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

/// Fetch a page body with curl.
pub fn fetch_html(url: &str, curl: &OsString) -> Result<String> {
    let output = Command::new(curl)
        .args([
            "-fsSL",
            "--compressed",
            "--max-time",
            "60",
            "-A",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
            "--",
            url,
        ])
        .output()
        .map_err(|err| spawn_error("curl", curl, &err, ""))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AimdError::failed(format!(
            "failed to fetch {url}: {}",
            stderr.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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
pub fn extract_readable_html(
    source: &str,
    backend: &ReadableBackend,
    curl: &OsString,
) -> Result<ReadableHtml> {
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
    use super::*;

    #[test]
    fn trigger_is_opt_in() {
        assert!(!readable_html_requested(
            Some("transcript"),
            false,
            "https://x.com"
        ));
        assert!(readable_html_requested(
            Some("readable_html"),
            false,
            "https://x.com"
        ));
        assert!(readable_html_requested(None, true, "page.HTM"));
        assert!(!readable_html_requested(None, true, "doc.pdf"));
    }

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
}
