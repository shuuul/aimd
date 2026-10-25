//! Media metadata extraction and ASR metadata context (ports
//! `aimd.plugins.url.metadata`).

use std::ffi::OsString;
use std::path::Path;

use serde_json::Value;

use aimd_core::{AimdError, Result};

use crate::cookies::{
    AUTH_REQUIRED_PLATFORMS, CookieSource, build_cookie_sources, is_auth_required_error,
    is_cookie_source_unavailable_error, is_invalid_cookie_option_error, is_keyring_error,
    is_unsupported_url_error,
};
use crate::pytext::{prefix_chars, strip};
use crate::ytdlp::{self, YtDlp, with_stale_hint};

const MAX_DESCRIPTION_CHARS: usize = 800;
const MAX_TAG_COUNT: usize = 20;
const MAX_CHAPTER_COUNT: usize = 30;
/// Cap on the injected ASR context, in characters.
pub const MAX_CONTEXT_CHARS: usize = 2000;

fn non_blank(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !strip(s).is_empty())
}

/// Python truthiness for `info.get("uploader") or info.get("channel")`.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// Free-form ASR biasing text from URL metadata (title, author, description,
/// tags, chapters), capped at [`MAX_CONTEXT_CHARS`]. `None` when empty.
pub fn build_metadata_context(info: &Value) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(title) = non_blank(info.get("title")) {
        parts.push(format!("Title: {}", strip(title)));
    }
    let uploader = if truthy(info.get("uploader")) {
        info.get("uploader")
    } else {
        info.get("channel")
    };
    if let Some(author) = non_blank(uploader) {
        parts.push(format!("Author: {}", strip(author)));
    }
    if let Some(description) = non_blank(info.get("description")) {
        parts.push(format!(
            "Description: {}",
            prefix_chars(strip(description), MAX_DESCRIPTION_CHARS)
        ));
    }
    let tags: Vec<&str> = info
        .get("tags")
        .and_then(Value::as_array)
        .map(|tags| {
            tags.iter()
                .filter_map(Value::as_str)
                .map(strip)
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !tags.is_empty() {
        let shown: Vec<&str> = tags.into_iter().take(MAX_TAG_COUNT).collect();
        parts.push(format!("Tags: {}", shown.join(", ")));
    }
    let chapters: Vec<&str> = info
        .get("chapters")
        .and_then(Value::as_array)
        .map(|chapters| {
            chapters
                .iter()
                .filter_map(|c| c.as_object()?.get("title")?.as_str())
                .map(strip)
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if !chapters.is_empty() {
        let shown: Vec<&str> = chapters.into_iter().take(MAX_CHAPTER_COUNT).collect();
        parts.push(format!("Chapters: {}", shown.join("; ")));
    }
    if parts.is_empty() {
        return None;
    }
    let context = format!(
        "The following background information describes the audio being transcribed. Use it to recognize proper nouns, names, and domain terminology accurately.\n{}",
        parts.join("\n")
    );
    Some(prefix_chars(&context, MAX_CONTEXT_CHARS).to_string())
}

/// Metadata plus the cookie source that produced it (Python stored it as
/// `info_dict["_aimd_cookie_source"]`).
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub info: Value,
    /// Raw `--dump-single-json` output, reused by `--load-info-json`.
    pub raw_json: String,
    pub cookie_source: CookieSource,
}

fn info_args(ytdlp: &YtDlp, platform: &str, source: &CookieSource, url: &str) -> Vec<OsString> {
    let mut args = ytdlp.base_args(platform);
    for a in [
        "--dump-single-json",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
    ] {
        args.push(a.into());
    }
    args.extend(ytdlp::cookie_args(source));
    args.push("--".into());
    args.push(url.into());
    args
}

/// `extract_video_info`: try each cookie source in order.
///
/// - unsupported URL -> `UnsupportedInput` immediately;
/// - an explicit cookie option yt-dlp rejects -> `UnsupportedInput` (fail fast);
/// - auth-required platforms (bilibili, xiaohongshu) that saw auth or
///   cookie-source errors -> "Authenticated cookies are required";
/// - otherwise `ProcessingFailed` with yt-dlp's last error (plus the
///   `brew upgrade yt-dlp` hint on stale-extractor signatures).
pub fn extract_video_info(
    ytdlp: &YtDlp,
    url: &str,
    platform: &str,
    cookies_file: Option<&Path>,
    cookies_from_browser: Option<&str>,
) -> Result<VideoInfo> {
    let sources = build_cookie_sources(platform, cookies_file, cookies_from_browser)?;
    if sources.is_empty() {
        return Err(AimdError::failed(
            "No available cookie source configuration",
        ));
    }

    let mut last_error: Option<String> = None;
    let mut auth_required_seen = false;
    let mut cookie_source_issue_seen = false;

    for source in sources {
        let run = ytdlp.run(&info_args(ytdlp, platform, &source, url))?;
        if run.success {
            let parsed: std::result::Result<Value, _> = serde_json::from_str(run.stdout.trim());
            match parsed {
                Ok(info) if truthy(Some(&info)) && info.is_object() => {
                    return Ok(VideoInfo {
                        info,
                        raw_json: run.stdout,
                        cookie_source: source,
                    });
                }
                Ok(_) => continue,
                Err(err) => {
                    last_error = Some(format!("yt-dlp returned invalid JSON: {err}"));
                    continue;
                }
            }
        }
        let message = run.error_message();
        last_error = Some(message.clone());

        if is_unsupported_url_error(&message) {
            return Err(AimdError::unsupported(format!("Unsupported URL: {url}")));
        }
        if source.is_explicit() && is_invalid_cookie_option_error(&message) {
            return Err(AimdError::unsupported(format!(
                "Invalid cookie option ({}): {message}",
                source.name()
            )));
        }
        if is_auth_required_error(&message) {
            auth_required_seen = true;
            if AUTH_REQUIRED_PLATFORMS.contains(&platform) && source == CookieSource::NoCookie {
                break;
            }
            continue;
        }
        if is_keyring_error(&message) || is_cookie_source_unavailable_error(&message) {
            cookie_source_issue_seen = true;
        }
    }

    if AUTH_REQUIRED_PLATFORMS.contains(&platform)
        && (auth_required_seen || cookie_source_issue_seen)
    {
        return Err(AimdError::failed(
            "Authenticated cookies are required for this URL. Provide --cookies (Netscape file) or --cookies-from-browser.",
        ));
    }
    match last_error {
        Some(err) => Err(AimdError::failed(with_stale_hint(&format!(
            "Failed to extract video information: {err}"
        )))),
        None => Err(AimdError::failed("Failed to extract video information")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn metadata_context_includes_core_fields() {
        let info = json!({
            "title": "聊聊 PostgreSQL 与 Redis",
            "uploader": "SomeChannel",
            "description": "本期嘉宾: 张三, 讨论 CUDA 优化。",
            "tags": ["数据库", "PostgreSQL", "Redis"],
            "chapters": [{"title": "开场"}, {"title": "深入 CUDA"}],
        });
        let context = build_metadata_context(&info).unwrap();
        for needle in [
            "Title: 聊聊 PostgreSQL 与 Redis",
            "Author: SomeChannel",
            "本期嘉宾: 张三",
            "Tags: 数据库, PostgreSQL, Redis",
            "Chapters: 开场; 深入 CUDA",
        ] {
            assert!(context.contains(needle), "{needle}");
        }
    }

    #[test]
    fn metadata_context_empty_and_truncated() {
        assert_eq!(build_metadata_context(&json!({})), None);
        assert_eq!(
            build_metadata_context(&json!({"title": "  ", "tags": [], "chapters": null})),
            None
        );
        let long = build_metadata_context(&json!({"title": "t", "description": "x".repeat(5000)}))
            .unwrap();
        assert!(long.chars().count() <= MAX_CONTEXT_CHARS);
        let channel = build_metadata_context(&json!({"uploader": "", "channel": "Chan"})).unwrap();
        assert!(channel.ends_with("Author: Chan"));
    }
}
