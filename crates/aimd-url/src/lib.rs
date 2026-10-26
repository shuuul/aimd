//! URL extraction pipeline (WS-18). Ports `aimd.plugins.url`.
//!
//! [`convert_url`] reproduces `get_text_from_url`:
//!
//! 1. `extracting`: `yt-dlp --dump-single-json` across the cookie sources
//!    (explicit `--cookies` file, explicit `--cookies-from-browser`, else the
//!    automatic browser probe `chrome:default, chromium, brave, edge, vivaldi,
//!    opera, firefox, safari, whale`; then `no-cookie` unless the platform is
//!    bilibili/xiaohongshu).
//! 2. Subtitles first (manual before auto, `*-orig` first by default,
//!    translation filtering by inferred language, json3/ttml/vtt/srv1/srt
//!    format order, downloaded by yt-dlp with the metadata's cookie source).
//!    Cancellation is checked after the lookup. Unless `raw_transcript`, the
//!    subtitle text goes through [`strip_subtitle_formatting`].
//! 3. `downloading` + `transcribing`: audio download through the format
//!    strategies, then [`AudioTranscriber`] with the inferred language and
//!    the ASR context (explicit `context`, plus [`build_metadata_context`]
//!    when `metadata_context` is on: `"{context}\n\n{metadata}"`).
//! 4. Both empty -> `ProcessingFailed` (never a metadata-only success).
//!
//! Executables: `AIMD_YT_DLP` overrides `yt-dlp` on `PATH` (missing ->
//! `BackendUnavailable` with `brew install yt-dlp`); errors matching known
//! stale-extractor signatures carry a `brew upgrade yt-dlp` hint.
//!
//! # Deviations from the Python plugin
//!
//! - Explicit cookie options are validated up front: a missing or
//!   non-Netscape `--cookies` file, and browser/keyring names yt-dlp rejects,
//!   are `UnsupportedInput` (Python silently fell through to other sources).
//! - Audio download no longer raises "Authenticated cookies are required"
//!   when a later cookie source succeeded after an auth/cookie-db failure.
//! - JSON `null` metadata renders as the missing-field default (yt-dlp's JSON
//!   output never contains nulls; the Python library dict could).
//! - Subtitle fetch retries are yt-dlp's own (`--retries 2`, 0.5s/1.5s
//!   linear sleep, `--socket-timeout 40`) instead of a Python retry loop.

mod audio;
mod cookies;
mod markdown;
mod metadata;
mod pytext;
mod readable;
mod subtitles;
mod tempdir;
mod xml;
mod ytdlp;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use aimd_core::{AimdError, ProcessHooks, ProcessInput, Result, Segment, anchors};

pub use audio::{
    AudioFallback, audio_filename, download_audio, extract_content_from_audio, format_strategies,
    truncate_to_bytes, try_download_with_format,
};
pub use cookies::{
    AUTH_REQUIRED_PLATFORMS, BrowserSpec, CookieSource, DEFAULT_BROWSER_SPECS, NetscapeCookie,
    build_cookie_sources, is_auth_required_error, is_cookie_source_unavailable_error,
    is_keyring_error, is_unsupported_url_error, parse_cookies_from_browser, parse_netscape_cookies,
    validate_cookie_file,
};
pub use markdown::{
    format_content, format_linear_transcript, markitdown_normalize, merge_rolling_caption_lines,
    strip_subtitle_formatting,
};
pub use metadata::{MAX_CONTEXT_CHARS, VideoInfo, build_metadata_context, extract_video_info};
pub use readable::{
    ReadableBackend, ReadableHtml, extract_readable_html, readable_from_html,
    readable_html_requested,
};
pub use subtitles::{
    SubtitleCandidate, detect_content_language, extract_subtitles, get_preferred_languages,
    json3_to_srt, normalize_metadata_language, normalize_subtitle_payload,
    resolve_subtitle_language, srv1_to_srt, srv3_to_srt, subtitle_candidates, subtitle_urls,
};
pub use ytdlp::{YT_DLP_INSTALL_HINT, YtDlp, with_stale_hint};

/// What the audio fallback hands to the transcriber (`transcribe_file`
/// keyword arguments in Python).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranscribeRequest {
    /// Explicit language, else inferred from metadata/title/description.
    pub language: Option<String>,
    /// Already merged explicit + metadata context.
    pub context: Option<String>,
    pub model: Option<String>,
    pub precision: Option<String>,
    /// `ProcessInput.temp_dir` (ASR scratch files).
    pub temp_dir: Option<PathBuf>,
    /// Remote OpenAI-compatible ASR (`asr_base_url`/`asr_model`/`asr_api_key`).
    pub asr_base_url: Option<String>,
    pub asr_model: Option<String>,
    pub asr_api_key: Option<String>,
    /// `ProcessInput.anchors`: ask for timed sentences
    /// ([`AudioTranscriber::transcribe_timed`]).
    pub anchors: bool,
}

/// A transcript with optional timed sentences (spec 005).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimedText {
    pub text: String,
    pub sentences: Option<Vec<Segment>>,
}

/// Transcribes downloaded URL audio. Implemented by `aimd-process` (WS-16).
pub trait AudioTranscriber: Send + Sync {
    fn transcribe(
        &self,
        audio_path: &Path,
        request: &TranscribeRequest,
        hooks: &dyn ProcessHooks,
    ) -> Result<String>;

    /// [`Self::transcribe`] plus timed sentences when the backend has them;
    /// the default returns the plain transcript.
    fn transcribe_timed(
        &self,
        audio_path: &Path,
        request: &TranscribeRequest,
        hooks: &dyn ProcessHooks,
    ) -> Result<TimedText> {
        Ok(TimedText {
            text: self.transcribe(audio_path, request, hooks)?,
            sentences: None,
        })
    }
}

/// Markdown extracted from a URL.
#[derive(Debug, Clone, PartialEq)]
pub struct UrlOutput {
    pub markdown: String,
    pub title: String,
    pub platform: String,
    /// Timed transcript sentences (subtitle cues or aligned ASR) when
    /// anchors were requested.
    pub segments: Option<Vec<Segment>>,
}

/// External tools used by the URL pipeline.
#[derive(Debug, Clone)]
pub struct UrlConfig {
    pub ytdlp: YtDlp,
    pub readable_backend: ReadableBackend,
    /// `curl` for readable-HTML page fetches.
    pub curl: OsString,
}

impl UrlConfig {
    /// `AIMD_YT_DLP`, `AIMD_READABLE_BACKEND`/`AIMD_NPX`, `AIMD_CURL`.
    pub fn from_env() -> Self {
        Self {
            ytdlp: YtDlp::from_env(),
            readable_backend: ReadableBackend::from_env(),
            curl: std::env::var_os("AIMD_CURL")
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "curl".into()),
        }
    }
}

/// `urlparse(value)`: scheme is http/https and the netloc is non-empty.
pub fn is_url(value: &str) -> bool {
    let Some((scheme, rest)) = value.split_once(':') else {
        return false;
    };
    let valid_scheme = scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !valid_scheme {
        return false;
    }
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return false;
    }
    let Some(after) = rest.strip_prefix("//") else {
        return false;
    };
    let netloc = after.split(['/', '?', '#']).next().unwrap_or("");
    // urlparse raises ValueError on unbalanced IPv6 brackets.
    if netloc.contains('[') != netloc.contains(']') {
        return false;
    }
    !netloc.is_empty()
}

/// Transcript platform key from a URL.
pub fn detect_platform(url: &str) -> String {
    let lower = url.to_lowercase();
    let platform = if lower.contains("youtube.com") || lower.contains("youtu.be") {
        "youtube"
    } else if lower.contains("bilibili.com") {
        "bilibili"
    } else if lower.contains("xiaohongshu.com") || lower.contains("xhslink.com") {
        "xiaohongshu"
    } else if lower.contains("xiaoyuzhoufm.com") {
        "xiaoyuzhoufm"
    } else {
        "unknown"
    };
    platform.to_string()
}

/// Merge explicit and metadata context as `get_text_from_url` does.
pub fn merge_asr_context(
    context: Option<&str>,
    metadata_context: bool,
    info: &serde_json::Value,
) -> Option<String> {
    if metadata_context && let Some(built) = build_metadata_context(info) {
        return Some(match context {
            Some(c) if !c.is_empty() => format!("{c}\n\n{built}"),
            _ => built,
        });
    }
    context.map(str::to_string)
}

/// URL -> Markdown with tools resolved from the environment.
pub fn convert_url(
    input: &ProcessInput,
    transcriber: &dyn AudioTranscriber,
    hooks: &dyn ProcessHooks,
) -> Result<UrlOutput> {
    convert_url_with(input, &UrlConfig::from_env(), transcriber, hooks)
}

/// [`convert_url`] with explicit tool configuration.
pub fn convert_url_with(
    input: &ProcessInput,
    config: &UrlConfig,
    transcriber: &dyn AudioTranscriber,
    hooks: &dyn ProcessHooks,
) -> Result<UrlOutput> {
    let url = input.input_source.as_str();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(AimdError::unsupported(format!("Invalid URL: {url}")));
    }
    hooks.progress("extracting", None, None, Some("Extracting URL metadata"));

    let platform = detect_platform(url);
    let cookies_file = input.cookies.as_deref();
    let cookies_from_browser = input.cookies_from_browser.as_deref();
    let video = extract_video_info(
        &config.ytdlp,
        url,
        &platform,
        cookies_file,
        cookies_from_browser,
    )?;
    let info = &video.info;
    let title = match info.get("title").filter(|v| !v.is_null()) {
        Some(v) => markdown::py_str(v),
        None => "Unknown Title".to_string(),
    };
    let language = input.language.as_deref().filter(|l| !l.is_empty());
    let work = tempdir::TempDir::new(input.temp_dir.as_deref(), "aimd-url")?;
    let subtitle = subtitles::extract_subtitles(
        &config.ytdlp,
        info,
        &video.raw_json,
        &platform,
        language,
        Some(&video.cookie_source),
        work.path(),
    )?;
    if hooks.is_cancelled() {
        return Err(AimdError::cancelled(
            "URL processing cancelled after subtitle lookup",
        ));
    }
    if let Some(sub) = subtitle.filter(|s| !pytext::strip(s).is_empty()) {
        let sentences = (input.anchors && !input.raw_transcript)
            .then(|| anchors::sentences_from_cues(&anchors::parse_srt(&sub)))
            .filter(|s| !s.is_empty());
        let body = match &sentences {
            _ if input.raw_transcript => sub,
            Some(sentences) => anchors::timed_paragraphs(sentences),
            None => strip_subtitle_formatting(&sub),
        };
        return Ok(UrlOutput {
            markdown: markitdown_normalize(&format_content(info, Some(&body), Some(&platform))),
            title,
            platform,
            segments: sentences,
        });
    }

    hooks.progress("downloading", None, None, Some("Downloading source audio"));
    // Only an explicit language forces ASR; otherwise Qwen3-ASR detects it
    // (metadata/title heuristics only steer subtitle selection).
    let request = TranscribeRequest {
        language: language.map(str::to_string),
        context: merge_asr_context(input.context.as_deref(), input.metadata_context, info),
        model: input.model.clone(),
        precision: input.precision.clone(),
        temp_dir: input.temp_dir.clone(),
        asr_base_url: input.asr_base_url.clone(),
        asr_model: input.asr_model.clone(),
        asr_api_key: input.asr_api_key.clone(),
        anchors: input.anchors,
    };
    let fallback = audio::AudioFallback {
        ytdlp: &config.ytdlp,
        info,
        url,
        save_original: input.save_original.as_deref(),
        cookies_file,
        cookies_from_browser,
        temp_dir: input.temp_dir.as_deref(),
        request,
    };
    let audio_text = audio::extract_content_from_audio(&fallback, transcriber, hooks)?;
    if let Some(timed) = audio_text.filter(|t| !pytext::strip(&t.text).is_empty()) {
        let sentences = timed.sentences.filter(|s| input.anchors && !s.is_empty());
        let body = match &sentences {
            Some(sentences) => anchors::timed_paragraphs(sentences),
            None => timed.text,
        };
        return Ok(UrlOutput {
            markdown: markitdown_normalize(&format_content(info, Some(&body), Some(&platform))),
            title,
            platform,
            segments: sentences,
        });
    }
    Err(AimdError::failed(
        "Failed to extract subtitles or transcribe audio for URL",
    ))
}

/// Readable HTML for a URL or local `.html` file, shaped like the Python
/// converter result (`title` falls back to `""`).
pub fn convert_readable_html(source: &str, config: &UrlConfig) -> Result<UrlOutput> {
    let page = extract_readable_html(source, &config.readable_backend, &config.curl)?;
    Ok(UrlOutput {
        markdown: markitdown_normalize(&page.markdown),
        title: page.title.unwrap_or_default(),
        platform: detect_platform(source),
        segments: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_url_matches_urlparse() {
        assert!(is_url("https://youtube.com/watch?v=x"));
        assert!(is_url("HTTP://example.com"));
        assert!(!is_url("ftp://example.com"));
        assert!(!is_url("https://"));
        assert!(!is_url("example.com/video"));
        assert!(!is_url("http://[::1"));
        assert!(!is_url("/local/file.mp3"));
    }

    #[test]
    fn detect_platform_keys() {
        assert_eq!(detect_platform("https://youtu.be/x"), "youtube");
        assert_eq!(
            detect_platform("https://www.bilibili.com/video/BV1"),
            "bilibili"
        );
        assert_eq!(detect_platform("https://xhslink.com/a"), "xiaohongshu");
        assert_eq!(
            detect_platform("https://www.xiaoyuzhoufm.com/episode/1"),
            "xiaoyuzhoufm"
        );
        assert_eq!(detect_platform("https://example.com/video"), "unknown");
    }

    #[test]
    fn asr_context_merging() {
        let info = serde_json::json!({"title": "Interview with Linus Torvalds"});
        let merged = merge_asr_context(Some("Vocabulary: Linus, Git."), true, &info).unwrap();
        assert!(merged.starts_with("Vocabulary: Linus, Git.\n\nThe following background"));
        assert_eq!(
            merge_asr_context(Some("x"), false, &info).as_deref(),
            Some("x")
        );
        assert_eq!(merge_asr_context(None, false, &info), None);
        assert!(
            merge_asr_context(Some(""), true, &info)
                .unwrap()
                .starts_with("The following")
        );
        assert_eq!(
            merge_asr_context(Some("c"), true, &serde_json::json!({})).as_deref(),
            Some("c")
        );
    }
}
