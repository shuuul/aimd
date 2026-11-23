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
pub(crate) mod parse;
mod readable;
mod subtitles;
mod xml;
mod ytdlp;

use std::path::{Path, PathBuf};

use crate::anchors;
use crate::error::{AimdError, Result};
use crate::hooks::ProcessHooks;
use crate::model::{ProcessInput, Segment};
use crate::sys::System;
use crate::util::pytext;

pub(crate) use markdown::markitdown_normalize;
use markdown::{format_content, strip_subtitle_formatting};
use metadata::{build_metadata_context, extract_video_info};
use readable::{ReadableBackend, extract_readable_html};
use ytdlp::YtDlp;

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
    /// Ask for timed sentences ([`AudioTranscriber::transcribe_timed`]):
    /// `ProcessInput::wants_positions`.
    pub anchors: bool,
}

/// A transcript with optional timed sentences (spec 005).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimedText {
    pub text: String,
    pub sentences: Option<Vec<Segment>>,
}

/// Transcribes downloaded URL audio.
pub(crate) trait AudioTranscriber: Send + Sync {
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

/// `urlparse(value)`: scheme is http/https and the netloc is non-empty.
#[cfg(test)]
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
pub(crate) fn convert_url(
    system: &dyn System,
    input: &ProcessInput,
    transcriber: &dyn AudioTranscriber,
    hooks: &dyn ProcessHooks,
) -> Result<UrlOutput> {
    convert_url_with(input, &YtDlp::from_env(system), transcriber, hooks)
}

/// [`convert_url`] with explicit tool configuration.
pub(crate) fn convert_url_with(
    input: &ProcessInput,
    ytdlp: &YtDlp<'_>,
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
    let video = extract_video_info(ytdlp, url, &platform, cookies_file, cookies_from_browser)?;
    let info = &video.info;
    let title = match info.get("title").filter(|v| !v.is_null()) {
        Some(v) => markdown::py_str(v),
        None => "Unknown Title".to_string(),
    };
    let language = input.language.as_deref().filter(|l| !l.is_empty());
    let work = crate::util::tempdir::TempDir::new(input.temp_dir.as_deref(), "aimd-url-")?;
    let subtitle = subtitles::extract_subtitles(
        ytdlp,
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
        let sentences = (input.wants_positions() && !input.raw_transcript)
            .then(|| anchors::sentences_from_cues(&anchors::parse_srt(&sub)))
            .filter(|s| !s.is_empty());
        let body = match &sentences {
            _ if input.raw_transcript => sub,
            Some(sentences) if input.anchors => anchors::timed_paragraphs(sentences),
            _ => strip_subtitle_formatting(&sub),
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
        anchors: input.wants_positions(),
    };
    let fallback = audio::AudioFallback {
        ytdlp,
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
        let sentences = timed.sentences.filter(|s| !s.is_empty());
        let body = match &sentences {
            Some(sentences) if input.anchors => anchors::timed_paragraphs(sentences),
            _ => timed.text,
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

/// Readable HTML for a URL or local `.html` file (`title` falls back to
/// `""`).
pub(crate) fn convert_readable_html(system: &dyn System, source: &str) -> Result<UrlOutput> {
    let page = extract_readable_html(system, source, &ReadableBackend::from_env(system))?;
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

#[cfg(test)]
#[path = "../../tests/url/e2e.rs"]
mod e2e;

#[cfg(test)]
#[path = "../../tests/url/golden_parity.rs"]
mod golden_parity;

#[cfg(test)]
#[path = "../../tests/url/url_pipeline.rs"]
mod url_pipeline;
