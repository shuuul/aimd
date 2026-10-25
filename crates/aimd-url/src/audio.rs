//! Audio fallback: yt-dlp download with format strategies, then
//! transcription through [`AudioTranscriber`] (ports
//! `aimd.plugins.url.audio`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde_json::Value;

use aimd_core::{AimdError, ProcessHooks, Result};

use crate::cookies::{
    AUTH_REQUIRED_PLATFORMS, build_cookie_sources, is_auth_required_error,
    is_cookie_source_unavailable_error, is_keyring_error,
};
use crate::tempdir::TempDir;
use crate::ytdlp::{self, YtDlp};
use crate::{AudioTranscriber, TranscribeRequest, detect_platform};

const MAX_FILENAME_BYTES: usize = 200;

/// Truncate to at most `max_bytes` UTF-8 bytes without splitting a character.
pub fn truncate_to_bytes(name: &str, max_bytes: usize) -> &str {
    if name.len() <= max_bytes {
        return name;
    }
    let mut end = 0;
    for (i, c) in name.char_indices() {
        if i + c.len_utf8() > max_bytes {
            break;
        }
        end = i + c.len_utf8();
    }
    &name[..end]
}

/// One `(format selector, ffmpeg codec, description)` attempt.
pub type FormatStrategy = (&'static str, Option<&'static str>, &'static str);

/// Platform-specific yt-dlp format fallbacks (audio-only first).
pub fn format_strategies(platform: &str) -> &'static [FormatStrategy] {
    match platform {
        "bilibili" => &[
            ("bestaudio", None, "best audio-only stream"),
            ("bestaudio[ext=m4a]", Some("m4a"), "best m4a audio stream"),
            (
                "best[acodec!=none]",
                Some("m4a"),
                "best format with audio track",
            ),
            ("best", Some("m4a"), "best combined format"),
        ],
        "xiaoyuzhoufm" => &[
            ("best", None, "best available format"),
            ("bestaudio", None, "best audio-only stream"),
        ],
        _ => &[
            (
                "bestaudio[acodec^=opus][abr<=128]/bestaudio[abr<=128]/bestaudio",
                None,
                "audio-only stream (prefer opus <=128kbps)",
            ),
            (
                "bestaudio[ext=m4a]/bestaudio",
                Some("m4a"),
                "audio-only m4a stream",
            ),
            (
                "best[acodec!=none]",
                Some("m4a"),
                "best format with audio track",
            ),
            ("best", Some("m4a"), "best combined format"),
        ],
    }
}

/// Output filename stem: the `save_original` stem when it names a file,
/// else the sanitized title, else `audio_<id>`.
pub fn audio_filename(info: &Value, save_original: Option<&Path>) -> String {
    if let Some(path) = save_original
        && path.extension().is_some()
        && let Some(stem) = path.file_stem()
    {
        return stem.to_string_lossy().into_owned();
    }
    let id = info
        .get("id")
        .filter(|v| !v.is_null())
        .map_or("unknown".to_string(), crate::markdown::py_str);
    let title = info.get("title").and_then(Value::as_str).unwrap_or("");
    if title.is_empty() {
        return format!("audio_{id}");
    }
    let safe: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let truncated = truncate_to_bytes(crate::pytext::strip(&safe), MAX_FILENAME_BYTES);
    if truncated.is_empty() {
        format!("audio_{id}")
    } else {
        truncated.to_string()
    }
}

fn find_output(dir: &Path, stem: &str) -> Option<PathBuf> {
    let mut by_stem = Vec::new();
    let mut fallback = Vec::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.ends_with(".part") || name.ends_with(".ytdl") {
            continue;
        }
        if name.starts_with(&format!("{stem}.")) {
            by_stem.push(path);
        } else if name.starts_with("audio_") && name[6..].contains('.') {
            fallback.push(path);
        }
    }
    by_stem.sort();
    fallback.sort();
    by_stem
        .into_iter()
        .next()
        .or_else(|| fallback.into_iter().next())
}

/// `_try_download_with_format`: one format strategy across the cookie
/// sources. A later source runs only after the previous one failed.
#[allow(clippy::too_many_arguments)]
pub fn try_download_with_format(
    ytdlp: &YtDlp,
    url: &str,
    download_dir: &Path,
    filename: &str,
    format_selector: &str,
    preferred_codec: Option<&str>,
    platform: &str,
    cookies_file: Option<&Path>,
    cookies_from_browser: Option<&str>,
) -> Result<Option<PathBuf>> {
    let sources = build_cookie_sources(platform, cookies_file, cookies_from_browser)?;
    let template = download_dir.join(format!("{}.%(ext)s", filename.replace('%', "%%")));

    let mut auth_required_seen = false;
    let mut cookie_source_issue_seen = false;
    let mut succeeded = false;
    for source in &sources {
        let mut args = ytdlp.base_args(platform);
        args.push("-f".into());
        args.push(format_selector.into());
        args.push("-o".into());
        args.push(template.clone().into_os_string());
        if let Some(codec) = preferred_codec {
            for a in ["-x", "--audio-format", codec, "--audio-quality", "192"] {
                args.push(OsString::from(a));
            }
        }
        args.extend(ytdlp::cookie_args(source));
        args.push("--".into());
        args.push(url.into());
        let run = ytdlp.run(&args)?;
        if run.success {
            succeeded = true;
            break;
        }
        let message = run.error_message();
        if is_auth_required_error(&message) {
            auth_required_seen = true;
        } else if is_keyring_error(&message) || is_cookie_source_unavailable_error(&message) {
            cookie_source_issue_seen = true;
        }
    }

    // Python raised this even when a later source succeeded; only raise when
    // every source failed (see crate docs, "Deviations").
    if !succeeded
        && AUTH_REQUIRED_PLATFORMS.contains(&platform)
        && (auth_required_seen || cookie_source_issue_seen)
    {
        return Err(AimdError::failed(
            "Authenticated cookies are required for this download. Provide --cookies or --cookies-from-browser.",
        ));
    }
    Ok(find_output(download_dir, filename))
}

/// `download_audio`: walk the format strategies until a file appears.
/// Domain errors propagate; other failures fall through to the next strategy.
pub fn download_audio(
    ytdlp: &YtDlp,
    info: &Value,
    url: &str,
    download_dir: &Path,
    save_original: Option<&Path>,
    cookies_file: Option<&Path>,
    cookies_from_browser: Option<&str>,
) -> Result<Option<PathBuf>> {
    let filename = audio_filename(info, save_original);
    let platform = detect_platform(url);
    for (format_selector, codec, _description) in format_strategies(&platform) {
        let found = try_download_with_format(
            ytdlp,
            url,
            download_dir,
            &filename,
            format_selector,
            *codec,
            &platform,
            cookies_file,
            cookies_from_browser,
        )?;
        if let Some(path) = found
            && path.exists()
        {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

/// Inputs of the audio fallback besides the transcriber.
pub struct AudioFallback<'a> {
    pub ytdlp: &'a YtDlp,
    pub info: &'a Value,
    pub url: &'a str,
    pub save_original: Option<&'a Path>,
    pub cookies_file: Option<&'a Path>,
    pub cookies_from_browser: Option<&'a str>,
    pub temp_dir: Option<&'a Path>,
    pub request: TranscribeRequest,
}

fn download_and_transcribe(
    fb: &AudioFallback<'_>,
    download_dir: &Path,
    transcriber: &dyn AudioTranscriber,
    hooks: &dyn ProcessHooks,
) -> Result<Option<String>> {
    let audio = download_audio(
        fb.ytdlp,
        fb.info,
        fb.url,
        download_dir,
        fb.save_original,
        fb.cookies_file,
        fb.cookies_from_browser,
    )?;
    let Some(audio) = audio.filter(|p| p.exists()) else {
        return Ok(None);
    };
    hooks.progress(
        "transcribing",
        None,
        None,
        Some("Transcribing source audio"),
    );
    let text = transcriber.transcribe(&audio, &fb.request, hooks)?;
    if text.chars().count() > 10 {
        Ok(Some(text))
    } else {
        Ok(None)
    }
}

/// `extract_content_from_audio`: download into `save_original`'s directory
/// (kept) or a temporary directory under `temp_dir` (removed afterwards).
pub fn extract_content_from_audio(
    fb: &AudioFallback<'_>,
    transcriber: &dyn AudioTranscriber,
    hooks: &dyn ProcessHooks,
) -> Result<Option<String>> {
    if let Some(save) = fb.save_original {
        let download_dir = if save.is_dir() || (!save.exists() && save.extension().is_none()) {
            save.to_path_buf()
        } else {
            save.parent()
                .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
        };
        let download_dir = if download_dir.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            download_dir
        };
        std::fs::create_dir_all(&download_dir).map_err(|err| {
            AimdError::failed(format!(
                "Cannot create save_original directory {}: {err}",
                download_dir.display()
            ))
        })?;
        return download_and_transcribe(fb, &download_dir, transcriber, hooks);
    }
    let tmp = TempDir::new(fb.temp_dir, "aimd-url-audio")?;
    download_and_transcribe(fb, tmp.path(), transcriber, hooks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn filename_rules() {
        assert_eq!(
            audio_filename(&json!({"title": "a/b: c?", "id": "x"}), None),
            "a_b_ c_"
        );
        assert_eq!(audio_filename(&json!({"id": "x"}), None), "audio_x");
        assert_eq!(
            audio_filename(&json!({"title": "///", "id": 7}), None),
            "___"
        );
        assert_eq!(
            audio_filename(&json!({"title": "t"}), Some(Path::new("/tmp/out/keep.m4a"))),
            "keep"
        );
        let long = "长".repeat(100);
        let name = audio_filename(&json!({"title": long}), None);
        assert!(name.len() <= MAX_FILENAME_BYTES && name.len() == 198);
    }

    #[test]
    fn strategies_prefer_audio_only() {
        assert_eq!(
            format_strategies("youtube")[0],
            (
                "bestaudio[acodec^=opus][abr<=128]/bestaudio[abr<=128]/bestaudio",
                None,
                "audio-only stream (prefer opus <=128kbps)"
            )
        );
        assert_eq!(format_strategies("bilibili")[0].0, "bestaudio");
        assert_eq!(format_strategies("bilibili")[0].1, None);
    }
}
