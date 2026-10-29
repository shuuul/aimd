//! ffmpeg/ffprobe helpers (ports `aimd.plugins.asr.audio_utils`).

use std::path::{Path, PathBuf};

use aimd_core::ext::{SUPPORTED_AUDIO_FORMATS, suffix_lower};
use aimd_core::{AimdError, Result, log_info};

use crate::Runtime;
use crate::tempdir::{TempDir, TempFile};

fn sorted_supported_formats() -> String {
    let mut formats = SUPPORTED_AUDIO_FORMATS.to_vec();
    formats.sort_unstable();
    formats.join(", ")
}

fn lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Whether the native wav loader reads `path` as-is (16 kHz mono PCM wav).
pub fn is_native_wav(path: &Path) -> bool {
    suffix_lower(path) == ".wav" && aimd_engine::audio::load_wav(path).is_ok()
}

/// `ffmpeg -y -i src -ar 16000 -ac 1 out.wav` into a temporary file.
pub fn convert_to_wav(
    rt: &dyn Runtime,
    source: &Path,
    temp_dir: Option<&Path>,
) -> Result<TempFile> {
    let suffix = suffix_lower(source);
    let Some(ffmpeg) = rt.which("ffmpeg") else {
        return Err(AimdError::backend_unavailable(format!(
            "Cannot convert {suffix} to WAV: ffmpeg not found. Install ffmpeg (`brew install \
             ffmpeg`) or provide a supported format ({}).",
            sorted_supported_formats()
        )));
    };
    let wav = TempFile::reserve(temp_dir, "aimd-asr-", ".wav")?;
    log_info!("Converting {suffix} to WAV for transcription backend");
    let args = [
        "-y".to_string(),
        "-i".to_string(),
        lossy(source),
        "-ar".to_string(),
        "16000".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        lossy(wav.path()),
    ];
    let out = rt
        .run(&ffmpeg, &args, None)
        .map_err(|e| AimdError::failed(format!("ffmpeg conversion to WAV failed: {e}")))?;
    if !out.success() {
        return Err(AimdError::failed(format!(
            "ffmpeg conversion to WAV failed: {}",
            out.stderr_text()
        )));
    }
    Ok(wav)
}

/// Remote uploads: convert only formats outside `SUPPORTED_AUDIO_FORMATS`.
pub fn convert_to_wav_if_needed(
    rt: &dyn Runtime,
    source: &Path,
    temp_dir: Option<&Path>,
) -> Result<Option<TempFile>> {
    if SUPPORTED_AUDIO_FORMATS.contains(&suffix_lower(source).as_str()) {
        return Ok(None);
    }
    convert_to_wav(rt, source, temp_dir).map(Some)
}

/// Parse `Duration: hh:mm:ss.ss` from ffmpeg stderr.
fn parse_ffmpeg_duration(stderr: &str) -> Option<f64> {
    let mut rest = stderr;
    while let Some(pos) = rest.find("Duration:") {
        let after = rest[pos + "Duration:".len()..].trim_start();
        let token: String = after
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ':' || *c == '.')
            .collect();
        let parts: Vec<&str> = token.split(':').collect();
        if parts.len() >= 3 {
            let (h, m) = (parts[0], parts[1]);
            // Python regex: (\d+):(\d+):(\d+\.\d+)
            let s: String = {
                let sec = parts[2];
                let mut seen_dot = false;
                let mut out = String::new();
                for c in sec.chars() {
                    if c == '.' {
                        if seen_dot {
                            break;
                        }
                        seen_dot = true;
                    }
                    out.push(c);
                }
                out
            };
            let valid = !h.is_empty()
                && !m.is_empty()
                && s.contains('.')
                && !s.starts_with('.')
                && !s.ends_with('.');
            if valid
                && let (Ok(h), Ok(m), Ok(s)) =
                    (h.parse::<u64>(), m.parse::<u64>(), s.parse::<f64>())
            {
                return Some(h as f64 * 3600.0 + m as f64 * 60.0 + s);
            }
        }
        rest = &rest[pos + 1..];
    }
    None
}

/// `get_audio_duration`: ffprobe, then ffmpeg `-i` stderr.
pub fn get_audio_duration(rt: &dyn Runtime, source: &Path) -> Result<f64> {
    if let Some(ffprobe) = rt.which("ffprobe") {
        let args = [
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ]
        .iter()
        .map(|s| s.to_string())
        .chain([lossy(source)])
        .collect::<Vec<_>>();
        if let Ok(out) = rt.run(&ffprobe, &args, None)
            && out.success()
            && let Ok(value) = aimd_core::pyfmt::strip(&out.stdout_text()).parse::<f64>()
        {
            return Ok(value);
        }
    }
    let Some(ffmpeg) = rt.which("ffmpeg") else {
        return Err(AimdError::failed(
            "ffmpeg/ffprobe not found. Cannot determine audio duration.",
        ));
    };
    let out = rt
        .run(&ffmpeg, &["-i".to_string(), lossy(source)], None)
        .map_err(|e| AimdError::failed(e.to_string()))?;
    let stderr = out.stderr_text();
    parse_ffmpeg_duration(&stderr).ok_or_else(|| {
        AimdError::failed(format!(
            "Could not determine audio duration using ffmpeg/ffprobe. stderr: {stderr}"
        ))
    })
}

/// Request-owned segment directory, removed with its files on drop.
#[derive(Debug)]
pub struct Segments {
    pub paths: Vec<PathBuf>,
    _dir: TempDir,
}

fn list_segments(parent: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".wav"))
        })
        .collect();
    found.sort();
    found
}

/// `segment_audio`: 16 kHz mono PCM wav chunks of `segment_secs`.
pub fn segment_audio(
    rt: &dyn Runtime,
    source: &Path,
    segment_secs: f64,
    temp_dir: Option<&Path>,
) -> Result<Segments> {
    let Some(ffmpeg) = rt.which("ffmpeg") else {
        return Err(AimdError::backend_unavailable(
            "ffmpeg not found. Cannot segment audio.",
        ));
    };
    let prefix = format!("seg_{}_", crate::file_stem(source));
    let dir = TempDir::new(temp_dir, "aimd-segments-")?;
    let template = dir.path().join(format!("{prefix}%03d.wav"));
    let args: Vec<String> = [
        "-y".to_string(),
        "-i".to_string(),
        lossy(source),
        "-f".to_string(),
        "segment".to_string(),
        "-segment_time".to_string(),
        format!("{segment_secs:?}"),
        "-c:a".to_string(),
        "pcm_s16le".to_string(),
        "-ar".to_string(),
        "16000".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        lossy(&template),
    ]
    .to_vec();
    log_info!("Segmenting audio into {segment_secs:?}s chunks");
    let out = rt.run(&ffmpeg, &args, None);
    let failed = match out {
        Ok(ref o) if o.success() => None,
        Ok(o) => Some(o.stderr_text()),
        Err(e) => Some(e.to_string()),
    };
    if let Some(stderr) = failed {
        return Err(AimdError::failed(format!(
            "ffmpeg audio segmentation failed: {stderr}"
        )));
    }
    let paths = list_segments(dir.path(), &prefix);
    if paths.is_empty() {
        return Err(AimdError::failed(
            "ffmpeg segmentation succeeded but no segment files were created.",
        ));
    }
    Ok(Segments { paths, _dir: dir })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_duration_regex() {
        let stderr = "Input #0, wav\n  Duration: 00:12:03.50, bitrate: 256 kb/s";
        assert_eq!(parse_ffmpeg_duration(stderr), Some(723.5));
        assert_eq!(parse_ffmpeg_duration("Duration: N/A"), None);
        assert_eq!(parse_ffmpeg_duration("Duration: 01:00:05, x"), None);
    }
}
