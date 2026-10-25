//! Subtitle language selection, native-caption normalization and download
//! (ports `aimd.plugins.url.subtitles`).
//!
//! Python fetched the chosen track URL through `YoutubeDL.urlopen`; here the
//! same yt-dlp process (cookies, impersonation, socket timeout, same-URL
//! retries) downloads it via `--load-info-json` plus subtitle flags that
//! select exactly one language, one track kind (manual or auto) and one
//! format.

use std::collections::HashSet;
use std::path::Path;

use serde_json::{Map, Value};

use aimd_core::Result;

use crate::cookies::CookieSource;
use crate::pytext::{lower, prefix_chars, strip};
use crate::xml;
use crate::ytdlp::{self, YtDlp};

pub const ENGLISH_SUBTITLE_LANGUAGES: &[&str] = &["en-orig", "en", "en-US", "en-GB", "ai-en"];
pub const CHINESE_SUBTITLE_LANGUAGES: &[&str] = &[
    "zh-orig", "zh-Hans", "zh-Hant", "zh-CN", "zh", "ai-zh", "zh-TW", "zh-HK",
];
pub const FORBIDDEN_SUBTITLE_LANGUAGES: &[&str] = &["danmaku"];

/// Download preference: native json3/srv1 are served directly, converted
/// formats (`fmt=srt`/srv2/srv3) often stall on long ASR tracks.
pub const PREFERRED_SUBTITLE_FORMATS: &[&str] = &["json3", "ttml", "vtt", "srv1", "srt"];
const DEFAULT_CUE_DURATION_MS: i64 = 1000;
const CONTENT_LANGUAGE_SAMPLE_CHARS: usize = 4000;

fn is_original_language(lang: &str) -> bool {
    let n = lower(lang);
    n == "orig" || n.ends_with("-orig") || n.contains("original")
}

fn normalize_lang_code(lang: &str) -> String {
    lower(lang).replace('_', "-")
}

/// zh/en family of a language code.
pub fn language_family(lang: &str) -> Option<&'static str> {
    let n = normalize_lang_code(lang);
    if n == "en" || n == "english" || n.starts_with("en-") || n.starts_with("ai-en") {
        return Some("en");
    }
    if n == "zh" || n == "chinese" || n.starts_with("zh-") || n.starts_with("ai-zh") {
        return Some("zh");
    }
    None
}

/// Normalize a yt-dlp `language` field to `zh`/`en`.
pub fn normalize_metadata_language(language: Option<&str>) -> Option<&'static str> {
    let l = strip(language?);
    if l.is_empty() {
        return None;
    }
    language_family(l)
}

/// Infer zh/en from title/description script signals.
pub fn detect_content_language(texts: &[Option<&str>]) -> Option<&'static str> {
    let combined = texts
        .iter()
        .flatten()
        .map(|t| strip(t))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if combined.is_empty() {
        return None;
    }
    let sample = prefix_chars(&combined, CONTENT_LANGUAGE_SAMPLE_CHARS);
    let cjk = sample
        .chars()
        .filter(|&c| {
            ('\u{3400}'..='\u{4dbf}').contains(&c)
                || ('\u{4e00}'..='\u{9fff}').contains(&c)
                || ('\u{f900}'..='\u{faff}').contains(&c)
        })
        .count();
    let latin = sample.chars().filter(char::is_ascii_alphabetic).count();
    if cjk >= 2 || (cjk == 1 && latin < 20) {
        return Some("zh");
    }
    if latin >= 3 {
        return Some("en");
    }
    None
}

/// Explicit language, else metadata language, else script heuristics.
pub fn resolve_subtitle_language(
    language: Option<&str>,
    title: Option<&str>,
    description: Option<&str>,
    metadata_language: Option<&str>,
) -> Option<String> {
    if let Some(l) = language.filter(|l| !l.is_empty()) {
        return Some(l.to_string());
    }
    if let Some(m) = normalize_metadata_language(metadata_language) {
        return Some(m.to_string());
    }
    detect_content_language(&[title, description]).map(str::to_string)
}

fn dedupe<I: IntoIterator<Item = String>>(items: I) -> Vec<String> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|i| seen.insert(i.clone()))
        .collect()
}

fn ordered_family_matches(available: &[String], family: &str) -> Vec<String> {
    let family_langs: Vec<&String> = available
        .iter()
        .filter(|l| language_family(l) == Some(family))
        .collect();
    let originals: Vec<&String> = family_langs
        .iter()
        .copied()
        .filter(|l| is_original_language(l))
        .collect();
    let non_originals: Vec<&String> = family_langs
        .iter()
        .copied()
        .filter(|l| !is_original_language(l))
        .collect();
    let curated = if family == "en" {
        ENGLISH_SUBTITLE_LANGUAGES
    } else {
        CHINESE_SUBTITLE_LANGUAGES
    };
    let curated_originals: Vec<String> = curated
        .iter()
        .filter(|c| originals.iter().any(|o| o == c))
        .map(|c| c.to_string())
        .collect();
    let curated_non: Vec<String> = curated
        .iter()
        .filter(|c| non_originals.iter().any(|o| o == c))
        .map(|c| c.to_string())
        .collect();
    let other_originals = originals
        .iter()
        .filter(|o| !curated_originals.contains(o))
        .map(|s| s.to_string());
    let other_non = non_originals
        .iter()
        .filter(|o| !curated_non.contains(o))
        .map(|s| s.to_string());
    dedupe(
        curated_originals
            .clone()
            .into_iter()
            .chain(other_originals)
            .chain(curated_non.clone())
            .chain(other_non),
    )
}

/// Preferred subtitle languages for an explicit preference (`None`/`orig`
/// prioritizes `*-orig`, then English, Chinese, the rest).
pub fn get_preferred_languages(language: Option<&str>, available: &[String]) -> Vec<String> {
    let available = dedupe(available.iter().cloned());
    let originals: Vec<String> = available
        .iter()
        .filter(|l| is_original_language(l))
        .cloned()
        .collect();
    let english = ordered_family_matches(&available, "en");
    let chinese = ordered_family_matches(&available, "zh");
    let remaining: Vec<String> = available
        .iter()
        .filter(|l| !originals.contains(l) && !english.contains(l) && !chinese.contains(l))
        .cloned()
        .collect();
    let curated =
        |first: &[&str], second: &[&str]| dedupe(first.iter().chain(second).map(|s| s.to_string()));
    let default_priority = if available.is_empty() {
        curated(ENGLISH_SUBTITLE_LANGUAGES, CHINESE_SUBTITLE_LANGUAGES)
    } else {
        dedupe(
            originals
                .iter()
                .chain(&english)
                .chain(&chinese)
                .chain(&remaining)
                .cloned(),
        )
    };

    let Some(language) = language.filter(|l| !l.is_empty()) else {
        return default_priority;
    };
    let lang = lower(language);
    match lang.as_str() {
        "orig" | "original" => default_priority,
        "zh" | "chinese" | "zh-hans" | "zh-hant" => {
            if available.is_empty() {
                curated(CHINESE_SUBTITLE_LANGUAGES, ENGLISH_SUBTITLE_LANGUAGES)
            } else {
                dedupe(
                    chinese
                        .iter()
                        .chain(&originals)
                        .chain(&english)
                        .chain(&remaining)
                        .cloned(),
                )
            }
        }
        "en" | "english" => {
            if available.is_empty() {
                curated(ENGLISH_SUBTITLE_LANGUAGES, CHINESE_SUBTITLE_LANGUAGES)
            } else {
                // Chinese stays behind other leftovers so English auto-captions
                // win even when YouTube lists zh-Hans before en/en-orig.
                let non_chinese: Vec<String> = remaining
                    .iter()
                    .filter(|l| language_family(l) != Some("zh"))
                    .cloned()
                    .collect();
                dedupe(
                    english
                        .iter()
                        .chain(&originals)
                        .chain(&non_chinese)
                        .chain(&chinese)
                        .cloned(),
                )
            }
        }
        _ => default_priority,
    }
}

/// `parse_qs(urlparse(url).query)["tlang"][0]`.
fn translation_target(entry: &Value) -> Option<String> {
    let url = entry.get("url")?.as_str()?;
    let without_fragment = url.split('#').next().unwrap_or(url);
    let query = without_fragment.split_once('?')?.1;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if percent_decode(k) == "tlang" && !v.is_empty() {
            return Some(percent_decode(v));
        }
    }
    None
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let hex = |b: u8| (b as char).to_digit(16);
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push((h * 16 + l) as u8);
                    i += 2;
                }
                _ => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn entries_match_resolved_language(
    lang: &str,
    entries: &[Value],
    resolved: Option<&str>,
    accept_original_tracks: bool,
) -> bool {
    if FORBIDDEN_SUBTITLE_LANGUAGES.contains(&lang) {
        return false;
    }
    if accept_original_tracks && is_original_language(lang) {
        return true;
    }
    let Some(family) = resolved.and_then(language_family) else {
        return true;
    };
    let targets: Vec<String> = entries.iter().filter_map(translation_target).collect();
    if !targets.is_empty() {
        return targets.iter().any(|t| language_family(t) == Some(family));
    }
    language_family(lang) == Some(family)
}

/// One subtitle track to try.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleCandidate {
    pub lang: String,
    pub entries: Vec<Value>,
    pub is_manual: bool,
}

fn entries_of(v: &Value) -> Vec<Value> {
    v.as_array().cloned().unwrap_or_default()
}

/// Manual tracks first, then auto, each in preference order; when nothing
/// passes the language filter, every non-forbidden track in listing order.
pub fn subtitle_candidates(
    preferred: &[String],
    subtitles: &Map<String, Value>,
    auto_subtitles: &Map<String, Value>,
    resolved: Option<&str>,
    accept_original_tracks: bool,
) -> Vec<SubtitleCandidate> {
    let mut candidates = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (source, is_manual) in [(subtitles, true), (auto_subtitles, false)] {
        for lang in preferred {
            if seen.contains(lang) || FORBIDDEN_SUBTITLE_LANGUAGES.contains(&lang.as_str()) {
                continue;
            }
            if let Some(v) = source.get(lang) {
                let entries = entries_of(v);
                if entries_match_resolved_language(lang, &entries, resolved, accept_original_tracks)
                {
                    candidates.push(SubtitleCandidate {
                        lang: lang.clone(),
                        entries,
                        is_manual,
                    });
                    seen.insert(lang.clone());
                }
            }
        }
    }
    if !candidates.is_empty() {
        return candidates;
    }
    for (source, is_manual) in [(subtitles, true), (auto_subtitles, false)] {
        for (lang, v) in source {
            if seen.contains(lang) || FORBIDDEN_SUBTITLE_LANGUAGES.contains(&lang.as_str()) {
                continue;
            }
            candidates.push(SubtitleCandidate {
                lang: lang.clone(),
                entries: entries_of(v),
                is_manual,
            });
            seen.insert(lang.clone());
        }
    }
    candidates
}

/// Subtitle `(ext, url)` pairs in download-preference order (`_iter_subtitle_urls`).
pub fn subtitle_urls(entries: &[Value]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen = HashSet::new();
    for fmt in PREFERRED_SUBTITLE_FORMATS {
        for entry in entries {
            if entry.get("ext").and_then(Value::as_str) != Some(fmt) {
                continue;
            }
            if let Some(url) = entry.get("url").and_then(Value::as_str)
                && !url.is_empty()
                && seen.insert(url.to_string())
            {
                out.push((fmt.to_string(), url.to_string()));
            }
        }
    }
    out
}

fn ms_to_srt_timestamp(ms: i64) -> String {
    let ms = ms.max(0);
    let (h, r) = (ms / 3_600_000, ms % 3_600_000);
    let (m, r) = (r / 60_000, r % 60_000);
    let (s, millis) = (r / 1000, r % 1000);
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

fn cues_to_srt(cues: &[(i64, i64, String)]) -> Option<String> {
    if cues.is_empty() {
        return None;
    }
    let blocks: Vec<String> = cues
        .iter()
        .enumerate()
        .map(|(i, (start, end, text))| {
            let end = if end > start {
                *end
            } else {
                start + DEFAULT_CUE_DURATION_MS
            };
            format!(
                "{}\n{} --> {}\n{}",
                i + 1,
                ms_to_srt_timestamp(*start),
                ms_to_srt_timestamp(end),
                text
            )
        })
        .collect();
    Some(blocks.join("\n\n") + "\n")
}

/// Python `int(x)` for a JSON number (floats truncate toward zero).
fn json_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.is_finite())
                .map(|f| f.trunc() as i64)
        }),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

fn json_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// YouTube json3 timedtext -> SRT.
pub fn json3_to_srt(text: &str) -> Option<String> {
    let data: Value = serde_json::from_str(text).ok()?;
    let events = data.as_object()?.get("events")?.as_array()?;
    let mut cues = Vec::new();
    for event in events {
        let Some(event) = event.as_object() else {
            continue;
        };
        let Some(segs) = event.get("segs").and_then(Value::as_array) else {
            continue;
        };
        let mut body = String::new();
        for seg in segs {
            if let Some(utf8) = seg
                .as_object()
                .and_then(|s| s.get("utf8"))
                .and_then(Value::as_str)
                && utf8 != "\n"
                && utf8 != "\r\n"
            {
                body.push_str(utf8);
            }
        }
        let body = strip(&body.replace('\n', " ")).to_string();
        if body.is_empty() {
            continue;
        }
        let Some(start) = event.get("tStartMs").and_then(json_int) else {
            continue;
        };
        let duration = event
            .get("dDurationMs")
            .filter(|d| json_f64(d).is_some_and(|f| f > 0.0))
            .and_then(json_int)
            .unwrap_or(DEFAULT_CUE_DURATION_MS);
        cues.push((start, start + duration, body));
    }
    cues_to_srt(&cues)
}

/// Python `float(str)` for XML attributes.
fn py_float(s: &str) -> Option<f64> {
    let t = strip(s).replace('_', "");
    match lower(&t).as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => t.parse::<f64>().ok(),
    }
}

fn finite_int(f: f64) -> Option<i64> {
    f.is_finite().then(|| f.trunc() as i64)
}

/// YouTube srv1 `<transcript>` -> SRT.
pub fn srv1_to_srt(text: &str) -> Option<String> {
    let root = xml::parse(text).ok()?;
    let mut cues = Vec::new();
    for el in root.iter().into_iter().filter(|e| e.name == "text") {
        let body = strip(&el.text().replace('\n', " ")).to_string();
        if body.is_empty() {
            continue;
        }
        let Some(start) =
            py_float(el.attr("start").unwrap_or("0")).and_then(|f| finite_int(f * 1000.0))
        else {
            continue;
        };
        let dur_raw = el.attr("dur").filter(|d| !d.is_empty()).unwrap_or("0");
        let Some(dur) = py_float(dur_raw) else {
            continue;
        };
        let duration = if dur > 0.0 {
            match finite_int(dur * 1000.0) {
                Some(d) => d,
                None => continue,
            }
        } else {
            DEFAULT_CUE_DURATION_MS
        };
        cues.push((start, start + duration, body));
    }
    cues_to_srt(&cues)
}

/// YouTube srv3 `<timedtext>` -> SRT.
pub fn srv3_to_srt(text: &str) -> Option<String> {
    let root = xml::parse(text).ok()?;
    let mut cues = Vec::new();
    for el in root.iter().into_iter().filter(|e| e.name == "p") {
        let body = strip(&el.text().replace('\n', " ")).to_string();
        if body.is_empty() {
            continue;
        }
        let Some(start) = py_float(el.attr("t").unwrap_or("0")).and_then(finite_int) else {
            continue;
        };
        let d_raw = el.attr("d").filter(|d| !d.is_empty()).unwrap_or("0");
        let Some(mut duration) = py_float(d_raw).and_then(finite_int) else {
            continue;
        };
        if duration <= 0 {
            duration = DEFAULT_CUE_DURATION_MS;
        }
        cues.push((start, start + duration, body));
    }
    cues_to_srt(&cues)
}

/// Normalize native YouTube captions to SRT; leave SRT/VTT/TTML unchanged.
/// `None` when the payload looks native but yields no cues.
pub fn normalize_subtitle_payload(text: &str) -> Option<String> {
    let stripped = strip(text);
    if stripped.is_empty() {
        return None;
    }
    let head = prefix_chars(stripped, 2000);
    if stripped.starts_with('{') && (head.contains("\"events\"") || head.contains("\"wireMagic\""))
    {
        return json3_to_srt(stripped);
    }
    if head.contains("<transcript") {
        return srv1_to_srt(stripped);
    }
    if head.contains("<timedtext") {
        return srv3_to_srt(stripped);
    }
    Some(stripped.to_string())
}

/// Escape a language code for yt-dlp's `--sub-langs` (case-insensitive regex
/// fullmatch).
fn sub_langs_pattern(lang: &str) -> String {
    let mut out = String::new();
    for c in lang.chars() {
        if "\\.^$|?*+()[]{},".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Download one track format through yt-dlp. `None` on any failure (Python
/// logged and moved on to the next format/language).
#[allow(clippy::too_many_arguments)]
fn download_subtitle(
    ytdlp: &YtDlp,
    platform: &str,
    cookie_source: Option<&CookieSource>,
    info_json: &Path,
    work_dir: &Path,
    attempt: usize,
    candidate: &SubtitleCandidate,
    ext: &str,
) -> Result<Option<String>> {
    let stem = format!("sub-{attempt}");
    let template = work_dir.join(format!("{stem}.%(ext)s"));
    let mut args = ytdlp.base_args(platform);
    for a in [
        "--socket-timeout",
        ytdlp::SUBTITLE_SOCKET_TIMEOUT_SECS,
        "--retries",
        ytdlp::SUBTITLE_RETRIES,
        "--retry-sleep",
        ytdlp::SUBTITLE_RETRY_SLEEP,
        "--skip-download",
    ] {
        args.push(a.into());
    }
    args.push("--ignore-no-formats-error".into());
    args.push(
        if candidate.is_manual {
            "--write-subs"
        } else {
            "--write-auto-subs"
        }
        .into(),
    );
    args.push("--sub-langs".into());
    args.push(sub_langs_pattern(&candidate.lang).into());
    args.push("--sub-format".into());
    args.push(ext.into());
    args.push("--load-info-json".into());
    args.push(info_json.as_os_str().to_owned());
    args.push("-o".into());
    args.push(template.into_os_string());
    if let Some(source) = cookie_source {
        args.extend(ytdlp::cookie_args(source));
    }
    // Failures only mattered for logging in Python; the caller moves on.
    if ytdlp.run(&args).is_err() {
        return Ok(None);
    }
    // Prefer the exact `<stem>.<lang>.<ext>` file; a case-insensitive regex
    // could have matched a sibling language.
    let exact = work_dir.join(format!("{stem}.{}.{ext}", candidate.lang));
    let path = if exact.is_file() {
        Some(exact)
    } else {
        std::fs::read_dir(work_dir).ok().and_then(|rd| {
            let mut files: Vec<_> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                        n.starts_with(&format!("{stem}.")) && !n.ends_with(".part")
                    })
                })
                .collect();
            files.sort();
            files.into_iter().next()
        })
    };
    let Some(path) = path else {
        return Ok(None);
    };
    let bytes = std::fs::read(&path).ok();
    Ok(bytes.map(|b| String::from_utf8_lossy(&b).into_owned()))
}

fn str_field<'a>(info: &'a Value, key: &str) -> Option<&'a str> {
    info.get(key).and_then(Value::as_str)
}

/// Extract subtitle text from yt-dlp metadata (`extract_subtitles`).
///
/// `language` is the explicit user preference; when unset, `*-orig` tracks
/// are preferred and metadata/title/description only filter translations.
/// `info_json` is the raw `--dump-single-json` payload (written to a temp
/// file for `--load-info-json`), `work_dir` a scratch directory.
#[allow(clippy::too_many_arguments)]
pub fn extract_subtitles(
    ytdlp: &YtDlp,
    info: &Value,
    info_json: &str,
    platform: &str,
    language: Option<&str>,
    cookie_source: Option<&CookieSource>,
    work_dir: &Path,
) -> Result<Option<String>> {
    let empty = Map::new();
    let subtitles = info
        .get("subtitles")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let auto = info
        .get("automatic_captions")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    if subtitles.is_empty() && auto.is_empty() {
        return Ok(None);
    }

    let content_language = resolve_subtitle_language(
        language,
        str_field(info, "title"),
        str_field(info, "description"),
        str_field(info, "language"),
    );
    let accept_original_tracks = match language {
        None => true,
        Some(l) => matches!(lower(l).as_str(), "orig" | "original"),
    };
    let available: Vec<String> = subtitles.keys().chain(auto.keys()).cloned().collect();
    let preferred = get_preferred_languages(language, &available);
    let candidates = subtitle_candidates(
        &preferred,
        subtitles,
        auto,
        content_language.as_deref(),
        accept_original_tracks,
    );
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut info_path = None;
    let mut attempt = 0usize;
    for candidate in &candidates {
        if platform == "bilibili" {
            let Some(first) = candidate.entries.first() else {
                continue;
            };
            if let Some(data) = first.get("data").and_then(Value::as_str)
                && !strip(data).is_empty()
            {
                return Ok(Some(data.to_string()));
            }
            continue;
        }
        if platform != "youtube" && platform != "unknown" {
            // "Unsupported platform for subtitles"
            return Ok(None);
        }
        let urls = subtitle_urls(&candidate.entries);
        for (ext, _url) in urls {
            let path = match &info_path {
                Some(p) => p,
                None => {
                    let p = work_dir.join("info.json");
                    if std::fs::write(&p, info_json).is_err() {
                        return Ok(None);
                    }
                    info_path.insert(p)
                }
            };
            attempt += 1;
            let Some(content) = download_subtitle(
                ytdlp,
                platform,
                cookie_source,
                path,
                work_dir,
                attempt,
                candidate,
                &ext,
            )?
            else {
                continue;
            };
            if strip(&content).is_empty() {
                continue;
            }
            if let Some(normalized) = normalize_subtitle_payload(&content)
                && !strip(&normalized).is_empty()
            {
                return Ok(Some(normalized));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn preferred_languages_default_and_explicit() {
        let p = get_preferred_languages(None, &s(&["zh-Hans", "en-orig", "fr-orig", "en", "zh"]));
        assert_eq!(p[..5], s(&["en-orig", "fr-orig", "en", "zh-Hans", "zh"]));
        let p = get_preferred_languages(Some("orig"), &s(&["zh-Hans", "en-orig", "en"]));
        assert_eq!(p[..3], s(&["en-orig", "en", "zh-Hans"]));
        let p = get_preferred_languages(
            Some("en"),
            &s(&["ab", "zh-Hans", "zh-Hant", "en-orig", "en", "fr"]),
        );
        assert_eq!(p[..2], s(&["en-orig", "en"]));
        let pos = |x: &str| p.iter().position(|l| l == x).unwrap();
        assert!(pos("en") < pos("zh-Hans"));
        let p = get_preferred_languages(
            None,
            &s(&["zh-Hans", "zh-Hant", "en-orig", "en", "fr-orig"]),
        );
        assert_eq!(p[0], "en-orig");
    }

    #[test]
    fn language_resolution() {
        assert_eq!(normalize_metadata_language(Some("en-US")), Some("en"));
        assert_eq!(normalize_metadata_language(Some("zh-CN")), Some("zh"));
        assert_eq!(
            resolve_subtitle_language(
                None,
                Some("未使用的中文标题足够汉字"),
                Some("未使用的中文描述足够汉字"),
                Some("en")
            )
            .as_deref(),
            Some("en")
        );
        assert_eq!(
            resolve_subtitle_language(
                Some("en"),
                Some("中文标题带有足够汉字"),
                Some("这是中文描述"),
                None
            )
            .as_deref(),
            Some("en")
        );
        assert_eq!(
            resolve_subtitle_language(
                None,
                Some("中文标题带有足够汉字"),
                Some("English description only as filler text"),
                None
            )
            .as_deref(),
            Some("zh")
        );
        let cases: &[(&[Option<&str>], Option<&str>)] = &[
            (
                &[Some("如何用 AIMD 把 YouTube 视频转成 Markdown")],
                Some("zh"),
            ),
            (
                &[Some("How to convert YouTube videos to Markdown with AIMD")],
                Some("en"),
            ),
            (
                &[
                    Some("【深度访谈】OpenAI CEO 谈 AGI 的未来"),
                    Some("English transcript available"),
                ],
                Some("zh"),
            ),
            (
                &[Some("Deep Dive with OpenAI CEO on the Future of AGI")],
                Some("en"),
            ),
            (
                &[Some("B站 UP主 推荐：Python 异步编程实战教程")],
                Some("zh"),
            ),
            (&[Some("ChatGPT 使用指南：从入门到精通")], Some("zh")),
            (&[Some("使用 ChatGPT 写代码")], Some("zh")),
            (&[Some("WWDC 2024 Keynote Highlights")], Some("en")),
            (&[Some(""), None], None),
            (&[None, None], None),
            (&[Some("!!")], None),
        ];
        for (texts, expected) in cases {
            assert_eq!(detect_content_language(texts), *expected, "{texts:?}");
        }
    }

    #[test]
    fn subtitle_urls_prefer_native_formats() {
        let entries: Vec<Value> = ["json3", "srv1", "srt", "vtt", "ttml", "srv2", "srv3"]
            .iter()
            .map(|e| json!({"ext": e, "url": format!("https://example.com/en?fmt={e}")}))
            .collect();
        let urls: Vec<String> = subtitle_urls(&entries)
            .into_iter()
            .map(|(_, u)| u)
            .collect();
        assert_eq!(
            urls,
            ["json3", "ttml", "vtt", "srv1", "srt"]
                .iter()
                .map(|e| format!("https://example.com/en?fmt={e}"))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn translation_filter_skips_chinese_translations_for_english_video() {
        let auto = json!({
            "zh-Hans": [{"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=x&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"}],
            "zh-Hant": [{"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=x&kind=asr&lang=en&tlang=zh-Hant&fmt=srt"}],
            "en-orig": [{"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=x&kind=asr&lang=en&fmt=srt"}],
            "en": [{"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=x&kind=asr&lang=en&fmt=srt&src=en"}],
        });
        let auto = auto.as_object().unwrap();
        let available: Vec<String> = auto.keys().cloned().collect();
        let preferred = get_preferred_languages(None, &available);
        let c = subtitle_candidates(&preferred, &Map::new(), auto, Some("en"), true);
        assert_eq!(c[0].lang, "en-orig");
        assert!(c.iter().all(|c| !c.lang.starts_with("zh")));

        // Explicit zh keeps the zh-Hans translation first.
        let preferred = get_preferred_languages(Some("zh"), &available);
        let c = subtitle_candidates(&preferred, &Map::new(), auto, Some("zh"), false);
        assert_eq!(c[0].lang, "zh-Hans");
    }

    #[test]
    fn native_caption_conversion() {
        let json3 = r#"{"wireMagic": "pb3", "events": [
            {"tStartMs": 0, "dDurationMs": 0, "id": 1},
            {"tStartMs": 80, "dDurationMs": 3200, "segs": [{"utf8": "you've "}, {"utf8": "discovered", "tOffsetMs": 400}]},
            {"tStartMs": 1360, "segs": [{"utf8": "\n"}]},
            {"tStartMs": 3280, "dDurationMs": 2000, "segs": [{"utf8": "the brain"}]}]}"#;
        let srt = json3_to_srt(json3).unwrap();
        assert!(srt.starts_with("1\n00:00:00,080 --> 00:00:03,280\nyou've discovered"));
        assert!(srt.contains("00:00:03,280 --> 00:00:05,280\nthe brain"));
        assert_eq!(
            crate::markdown::strip_subtitle_formatting(&srt),
            "you've discovered the brain"
        );

        let srv1 = "<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n<transcript>\n  <text start=\"0.08\" dur=\"3.2\">you've discovered</text>\n  <text start=\"3.28\" dur=\"2\">the brain</text>\n</transcript>\n";
        let srt = srv1_to_srt(srv1).unwrap();
        assert!(srt.contains("00:00:00,080 --> 00:00:03,280"), "{srt}");
        assert!(srt.contains("the brain"));

        let srv3 = "<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n<timedtext format=\"3\">\n  <body>\n    <p t=\"80\" d=\"3200\">you've discovered</p>\n    <p t=\"3280\" d=\"2000\"><s>the </s><s t=\"400\">brain</s></p>\n  </body>\n</timedtext>\n";
        let srt = srv3_to_srt(srv3).unwrap();
        assert!(srt.contains("00:00:00,080 --> 00:00:03,280") && srt.contains("the brain"));

        let srt_in = "1\n00:00:00,000 --> 00:00:01,000\nhello\n";
        assert_eq!(
            normalize_subtitle_payload(srt_in).as_deref(),
            Some(strip(srt_in))
        );
        assert_eq!(
            normalize_subtitle_payload("{\"wireMagic\": \"pb3\", \"events\": [{\"tStartMs\": 0}]}"),
            None
        );
    }

    #[test]
    fn sub_langs_pattern_escapes_regex() {
        assert_eq!(sub_langs_pattern("en-orig"), "en-orig");
        assert_eq!(sub_langs_pattern("a.b+c"), "a\\.b\\+c");
    }
}
