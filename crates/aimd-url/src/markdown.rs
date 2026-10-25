//! Transcript cleanup and Markdown formatting (ports `aimd.plugins.url.markdown`).
//!
//! The regexes of the Python module are hand-written matchers here; the
//! golden fixtures in `tests/golden/subtitles` lock byte-identical output.

use serde_json::Value;

use crate::pytext::{
    float_repr, group_thousands_float, group_thousands_int, is_decimal, is_space, lower,
    prefix_chars, split_ws, splitlines, strip,
};
use crate::xml;

const SPEAKER_MARKERS: [&str; 2] = [">>", "<<"];
const ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "inc.", "ltd.", "st.",
    "u.s.", "u.k.", "e.g.", "i.e.", "ph.d.", "phd.",
];

/// Scans `\d{min,max}` greedily from `chars[i]`; returns the end index.
fn digits(chars: &[char], i: usize, min: usize, max: usize) -> Option<usize> {
    let mut n = 0;
    while n < max && i + n < chars.len() && is_decimal(chars[i + n]) {
        n += 1;
    }
    (n >= min).then_some(i + n)
}

/// `\d{1,2}:\d{2}:\d{2}[.,]\d{2,3}` anchored at `i` (with backtracking over
/// the greedy quantifiers). Returns every possible end position.
fn timestamp_ends(chars: &[char], i: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    for h in [2usize, 1] {
        let Some(j) = digits(chars, i, h, h) else {
            continue;
        };
        if chars.get(j) != Some(&':') {
            continue;
        }
        let Some(j) = digits(chars, j + 1, 2, 2) else {
            continue;
        };
        if chars.get(j) != Some(&':') {
            continue;
        }
        let Some(j) = digits(chars, j + 1, 2, 2) else {
            continue;
        };
        if !matches!(chars.get(j), Some('.' | ',')) {
            continue;
        }
        for f in [3usize, 2] {
            if let Some(k) = digits(chars, j + 1, f, f) {
                ends.push(k);
            }
        }
    }
    ends
}

/// `_TIMESTAMP_RE.search(line)`.
fn has_timestamp_range(line: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    for start in 0..chars.len() {
        for end in timestamp_ends(&chars, start) {
            let mut k = end;
            while k < chars.len() && is_space(chars[k]) {
                k += 1;
            }
            // `\s*` backtracking cannot help: `-->` is not whitespace.
            if chars.get(k..k + 3) != Some(&['-', '-', '>'][..]) {
                continue;
            }
            let mut k = k + 3;
            while k < chars.len() && is_space(chars[k]) {
                k += 1;
            }
            if !timestamp_ends(&chars, k).is_empty() {
                return true;
            }
        }
    }
    false
}

/// `^\d+$` on an already-stripped line.
fn is_sequence(line: &str) -> bool {
    !line.is_empty() && line.chars().all(is_decimal)
}

fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len()
        && s.is_char_boundary(prefix.len())
        && s[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Removes `<\d{1,2}:\d{2}:\d{2}\.\d{3}>` word timestamps.
fn remove_word_timestamps(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let hit = timestamp_ends(&chars, i + 1).into_iter().find(|&end| {
                // only the `.` separator with exactly 3 fraction digits
                end >= 4 && chars[end - 4] == '.' && chars.get(end) == Some(&'>')
            });
            if let Some(end) = hit {
                i = end + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Length of a `_SUBTITLE_TAG_RE` match starting at `chars[i] == '<'`.
fn subtitle_tag_len(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if chars.get(j) == Some(&'/') {
        j += 1;
    }
    let has = |name: &str| -> bool {
        name.chars().enumerate().all(|(n, want)| {
            chars
                .get(j + n)
                .is_some_and(|c| c.eq_ignore_ascii_case(&want))
        })
    };
    let close_after = |k: usize| -> Option<usize> {
        chars[k..]
            .iter()
            .position(|&c| c == '>')
            .map(|p| k + p + 1 - i)
    };
    let exact = |name: &str| -> Option<usize> {
        let k = j + name.len();
        (has(name) && chars.get(k) == Some(&'>')).then(|| k + 1 - i)
    };
    let with_ws = |name: &str| -> Option<usize> {
        let k = j + name.len();
        if has(name) && chars.get(k).is_some_and(|&c| is_space(c)) {
            close_after(k)
        } else {
            None
        }
    };
    // Alternation order of the Python regex:
    // b | c(\.[^>]*)? | font(\s+[^>]*)? | i | lang(\s+[^>]*)? | rt | ruby | u | v(\s+[^>]*)?
    exact("b")
        .or_else(|| if has("c.") { close_after(j + 2) } else { None })
        .or_else(|| exact("c"))
        .or_else(|| with_ws("font"))
        .or_else(|| exact("font"))
        .or_else(|| exact("i"))
        .or_else(|| with_ws("lang"))
        .or_else(|| exact("lang"))
        .or_else(|| exact("rt"))
        .or_else(|| exact("ruby"))
        .or_else(|| exact("u"))
        .or_else(|| with_ws("v"))
        .or_else(|| exact("v"))
}

fn remove_subtitle_tags(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<'
            && let Some(len) = subtitle_tag_len(&chars, i)
        {
            i += len;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `_BR_RE.sub(" ", text)`: `<br\s*/?>` case-insensitive.
fn replace_br(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<'
            && chars
                .get(i + 1)
                .is_some_and(|c| c.eq_ignore_ascii_case(&'b'))
            && chars
                .get(i + 2)
                .is_some_and(|c| c.eq_ignore_ascii_case(&'r'))
        {
            let mut k = i + 3;
            while k < chars.len() && is_space(chars[k]) {
                k += 1;
            }
            if chars.get(k) == Some(&'/') {
                k += 1;
            }
            if chars.get(k) == Some(&'>') {
                out.push(' ');
                i = k + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn split_speaker_marker(token: &str) -> (bool, &str) {
    if SPEAKER_MARKERS.contains(&token) {
        return (true, "");
    }
    for marker in SPEAKER_MARKERS {
        if let Some(rest) = token.strip_prefix(marker) {
            return (true, rest);
        }
    }
    (false, token)
}

/// `[.!?。！？…]["'”’)\]]*$`
fn has_sentence_end(token: &str) -> bool {
    let trimmed = token.trim_end_matches(['"', '\'', '”', '’', ')', ']']);
    trimmed
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '…'))
}

/// `^["'“‘(\[]?(?:[A-Z]|[㐀-鿿])`
fn starts_next_sentence(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    let starter = |c: char| c.is_ascii_uppercase() || ('\u{3400}'..='\u{9fff}').contains(&c);
    if matches!(first, '"' | '\'' | '“' | '‘' | '(' | '[') {
        return chars.next().is_some_and(starter);
    }
    starter(first)
}

fn is_sentence_end(token: &str, next_token: Option<&str>) -> bool {
    if ABBREVIATIONS.contains(&lower(token).as_str()) {
        return false;
    }
    let mut cs = token.chars();
    if let (Some(a), Some('.'), None) = (cs.next(), cs.next(), cs.next())
        && a.is_ascii_alphabetic()
    {
        return false;
    }
    if !has_sentence_end(token) {
        return false;
    }
    let Some(next) = next_token else {
        return true;
    };
    let (speaker_change, remainder) = split_speaker_marker(next);
    if speaker_change {
        return true;
    }
    starts_next_sentence(if remainder.is_empty() {
        next
    } else {
        remainder
    })
}

/// One sentence per line; YouTube `>>` speaker markers start their turn's line.
pub fn format_linear_transcript(tokens: &[&str]) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut prefix: Option<&'static str> = None;

    let flush =
        |current: &mut Vec<String>, prefix: &mut Option<&'static str>, lines: &mut Vec<String>| {
            if current.is_empty() {
                return;
            }
            let text = current.join(" ");
            match prefix.take() {
                Some(p) => lines.push(format!("{p} {text}")),
                None => lines.push(text),
            }
            current.clear();
        };

    let mut pending: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    let mut index = 0;
    while index < pending.len() {
        let token = pending[index].clone();
        let (speaker_change, remainder) = split_speaker_marker(&token);
        if speaker_change {
            flush(&mut current, &mut prefix, &mut lines);
            prefix = Some(if token.starts_with(">>") { ">>" } else { "<<" });
            if !remainder.is_empty() {
                pending[index] = remainder.to_string();
                continue;
            }
            index += 1;
            continue;
        }
        current.push(token.clone());
        let next = pending.get(index + 1).map(String::as_str);
        if is_sentence_end(&token, next) {
            flush(&mut current, &mut prefix, &mut lines);
        }
        index += 1;
    }
    flush(&mut current, &mut prefix, &mut lines);
    lines.join("\n")
}

/// Join caption cues, dropping YouTube ASR rolling-window overlap.
pub fn merge_rolling_caption_lines<S: AsRef<str>>(lines: &[S]) -> String {
    let mut tokens: Vec<&str> = Vec::new();
    for line in lines {
        let current = split_ws(line.as_ref());
        if current.is_empty() {
            continue;
        }
        if tokens.is_empty() {
            tokens.extend(current);
            continue;
        }
        let max_overlap = tokens.len().min(current.len());
        let mut overlap = 0;
        for size in (1..=max_overlap).rev() {
            if tokens[tokens.len() - size..] == current[..size] {
                overlap = size;
                break;
            }
        }
        tokens.extend_from_slice(&current[overlap..]);
    }
    format_linear_transcript(&tokens)
}

fn ttml_cue_texts(text: &str) -> Option<Vec<String>> {
    let normalized = replace_br(text);
    let root = xml::parse(&normalized).ok()?;
    let cues = root
        .iter()
        .into_iter()
        .filter(|e| e.local_name() == "p")
        .map(|e| split_ws(&e.text()).join(" "))
        .filter(|body| !body.is_empty())
        .collect();
    Some(cues)
}

fn timed_cue_texts(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in splitlines(strip(text)) {
        let stripped = strip(line);
        if stripped.is_empty()
            || starts_with_ignore_ascii_case(stripped, "WEBVTT")
            || starts_with_ignore_ascii_case(stripped, "Kind:")
            || starts_with_ignore_ascii_case(stripped, "Language:")
            || has_timestamp_range(stripped)
            || is_sequence(stripped)
            || stripped.starts_with("NOTE ")
        {
            continue;
        }
        let cue = remove_word_timestamps(stripped);
        let cue = remove_subtitle_tags(&cue);
        let cue = strip(&cue);
        if !cue.is_empty() {
            out.push(cue.to_string());
        }
    }
    out
}

/// Strip SRT/WebVTT/TTML formatting into a format-independent transcript,
/// one sentence per line. Returns the input unchanged when no subtitle
/// formatting is detected.
pub fn strip_subtitle_formatting(text: &str) -> String {
    let stripped = strip(text);
    if stripped.is_empty() {
        return String::new();
    }
    let head = prefix_chars(stripped, 500);
    if (head.contains("<tt ") || head.contains("xmlns=\"http://www.w3.org/ns/ttml\""))
        && let Some(cues) = ttml_cue_texts(stripped)
        && !cues.is_empty()
    {
        return merge_rolling_caption_lines(&cues);
    }
    let lines = splitlines(stripped);
    if !lines.iter().take(30).any(|l| has_timestamp_range(l)) {
        return text.to_string();
    }
    merge_rolling_caption_lines(&timed_cue_texts(stripped))
}

/// Python `str(value)` for JSON metadata values.
/// MarkItDown's post-conversion normalization, applied to every converter
/// result in Python: `"\n".join(line.rstrip() for line in re.split(r"\r?\n",
/// text))`, then `\n{3,}` collapsed to `\n\n`. Idempotent.
pub fn markitdown_normalize(text: &str) -> String {
    let joined = text
        .split('\n')
        .map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            line.trim_end_matches(crate::pytext::is_space)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut out = String::with_capacity(joined.len());
    let mut newlines = 0usize;
    for c in joined.chars() {
        if c == '\n' {
            newlines += 1;
            continue;
        }
        if newlines > 0 {
            out.push_str(if newlines >= 3 {
                "\n\n"
            } else {
                &"\n\n"[..newlines]
            });
            newlines = 0;
        }
        out.push(c);
    }
    if newlines > 0 {
        out.push_str(if newlines >= 3 {
            "\n\n"
        } else {
            &"\n\n"[..newlines]
        });
    }
    out
}

pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => match n.as_u64() {
                Some(u) => u.to_string(),
                None => float_repr(n.as_f64().unwrap_or(0.0)),
            },
        },
        // Lists/dicts are not used by the formatter in practice.
        other => other.to_string(),
    }
}

fn py_thousands(value: &Value) -> String {
    match value {
        Value::Number(n) => match n.as_i64() {
            Some(i) => group_thousands_int(i128::from(i)),
            None => match n.as_u64() {
                Some(u) => group_thousands_int(i128::from(u)),
                None => group_thousands_float(n.as_f64().unwrap_or(0.0)),
            },
        },
        Value::Bool(b) => group_thousands_int(i128::from(*b)),
        // Python raises on `format(None, ",")`; yt-dlp JSON never carries
        // nulls (`clean_infojson`), so treat anything else as missing.
        _ => "0".into(),
    }
}

/// `info.get(key, default)`; JSON `null` counts as missing (yt-dlp's
/// `--dump-single-json` drops `None` fields, so it only appears in hand-made
/// inputs).
fn get<'a>(info: &'a Value, key: &str) -> Option<&'a Value> {
    info.get(key).filter(|v| !v.is_null())
}

/// Format video metadata and extracted content into Markdown.
pub fn format_content(info: &Value, content: Option<&str>, platform: Option<&str>) -> String {
    let text = |key: &str, default: &str| get(info, key).map_or(default.to_string(), py_str);
    let title = text("title", "Unknown Title");
    let description = text("description", "No description available");
    let channel = match get(info, "channel") {
        Some(v) => py_str(v),
        None => text("uploader", "Unknown"),
    };
    let duration = text("duration", "0");
    let view_count = get(info, "view_count").map_or("0".to_string(), py_thousands);
    let upload_date = text("upload_date", "");
    let webpage_url = text("webpage_url", "");
    let platform = match platform {
        Some(p) if !p.is_empty() => p,
        _ => "unknown",
    };

    let mut out = format!(
        "# {title}\n\n**Channel:** {channel}\n**Platform:** {platform}\n**Duration:** {duration} seconds\n**Upload Date:** {upload_date}\n**View Count:** {view_count} views\n**URL:** {webpage_url}\n\n## Description\n\n{description}\n\n## Content\n\n"
    );
    match content {
        Some(c) if !strip(c).is_empty() => out.push_str(c),
        _ => out.push_str("*No subtitles or transcription available for this video.*"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::markitdown_normalize;

    #[test]
    fn markitdown_normalize_matches_python() {
        assert_eq!(
            markitdown_normalize("a \t\r\nb\u{3000}\n\n\n\nc\n\n"),
            "a\nb\n\nc\n\n"
        );
        assert_eq!(markitdown_normalize("x\n\n\n"), "x\n\n");
        let once = markitdown_normalize("p \n\n\n\nq  ");
        assert_eq!(markitdown_normalize(&once), once);
    }

    use super::*;

    #[test]
    fn timestamp_matcher() {
        assert!(has_timestamp_range("00:00:00,080 --> 00:00:03,280"));
        assert!(has_timestamp_range("0:00:00.08-->0:00:03.28 align:start"));
        assert!(!has_timestamp_range("00:00 --> 00:01"));
        assert!(!has_timestamp_range("hello"));
    }

    #[test]
    fn tag_and_word_timestamp_removal() {
        assert_eq!(
            remove_subtitle_tags(&remove_word_timestamps(
                "你<00:00:00.340><c>发现</c><00:00:00.600><c.color>了</c>"
            )),
            "你发现了"
        );
        assert_eq!(
            remove_subtitle_tags("<v Bob>hi</v> <font color=\"r\">x</font>"),
            "hi x"
        );
        assert_eq!(remove_subtitle_tags("<bx>keep</bx>"), "<bx>keep</bx>");
        assert_eq!(replace_br("a<BR/>b<br >c"), "a b c");
    }

    #[test]
    fn merge_drops_rolling_overlap() {
        assert_eq!(
            merge_rolling_caption_lines(&[
                "you've discovered something about",
                "discovered something about the brain",
            ]),
            "you've discovered something about the brain"
        );
    }

    #[test]
    fn linear_transcript_splits_sentences_and_speakers() {
        let text = "you've discovered something about the brain I think is quite surprising. >> So we found that in fathers the brain is actually shrinking. Dr. Darby found that. >> Yeah, go for it.";
        let tokens: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(
            format_linear_transcript(&tokens),
            "you've discovered something about the brain I think is quite surprising.\n>> So we found that in fathers the brain is actually shrinking.\nDr. Darby found that.\n>> Yeah, go for it."
        );
    }

    #[test]
    fn strip_handles_srt_vtt_and_plain_text() {
        let srt = "1\n00:00:00,080 --> 00:00:03,280\nyou've discovered something surprising.\n\n2\n00:00:03,280 --> 00:00:06,000\n>> So we found that the brain is shrinking.\n";
        assert_eq!(
            strip_subtitle_formatting(srt),
            "you've discovered something surprising.\n>> So we found that the brain is shrinking."
        );
        let vtt = "WEBVTT\nKind: captions\nLanguage: zh-Hans\n\n00:00:00.000 --> 00:00:02.000\n你<00:00:00.340><c>发现</c><00:00:00.600><c>了一些</c><00:00:01.120><c>关于</c>\n\n00:00:01.000 --> 00:00:03.000\n你发现了一些关于 大脑<00:00:01.660><c>的惊人秘密。</c>\n\n00:00:02.000 --> 00:00:04.000\n大脑的惊人秘密。\n";
        assert_eq!(
            strip_subtitle_formatting(vtt),
            "你发现了一些关于 大脑的惊人秘密。"
        );
        assert_eq!(
            strip_subtitle_formatting(" plain text \n"),
            " plain text \n"
        );
        assert_eq!(strip_subtitle_formatting("  \n"), "");
    }

    #[test]
    fn format_content_matches_python_template() {
        let info = serde_json::json!({
            "title": "Example", "channel": "Demo Channel", "webpage_url": "https://x",
            "duration": 12, "upload_date": "20260307", "view_count": 1234567, "description": "desc"
        });
        assert_eq!(
            format_content(&info, Some("subtitle text"), Some("youtube")),
            "# Example\n\n**Channel:** Demo Channel\n**Platform:** youtube\n**Duration:** 12 seconds\n**Upload Date:** 20260307\n**View Count:** 1,234,567 views\n**URL:** https://x\n\n## Description\n\ndesc\n\n## Content\n\nsubtitle text"
        );
        let empty = format_content(&serde_json::json!({}), None, None);
        assert!(empty.starts_with("# Unknown Title\n\n**Channel:** Unknown\n**Platform:** unknown\n**Duration:** 0 seconds"));
        assert!(empty.ends_with("*No subtitles or transcription available for this video.*"));
    }
}
