//! Source anchors (spec 005): OCR position comments, timed transcript
//! paragraphs, and SRT/VTT rendering and parsing.
//!
//! OCR anchors are HTML comments on their own line before each block
//! (`<!-- aimd:loc page=1 bbox=0.050,0.030,0.400,0.070 kind=title -->`), so
//! Markdown viewers hide them while `chunk_list` keeps them for citation.
//! Transcript paragraphs start with a visible `[hh:mm:ss] ` prefix.

use crate::model::Segment;

/// New paragraph after a pause this long between sentences (seconds).
pub const PARAGRAPH_GAP_SECONDS: f64 = 2.0;
/// New paragraph once the current one reaches this many characters.
pub const PARAGRAPH_MAX_CHARS: usize = 300;
/// Longest subtitle cue before it is split (seconds).
pub const CUE_MAX_SECONDS: f64 = 7.0;
/// Longest subtitle cue in characters (Latin / CJK text).
pub const CUE_MAX_CHARS_LATIN: usize = 84;
pub const CUE_MAX_CHARS_CJK: usize = 42;
/// Shortest subtitle cue (seconds).
pub const CUE_MIN_SECONDS: f64 = 0.5;

/// `<!-- aimd:loc page=P bbox=x0,y0,x1,y1 kind=K -->`.
pub fn loc_comment(page: usize, bbox: [f32; 4], kind: &str) -> String {
    let kind = kind.replace(|c: char| c.is_whitespace() || c == '-', "_");
    format!(
        "<!-- aimd:loc page={page} bbox={:.3},{:.3},{:.3},{:.3} kind={kind} -->",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )
}

/// Axis-aligned bounds of a quad.
pub fn quad_bbox(quad: &[[f32; 2]; 4]) -> [f32; 4] {
    let xs = quad.map(|p| p[0]);
    let ys = quad.map(|p| p[1]);
    let min = |v: [f32; 4]| v.into_iter().fold(f32::INFINITY, f32::min);
    let max = |v: [f32; 4]| v.into_iter().fold(f32::NEG_INFINITY, f32::max);
    [min(xs), min(ys), max(xs), max(ys)]
}

/// Located blocks as Markdown: each block preceded by its anchor comment
/// (blocks without a page/bbox are emitted bare), joined by blank lines.
pub fn located_markdown(blocks: &[Segment]) -> String {
    blocks
        .iter()
        .filter(|b| !b.text.trim().is_empty())
        .map(|b| match (b.page, b.bbox) {
            (Some(page), Some(bbox)) => format!(
                "{}\n{}",
                loc_comment(page, bbox, b.kind.as_deref().unwrap_or("text")),
                b.text.trim()
            ),
            _ => b.text.trim().to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `hh:mm:ss` (seconds floored).
pub fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

fn timecode(seconds: f64, sep: char) -> String {
    let ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{sep}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// CJK ideographs, kana, hangul and fullwidth punctuation: text written
/// without spaces between words.
pub fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x303F | 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF
        | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F)
}

fn mostly_cjk(text: &str) -> bool {
    let (cjk, other) =
        text.chars()
            .filter(|c| !c.is_whitespace())
            .fold((0usize, 0usize), |(a, b), c| {
                if is_cjk(c) { (a + 1, b) } else { (a, b + 1) }
            });
    cjk > other
}

/// Joins two pieces of running text: no space between CJK neighbors.
fn join_text(out: &mut String, next: &str) {
    let next = next.trim();
    if next.is_empty() {
        return;
    }
    let glue = match (out.chars().last(), next.chars().next()) {
        (None, _) => "",
        (Some(a), Some(b)) if is_cjk(a) || is_cjk(b) => "",
        _ => " ",
    };
    out.push_str(glue);
    out.push_str(next);
}

const CLOSERS: &[char] = &['"', '\'', '”', '’', '」', '』', ')', '）', ']', '】', '》'];

/// Byte spans of the sentences in `text`: a sentence ends at `。！？!?；`,
/// at `…`, at `.` followed by whitespace or the end, or at a newline;
/// closing quotes/brackets stay with the sentence. Spans are trimmed.
pub fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let push = |s: usize, e: usize, spans: &mut Vec<(usize, usize)>| {
        let piece = &text[s..e];
        let lead = piece.len() - piece.trim_start().len();
        let trimmed = piece.trim();
        if !trimmed.is_empty() {
            spans.push((s + lead, s + lead + trimmed.len()));
        }
    };
    while i < chars.len() {
        let (_, c) = chars[i];
        let next = chars.get(i + 1).map(|&(_, n)| n);
        let ends = match c {
            '。' | '！' | '？' | '!' | '?' | '；' | '…' | '\n' => true,
            '.' => {
                next.is_none_or(char::is_whitespace) || next.is_some_and(|n| CLOSERS.contains(&n))
            }
            _ => false,
        };
        if ends {
            let mut j = i + 1;
            while j < chars.len()
                && (CLOSERS.contains(&chars[j].1)
                    || matches!(chars[j].1, '。' | '！' | '？' | '!' | '?' | '…' | '.'))
            {
                j += 1;
            }
            let end = chars.get(j).map_or(text.len(), |&(b, _)| b);
            push(start, end, &mut spans);
            start = end;
            i = j;
        } else {
            i += 1;
        }
    }
    push(start, text.len(), &mut spans);
    spans
}

fn is_unit_char(c: char) -> bool {
    c.is_alphanumeric()
}

/// Timed sentences from a transcript and its forced-alignment units
/// (`(text, start, end)`, in transcript order). Units are matched to the
/// transcript by their alphanumeric characters; a sentence takes the start
/// of its first unit and the end of its last. Sentences with no matched
/// unit inherit the previous sentence's end (or `offset`). Times are
/// shifted by `offset` seconds.
pub fn timed_sentences(text: &str, units: &[(String, f64, f64)], offset: f64) -> Vec<Segment> {
    // Character position (index into the alphanumeric stream) -> unit index.
    let stream: Vec<char> = text
        .chars()
        .filter(|&c| is_unit_char(c))
        .flat_map(char::to_lowercase)
        .collect();
    let mut owner: Vec<Option<usize>> = vec![None; stream.len()];
    let mut pos = 0usize;
    for (u, (unit, _, _)) in units.iter().enumerate() {
        let needle: Vec<char> = unit
            .chars()
            .filter(|&c| is_unit_char(c))
            .flat_map(char::to_lowercase)
            .collect();
        if needle.is_empty() {
            continue;
        }
        // Search forward from the cursor (alignment units are in order).
        let found = (pos..=stream.len().saturating_sub(needle.len()))
            .find(|&p| stream[p..p + needle.len()] == needle[..]);
        if let Some(p) = found {
            owner[p..p + needle.len()]
                .iter_mut()
                .for_each(|o| *o = Some(u));
            pos = p + needle.len();
        }
    }
    let mut out = Vec::new();
    let mut cursor = 0usize;
    let mut last_end = 0.0f64;
    for (s, e) in sentence_spans(text) {
        let sentence = &text[s..e];
        let n = sentence
            .chars()
            .filter(|&c| is_unit_char(c))
            .flat_map(char::to_lowercase)
            .count();
        let owners: Vec<usize> = owner[cursor..cursor + n]
            .iter()
            .flatten()
            .copied()
            .collect();
        cursor += n;
        let (start, end) = match (owners.first(), owners.last()) {
            (Some(&a), Some(&b)) => (units[a].1, units[b].2.max(units[a].1)),
            _ => (last_end, last_end),
        };
        last_end = end;
        out.push(Segment {
            text: sentence.to_string(),
            start: Some(start + offset),
            end: Some(end + offset),
            ..Segment::default()
        });
    }
    out
}

/// Sentences of `text` spread evenly over `[start, end]` by character count
/// (chunk-level fallback when no alignment is available).
pub fn spread_sentences(text: &str, start: f64, end: f64) -> Vec<Segment> {
    let spans = sentence_spans(text);
    let total: usize = spans.iter().map(|&(s, e)| text[s..e].chars().count()).sum();
    let span = (end - start).max(0.0);
    let mut acc = 0usize;
    spans
        .into_iter()
        .map(|(s, e)| {
            let len = text[s..e].chars().count();
            let a = start + span * acc as f64 / total.max(1) as f64;
            acc += len;
            let b = start + span * acc as f64 / total.max(1) as f64;
            Segment {
                text: text[s..e].to_string(),
                start: Some(a),
                end: Some(b),
                ..Segment::default()
            }
        })
        .collect()
}

/// Groups timed sentences into paragraphs (pause >= [`PARAGRAPH_GAP_SECONDS`]
/// or [`PARAGRAPH_MAX_CHARS`] reached) and renders `[hh:mm:ss] text` blocks
/// joined by blank lines.
pub fn timed_paragraphs(sentences: &[Segment]) -> String {
    let mut paragraphs: Vec<(f64, String)> = Vec::new();
    let mut prev_end: Option<f64> = None;
    for s in sentences.iter().filter(|s| !s.text.trim().is_empty()) {
        let start = s.start.unwrap_or(0.0);
        let gap = prev_end.is_some_and(|e| start - e >= PARAGRAPH_GAP_SECONDS);
        let full = paragraphs
            .last()
            .is_some_and(|(_, t)| t.chars().count() >= PARAGRAPH_MAX_CHARS);
        if paragraphs.is_empty() || gap || full {
            paragraphs.push((start, String::new()));
        }
        join_text(&mut paragraphs.last_mut().expect("paragraph").1, &s.text);
        prev_end = s.end.or(s.start);
    }
    paragraphs
        .into_iter()
        .map(|(start, text)| format!("[{}] {text}", clock(start)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Splits `text` near its middle at a clause boundary (`，、,;；：:` or a
/// space); `None` when there is no boundary.
fn split_clause(text: &str) -> Option<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mid = chars.len() / 2;
    let is_break = |c: char| matches!(c, '，' | '、' | ',' | ';' | '；' | '：' | ':' | '—');
    let candidates = |pred: &dyn Fn(char) -> bool| {
        (1..chars.len().saturating_sub(1))
            .filter(|&i| pred(chars[i]))
            .min_by_key(|&i| i.abs_diff(mid))
    };
    let at = candidates(&is_break)
        .map(|i| i + 1)
        .or_else(|| candidates(&|c: char| c == ' '))
        .or_else(|| mostly_cjk(text).then_some(mid))?;
    let a: String = chars[..at].iter().collect();
    let b: String = chars[at..].iter().collect();
    let (a, b) = (a.trim().to_string(), b.trim().to_string());
    (!a.is_empty() && !b.is_empty()).then_some((a, b))
}

/// Subtitle cues from timed sentences: sentences longer than
/// [`CUE_MAX_SECONDS`] or the character limit are split at clause
/// boundaries with time shared by character count; cues last at least
/// [`CUE_MIN_SECONDS`].
pub fn subtitle_cues(sentences: &[Segment]) -> Vec<Segment> {
    fn split(seg: Segment, out: &mut Vec<Segment>) {
        let (start, end) = (seg.start.unwrap_or(0.0), seg.end.unwrap_or(0.0));
        let limit = if mostly_cjk(&seg.text) {
            CUE_MAX_CHARS_CJK
        } else {
            CUE_MAX_CHARS_LATIN
        };
        let too_long = seg.text.chars().count() > limit || end - start > CUE_MAX_SECONDS;
        match too_long.then(|| split_clause(&seg.text)).flatten() {
            Some((a, b)) => {
                let (na, nb) = (a.chars().count() as f64, b.chars().count() as f64);
                let mid = start + (end - start) * na / (na + nb);
                split(
                    Segment {
                        text: a,
                        start: Some(start),
                        end: Some(mid),
                        ..Segment::default()
                    },
                    out,
                );
                split(
                    Segment {
                        text: b,
                        start: Some(mid),
                        end: Some(end),
                        ..Segment::default()
                    },
                    out,
                );
            }
            None => out.push(seg),
        }
    }
    let mut out = Vec::new();
    for s in sentences.iter().filter(|s| !s.text.trim().is_empty()) {
        split(
            Segment {
                text: s.text.trim().to_string(),
                start: s.start,
                end: s.end,
                ..Segment::default()
            },
            &mut out,
        );
    }
    for cue in &mut out {
        let start = cue.start.unwrap_or(0.0);
        cue.end = Some(cue.end.unwrap_or(start).max(start + CUE_MIN_SECONDS));
    }
    out
}

/// SubRip text for cues.
pub fn to_srt(cues: &[Segment]) -> String {
    cues.iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "{}\n{} --> {}\n{}\n",
                i + 1,
                timecode(c.start.unwrap_or(0.0), ','),
                timecode(c.end.unwrap_or(0.0), ','),
                c.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// WebVTT text for cues.
pub fn to_vtt(cues: &[Segment]) -> String {
    let body: Vec<String> = cues
        .iter()
        .map(|c| {
            format!(
                "{} --> {}\n{}\n",
                timecode(c.start.unwrap_or(0.0), '.'),
                timecode(c.end.unwrap_or(0.0), '.'),
                c.text
            )
        })
        .collect();
    format!("WEBVTT\n\n{}", body.join("\n"))
}

fn parse_timecode(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, sec) = match parts.as_slice() {
        [h, m, s] => (
            h.parse::<f64>().ok()?,
            m.parse::<f64>().ok()?,
            s.parse::<f64>().ok()?,
        ),
        [m, s] => (0.0, m.parse::<f64>().ok()?, s.parse::<f64>().ok()?),
        _ => return None,
    };
    Some(h * 3600.0 + m * 60.0 + sec)
}

fn strip_markup(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut depth = 0usize;
    for c in line.chars() {
        match c {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            '{' if depth == 0 && line.contains("{\\") => depth += 1,
            '}' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

/// Cues of an SRT (or WebVTT) document: timing line `a --> b`, following
/// non-empty lines as the text (markup removed, lines joined). Rolling
/// captions that repeat the previous cue's last line keep only the new
/// text; cues that end up empty are dropped.
pub fn parse_srt(text: &str) -> Vec<Segment> {
    let mut cues: Vec<Segment> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((a, b)) = line.split_once("-->") else {
            continue;
        };
        let start = parse_timecode(a);
        let end = parse_timecode(b.split_whitespace().next().unwrap_or(""));
        let (Some(start), Some(end)) = (start, end) else {
            continue;
        };
        let mut body: Vec<String> = Vec::new();
        while let Some(next) = lines.peek() {
            if next.trim().is_empty() {
                break;
            }
            let l = strip_markup(lines.next().unwrap_or_default());
            if !l.is_empty() {
                body.push(l);
            }
        }
        // Rolling captions repeat earlier lines: drop lines already shown
        // as the tail of the previous cue.
        if let Some(prev) = cues.last() {
            body.retain(|l| !prev.text.ends_with(l.as_str()) && !prev.text.contains(l.as_str()));
        }
        let mut joined = String::new();
        for l in &body {
            join_text(&mut joined, l);
        }
        if !joined.is_empty() {
            cues.push(Segment {
                text: joined,
                start: Some(start),
                end: Some(end),
                ..Segment::default()
            });
        }
    }
    cues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str, start: f64, end: f64) -> Segment {
        Segment {
            text: text.into(),
            start: Some(start),
            end: Some(end),
            ..Segment::default()
        }
    }

    #[test]
    fn loc_comment_and_located_markdown() {
        assert_eq!(
            loc_comment(2, [0.05, 0.0304, 0.4, 0.07], "section header"),
            "<!-- aimd:loc page=2 bbox=0.050,0.030,0.400,0.070 kind=section_header -->"
        );
        let blocks = [
            Segment {
                text: "Title\n".into(),
                kind: Some("title".into()),
                page: Some(1),
                bbox: Some([0.1, 0.1, 0.5, 0.2]),
                ..Segment::default()
            },
            Segment {
                text: "  ".into(),
                ..Segment::default()
            },
            Segment {
                text: "bare".into(),
                ..Segment::default()
            },
        ];
        assert_eq!(
            located_markdown(&blocks),
            "<!-- aimd:loc page=1 bbox=0.100,0.100,0.500,0.200 kind=title -->\nTitle\n\nbare"
        );
        assert_eq!(
            quad_bbox(&[[0.3, 0.1], [0.9, 0.2], [0.8, 0.4], [0.2, 0.3]]),
            [0.2, 0.1, 0.9, 0.4]
        );
    }

    #[test]
    fn clocks_and_timecodes() {
        assert_eq!(clock(3725.9), "01:02:05");
        assert_eq!(timecode(61.2345, ','), "00:01:01,235");
        assert_eq!(parse_timecode("01:02:03,500"), Some(3723.5));
        assert_eq!(parse_timecode("02:03.250"), Some(123.25));
    }

    #[test]
    fn sentences_split_on_terminators() {
        let text = "Hello world. It is 3.5 km! 你好。“真的吗？”他说。tail";
        let got: Vec<&str> = sentence_spans(text)
            .into_iter()
            .map(|(s, e)| &text[s..e])
            .collect();
        assert_eq!(
            got,
            [
                "Hello world.",
                "It is 3.5 km!",
                "你好。",
                "“真的吗？”",
                "他说。",
                "tail"
            ]
        );
    }

    #[test]
    fn sentences_take_unit_times() {
        let text = "今天天气很好。我们去公园吧！";
        let units: Vec<(String, f64, f64)> = "今天天气很好我们去公园吧"
            .chars()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as f64 * 0.5, i as f64 * 0.5 + 0.4))
            .collect();
        let got = timed_sentences(text, &units, 10.0);
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].start, got[0].end), (Some(10.0), Some(12.9)));
        assert_eq!((got[1].start, got[1].end), (Some(13.0), Some(15.9)));

        let text = "Hi there, Bob. Unaligned?";
        let units = vec![
            ("hi".into(), 1.0, 1.2),
            ("there".into(), 1.3, 1.6),
            ("Bob".into(), 1.8, 2.0),
        ];
        let got = timed_sentences(text, &units, 0.0);
        assert_eq!((got[0].start, got[0].end), (Some(1.0), Some(2.0)));
        assert_eq!((got[1].start, got[1].end), (Some(2.0), Some(2.0)));
    }

    #[test]
    fn spread_fallback_covers_the_chunk() {
        let got = spread_sentences("ab. cd.", 300.0, 600.0);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].start, Some(300.0));
        assert_eq!(got[1].end, Some(600.0));
    }

    #[test]
    fn paragraphs_break_on_pauses_and_length() {
        let sentences = [
            seg("第一句。", 0.0, 1.0),
            seg("第二句。", 1.5, 2.0),
            seg("Next part.", 5.0, 6.0),
            seg("More.", 6.1, 7.0),
        ];
        assert_eq!(
            timed_paragraphs(&sentences),
            "[00:00:00] 第一句。第二句。\n\n[00:00:05] Next part. More."
        );
        let long: Vec<Segment> = (0..40)
            .map(|i| {
                seg(
                    "A sentence of ten words or so here.",
                    i as f64,
                    i as f64 + 0.9,
                )
            })
            .collect();
        assert!(timed_paragraphs(&long).matches('[').count() > 1);
    }

    #[test]
    fn cues_split_long_sentences_and_render() {
        let cues = subtitle_cues(&[seg(
            "这是一个非常非常长的句子，它需要被拆分成两条字幕，否则观众读不完这么多文字内容。",
            0.0,
            10.0,
        )]);
        assert!(cues.len() >= 2, "{cues:?}");
        assert!(
            cues.iter()
                .all(|c| c.text.chars().count() <= CUE_MAX_CHARS_CJK)
        );
        let short = subtitle_cues(&[seg("Hi.", 1.0, 1.1)]);
        assert_eq!(short[0].end, Some(1.5));
        assert_eq!(
            to_srt(&[seg("One.", 0.0, 1.5), seg("Two.", 2.0, 3.0)]),
            "1\n00:00:00,000 --> 00:00:01,500\nOne.\n\n2\n00:00:02,000 --> 00:00:03,000\nTwo.\n"
        );
        assert_eq!(
            to_vtt(&[seg("One.", 0.0, 1.5)]),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nOne.\n"
        );
    }

    #[test]
    fn srt_parse_round_trips_and_dedupes_rolling_lines() {
        let cues = [seg("One.", 0.0, 1.5), seg("Two words.", 2.0, 3.0)];
        assert_eq!(parse_srt(&to_srt(&cues)), cues);
        let rolling = "1\n00:00:01,000 --> 00:00:02,000\n<c>hello there</c>\n\n\
                       2\n00:00:02,000 --> 00:00:03,000\nhello there\ngeneral Kenobi\n";
        let got = parse_srt(rolling);
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].text, "general Kenobi");
    }
}
