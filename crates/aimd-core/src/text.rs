//! Markdown title extraction and context-window chunking.
//!
//! Exact port of `aimd.core.process` (`_extract_title_from_content`,
//! `_split_markdown_by_header_level`, `_split_markdown_by_headers`,
//! `_split_text_by_paragraphs`, `_combine_sections_for_processing`,
//! `_text_context_from_markdown`). The Python regexes are hand-expanded;
//! every length is counted in Unicode scalar values like Python `len()`.

use crate::model::TextContext;
use crate::pyfmt::{is_space, strip};

/// Default context-window chunk size (characters).
pub const MAX_CHUNK_SIZE: usize = 40000;

/// Separator placed between combined sections.
fn section_separator() -> String {
    format!("\n\n{}\n\n", "=".repeat(80))
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Extract and clean a title from Markdown text.
pub fn extract_title_from_content(content: &str, fallback_title: &str) -> String {
    if strip(content).is_empty() {
        return fallback_title.to_string();
    }
    let lines: Vec<&str> = strip(content)
        .split('\n')
        .map(strip)
        .filter(|l| !l.is_empty())
        .collect();

    let mut extracted: Option<String> = None;
    for line in &lines {
        if let Some(rest) = line.strip_prefix("# ") {
            extracted = Some(strip(rest).to_string());
            break;
        }
    }
    if extracted.as_deref().is_none_or(str::is_empty) {
        extracted = None;
        for (i, line) in lines.iter().enumerate() {
            if let Some(next) = lines.get(i + 1)
                && (next.chars().all(|c| c == '=') || next.chars().all(|c| c == '-'))
            {
                extracted = Some((*line).to_string());
                break;
            }
        }
    }
    if extracted.as_deref().is_none_or(str::is_empty) {
        extracted = None;
        for line in lines.iter().take(5) {
            let blocked = ["#", "!", "[", "<", "{", ":::", "---"]
                .iter()
                .any(|p| line.starts_with(p));
            if !line.is_empty() && !blocked && !line.to_lowercase().starts_with("http") {
                extracted = Some((*line).to_string());
                break;
            }
        }
    }
    let Some(extracted) = extracted.filter(|e| !e.is_empty()) else {
        return fallback_title.to_string();
    };

    let clean = strip_leading_hashes(&extracted);
    let clean = unwrap_emphasis(&clean);
    let clean = unwrap_links(&clean);
    let clean = remove_braces_parens_tags(&clean);
    let clean = clean
        .trim_start_matches(['"', '\''])
        .trim_end_matches(['"', '\''])
        .to_string();
    let clean = collapse_whitespace(&clean);
    let clean = strip(&clean).to_string();
    let clean = clean
        .trim_end_matches(['。', '，', '、', '；', '：', '！', '？'])
        .to_string();

    if clean.is_empty() {
        fallback_title.to_string()
    } else {
        clean
    }
}

/// `re.sub(r"^#+\s*", "", s)`
fn strip_leading_hashes(s: &str) -> String {
    if !s.starts_with('#') {
        return s.to_string();
    }
    s.trim_start_matches('#')
        .trim_start_matches(is_space)
        .to_string()
}

/// `re.sub(r"\*+([^*]+)\*+", r"\1", s)`
fn unwrap_emphasis(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '*' {
            let mut j = i;
            while j < chars.len() && chars[j] == '*' {
                j += 1;
            }
            let body_start = j;
            while j < chars.len() && chars[j] != '*' {
                j += 1;
            }
            if j > body_start && j < chars.len() {
                // chars[j] is a star: consume the closing run greedily.
                out.extend(&chars[body_start..j]);
                while j < chars.len() && chars[j] == '*' {
                    j += 1;
                }
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)`
fn unwrap_links(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '[' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != ']' {
                j += 1;
            }
            let text_end = j;
            if text_end > i + 1 && j < chars.len() && chars.get(j + 1) == Some(&'(') {
                let mut k = j + 2;
                while k < chars.len() && chars[k] != ')' {
                    k += 1;
                }
                if k > j + 2 && k < chars.len() {
                    out.extend(&chars[i + 1..text_end]);
                    i = k + 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `re.sub(r"\{[^}]*\}|\([^)]*\)|#\S+", "", s)`
fn remove_braces_parens_tags(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let close = match c {
            '{' => Some('}'),
            '(' => Some(')'),
            _ => None,
        };
        if let Some(close) = close
            && let Some(offset) = chars[i + 1..].iter().position(|&x| x == close)
        {
            i += offset + 2;
            continue;
        }
        if c == '#' && chars.get(i + 1).is_some_and(|&n| !is_space(n)) {
            let mut j = i + 1;
            while j < chars.len() && !is_space(chars[j]) {
                j += 1;
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// `re.sub(r"\s+", " ", s)`
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_space = false;
    for c in s.chars() {
        if is_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// Combine sections into chunks of at most `max_chunk_size` characters
/// (single oversized sections stay whole).
pub fn combine_sections_for_processing(
    section_data: &[(Option<String>, String)],
    max_chunk_size: usize,
) -> Vec<String> {
    let separator = section_separator();
    let separator_size = char_len(&separator);
    let mut combined = Vec::new();
    let mut parts: Vec<&str> = Vec::new();
    let mut size = 0usize;

    for (_, content) in section_data {
        let section_size = char_len(content);
        let would_exceed = size + section_size + if parts.is_empty() { 0 } else { separator_size }
            > max_chunk_size;
        if would_exceed && !parts.is_empty() {
            combined.push(parts.join(&separator));
            parts.clear();
            size = 0;
        }
        parts.push(content);
        size += section_size;
        if parts.len() > 1 {
            size += separator_size;
        }
    }
    if !parts.is_empty() {
        combined.push(parts.join(&separator));
    }
    combined
}

/// `re.split(r"\n\s*\n", text)`.
fn split_blank_lines(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while idx < text.len() {
        if text.as_bytes()[idx] == b'\n' {
            let after = idx + 1;
            let mut last_newline: Option<usize> = None;
            for (off, c) in text[after..].char_indices() {
                if !is_space(c) {
                    break;
                }
                if c == '\n' {
                    last_newline = Some(after + off);
                }
            }
            if let Some(end) = last_newline {
                pieces.push(&text[start..idx]);
                start = end + 1;
                idx = end + 1;
                continue;
            }
        }
        idx += text[idx..].chars().next().map_or(1, char::len_utf8);
    }
    pieces.push(&text[start..]);
    pieces
}

/// Split text on paragraph boundaries, hard-wrapping oversized blocks.
pub fn split_text_by_paragraphs(text: &str, max_chunk_size: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = split_blank_lines(text)
        .into_iter()
        .map(strip)
        .filter(|b| !b.is_empty())
        .collect();
    if paragraphs.is_empty() {
        let stripped = strip(text);
        return if stripped.is_empty() {
            Vec::new()
        } else {
            vec![stripped.to_string()]
        };
    }

    let mut chunks = Vec::new();
    let mut parts: Vec<&str> = Vec::new();
    let mut size = 0usize;
    for paragraph in paragraphs {
        let paragraph_len = char_len(paragraph);
        let separator = if parts.is_empty() { 0 } else { 2 };
        if !parts.is_empty() && size + separator + paragraph_len > max_chunk_size {
            chunks.push(parts.join("\n\n"));
            parts.clear();
            size = 0;
        }
        if paragraph_len <= max_chunk_size {
            parts.push(paragraph);
            size += if size > 0 { 2 } else { 0 } + paragraph_len;
            continue;
        }
        if !parts.is_empty() {
            chunks.push(parts.join("\n\n"));
            parts.clear();
            size = 0;
        }
        let chars: Vec<char> = paragraph.chars().collect();
        for piece in chars.chunks(max_chunk_size.max(1)) {
            let piece: String = piece.iter().collect();
            let piece = strip(&piece);
            if !piece.is_empty() {
                chunks.push(piece.to_string());
            }
        }
    }
    if !parts.is_empty() {
        chunks.push(parts.join("\n\n"));
    }
    chunks
}

/// `re.match(f"^{'#' * level}\\s+(.+)$", line)` -> header title (lines
/// never contain `\n` here: they come from `split('\n')`).
fn match_header(line: &str, level: usize) -> Option<&str> {
    let bytes = line.as_bytes();
    if bytes.len() < level || !bytes[..level].iter().all(|&b| b == b'#') {
        return None;
    }
    let rest = &line[level..];
    let first = rest.chars().next()?;
    if !is_space(first) {
        return None;
    }
    // `\s+` takes at least `first`; `.+` needs at least one more character.
    let remainder = &rest[first.len_utf8()..];
    if remainder.is_empty() {
        return None;
    }
    Some(remainder)
}

/// Split Markdown at one header level into `(title, content)` sections.
pub fn split_markdown_by_header_level(
    markdown_content: &str,
    header_level: usize,
) -> Vec<(Option<String>, String)> {
    let mut sections = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut title: Option<String> = None;

    fn save(sections: &mut Vec<(Option<String>, String)>, lines: &[&str], title: &Option<String>) {
        if !lines.is_empty() {
            let content = strip(&lines.join("\n")).to_string();
            if !content.is_empty() {
                sections.push((title.clone(), content));
            }
        }
    }

    for line in markdown_content.split('\n') {
        if let Some(header) = match_header(line, header_level) {
            save(&mut sections, &current, &title);
            title = Some(strip(header).to_string());
            current = vec![line];
        } else {
            current.push(line);
        }
    }
    save(&mut sections, &current, &title);
    sections
}

/// Split Markdown at the first header level (1-6) whose sections all fit,
/// else fall back to paragraph splitting (`header_level = None`).
pub fn split_markdown_by_headers(
    markdown_content: &str,
    max_chunk_size: usize,
) -> (Vec<(Option<String>, String)>, Option<u8>) {
    for level in 1..=6usize {
        let sections = split_markdown_by_header_level(markdown_content, level);
        if sections.len() <= 1 {
            continue;
        }
        let max_section = sections.iter().map(|(_, c)| char_len(c)).max().unwrap_or(0);
        if max_section <= max_chunk_size {
            crate::log_info!(
                "Using split level {level} - all chunks under {max_chunk_size} chars (max: {max_section})"
            );
            return (sections, Some(level as u8));
        }
    }
    let chunks = split_text_by_paragraphs(markdown_content, max_chunk_size);
    (chunks.into_iter().map(|c| (None, c)).collect(), None)
}

/// Wrap Markdown output into the [`TextContext`] shape.
pub fn text_context_from_markdown(
    markdown: &str,
    fallback_title: &str,
    title: Option<&str>,
    max_chunk_size: usize,
) -> TextContext {
    let resolved_title = match title {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => extract_title_from_content(markdown, fallback_title),
    };
    let stripped = strip(markdown);
    if char_len(stripped) <= max_chunk_size {
        return TextContext {
            title: resolved_title,
            chunk_list: if stripped.is_empty() {
                Vec::new()
            } else {
                vec![stripped.to_string()]
            },
            split_header_level: None,
        };
    }
    let (sections, header_level) = split_markdown_by_headers(stripped, max_chunk_size);
    let section_data: Vec<(Option<String>, String)> = sections
        .into_iter()
        .filter_map(|(t, c)| {
            let c = strip(&c).to_string();
            (!c.is_empty()).then_some((t, c))
        })
        .collect();
    TextContext {
        title: resolved_title,
        chunk_list: combine_sections_for_processing(&section_data, max_chunk_size),
        split_header_level: header_level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_prefers_atx_header() {
        assert_eq!(
            extract_title_from_content("intro\n# **My** [Doc](x.md) {#id}\n", "fb"),
            "My Doc"
        );
    }

    #[test]
    fn title_uses_setext_then_first_plain_line() {
        assert_eq!(
            extract_title_from_content("Title\n=====\nbody", "fb"),
            "Title"
        );
        assert_eq!(
            extract_title_from_content("![img](a.png)\nhttp://x\n\"Quoted line。\"", "fb"),
            "Quoted line"
        );
        assert_eq!(extract_title_from_content("   \n", "fb"), "fb");
        assert_eq!(extract_title_from_content("# \n", "fb"), "fb");
        assert_eq!(extract_title_from_content("[link only](u)", "fb"), "fb");
    }

    #[test]
    fn title_cleanup_regexes() {
        assert_eq!(unwrap_emphasis("a **b** c ***d***"), "a b c d");
        assert_eq!(unwrap_emphasis("**"), "**");
        assert_eq!(unwrap_emphasis("a*b"), "a*b");
        assert_eq!(unwrap_links("[a](b) [c] [](d) [e]("), "a [c] [](d) [e](");
        assert_eq!(
            remove_braces_parens_tags("x (y) {z} #tag # w ("),
            "x    # w ("
        );
        assert_eq!(strip_leading_hashes("##  Title"), "Title");
        assert_eq!(collapse_whitespace("a \t\n b"), "a b");
    }

    #[test]
    fn split_markdown_by_headers_fallback_without_headers() {
        // test_file_processing.py::test_split_markdown_by_headers_fallback_without_headers
        let markdown = "Paragraph ".repeat(6000);
        let (sections, level) = split_markdown_by_headers(&markdown, 4000);
        assert_eq!(level, None);
        assert!(sections.len() > 1);
        assert!(sections.iter().all(|(_, c)| c.chars().count() <= 4000));
    }

    #[test]
    fn split_picks_first_level_that_fits() {
        let body = "x".repeat(50);
        let md = format!("# A\n{body}\n## A1\n{body}\n# B\n{body}");
        let (sections, level) = split_markdown_by_headers(&md, 200);
        assert_eq!(level, Some(1));
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].0.as_deref(), Some("A"));
        assert_eq!(sections[1].0.as_deref(), Some("B"));
    }

    #[test]
    fn header_match_requires_exact_level_and_text() {
        assert_eq!(match_header("# Title", 1), Some("Title"));
        assert_eq!(match_header("## Title", 1), None);
        assert_eq!(match_header("#Title", 1), None);
        assert_eq!(match_header("# ", 1), None);
        assert_eq!(match_header("#  ", 1), Some(" "));
    }

    #[test]
    fn blank_line_split_matches_python_re_split() {
        assert_eq!(
            split_blank_lines("a\n\nb\n \t\n\nc\nd"),
            vec!["a", "b", "c\nd"]
        );
        assert_eq!(split_blank_lines("a\n"), vec!["a\n"]);
        assert_eq!(split_blank_lines("\n\n"), vec!["", ""]);
    }

    #[test]
    fn paragraphs_hard_wrap_oversized_blocks() {
        let text = format!("short\n\n{}\n\ntail", "y".repeat(25));
        assert_eq!(
            split_text_by_paragraphs(&text, 10),
            vec!["short", "yyyyyyyyyy", "yyyyyyyyyy", "yyyyy", "tail"]
        );
        assert_eq!(split_text_by_paragraphs("a\n\nb", 10), vec!["a\n\nb"]);
    }

    #[test]
    fn combine_respects_separator_budget() {
        let sections: Vec<(Option<String>, String)> =
            ["a".repeat(10), "b".repeat(10), "c".repeat(100)]
                .into_iter()
                .map(|c| (None, c))
                .collect();
        let sep = section_separator();
        let chunks = combine_sections_for_processing(&sections, 104);
        assert_eq!(
            chunks,
            vec![
                format!("{}{sep}{}", "a".repeat(10), "b".repeat(10)),
                "c".repeat(100)
            ]
        );
    }

    #[test]
    fn short_markdown_is_one_chunk_without_split_metadata() {
        // test_ocr.py::test_process_input_ocr_short_markdown_has_no_split_metadata
        let ctx =
            text_context_from_markdown("  short ocr text \n", "page", Some("page"), MAX_CHUNK_SIZE);
        assert_eq!(ctx.title, "page");
        assert_eq!(ctx.chunk_list, vec!["short ocr text"]);
        assert_eq!(ctx.split_header_level, None);
        let empty = text_context_from_markdown("", "stem", None, MAX_CHUNK_SIZE);
        assert_eq!(empty.title, "stem");
        assert!(empty.chunk_list.is_empty());
    }

    #[test]
    fn long_ocr_markdown_splits_on_page_headers() {
        // test_ocr.py::test_process_input_ocr_text_context_owned_by_core
        let md = format!(
            "## Page 1\n\n{}\n\n## Page 2\n\n{}",
            "A".repeat(300),
            "B".repeat(300)
        );
        let ctx = text_context_from_markdown(&md, "scan", Some("scan"), 400);
        assert_eq!(ctx.title, "scan");
        assert_eq!(ctx.split_header_level, Some(2));
        assert_eq!(ctx.chunk_list.len(), 2);
        assert!(ctx.chunk_list[0].contains("## Page 1"));
        assert!(ctx.chunk_list[1].contains("## Page 2"));
    }
}
