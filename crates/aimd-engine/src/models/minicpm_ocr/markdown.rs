//! Region text post-processing and page Markdown assembly, a port of the
//! MiniCPM-OCR SDK's `postprocess/markdown.py` (`postprocess_content`,
//! `split_text_block`, `assemble_markdown`, ...), `pipeline/_text.py`
//! (`_truncate_repetitive_suffix`, `_normalize_crop_text`,
//! `_normalize_and_dedupe_blocks`) and `_utils.classify_engine_text`.
//!
//! Python semantics kept on purpose:
//!
//! - `str.strip()` / `\s` use Python's whitespace set, which is Unicode
//!   `White_Space` plus U+001C..U+001F ([`py_is_space`]); regexes spell
//!   `\s` as `[\s\x1C-\x1F]`.
//! - Lookarounds (`(?<!\\)\$(?!\$)` and friends) are hand-written scans;
//!   everything else goes through the `regex` crate, whose leftmost-first
//!   matching (lazy quantifiers included) equals Python's backtracking
//!   result for these lookaround-free patterns.
//! - Lengths that Python measures in code points (`truncate_repetitive_content`)
//!   are measured in `char`s.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::{Captures, NoExpand, Regex};

use super::latex::repair_latex_formula;
use super::otsl::convert_otsl_to_html;

/// Python's `str.isspace()` for one character.
pub(crate) fn py_is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.strip()`.
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(py_is_space)
}

/// Python's `str.rstrip()`.
pub(crate) fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_is_space)
}

macro_rules! static_regex {
    ($name:ident, $pat:expr) => {
        fn $name() -> &'static Regex {
            static RE: OnceLock<Regex> = OnceLock::new();
            RE.get_or_init(|| Regex::new($pat).expect("static regex"))
        }
    };
}

static_regex!(re_spaces_tabs, r"[ \t]+");
static_regex!(re_many_newlines, r"\n{3,}");
static_regex!(re_inline_latex, r"\\\((.+?)\\\)");
static_regex!(re_display_bracket_sep, r"\\\][\s\x1C-\x1F]*\\\[");
static_regex!(
    re_closing_tokens,
    r"\\end\{[^}]*\}|\\\]|\$\$|</(?:table|tr|td|th)>"
);
static_regex!(
    re_empty_bracket_math,
    r"\\\[[\s\x1C-\x1F]*\\\]|\\\([\s\x1C-\x1F]*\\\)"
);
static_regex!(re_display_span, r"(?s)\\\[(.*?)\\\]|\$\$(.*?)\$\$");
static_regex!(re_spacing_punct, r"\\[,;:!]");
static_regex!(
    re_spacing_hv,
    r"\\(?:hspace|vspace)[\s\x1C-\x1F]*\{[^{}]*\}"
);
static_regex!(re_space_braces, r"[\s\x1C-\x1F{}]");
static_regex!(
    re_engine_failure,
    r"(?s)\A[\s\x1C-\x1F]*<!--[\s\x1C-\x1F]*(.*?)[\s\x1C-\x1F]*-->[\s\x1C-\x1F]*\z"
);

/// A region's prompt-map `output_format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionFormat {
    Text,
    Html,
    Latex,
}

impl RegionFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Html => "html",
            Self::Latex => "latex",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "html" => Some(Self::Html),
            "latex" => Some(Self::Latex),
            _ => None,
        }
    }
}

/// `TEXT_BLOCK_LABELS`: labels whose lines are split into paragraphs.
pub const TEXT_BLOCK_LABELS: [&str; 12] = [
    "text",
    "content",
    "abstract",
    "reference",
    "reference_content",
    "vertical_text",
    "vision_footnote",
    "algorithm",
    "header",
    "footer",
    "footnote",
    "aside_text",
];

/// `constants.OTSL_LABELS`: labels whose text goes through OTSL -> HTML.
pub const OTSL_LABELS: [&str; 2] = ["table", "chart"];

/// `IMAGE_LABELS`.
pub const IMAGE_LABELS: [&str; 3] = ["image", "header_image", "footer_image"];

/// `markdown_assembly.ignore_labels`.
pub const IGNORE_LABELS: [&str; 5] = ["number", "footnote", "header", "footer", "aside_text"];

/// `markdown_assembly.image_format`.
pub const IMAGE_PLACEHOLDER: &str = "![image]()";

// --- _utils.classify_engine_text ---------------------------------------------------

const ENGINE_FAILURE_MARKERS: [&str; 5] = [
    "failed",
    "cuda out of memory",
    "outofmemoryerror",
    "enginecore",
    "traceback",
];

/// `classify_engine_text`: a whole-output `<!-- ... -->` comment naming an
/// engine failure becomes `("", Some(message))`.
pub fn classify_engine_text(raw: &str) -> (String, Option<String>) {
    if let Some(caps) = re_engine_failure().captures(raw) {
        let msg = py_strip(caps.get(1).map_or("", |m| m.as_str()));
        let lowered = msg.to_lowercase();
        if ENGINE_FAILURE_MARKERS.iter().any(|m| lowered.contains(m)) {
            return (String::new(), Some(msg.to_string()));
        }
    }
    (raw.to_string(), None)
}

// --- markdown.py ---------------------------------------------------------------

/// `truncate_repetitive_content(content, min_count)`.
pub fn truncate_repetitive_content(content: &str, min_count: usize) -> String {
    if content.is_empty() || content.chars().count() < min_count {
        return content.to_string();
    }
    let stripped = py_strip(content);
    if stripped.is_empty() {
        return content.to_string();
    }
    let chars: Vec<char> = stripped.chars().collect();
    let n = chars.len();
    let single_line = !stripped.contains('\n');

    if single_line && n > 100 {
        for unit_len in 8..=n / 5 {
            let unit = &chars[n - unit_len..];
            let mut count = 0;
            let mut pos = n - unit_len;
            loop {
                if chars[pos..pos + unit_len] != *unit {
                    break;
                }
                count += 1;
                if pos < unit_len {
                    break;
                }
                pos -= unit_len;
            }
            if count >= 5 {
                let start = n - count * unit_len;
                if 2 * unit_len * count > n {
                    return chars[..start].iter().collect();
                }
            }
        }
    }

    if single_line && n > 10 {
        for unit_len in 1..=n / 2 {
            let k = n / unit_len;
            let periodic = (unit_len..unit_len * k).all(|i| chars[i] == chars[i % unit_len]);
            if periodic {
                if k >= 10 {
                    return chars[..unit_len].iter().collect();
                }
                break;
            }
        }
    }

    let lines: Vec<&str> = stripped
        .split('\n')
        .map(py_strip)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() >= 10 {
        // Counter.most_common(1): the first-inserted line among the most
        // frequent.
        let mut counts: Vec<(&str, usize)> = Vec::new();
        for &line in &lines {
            match counts.iter_mut().find(|(l, _)| *l == line) {
                Some(entry) => entry.1 += 1,
                None => counts.push((line, 1)),
            }
        }
        let mut best = counts[0];
        for &entry in &counts[1..] {
            if entry.1 > best.1 {
                best = entry;
            }
        }
        if best.1 >= 10 && best.1 as f64 / lines.len() as f64 >= 0.8 {
            return best.0.to_string();
        }
    }
    content.to_string()
}

const SPACING_COMMANDS: [&str; 9] = [
    "quad",
    "qquad",
    "enspace",
    "thinspace",
    "medspace",
    "thickspace",
    "negthinspace",
    "negmedspace",
    "negthickspace",
];

/// `_strip_spacing_tokens`: what is left of a formula once every spacing
/// command, brace and whitespace is removed.
fn strip_spacing_tokens(content: &str) -> String {
    // `\\(?:quad|...)(?![A-Za-z])` -> "".
    let b = content.as_bytes();
    let mut step1 = String::with_capacity(content.len());
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' {
            let hit = SPACING_COMMANDS.iter().find(|name| {
                b[i + 1..].starts_with(name.as_bytes())
                    && !b
                        .get(i + 1 + name.len())
                        .is_some_and(u8::is_ascii_alphabetic)
            });
            if let Some(name) = hit {
                step1.push_str(&content[last..i]);
                i += 1 + name.len();
                last = i;
                continue;
            }
        }
        i += 1;
    }
    step1.push_str(&content[last..]);
    let s = re_spacing_punct().replace_all(&step1, "");
    let s = s.replace("\\ ", "");
    let s = re_spacing_hv().replace_all(&s, "");
    let s = s.replace('~', "");
    re_space_braces().replace_all(&s, "").into_owned()
}

/// `_repair_display_dollars`: drop `$$` delimiters that open nothing.
fn repair_display_dollars(markdown: &str) -> String {
    let mut tokens: Vec<&str> = Vec::new();
    let mut last = 0;
    while let Some(k) = markdown[last..].find("$$") {
        let at = last + k;
        tokens.push(&markdown[last..at]);
        tokens.push("$$");
        last = at + 2;
    }
    tokens.push(&markdown[last..]);

    let mut output: Vec<&str> = Vec::new();
    let mut in_display = false;
    for (index, &token) in tokens.iter().enumerate() {
        if token != "$$" {
            output.push(token);
            continue;
        }
        let next_segment = tokens.get(index + 1).copied().unwrap_or("");
        let next_is_delimiter = tokens.get(index + 2) == Some(&"$$");
        let has_later_delimiter = tokens.iter().skip(index + 2).any(|&t| t == "$$");
        if !in_display {
            if (py_strip(next_segment).is_empty() && next_is_delimiter) || !has_later_delimiter {
                continue;
            }
            output.push(token);
            in_display = true;
        } else {
            output.push(token);
            in_display = false;
        }
    }
    if in_display {
        output.pop();
    }
    output.concat()
}

/// `repair_display_formula`: repair malformed display math delimiters.
pub fn repair_display_formula(content: &str) -> String {
    let content = repair_display_dollars(content);
    let content = re_empty_bracket_math().replace_all(&content, " ");
    re_display_span()
        .replace_all(&content, |caps: &Captures| {
            let whole = caps.get(0).map_or("", |m| m.as_str());
            let is_bracket = whole.starts_with("\\[");
            let body =
                if is_bracket { caps.get(1) } else { caps.get(2) }.map_or("", |m| m.as_str());
            let stripped = py_rstrip(body);
            let trailing = &body[stripped.len()..];
            let mut body = body.to_string();
            // `(?<!\\)\\$`: one (unescaped) trailing backslash.
            if stripped.ends_with('\\') && !stripped[..stripped.len() - 1].ends_with('\\') {
                body = format!("{}{trailing}", &stripped[..stripped.len() - 1]);
            }
            if !py_strip(&body).is_empty() && strip_spacing_tokens(&body).is_empty() {
                return " ".to_string();
            }
            if is_bracket {
                format!("\\[{body}\\]")
            } else {
                format!("$${body}$$")
            }
        })
        .into_owned()
}

/// First alternative of `INLINE_MATH_RE` at `p`:
/// `(?<!\\)\$(?!\$)(?:[^$\n]|\\.)+?(?<!\\)\$(?!\$)`. Returns the match end.
/// The lazy body is a depth-first search over positions in the order the
/// backtracking engine visits them (exit first, then `[^$\n]`, then `\\.`);
/// a position that failed once fails again, so it is visited once.
fn inline_dollar_at(s: &[char], p: usize, full: bool) -> Option<usize> {
    let n = s.len();
    if s.get(p) != Some(&'$') || (p > 0 && s[p - 1] == '\\') || s.get(p + 1) == Some(&'$') {
        return None;
    }
    let closes = |q: usize| {
        q < n
            && s[q] == '$'
            && s[q - 1] != '\\'
            && s.get(q + 1) != Some(&'$')
            && (!full || q + 1 == n)
    };
    let children = |pos: usize, stack: &mut Vec<usize>| {
        // Pushed in reverse so `[^$\n]` is explored first.
        if pos + 1 < n && s[pos] == '\\' && s[pos + 1] != '\n' {
            stack.push(pos + 2);
        }
        if pos < n && s[pos] != '$' && s[pos] != '\n' {
            stack.push(pos + 1);
        }
    };
    let mut visited = vec![false; n + 2];
    let mut stack = Vec::new();
    children(p + 1, &mut stack);
    while let Some(pos) = stack.pop() {
        if visited[pos] {
            continue;
        }
        visited[pos] = true;
        if closes(pos) {
            return Some(pos + 1);
        }
        children(pos, &mut stack);
    }
    None
}

/// Second alternative of `INLINE_MATH_RE` at `p`: `\\\([^\n]*?\\\)`.
fn inline_paren_at(s: &[char], p: usize, full: bool) -> Option<usize> {
    let n = s.len();
    if !(p + 1 < n && s[p] == '\\' && s[p + 1] == '(') {
        return None;
    }
    let mut q = p + 2;
    while q + 1 < n {
        if s[q] == '\\' && s[q + 1] == ')' && (!full || q + 2 == n) {
            return Some(q + 2);
        }
        if s[q] == '\n' {
            return None;
        }
        q += 1;
    }
    None
}

fn inline_math_at(s: &[char], p: usize, full: bool) -> Option<usize> {
    inline_dollar_at(s, p, full).or_else(|| inline_paren_at(s, p, full))
}

/// `has_mixed_inline_formula_content`: an inline-formula crop that also
/// holds ordinary text.
pub fn has_mixed_inline_formula_content(content: &str) -> bool {
    let stripped = py_strip(content);
    if stripped.is_empty() {
        return false;
    }
    let s: Vec<char> = stripped.chars().collect();
    if inline_math_at(&s, 0, true) == Some(s.len()) {
        return false;
    }
    let mut without = String::new();
    let mut p = 0;
    while p < s.len() {
        if let Some(end) = inline_math_at(&s, p, false) {
            p = end;
        } else {
            without.push(s[p]);
            p += 1;
        }
    }
    !py_strip(&without).is_empty()
}

fn is_pipe_table_row(line: &str) -> bool {
    let stripped = py_strip(line);
    let pipes = stripped.matches('|').count();
    (pipes >= 2 && stripped.starts_with('|')) || (pipes >= 2 && stripped.contains("---"))
}

/// `len(re.findall(r"(?<!\\)\$\$", line))`.
fn display_dollar_count(line: &str) -> usize {
    let b = line.as_bytes();
    let (mut i, mut count) = (0, 0);
    while i + 1 < b.len() {
        if b[i] == b'$' && b[i + 1] == b'$' && (i == 0 || b[i - 1] != b'\\') {
            count += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    count
}

/// `len(re.findall(r"(?<!\\)(?<!\$)\$(?!\$)", line))`.
fn inline_dollar_count(line: &str) -> usize {
    let b = line.as_bytes();
    (0..b.len())
        .filter(|&i| {
            b[i] == b'$'
                && (i == 0 || (b[i - 1] != b'\\' && b[i - 1] != b'$'))
                && b.get(i + 1) != Some(&b'$')
        })
        .count()
}

/// `open` occurs and the text after its first occurrence lacks `close`.
fn opens_unclosed(line: &str, open: &str, close: &str) -> bool {
    line.find(open)
        .is_some_and(|at| !line[at + open.len()..].contains(close))
}

/// `split_text_block`: one paragraph per text line, keeping code fences,
/// HTML / pipe tables, display math and environments intact.
pub fn split_text_block(content: &str, protect_inline_math: bool) -> String {
    let lines: Vec<&str> = content.split('\n').collect();
    let mut output: Vec<&str> = Vec::new();
    let (mut in_code, mut in_dollar, mut in_bracket, mut in_html_table) =
        (false, false, false, false);
    let (mut in_inline_dollar, mut in_inline_paren) = (false, false);
    let mut env_depth: i64 = 0;
    let last_nonblank = |output: &Vec<&str>| output.last().is_some_and(|l| !py_strip(l).is_empty());

    for (index, &line) in lines.iter().enumerate() {
        let stripped = py_strip(line);
        if stripped.starts_with("```") {
            output.push(line);
            in_code = !in_code;
            continue;
        }
        if in_code {
            output.push(line);
            continue;
        }
        let lower = stripped.to_lowercase();
        if lower.contains("<table") {
            in_html_table = true;
        }
        if in_html_table {
            output.push(line);
            if lower.contains("</table>") {
                in_html_table = false;
            }
            continue;
        }
        if is_pipe_table_row(line) {
            output.push(line);
            let next_is_pipe_row = lines.get(index + 1).is_some_and(|l| is_pipe_table_row(l));
            if !next_is_pipe_row && last_nonblank(&output) {
                output.push("");
            }
            continue;
        }

        let display_dollars = display_dollar_count(line);
        let bracket_opens = opens_unclosed(line, "\\[", "\\]");
        let inline_dollars = inline_dollar_count(line);
        let inline_paren_opens = opens_unclosed(line, "\\(", "\\)");
        let net_env =
            line.matches("\\begin{").count() as i64 - line.matches("\\end{").count() as i64;

        let protected = in_code
            || in_dollar
            || in_bracket
            || in_html_table
            || in_inline_dollar
            || in_inline_paren
            || env_depth > 0;
        if protected {
            output.push(line);
            if in_dollar && display_dollars % 2 == 1 {
                in_dollar = false;
            }
            if in_bracket && line.contains("\\]") {
                in_bracket = false;
            }
            if in_inline_dollar && inline_dollars % 2 == 1 {
                in_inline_dollar = false;
            }
            if in_inline_paren && line.contains("\\)") {
                in_inline_paren = false;
            }
            env_depth = (env_depth + net_env).max(0);
            continue;
        }
        if display_dollars % 2 == 1 || bracket_opens || net_env > 0 {
            output.push(line);
            in_dollar = display_dollars % 2 == 1;
            in_bracket = bracket_opens;
            env_depth = (env_depth + net_env).max(0);
            continue;
        }
        if protect_inline_math && (inline_dollars % 2 == 1 || inline_paren_opens) {
            output.push(line);
            in_inline_dollar = inline_dollars % 2 == 1;
            in_inline_paren = inline_paren_opens;
            continue;
        }
        if !stripped.is_empty() {
            if last_nonblank(&output) {
                output.push("");
            }
            output.push(stripped);
            output.push("");
        }
    }
    let joined = output.join("\n");
    re_many_newlines()
        .replace_all(&joined, "\n\n")
        .trim_matches('\n')
        .to_string()
}

/// `postprocess_content(content, output_format, label)`: per-format clean
/// up of one recognised crop.
pub fn postprocess_content(content: &str, format: RegionFormat, label: &str) -> String {
    let min_count = if label == "table" { 5000 } else { 50 };
    let mut content = truncate_repetitive_content(py_strip(content), min_count);
    match format {
        RegionFormat::Latex => {
            if label == "display_formula" {
                content = repair_display_formula(&content);
            }
            for (left, right) in [("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")] {
                if content.starts_with(left) && content.ends_with(right) {
                    let (start, end) = (left.len(), content.len() - right.len());
                    content = if start <= end {
                        py_strip(&content[start..end]).to_string()
                    } else {
                        String::new()
                    };
                }
            }
            if label == "display_formula" {
                content = re_display_bracket_sep()
                    .replace_all(&content, NoExpand("$$   $$"))
                    .replace("\\[", "")
                    .replace("\\]", "");
            }
            if label == "display_formula" || label == "inline_formula" {
                content = repair_latex_formula(&content, label);
            }
        }
        RegionFormat::Html => {
            let mut s = content.as_str();
            if let Some(rest) = s.strip_prefix("```html") {
                s = rest;
            }
            if let Some(rest) = s.strip_prefix("```") {
                s = rest;
            }
            if let Some(rest) = s.strip_suffix("```") {
                s = rest;
            }
            content = py_strip(s).to_string();
            if label == "table" {
                let open = content.matches("<table").count();
                let close = content.matches("</table>").count();
                if open > close {
                    content.push_str(&"</table>".repeat(open - close));
                }
            }
        }
        RegionFormat::Text => {}
    }
    content
}

// --- pipeline/_text.py -------------------------------------------------------------

/// `_truncate_repetitive_suffix`: cut the text where a run of 6..=64 words
/// first repeats three times in a row, keeping any closing tokens of the
/// cut tail.
pub fn truncate_repetitive_suffix(text: &str) -> String {
    let mut words: Vec<(usize, &str)> = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (py_is_space(c), start) {
            (false, None) => start = Some(i),
            (true, Some(s)) => {
                words.push((s, &text[s..i]));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        words.push((s, &text[s..]));
    }
    let n = words.len();
    if n < 18 {
        return text.to_string();
    }
    let normalized: Vec<String> = words.iter().map(|(_, w)| w.to_lowercase()).collect();
    let mut earliest_cut = text.len();
    let max_window = (n / 3).min(64);
    for window in 6..=max_window {
        let limit = n - window * 3 + 1;
        for start in 0..limit {
            let first = &normalized[start..start + window];
            if first == &normalized[start + window..start + 2 * window]
                && first == &normalized[start + 2 * window..start + 3 * window]
            {
                earliest_cut = earliest_cut.min(words[start].0);
                break;
            }
        }
    }
    if earliest_cut < text.len() {
        let mut truncated = py_rstrip(&text[..earliest_cut]).to_string();
        let closing: Vec<&str> = re_closing_tokens()
            .find_iter(&text[earliest_cut..])
            .map(|m| m.as_str())
            .collect();
        if !closing.is_empty() {
            truncated.push(' ');
            truncated.push_str(&closing.join(" "));
        }
        return truncated;
    }
    text.to_string()
}

/// `_normalize_crop_text(text, label)`.
pub fn normalize_crop_text(text: &str, label: &str) -> String {
    let mut out = truncate_repetitive_suffix(py_strip(text));
    if label != "display_formula" {
        out = re_inline_latex()
            .replace_all(&out, |caps: &Captures| {
                format!("${}$", py_strip(caps.get(1).map_or("", |m| m.as_str())))
            })
            .into_owned();
    } else {
        out = re_display_bracket_sep()
            .replace_all(&out, NoExpand("$$   $$"))
            .replace("\\[", "")
            .replace("\\]", "");
    }
    let out = re_spaces_tabs().replace_all(&out, " ");
    let out = re_many_newlines().replace_all(&out, "\n\n").into_owned();
    if OTSL_LABELS.contains(&label) {
        let html = convert_otsl_to_html(&out);
        if !html.is_empty() {
            return html;
        }
    }
    out
}

/// One page block after recognition (`result_block`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionBlock {
    /// Reading-order index (`order`).
    pub order: usize,
    pub label: String,
    pub content: String,
    /// `inline_formula_mixed`.
    pub inline_formula_mixed: bool,
}

/// `_normalize_and_dedupe_blocks`: order, normalize, paragraph-split and
/// drop exact `(order, label, content)` duplicates.
pub fn normalize_and_dedupe_blocks(blocks: &[RegionBlock]) -> Vec<RegionBlock> {
    let mut sorted: Vec<&RegionBlock> = blocks.iter().collect();
    sorted.sort_by_key(|b| b.order);
    let mut seen: HashSet<(usize, String, String)> = HashSet::new();
    let mut out = Vec::new();
    for block in sorted {
        let mut item = block.clone();
        item.content = normalize_crop_text(&item.content, &item.label);
        if TEXT_BLOCK_LABELS.contains(&item.label.as_str()) {
            item.content = split_text_block(&item.content, false);
        } else if item.label == "inline_formula" && item.inline_formula_mixed {
            item.content = split_text_block(&item.content, true);
        }
        if seen.insert((item.order, item.label.clone(), item.content.clone())) {
            out.push(item);
        }
    }
    out
}

/// `assemble_markdown`'s per-block parts: `(index into blocks, markdown)`
/// in reading order. With `emit_image_placeholder = false` (aimd), image
/// regions and empty chart / seal regions contribute nothing instead of
/// `![image]()`.
pub fn assemble_parts(
    blocks: &[RegionBlock],
    emit_image_placeholder: bool,
) -> Vec<(usize, String)> {
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by_key(|&i| blocks[i].order);
    let mut parts = Vec::new();
    for i in order {
        let block = &blocks[i];
        let label = block.label.as_str();
        let content = block.content.as_str();
        if IGNORE_LABELS.contains(&label) {
            continue;
        }
        let image_like = IMAGE_LABELS.contains(&label) || label == "chart" || label == "seal";
        if content.is_empty() && !image_like {
            continue;
        }
        let part = match label {
            "doc_title" => Some(format!("# {content}")),
            "paragraph_title" => Some(format!("## {content}")),
            "figure_title" => Some(format!("*{content}*")),
            "display_formula" => Some(format!("$$\n{content}\n$$")),
            "inline_formula" if block.inline_formula_mixed => Some(content.to_string()),
            "inline_formula" => Some(format!("${content}$")),
            "image" | "header_image" | "footer_image" => {
                emit_image_placeholder.then(|| IMAGE_PLACEHOLDER.to_string())
            }
            "chart" | "seal" if content.is_empty() => {
                emit_image_placeholder.then(|| IMAGE_PLACEHOLDER.to_string())
            }
            _ => Some(content.to_string()),
        };
        if let Some(part) = part {
            parts.push((i, part));
        }
    }
    parts
}

/// `assemble_markdown(blocks, DEFAULT_PROMPT_MAP)`.
pub fn assemble_markdown(blocks: &[RegionBlock], emit_image_placeholder: bool) -> String {
    assemble_parts(blocks, emit_image_placeholder)
        .into_iter()
        .map(|(_, p)| p)
        .collect::<Vec<_>>()
        .join("\n\n")
}
