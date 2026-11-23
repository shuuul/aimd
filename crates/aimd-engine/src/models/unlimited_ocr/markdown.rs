//! Markdown and located blocks from the Unlimited-OCR v1 layout protocol.

/// Removes the Unlimited-OCR layout protocol while preserving content.
///
/// Line-by-line port of `normalize_unlimited_ocr_markdown` in
/// `src/aimd/plugins/ocr/models/unlimited.py` (including its
/// `_DETECTION_LINE` regex, implemented by hand): lines that match
/// `(?:<\|det\|>)?([^<\s|]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)` open a new
/// block (the `image` category is dropped entirely); every other non-empty
/// line appends to the current block; blocks are joined with blank lines.
/// Text without any det marker passes through `str.strip()`-ed.
pub fn normalize_unlimited_ocr_markdown(raw: &str) -> String {
    if !has_det_markers(raw) {
        return raw.trim().to_string();
    }
    let joined: Vec<String> = scan_det_blocks(raw)
        .iter()
        .map(|block| block.lines.join("\n"))
        .collect();
    joined.join("\n\n").trim().to_string()
}

/// Side of the coordinate grid Unlimited-OCR boxes live on: the model
/// emits integers in `0..=999` relative to the whole original image
/// (independent of the gundam crop grid), and the checkpoint's reference
/// renderer maps them back with `x / 999 * width`
/// (`draw_bounding_boxes` in `modeling_unlimitedocr.py`).
pub const UNLIMITED_OCR_GRID: f32 = 999.0;

/// One located layout block of an Unlimited-OCR page.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutBlock {
    /// Layout category as emitted by the model (`title`, `text`, `table`,
    /// ...). Content outside any det block (before the first det line)
    /// gets `text`.
    pub kind: String,
    /// `[x0, y0, x1, y1]` normalized to `0..=1` of the original page image
    /// (`x0 <= x1`, `y0 <= y1`); `[0, 0, 1, 1]` (the whole page) when the
    /// block has no parseable four-number box.
    pub bbox: [f32; 4],
    /// The block's markdown, exactly as it appears in
    /// [`normalize_unlimited_ocr_markdown`]'s output.
    pub text: String,
}

/// Bbox of blocks without a parseable box: the whole page.
const FULL_PAGE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Parses the `<|det|>kind [x0, y0, x1, y1]<|/det|>text` protocol into
/// located blocks with the normalizer's line grammar (image blocks
/// dropped, continuation lines appended to the current block).
///
/// Invariant: when `raw` has det markers, the block texts joined with
/// `"\n\n"` equal [`normalize_unlimited_ocr_markdown`]`(raw)`; otherwise
/// the result is empty.
pub fn parse_unlimited_ocr_blocks(raw: &str) -> Vec<LayoutBlock> {
    if !has_det_markers(raw) {
        return Vec::new();
    }
    let mut blocks: Vec<LayoutBlock> = scan_det_blocks(raw)
        .into_iter()
        .map(|block| LayoutBlock {
            kind: block.category.unwrap_or_else(|| "text".to_string()),
            bbox: block
                .bracket
                .as_deref()
                .and_then(parse_grid_bbox)
                .unwrap_or(FULL_PAGE),
            text: block.lines.join("\n"),
        })
        .collect();
    // Mirror the normalizer's final `.trim()`: leading/trailing empty
    // blocks only add separators the trim removes. Block lines are
    // rstripped and non-empty, so a block text is either empty or holds a
    // non-whitespace character; inner empty blocks keep their separators.
    while blocks.first().is_some_and(|b| b.text.is_empty()) {
        blocks.remove(0);
    }
    while blocks.last().is_some_and(|b| b.text.is_empty()) {
        blocks.pop();
    }
    if let Some(first) = blocks.first_mut() {
        first.text = first.text.trim_start().to_string();
    }
    if let Some(last) = blocks.last_mut() {
        last.text = last.text.trim_end().to_string();
    }
    blocks
}

/// Normalizes a `[x0, y0, x1, y1]` grid box (bracket text, brackets
/// included) to `0..=1`; `None` unless it holds exactly four finite numbers.
fn parse_grid_bbox(bracket: &str) -> Option<[f32; 4]> {
    let inner = bracket.strip_prefix('[')?.strip_suffix(']')?;
    let values: Vec<f32> = inner
        .split(',')
        .map(|v| v.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
        .collect::<Option<_>>()?;
    let [x0, y0, x1, y1] = values[..] else {
        return None;
    };
    let n = |v: f32| (v / UNLIMITED_OCR_GRID).clamp(0.0, 1.0);
    Some([n(x0.min(x1)), n(y0.min(y1)), n(x0.max(x1)), n(y0.max(y1))])
}

fn has_det_markers(raw: &str) -> bool {
    raw.contains("<|det|>") || raw.contains("<|/det|>")
}

/// One block of the det protocol before joining.
struct DetBlock {
    /// Category of the opening det line (`None` for leading content).
    category: Option<String>,
    /// The optional `[...]` group of the opening det line.
    bracket: Option<String>,
    /// Non-empty, rstripped lines (the det line's content first).
    lines: Vec<String>,
}

/// The normalizer's line grammar, shared by
/// [`normalize_unlimited_ocr_markdown`] and [`parse_unlimited_ocr_blocks`].
fn scan_det_blocks(raw: &str) -> Vec<DetBlock> {
    let mut blocks: Vec<DetBlock> = Vec::new();
    let mut current: Option<DetBlock> = None;
    for line in python_splitlines(raw) {
        let line = line.rstrip();
        if line.is_empty() {
            continue;
        }
        if let Some(det) = parse_detection_line(&line) {
            let category = det.category.trim();
            if category == "image" {
                continue;
            }
            if let Some(prev) = current.take() {
                blocks.push(prev);
            }
            current = Some(DetBlock {
                category: Some(category.to_string()),
                bracket: det.bracket,
                lines: if det.content.is_empty() {
                    Vec::new()
                } else {
                    vec![det.content]
                },
            });
            continue;
        }
        current
            .get_or_insert_with(|| DetBlock {
                category: None,
                bracket: None,
                lines: Vec::new(),
            })
            .lines
            .push(line);
    }
    if let Some(last) = current {
        blocks.push(last);
    }
    blocks
}

/// Python `str.splitlines()` boundaries: `\n`, `\r\n`, `\r`, `\v`, `\f`,
/// `\x1c`, `\x1d`, `\x1e`, `\x85` (NEL), `\u{2028}`, `\u{2029}`. The
/// terminators are not kept.
fn python_splitlines(text: &str) -> Vec<String> {
    let is_break = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' && chars.peek() == Some(&'\n') {
            chars.next();
            lines.push(std::mem::take(&mut current));
        } else if is_break(c) {
            lines.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    lines.push(current);
    lines
}

/// `str.rstrip()` (Python strips unicode whitespace from the end).
trait PythonStr {
    fn rstrip(&self) -> String;
}

impl PythonStr for String {
    fn rstrip(&self) -> String {
        self.trim_end().to_string()
    }
}

/// A matched det line: the captures of
/// `(?:<\|det\|>)?([^<\s|]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)`.
struct DetLine {
    /// Group 1 (the category run).
    category: String,
    /// The optional bracket group, brackets included.
    bracket: Option<String>,
    /// Group 2 (everything after `<|/det|>`).
    content: String,
}

/// Hand-rolled matcher for the [`DetLine`] regex, anchored at the line start,
/// with the regex's single backtrack point (the optional bracket group).
fn parse_detection_line(line: &str) -> Option<DetLine> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let starts_with = |at: usize, needle: &str| -> bool {
        let nd: Vec<char> = needle.chars().collect();
        at + nd.len() <= n && chars[at..at + nd.len()] == nd[..]
    };
    let mut p = 0usize;
    if starts_with(p, "<|det|>") {
        p += "<|det|>".chars().count();
    }
    let group1_start = p;
    // Greedy `[^<\s|]+` scan (the run cannot contain <, |, or whitespace).
    let mut group1_end = group1_start;
    while group1_end < n
        && chars[group1_end] != '<'
        && chars[group1_end] != '|'
        && !chars[group1_end].is_whitespace()
    {
        group1_end += 1;
    }
    if group1_end == group1_start {
        return None;
    }

    // Continuation after group 1: optional `\s*\[[^\]]*\]` then `\s*<|/det|>`.
    // The regex engine backtracks group 1 from its longest match downwards,
    // and the backtrack matters: "title[1, 2]<|/det|>" only matches with
    // group 1 = "title" so the bracket group can consume "[1, 2]".
    // Returns the content start and the bracket group's char range.
    let try_continue = |end: usize| -> Option<(usize, Option<(usize, usize)>)> {
        // Attempt 1: `\s*\[[^\]]*\]` followed by `\s*<|/det|>`.
        let mut q = end;
        while q < n && chars[q].is_whitespace() {
            q += 1;
        }
        if q < n && chars[q] == '[' {
            let mut r = q + 1;
            while r < n && chars[r] != ']' {
                r += 1;
            }
            if r < n {
                let mut s = r + 1;
                while s < n && chars[s].is_whitespace() {
                    s += 1;
                }
                if starts_with(s, "<|/det|>") {
                    return Some((s + "<|/det|>".chars().count(), Some((q, r + 1))));
                }
            }
        }
        // Attempt 2 (backtrack past the bracket group): `\s*<|/det|>`.
        let mut s = end;
        while s < n && chars[s].is_whitespace() {
            s += 1;
        }
        if starts_with(s, "<|/det|>") {
            Some((s + "<|/det|>".chars().count(), None))
        } else {
            None
        }
    };

    for end in (group1_start + 1..=group1_end).rev() {
        if let Some((content_start, bracket)) = try_continue(end) {
            return Some(DetLine {
                category: chars[group1_start..end].iter().collect(),
                bracket: bracket.map(|(a, b)| chars[a..b].iter().collect()),
                content: chars[content_start..].iter().collect(),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected Markdown comes from the original Python normalizer.
    #[test]
    fn ocr_generate_markdown_normalizer_matches_python() {
        let cases: Vec<(&str, &str)> = vec![
            ("  Hello world\nsecond line  ", "Hello world\nsecond line"),
            (
                "<|det|>title[1, 2, 3, 4]<|/det|>Hello world\nsecond line\n\n<|det|>text[0, 0, 9, 9]<|/det|>More text",
                "Hello world\nsecond line\n\nMore text",
            ),
            (
                "<|det|>image[0, 0, 5, 5]  <|/det|>\n<|det|>text[1, 1, 8, 8]<|/det|>keep me",
                "keep me",
            ),
            (
                "header[x, y]<|/det|>the content\nnext line",
                "the content\nnext line",
            ),
            ("<|det|>title[1, 2]<|/det|>\nfiller line", "filler line"),
            (
                "<|det|>a[1]<|/det|>one\n\n\n<|det|>b[2]<|/det|>two",
                "one\n\ntwo",
            ),
            (
                "intro line\n<|det|>text[1]<|/det|>body",
                "intro line\n\nbody",
            ),
            (
                "<|det|>text[0,0,9,9]<|/det|>line1\nline2\n\nline3\n<|det|>text[1,1,8,8]<|/det|>B1\nB2",
                "line1\nline2\nline3\n\nB1\nB2",
            ),
            ("<|det|>image[0,0,1,1]<|/det|>", ""),
            ("text before\n<|det|>image[0,0,1,1]<|/det|>", "text before"),
            (
                // Category run stops before whitespace, so this line does
                // not match the det regex and passes through untouched.
                "<|det|> table  [1, 2, 3, 4]  <|/det|> cell text",
                "<|det|> table  [1, 2, 3, 4]  <|/det|> cell text",
            ),
            (
                // Bracket group consumed but the continuation fails and the
                // no-bracket fallback fails too: no match, line kept.
                "text [1,2] blah <|/det|>not matched",
                "text [1,2] blah <|/det|>not matched",
            ),
            (
                "<|det|>标题[1, 2, 3, 4]<|/det|>你好世界\n第二行",
                "你好世界\n第二行",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_unlimited_ocr_markdown(input),
                expected,
                "case input {input:?}"
            );
        }
    }

    /// Every normalizer case (plus block-shape edge cases): blocks join to
    /// the normalized markdown whenever det markers exist, else are empty.
    #[test]
    fn ocr_generate_layout_blocks_join_to_markdown() {
        let cases = [
            "  Hello world\nsecond line  ",
            "<|det|>title[1, 2, 3, 4]<|/det|>Hello world\nsecond line\n\n<|det|>text[0, 0, 9, 9]<|/det|>More text",
            "<|det|>image[0, 0, 5, 5]  <|/det|>\n<|det|>text[1, 1, 8, 8]<|/det|>keep me",
            "header[x, y]<|/det|>the content\nnext line",
            "<|det|>title[1, 2]<|/det|>\nfiller line",
            "<|det|>a[1]<|/det|>one\n\n\n<|det|>b[2]<|/det|>two",
            "intro line\n<|det|>text[1]<|/det|>body",
            "<|det|>text[0,0,9,9]<|/det|>line1\nline2\n\nline3\n<|det|>text[1,1,8,8]<|/det|>B1\nB2",
            "<|det|>image[0,0,1,1]<|/det|>",
            "text before\n<|det|>image[0,0,1,1]<|/det|>",
            "<|det|> table  [1, 2, 3, 4]  <|/det|> cell text",
            "text [1,2] blah <|/det|>not matched",
            "<|det|>标题[1, 2, 3, 4]<|/det|>你好世界\n第二行",
            "<|det|>title [1, 2, 3, 4]<|/det|>\n<|det|>text [5, 6, 7, 8]<|/det|>x",
            "<|det|>text [1, 2, 3, 4]<|/det|>a\n<|det|>text [5, 6, 7, 8]<|/det|>\n<|det|>text [1, 1, 2, 2]<|/det|>b",
            "<|det|>text [1, 2, 3, 4]<|/det|>   indented\n<|det|>title [0, 0, 1, 1]<|/det|>",
        ];
        for raw in cases {
            let blocks = parse_unlimited_ocr_blocks(raw);
            let markdown = normalize_unlimited_ocr_markdown(raw);
            if has_det_markers(raw) {
                let joined: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
                assert_eq!(joined.join("\n\n"), markdown, "case {raw:?}");
            } else {
                assert!(blocks.is_empty(), "case {raw:?}");
            }
            for b in &blocks {
                assert!(b.kind != "image", "case {raw:?}");
                let [x0, y0, x1, y1] = b.bbox;
                assert!(
                    (0.0..=1.0).contains(&x0) && x0 <= x1 && x1 <= 1.0,
                    "case {raw:?}"
                );
                assert!(
                    (0.0..=1.0).contains(&y0) && y0 <= y1 && y1 <= 1.0,
                    "case {raw:?}"
                );
            }
        }
    }

    #[test]
    fn ocr_generate_layout_blocks_kinds_and_boxes() {
        let g = |v: f32| v / 999.0;
        let raw = "<|det|>title [37, 26, 259, 53]<|/det|>Quarterly Report\n\
                   <|det|>image [0, 0, 999, 999]<|/det|>\n\
                   <|det|>text [33, 97, 655, 228]<|/det|>Line one\nline two\n\
                   <|det|>table[1200, 5, 3, 1]<|/det|>| a |\n\
                   <|det|>text<|/det|>no box\n\
                   <|det|>text [1, 2]<|/det|>short box";
        let blocks = parse_unlimited_ocr_blocks(raw);
        let summary: Vec<(&str, &str)> = blocks
            .iter()
            .map(|b| (b.kind.as_str(), b.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("title", "Quarterly Report"),
                ("text", "Line one\nline two"),
                ("table", "| a |"),
                ("text", "no box"),
                ("text", "short box"),
            ]
        );
        assert_eq!(blocks[0].bbox, [g(37.0), g(26.0), g(259.0), g(53.0)]);
        assert_eq!(blocks[1].bbox, [g(33.0), g(97.0), g(655.0), g(228.0)]);
        // Swapped corners are ordered; out-of-grid values clamp to 1.
        assert_eq!(blocks[2].bbox, [g(3.0), g(1.0), 1.0, g(5.0)]);
        // Missing or malformed boxes fall back to the whole page.
        assert_eq!(blocks[3].bbox, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(blocks[4].bbox, [0.0, 0.0, 1.0, 1.0]);
        // Leading content outside any det block is a full-page `text` block.
        let lead = parse_unlimited_ocr_blocks("intro\n<|det|>title [1, 2, 3, 4]<|/det|>T");
        assert_eq!(lead[0].kind, "text");
        assert_eq!(lead[0].bbox, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(lead[1].bbox, [g(1.0), g(2.0), g(3.0), g(4.0)]);
    }

    #[test]
    fn ocr_generate_markdown_normalizer_splitlines_matches_python() {
        // No det markers: plain strip — the \r is NOT a split boundary on
        // this path (Python returns raw.strip()).
        assert_eq!(normalize_unlimited_ocr_markdown("a\rb\rc"), "a\rb\rc");
        // With det markers present, splitlines applies (lone \r splits).
        assert_eq!(
            normalize_unlimited_ocr_markdown("<|det|>t[1]<|/det|>x\ry"),
            "x\ny"
        );
        // No det markers: plain strip.
        assert_eq!(normalize_unlimited_ocr_markdown("   "), "");
        assert_eq!(normalize_unlimited_ocr_markdown(""), "");
    }
}
