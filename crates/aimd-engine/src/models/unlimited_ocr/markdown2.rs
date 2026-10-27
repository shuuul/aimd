//! Unlimited-OCR 2 raw output -> Markdown.
//!
//! Faithful port of Baidu's stdlib-only `process_unlimitedOCR2.py`
//! (`convert`). Unlimited-OCR 2 emits layout blocks, not Markdown:
//!
//! ```text
//! <PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>text
//! <BLOCK>text[x1,y1,x2,y2]<FORK>body
//! ```
//!
//! The converter splits pages (blank pages keep their number), parses blocks
//! (LEVEL-first or bbox-first, bare blocks without `<BLOCK>` repaired),
//! groups `cross_`/`append_` chains and `<MERGE>` columns with union-find,
//! merges cross-page table fragments (repeated header rows de-duplicated),
//! renders headings / images / tables / text, then applies the stray-token,
//! single-root-heading and whitespace rules.
//!
//! Python semantics are matched where they matter: `\s` and `str.strip()` also
//! treat U+001C..U+001F as whitespace, `\d` is Unicode, and the one lookbehind
//! (`BARE_BLOCK_RE`) and lookahead (`\n\s*<FORK>\s*(?=<BLOCK>)`) the `regex`
//! crate lacks are checked by hand on the match boundary.
//!
//! Known deviations from upstream (pathological input only): digits outside
//! ASCII inside a bbox / level number parse as 0 (Python `int()` accepts every
//! Unicode decimal digit), and `<LEVELn>` is capped at 1000 so a corrupt
//! number cannot request a multi-gigabyte `#` run.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::{Captures, Regex};

/// One parsed layout block (`_parse_blocks` + `_merge_adjacent_titles`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ocr2Block {
    /// 1-based page number (blank pages still consume a number).
    pub page: usize,
    /// Label as emitted, lowercased (`text`, `cross_table`, ...). Text that
    /// has no `<BLOCK>` header (upstream `__raw__`) is reported as `text`.
    pub kind: String,
    /// `<LEVELn>` of a title.
    pub level: Option<u32>,
    /// Raw bbox on the model's 0..1000 grid (first four numbers).
    pub bbox: Option<[i32; 4]>,
    pub text: String,
}

const RAW: &str = "__raw__";
const MAX_LEVEL: i64 = 1000;
const MAX_HEADER_ROWS: usize = 5;
const IMAGE_LABELS: [&str; 6] = ["image", "figure", "picture", "chart", "img", "seal"];
const NOISE_CONTENT: [&str; 4] = ["[non-text]", "[unreadable]", "[image]", "[figure]"];
const BASE_LABELS: [&str; 17] = [
    "text",
    "title",
    "header",
    "footer",
    "page_number",
    "image",
    "figure",
    "chart",
    "seal",
    "table",
    "image_caption",
    "image_footnote",
    "page_footnote",
    "equation",
    "ref_text",
    "aside_text",
    "algorithm",
];

/// Compile a pattern with Python's `\s` (adds U+001C..U+001F to Unicode
/// whitespace).
fn re(flags: &str, pattern: &str) -> Regex {
    let p = format!("{flags}{}", pattern.replace(r"\s", r"[\s\x1c-\x1f]"));
    Regex::new(&p).unwrap_or_else(|e| panic!("bad regex {p}: {e}"))
}

static PAGE_SPLIT_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"<\s*/?\s*PAGE\s*\d*\s*>"));
static MERGE_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"<\s*/?\s*MERGE\s*>"));
static STRAY_TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        "(?i)",
        r"<[|｜]/?[a-zA-Z_]+[|｜]>|<image>|<\s*/?\s*PAGE\s*\d*\s*>|<\s*/?\s*BLOCK\s*>|<\s*/?\s*FORK\s*>|<\s*/?\s*LEVEL\s*\d*\s*[>}]?|<\s*/?\s*MERGE\s*>",
    )
});
static BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        "(?i)",
        concat!(
            r"<\s*BLOCK\s*>\s*([a-zA-Z_]+)",
            r"(?:",
            r"\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?(?:\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)",
            r"(?:\s*[,，]\s*-?\d+)*\s*\])?",
            r"|",
            r"\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)",
            r"(?:\s*[,，]\s*-?\d+)*\s*\](?:\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?)?",
            r")?",
            r"\s*(?:<\s*FORK\s*>)?",
        ),
    )
});
static TR_RE: LazyLock<Regex> = LazyLock::new(|| re("(?is)", r"<tr\b[^>]*>.*?</tr>"));
static TABLE_OPEN_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"<table\b[^>]*>"));
static SPAN_ATTR_RE: LazyLock<Regex> =
    LazyLock::new(|| re("(?i)", r#"\b(colspan|rowspan)\s*=\s*"([^"]*)""#));
/// `BARE_BLOCK_RE` without its leading lookbehind alternation; the boundary
/// condition is checked by [`bare_start_ok`].
static BARE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        "(?i)",
        concat!(
            r"(?:<[|｜]\s*det\s*[|｜]>)?",
            r"((?:[a-zA-Z_][a-zA-Z_0-9]*)(?:\s*<\s*/?\s*LEVEL\s*\d*\s*[>}]?)?\s*",
            r"\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3,}[^\]]*\])",
        ),
    )
});
static FORK_BEFORE_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"\n\s*<FORK>\s*"));
static BLOCK_TAG_AT_START_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"\A<BLOCK>"));
static FORK_PAGE_AT_END_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"(?:<FORK>|<PAGE>)\z"));
static HTML_TAG_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"<[^>]+>"));
static TD_TH_RE: LazyLock<Regex> = LazyLock::new(|| re("(?i)", r"<t[hd]\b"));
static DIGITS_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"\d+"));
static PY_INT_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"\A[+-]?\d+(?:_\d+)*\z"));
static BBOX_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        "(?m)",
        r"^\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*$",
    )
});
static TRAILING_WS_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"[ \t]+\n"));
static NEWLINES_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"\n{3,}"));
static WS_RUN_RE: LazyLock<Regex> = LazyLock::new(|| re("", r"\s+"));

// -- Python string helpers ---------------------------------------------------

fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `str.lstrip()`.
fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(is_py_space)
}

/// `int(s)` for the ASCII digit runs the block regexes capture (saturating;
/// non-ASCII decimal digits count as 0).
fn parse_int(s: &str) -> i64 {
    let (neg, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let mut v: i64 = 0;
    for c in digits.chars() {
        v = v
            .saturating_mul(10)
            .saturating_add(i64::from(c.to_digit(10).unwrap_or(0)));
    }
    if neg { -v } else { v }
}

fn clamp_i32(v: i64) -> i32 {
    v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

// -- block parsing -----------------------------------------------------------

#[derive(Debug, Clone)]
struct Block {
    label: String,
    bbox: Option<[i64; 4]>,
    level: Option<u32>,
    content: String,
    page: usize,
    merge_prev: bool,
}

fn strip_fence(raw: &str) -> String {
    let mut text = py_strip(raw);
    let unfenced;
    if let Some(rest) = text.strip_prefix("```markdown") {
        unfenced = py_strip(rest).to_string();
        text = &unfenced;
    }
    match text.strip_suffix("```") {
        Some(rest) => py_strip(rest).to_string(),
        None => text.to_string(),
    }
}

/// Python lookbehind `(?<=<FORK>)|(?<=<PAGE>)|(?<=\n)|^` at byte offset `at`.
fn bare_start_ok(text: &str, at: usize) -> bool {
    if at == 0 || text[..at].ends_with('\n') {
        return true;
    }
    // Last 6 characters cover `<FORK>` even where case folding widens a letter.
    let start = text[..at].char_indices().rev().nth(5).map_or(0, |(i, _)| i);
    FORK_PAGE_AT_END_RE.is_match(&text[start..at])
}

fn next_char_boundary(text: &str, at: usize) -> usize {
    at + text[at..].chars().next().map_or(1, char::len_utf8)
}

/// Add the missing `<BLOCK>` prefix to bare block declarations (known labels).
fn fix_bare_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    let (mut last, mut pos) = (0, 0);
    while pos <= text.len() {
        let Some(caps) = BARE_BLOCK_RE.captures_at(text, pos) else {
            break;
        };
        let whole = caps.get(0).expect("group 0");
        if !bare_start_ok(text, whole.start()) {
            pos = next_char_boundary(text, whole.start());
            continue;
        }
        let decl = caps.get(1).expect("group 1").as_str();
        let label_len = decl
            .char_indices()
            .find(|&(_, c)| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(decl.len(), |(i, _)| i);
        let label = decl[..label_len].to_lowercase();
        if BASE_LABELS.iter().any(|b| is_bare_label(&label, b)) {
            out.push_str(&text[last..whole.start()]);
            out.push_str("<BLOCK>");
            out.push_str(decl);
            last = whole.end();
        }
        pos = whole.end();
    }
    out.push_str(&text[last..]);

    // `re.sub(r'\n\s*<FORK>\s*(?=<BLOCK>)', '\n', text, flags=re.I)`
    let mut fixed = String::with_capacity(out.len());
    let (mut last, mut pos) = (0, 0);
    while pos <= out.len() {
        let Some(m) = FORK_BEFORE_BLOCK_RE.find_at(&out, pos) else {
            break;
        };
        if BLOCK_TAG_AT_START_RE.is_match(&out[m.end()..]) {
            fixed.push_str(&out[last..m.start()]);
            fixed.push('\n');
            last = m.end();
            pos = m.end();
        } else {
            pos = next_char_boundary(&out, m.start());
        }
    }
    fixed.push_str(&out[last..]);
    fixed
}

/// `label in _BARE_LABELS` where `_BARE_LABELS` is the base set plus the
/// `cross_` / `append_` prefixed forms.
fn is_bare_label(label: &str, base: &str) -> bool {
    label == base
        || label.strip_prefix("cross_") == Some(base)
        || label.strip_prefix("append_") == Some(base)
}

fn base_type(label: &str) -> &str {
    for p in ["append_", "cross_"] {
        if let Some(rest) = label.strip_prefix(p) {
            return rest;
        }
    }
    label
}

fn split_segments(text: &str) -> Vec<&str> {
    let mut segments: Vec<&str> = PAGE_SPLIT_RE.split(text).collect();
    if segments.first().is_some_and(|s| py_strip(s).is_empty()) {
        segments.remove(0);
    }
    segments
}

fn parse_level(s: &str) -> Option<u32> {
    (!s.is_empty()).then(|| parse_int(s).clamp(0, MAX_LEVEL) as u32)
}

fn bbox_from(caps: &Captures<'_>, first: usize) -> Option<[i64; 4]> {
    caps.get(first)?;
    let mut b = [0i64; 4];
    for (k, slot) in b.iter_mut().enumerate() {
        *slot = parse_int(caps.get(first + k).map_or("", |m| m.as_str()));
    }
    Some(b)
}

fn raw_block(content: String, page: usize) -> Block {
    Block {
        label: RAW.to_string(),
        bbox: None,
        level: None,
        content,
        page,
        merge_prev: false,
    }
}

fn parse_blocks(text: &str) -> Vec<Block> {
    let text = fix_bare_blocks(text);
    let mut blocks = Vec::new();
    for (i, seg) in split_segments(&text).into_iter().enumerate() {
        let page = 1 + i;
        if py_strip(seg).is_empty() {
            continue;
        }
        let caps: Vec<Captures<'_>> = BLOCK_RE.captures_iter(seg).collect();
        if caps.is_empty() {
            blocks.push(raw_block(
                py_strip(&MERGE_RE.replace_all(seg, "")).to_string(),
                page,
            ));
            continue;
        }
        for (j, c) in caps.iter().enumerate() {
            let m = c.get(0).expect("group 0");
            let gap = if j == 0 {
                &seg[..m.start()]
            } else {
                &seg[caps[j - 1].get(0).expect("group 0").end()..m.start()]
            };
            let merge_prev = MERGE_RE.is_match(gap);
            if j == 0 {
                let no_merge = MERGE_RE.replace_all(gap, "");
                let pre = STRAY_TOKEN_RE.replace_all(&no_merge, "");
                let pre = py_strip(&pre);
                if !pre.is_empty() {
                    blocks.push(raw_block(pre.to_string(), page));
                }
            }
            let end = caps
                .get(j + 1)
                .map_or(seg.len(), |n| n.get(0).expect("group 0").start());
            let content = py_strip(&MERGE_RE.replace_all(&seg[m.end()..end], "")).to_string();
            let (bbox, level) = if c.get(2).is_some() {
                // LEVEL-first
                (bbox_from(c, 3), parse_level(&c[2]))
            } else {
                // bbox-first (or neither)
                (
                    bbox_from(c, 7),
                    c.get(11).and_then(|m| parse_level(m.as_str())),
                )
            };
            blocks.push(Block {
                label: c[1].to_lowercase(),
                bbox,
                level,
                content,
                page,
                merge_prev,
            });
        }
    }
    blocks
}

/// Merge strictly adjacent, same-page, same-`<LEVELn>` title blocks.
fn merge_adjacent_titles(blocks: Vec<Block>) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::with_capacity(blocks.len());
    for blk in blocks {
        if let Some(prev) = out.last_mut()
            && blk.label == "title"
            && prev.label == "title"
            && blk.page == prev.page
            && blk.level.is_some()
            && blk.level == prev.level
        {
            let parts: Vec<&str> = [prev.content.as_str(), blk.content.as_str()]
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect();
            prev.content = parts.join("\n");
            continue;
        }
        out.push(blk);
    }
    out
}

// -- grouping ----------------------------------------------------------------

fn is_chain_type(base: &str) -> bool {
    base == "text" || base == "table"
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

/// Union-find grouping of append/cross chains and same-page `<MERGE>` columns.
fn build_groups(blocks: &[Block]) -> Vec<Vec<usize>> {
    let n = blocks.len();
    let mut parent: Vec<usize> = (0..n).collect();

    let table_pages: HashSet<usize> = blocks
        .iter()
        .filter(|b| b.label != RAW && base_type(&b.label) == "table")
        .map(|b| b.page)
        .collect();
    let table_adjacent = |tail_page: usize, cur_page: usize| {
        cur_page.saturating_sub(tail_page) <= 1
            || (tail_page + 1..cur_page).all(|p| !table_pages.contains(&p))
    };

    let mut open_chain: HashMap<&str, usize> = HashMap::new();
    for (i, blk) in blocks.iter().enumerate() {
        let (label, base) = (blk.label.as_str(), base_type(&blk.label));
        if label == RAW || !is_chain_type(base) {
            continue;
        }
        if label.starts_with("append_") {
            let Some(&tail) = open_chain.get(base) else {
                continue;
            };
            if base == "table" && !table_adjacent(blocks[tail].page, blk.page) {
                open_chain.remove(base);
                continue;
            }
            union(&mut parent, tail, i);
            open_chain.insert(base, i);
        } else if label.starts_with("cross_") || base == "table" {
            open_chain.insert(base, i);
        }
    }

    let text_like = |b: &Block| b.label == "title" || base_type(&b.label) == "text";
    for i in 1..n {
        let (blk, pblk) = (&blocks[i], &blocks[i - 1]);
        if blk.merge_prev && pblk.page == blk.page && text_like(blk) && text_like(pblk) {
            union(&mut parent, i - 1, i);
        }
    }

    let mut order: Vec<usize> = Vec::new();
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups
            .entry(r)
            .or_insert_with(|| {
                order.push(r);
                Vec::new()
            })
            .push(i);
    }
    order.sort_unstable();
    order
        .into_iter()
        .map(|r| groups.remove(&r).expect("group"))
        .collect()
}

// -- tables ------------------------------------------------------------------

fn row_key(tr: &str) -> (usize, String) {
    let txt = HTML_TAG_RE.replace_all(tr, " ");
    let folded: String = txt
        .chars()
        .map(|c| match c as u32 {
            0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .collect();
    let ncells = TD_TH_RE.find_iter(tr).count();
    let joined: String = folded.split(is_py_space).collect();
    (ncells, joined.to_lowercase())
}

/// Merge cross-page table fragments, skipping repeated leading header rows.
fn merge_table_fragments(frags: &[&str]) -> String {
    let frags: Vec<&str> = frags
        .iter()
        .copied()
        .filter(|f| !py_strip(f).is_empty())
        .collect();
    if frags.len() <= 1 {
        return frags.first().map_or(String::new(), |f| f.to_string());
    }
    let base_rows: Vec<&str> = TR_RE.find_iter(frags[0]).map(|m| m.as_str()).collect();
    if base_rows.is_empty() {
        return frags.join("\n\n");
    }
    let mut out_rows: Vec<&str> = base_rows.clone();
    let base_keys: Vec<(usize, String)> = base_rows.iter().map(|r| row_key(r)).collect();
    let mut extra: Vec<&str> = Vec::new();
    for frag in &frags[1..] {
        let rows: Vec<&str> = TR_RE.find_iter(frag).map(|m| m.as_str()).collect();
        if rows.is_empty() {
            extra.push(py_strip(frag));
            continue;
        }
        let mut skip = 0;
        for r in rows.iter().take(MAX_HEADER_ROWS) {
            if skip < base_keys.len() && row_key(r) == base_keys[skip] {
                skip += 1;
            } else {
                break;
            }
        }
        out_rows.extend_from_slice(&rows[skip..]);
    }
    let open = TABLE_OPEN_RE
        .find(frags[0])
        .map_or("<table>", |m| m.as_str());
    let mut merged = format!("{open}{}</table>", out_rows.concat());
    if !extra.is_empty() {
        merged.push_str("\n\n");
        merged.push_str(&extra.join("\n\n"));
    }
    merged
}

// -- rendering ---------------------------------------------------------------

fn scale(v: i64) -> i64 {
    v.clamp(0, 1000)
}

/// Rendered items, each with the index of its group's head block.
/// `images` keeps upstream's `![](pageN_...jpg)` crop placeholders.
fn render_blocks(blocks: &[Block], groups: &[Vec<usize>], images: bool) -> Vec<(usize, String)> {
    let mut items = Vec::new();
    for members in groups {
        let at = members[0];
        let head = &blocks[at];
        let mut push = |item: String| items.push((at, item));
        let base = if head.label == RAW {
            RAW
        } else {
            base_type(&head.label)
        };
        let contents: Vec<&str> = members
            .iter()
            .map(|&k| blocks[k].content.as_str())
            .filter(|c| !c.is_empty() && !NOISE_CONTENT.contains(&c.to_lowercase().as_str()))
            .collect();
        if base == RAW {
            if let Some(c) = contents.first() {
                push(c.to_string());
            }
            continue;
        }
        if IMAGE_LABELS.contains(&base) {
            if let Some(b) = head.bbox.filter(|_| images) {
                push(format!(
                    "![](page{}_{}_{}_{}_{}.jpg)",
                    head.page,
                    scale(b[0]),
                    scale(b[1]),
                    scale(b[2]),
                    scale(b[3])
                ));
            }
            if let Some(c) = contents.first() {
                push(c.to_string());
            }
            continue;
        }
        if contents.is_empty() {
            continue;
        }
        match base {
            "title" => {
                let lv = head.level.unwrap_or(1).max(1) as usize;
                let joined = contents.concat();
                if py_strip(&STRAY_TOKEN_RE.replace_all(&joined, "")).is_empty() {
                    continue;
                }
                push(format!("{} {joined}", "#".repeat(lv)));
            }
            "table" => push(if contents.len() > 1 {
                merge_table_fragments(&contents)
            } else {
                contents[0].to_string()
            }),
            _ => push(contents.concat()),
        }
    }
    items
}

fn head_level(item: &str) -> usize {
    let s = py_lstrip(item);
    s.len() - s.trim_start_matches('#').len()
}

/// Collapse multiple top-level headings under the first one (single root).
fn single_root_titles(items: Vec<String>) -> Vec<String> {
    let idxs: Vec<usize> = (0..items.len())
        .filter(|&i| py_lstrip(&items[i]).starts_with('#'))
        .collect();
    if idxs.len() <= 1 {
        return items;
    }
    let levels: Vec<usize> = idxs.iter().map(|&i| head_level(&items[i])).collect();
    let min_lv = *levels.iter().min().expect("non-empty");
    if levels.iter().filter(|&&lv| lv == min_lv).count() <= 1 {
        return items;
    }
    let first_lv = levels[0] as i64;
    let mut out = items.clone();
    for (k, &i) in idxs.iter().enumerate() {
        let body = py_lstrip(&items[i])[levels[k]..].trim_start_matches(' ');
        let new_lv = if k == 0 {
            1
        } else {
            (levels[k] as i64 - first_lv + 2).max(2) as usize
        };
        out[i] = format!("{} {body}", "#".repeat(new_lv));
    }
    out
}

fn fold_head_ws(items: Vec<(usize, String)>) -> Vec<(usize, String)> {
    items
        .into_iter()
        .map(|(at, it)| {
            let s = py_lstrip(&it);
            if s.starts_with('#') {
                let n = head_level(s);
                let rest = WS_RUN_RE.replace_all(&s[n..], " ");
                (at, format!("{} {}", "#".repeat(n), py_strip(&rest)))
            } else {
                (at, it)
            }
        })
        .collect()
}

fn drop_empty_heads(items: Vec<(usize, String)>) -> Vec<(usize, String)> {
    items
        .into_iter()
        .filter(|(_, it)| {
            let s = py_lstrip(it);
            !(s.starts_with('#') && py_strip(s.trim_start_matches('#')).is_empty())
        })
        .collect()
}

/// `colspan="x"` / `rowspan="x"` with a non-integer value -> first digit run
/// (or `1`).
fn fix_span_attrs(text: &str) -> String {
    SPAN_ATTR_RE
        .replace_all(text, |c: &Captures<'_>| {
            let (name, val) = (&c[1], &c[2]);
            if PY_INT_RE.is_match(py_strip(val)) {
                c[0].to_string()
            } else {
                let num = DIGITS_RE.find(val).map_or("1", |m| m.as_str());
                format!("{name}=\"{num}\"")
            }
        })
        .into_owned()
}

// -- public API --------------------------------------------------------------

/// Unlimited-OCR 2 raw document string -> evaluation Markdown (upstream
/// `convert`).
pub fn convert_unlimited_ocr2_markdown(raw: &str) -> String {
    let (_, items) = rendered_items(raw, true);
    finish(
        items
            .into_iter()
            .map(|(_, it)| it)
            .collect::<Vec<_>>()
            .join("\n\n"),
    )
}

/// One rendered item of [`render_unlimited_ocr2`]: a block, or a
/// cross-page chain / `<MERGE>` group under its first block.
#[derive(Debug, Clone, PartialEq)]
pub struct Ocr2Item {
    /// 1-based page of the group's first block.
    pub page: usize,
    /// Label of the first block (`text`, `cross_table`, ...).
    pub kind: String,
    /// First block's box normalized to `0..=1` of its page (the model's
    /// grid is `0..=999` like v1), `None` when the block has none.
    pub bbox: Option<[f32; 4]>,
    pub text: String,
}

/// aimd's rendering of a raw Unlimited-OCR 2 document.
#[derive(Debug, Clone, PartialEq)]
pub struct Ocr2Render {
    /// [`convert_unlimited_ocr2_markdown`] without the image-crop
    /// placeholders (aimd writes no crop files; image captions stay).
    pub markdown: String,
    /// The same document item by item, each cleaned like the whole
    /// document, for source anchors.
    pub items: Vec<Ocr2Item>,
}

/// [`convert_unlimited_ocr2_markdown`] for aimd: no image placeholders,
/// plus the located items.
pub fn render_unlimited_ocr2(raw: &str) -> Ocr2Render {
    let (blocks, items) = rendered_items(raw, false);
    let markdown = finish(
        items
            .iter()
            .map(|(_, it)| it.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    let mut located: Vec<(usize, String)> = items
        .into_iter()
        .map(|(at, it)| (at, finish(it)))
        .filter(|(_, it)| !it.is_empty())
        .collect();
    let texts = single_root_titles(located.iter().map(|(_, t)| t.clone()).collect());
    for ((_, t), new) in located.iter_mut().zip(texts) {
        *t = new;
    }
    let items = located
        .into_iter()
        .map(|(at, text)| {
            let b = &blocks[at];
            Ocr2Item {
                page: b.page,
                kind: if b.label == RAW {
                    "text".to_string()
                } else {
                    b.label.clone()
                },
                bbox: b.bbox.map(normalize_bbox),
                text,
            }
        })
        .collect();
    Ocr2Render { markdown, items }
}

fn rendered_items(raw: &str, images: bool) -> (Vec<Block>, Vec<(usize, String)>) {
    let text = strip_fence(raw);
    let blocks = merge_adjacent_titles(parse_blocks(&text));
    let groups = build_groups(&blocks);
    let items = fold_head_ws(drop_empty_heads(render_blocks(&blocks, &groups, images)));
    (blocks, items)
}

/// Upstream's whole-document cleanup after the items are joined.
fn finish(text: String) -> String {
    let text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:");
    let text = STRAY_TOKEN_RE.replace_all(&text, "");
    let text = fix_span_attrs(&text);
    let text = BBOX_LINE_RE.replace_all(&text, "");
    let text = TRAILING_WS_RE.replace_all(&text, "\n");
    let text = NEWLINES_RE.replace_all(&text, "\n\n");
    let text = py_strip(&text);
    single_root_titles(text.split("\n\n").map(str::to_string).collect()).join("\n\n")
}

/// Model grid (`0..=999`, upstream `x / 999 * width`) -> `0..=1`, ordered.
fn normalize_bbox(b: [i64; 4]) -> [f32; 4] {
    let n = |v: i64| (v.clamp(0, 999) as f32 / 999.0).min(1.0);
    let (x0, x1) = (n(b[0].min(b[2])), n(b[0].max(b[2])));
    let (y0, y1) = (n(b[1].min(b[3])), n(b[1].max(b[3])));
    [x0, y0, x1, y1]
}

/// The raw text split on `<PAGE>` markers, the leading empty segment dropped
/// and blank pages kept, so index `i` is page `i + 1`. Fence stripping matches
/// [`convert_unlimited_ocr2_markdown`].
pub fn split_unlimited_ocr2_pages(raw: &str) -> Vec<String> {
    let text = strip_fence(raw);
    split_segments(&text)
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Parsed layout blocks in reading order (upstream `_parse_blocks` followed by
/// `_merge_adjacent_titles`).
pub fn parse_unlimited_ocr2_blocks(raw: &str) -> Vec<Ocr2Block> {
    let text = strip_fence(raw);
    merge_adjacent_titles(parse_blocks(&text))
        .into_iter()
        .map(|b| Ocr2Block {
            page: b.page,
            kind: if b.label == RAW {
                "text".to_string()
            } else {
                b.label
            },
            level: b.level,
            bbox: b.bbox.map(|v| v.map(clamp_i32)),
            text: b.content,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        input: String,
        expected: String,
    }

    #[test]
    fn matches_upstream_golden_cases() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/ocr2_markdown/cases.json"
        );
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(path).expect("golden cases"))
                .expect("valid golden json");
        assert!(cases.len() >= 12);
        for c in cases {
            assert_eq!(
                convert_unlimited_ocr2_markdown(&c.input),
                c.expected,
                "case {}",
                c.name
            );
        }
    }

    const REAL: &str = "<PAGE><BLOCK>title<LEVEL2>[42, 65, 855, 101]<FORK>Preparing Context\n\
<BLOCK>text[40, 130, 954, 225]<FORK>Body.\n\
<BLOCK>cross_text[40, 884, 947, 955]<FORK>Starts\n\
<PAGE><BLOCK>append_text[40, 37, 955, 84]<FORK>ends.\n\
<BLOCK>title<LEVEL2>[42, 110, 221, 136]<FORK>Conclusion";

    #[test]
    fn exposes_pages_and_blocks() {
        let pages = split_unlimited_ocr2_pages(&format!("{REAL}<PAGE> <PAGE>x"));
        assert_eq!(pages.len(), 4);
        assert!(pages[2].trim().is_empty());

        let blocks = parse_unlimited_ocr2_blocks(REAL);
        let kinds: Vec<(&str, usize)> = blocks.iter().map(|b| (b.kind.as_str(), b.page)).collect();
        assert_eq!(
            kinds,
            [
                ("title", 1),
                ("text", 1),
                ("cross_text", 1),
                ("append_text", 2),
                ("title", 2)
            ]
        );
        assert_eq!(blocks[0].level, Some(2));
        assert_eq!(blocks[0].bbox, Some([42, 65, 855, 101]));
        assert_eq!(blocks[0].text, "Preparing Context");

        let r = render_unlimited_ocr2(&format!(
            "{REAL}\n<BLOCK>image[0, 500, 999, 999]<FORK>\n<BLOCK>figure_caption[0, 0, 10, 10]<FORK>Fig. 1"
        ));
        let located: Vec<(usize, &str, &str)> = r
            .items
            .iter()
            .map(|i| (i.page, i.kind.as_str(), i.text.as_str()))
            .collect();
        assert_eq!(
            located,
            [
                (1, "title", "# Preparing Context"),
                (1, "text", "Body."),
                (1, "cross_text", "Startsends."),
                (2, "title", "## Conclusion"),
                (2, "figure_caption", "Fig. 1"),
            ]
        );
        let texts: Vec<&str> = r.items.iter().map(|i| i.text.as_str()).collect();
        assert_eq!(r.markdown, texts.join("\n\n"));
        assert!(!r.markdown.contains("!["));
        assert_eq!(
            r.items[1].bbox,
            Some([40.0 / 999.0, 130.0 / 999.0, 954.0 / 999.0, 225.0 / 999.0])
        );

        let raw = parse_unlimited_ocr2_blocks("loose text");
        assert_eq!(raw.len(), 1);
        assert_eq!((raw[0].kind.as_str(), raw[0].bbox), ("text", None));
    }
}
