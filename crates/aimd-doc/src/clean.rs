//! EPUB Markdown post-processing (port of the inlined `cleaner.py` in
//! `aimd.plugins.doc.conversion`). Step order and regexes mirror Python.

use std::sync::LazyLock;

use regex::{Captures, Regex};

use crate::util::splitlines;

const IMAGE_EXTS: &str = "jpg|jpeg|png|gif|webp|svg";

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|e| panic!("bad regex {p:?}: {e}"))
}

macro_rules! lazy_re {
    ($name:ident, $pat:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| re($pat));
    };
}

lazy_re!(
    SPAN_PLACEHOLDER,
    r#"(?i)<span\b[^>]*class="[^"]*\bimage placeholder\b[^"]*"[^>]*>\s*</span>\n*"#
);
lazy_re!(SPAN_ID, r#"(?i)<span\b[^>]*id="[^"]*"[^>]*>\s*</span>\n*"#);

static IMG_IN_IMAGES: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r#"(?i)<img\b[^>]*src="[^"]*/images/([^"/]+\.(?:{IMAGE_EXTS}))"[^>]*alt="([^"]*)"[^>]*/?>"#
    ))
});
static IMG_ANY: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r#"(?i)<img\b[^>]*src="([^"]*?/)?([^"/]+\.(?:{IMAGE_EXTS}))"[^>]*alt="([^"]*)"[^>]*/?>"#
    ))
});
lazy_re!(
    MD_IMG_IMAGES,
    r#"(?i)!\[([^\]]*)\]\((?:[^)"]*/)?images/([^)"/]+)\)"#
);
static MD_IMG_EXT: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r#"(?i)!\[([^\]]*)\]\((?:[^)"]*/)?([^)/"]+\.(?:{IMAGE_EXTS}))\)"#
    ))
});

lazy_re!(SEPARATORS, r"\n[-]{5,}\n");

lazy_re!(FN_BRACKET, r"(?m)^\s*\^\[([^\]]+)\]\(#.+?\)\^\s*");
lazy_re!(FN_ANCHOR_NUM, r"(?mi)^\s*\^<a\b[^>]*>(\d+)</a>\^\s*");
lazy_re!(FN_ANCHOR_TXT, r"(?mi)^\s*\^<a\b[^>]*>([^<]+)</a>\^\s*");
lazy_re!(FN_INLINE, r"\[\^([^\]]+)\]\(#.+?\)");

lazy_re!(
    HASH_LINK,
    r##"(?i)<a\b[^>]*href="#[^"]+"[^>]*>([^<]+)</a>"##
);
lazy_re!(SPAN_TAG, r"(?i)</?span\b[^>]*>");

lazy_re!(
    TOC_LINK,
    r"(?i)\[([^\]]+)\]\([^)]*(?:\.html|\.xhtml)[^)]*\)"
);
lazy_re!(TOC_MARK, r"\s*@@TOC@@\s*");
lazy_re!(NEWLINES3, r"\n{3,}");
lazy_re!(TOC_LEFT, r"@@TOC@@\s*");

lazy_re!(H_NEST, r"^(#{1,6})\s+(#{1,6})(\s+.*)$");
lazy_re!(H_NOSPACE, r"^(#{1,6})(\S)");
lazy_re!(H_DUP, r"^(#{1,6})\s+(#{1,6})\s+");
lazy_re!(H_WS, r"^(#{1,6})\s+");
lazy_re!(WS_RUN, r"[ \t]+");
lazy_re!(H_START, r"^#{1,6}(\s|#|$)");
lazy_re!(H_PARSE, r"^(#{1,6})\s+(.*)");
lazy_re!(
    LABEL,
    r"(?i)^(Chapter|Part|Section|Book|Volume|Appendix)\s+[A-Za-z0-9\.]+$"
);
lazy_re!(SHORT_LABEL, r"(?i)^[A-Z0-9]+[\.\)]?$");
lazy_re!(H1, r"^#\s+");
lazy_re!(H2, r"^##\s+");
lazy_re!(SPACE_BEFORE_H, r"([^\n])\n(#{1,6}\s)");
lazy_re!(SPACE_AFTER_H, r"(#{1,6}\s[^\n]+)\n([^\n#])");
lazy_re!(TRAIL_WS, r"(?m)[ \t]+$");

/// Apply every EPUB-specific fixup to a chapter's Markdown.
pub(crate) fn clean_markdown(text: &str) -> String {
    let mut t = clean_spans(text);
    t = fix_image_refs(&t);
    t = SEPARATORS.replace_all(&t, "\n\n---\n\n").into_owned();
    t = convert_footnotes(&t);
    t = strip_remaining_html(&t);
    t = flatten_toc(&t);
    t = normalize_headings(&t);
    t = merge_consecutive_headings(&t);
    t = demote_headings(&t);
    t = dedup_headings(&t);
    t = ensure_heading_spacing(&t);
    final_whitespace(&t)
}

fn clean_spans(text: &str) -> String {
    let t = SPAN_PLACEHOLDER.replace_all(text, "");
    SPAN_ID.replace_all(&t, "").into_owned()
}

fn alt_or_image(s: &str) -> &str {
    if s.is_empty() { "Image" } else { s }
}

fn fix_image_refs(text: &str) -> String {
    let t = IMG_IN_IMAGES.replace_all(text, |c: &Captures| {
        format!("![{}](images/{})", alt_or_image(&c[2]), &c[1])
    });
    let t = IMG_ANY.replace_all(&t, |c: &Captures| {
        format!("![{}](images/{})", alt_or_image(&c[3]), &c[2])
    });
    let t = MD_IMG_IMAGES.replace_all(&t, "![${1}](images/${2})");
    MD_IMG_EXT
        .replace_all(&t, "![${1}](images/${2})")
        .into_owned()
}

fn convert_footnotes(text: &str) -> String {
    let t = FN_BRACKET.replace_all(text, |c: &Captures| format!("[^{}]: ", c[1].trim()));
    let t = FN_ANCHOR_NUM.replace_all(&t, "[^${1}]: ");
    let t = FN_ANCHOR_TXT.replace_all(&t, |c: &Captures| format!("[^{}]: ", c[1].trim()));
    FN_INLINE.replace_all(&t, "[^${1}]").into_owned()
}

fn strip_remaining_html(text: &str) -> String {
    let t = HASH_LINK.replace_all(text, "${1}");
    SPAN_TAG.replace_all(&t, "").into_owned()
}

fn flatten_toc(text: &str) -> String {
    let t = TOC_LINK.replace_all(text, "@@TOC@@ ${1}");
    let t = TOC_MARK.replace_all(&t, "\n@@TOC@@ ");
    let t = NEWLINES3.replace_all(&t, "\n\n");

    const MARK: &str = "@@TOC@@ ";
    let lines = splitlines(&t);
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim().starts_with(MARK) {
            let mut block: Vec<String> = Vec::new();
            while i < lines.len() && lines[i].trim().starts_with(MARK) {
                let title = lines[i].trim()[MARK.len()..].trim().to_string();
                if !title.is_empty() {
                    block.push(title);
                }
                i += 1;
            }
            if !block.is_empty() {
                out.push(format!("## {}", block.join(": ")));
                out.push(String::new());
            }
            continue;
        }
        out.push(lines[i].to_string());
        i += 1;
    }
    let joined = out.join("\n");
    TOC_LEFT.replace_all(&joined, "").into_owned()
}

fn normalize_heading_line(line: &str) -> String {
    let mut s = line.trim().to_string();
    if s.is_empty() || (s.len() <= 6 && s.chars().all(|c| c == '#')) {
        return String::new();
    }
    loop {
        let new_s = H_NEST
            .replace(&s, |c: &Captures| {
                format!("{}{}", "#".repeat(c[1].len() + c[2].len()), &c[3])
            })
            .into_owned();
        if new_s == s {
            break;
        }
        s = new_s;
    }
    s = H_NOSPACE.replace(&s, "${1} ${2}").into_owned();
    s = H_DUP.replace(&s, "${1} ").into_owned();
    s = H_WS.replace(&s, "${1} ").into_owned();
    s = WS_RUN.replace_all(&s, " ").into_owned();
    s.trim().to_string()
}

fn normalize_headings(text: &str) -> String {
    let mut fixed: Vec<String> = Vec::new();
    for line in splitlines(text) {
        if H_START.is_match(line.trim()) {
            let n = normalize_heading_line(line);
            if !n.is_empty() {
                fixed.push(n);
            }
        } else {
            fixed.push(line.trim_end().to_string());
        }
    }
    fixed.join("\n")
}

fn merge_consecutive_headings(text: &str) -> String {
    let fixed = splitlines(text);
    let mut merged: Vec<String> = Vec::new();
    let mut i = 0;
    while i < fixed.len() {
        let line = fixed[i];
        let Some(cur) = H_PARSE.captures(line) else {
            merged.push(line.to_string());
            i += 1;
            continue;
        };
        let level = cur[1].to_string();
        let mut parts = vec![cur[2].trim().to_string()];
        let mut j = i + 1;
        while j < fixed.len() {
            let look = fixed[j];
            if look.trim().is_empty() {
                j += 1;
                continue;
            }
            if let Some(next) = H_PARSE.captures(look) {
                let last = parts.last().map(String::as_str).unwrap_or("");
                if LABEL.is_match(last) || SHORT_LABEL.is_match(last) {
                    parts.push(next[2].trim().to_string());
                    j += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        merged.push(format!("{level} {}", parts.join(": ")));
        i = j;
    }
    merged.join("\n")
}

fn demote_headings(text: &str) -> String {
    let mut adjusted: Vec<String> = Vec::new();
    for line in splitlines(text) {
        let s = line.trim();
        let s = if H1.is_match(s) {
            H1.replace(s, "## ").into_owned()
        } else if H2.is_match(s) {
            H2.replace(s, "### ").into_owned()
        } else {
            s.to_string()
        };
        adjusted.push(s);
    }
    adjusted.join("\n")
}

/// Is `line` matched by `#{1,6}\s+.+` (the heading group of the Python
/// `_dedup_headings` pattern)?
fn is_dedup_heading(line: &str) -> bool {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return false;
    }
    let mut rest = line[hashes..].chars();
    matches!(rest.next(), Some(c) if c.is_whitespace()) && rest.next().is_some()
}

/// `re.sub(r"^(#{1,6}\s+.+)\n+\1$", r"\1", text, flags=re.M)`: a heading line
/// repeated verbatim right after (blank lines allowed) collapses to one.
fn dedup_headings(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        if is_dedup_heading(lines[i]) {
            let mut j = i + 1;
            while j < lines.len() && lines[j].is_empty() {
                j += 1;
            }
            if j < lines.len() && lines[j] == lines[i] {
                out.push(lines[i]);
                i = j + 1;
                continue;
            }
        }
        out.push(lines[i]);
        i += 1;
    }
    out.join("\n")
}

fn ensure_heading_spacing(text: &str) -> String {
    let t = SPACE_BEFORE_H.replace_all(text, "${1}\n\n${2}");
    SPACE_AFTER_H.replace_all(&t, "${1}\n\n${2}").into_owned()
}

fn final_whitespace(text: &str) -> String {
    let t = NEWLINES3.replace_all(text, "\n\n");
    let t = TRAIL_WS.replace_all(&t, "");
    format!("{}\n", t.trim())
}
