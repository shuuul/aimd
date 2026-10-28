//! LaTeX formula repairs, a port of the MiniCPM-OCR SDK's
//! `postprocess/latex_repair.py` (`repair_latex_formula`), which the SDK
//! runs on every display / inline formula crop.
//!
//! Python indexes strings by code point; the brace and `\left` / `\right`
//! walkers here work on `Vec<char>` for the same reason. Lookarounds the
//! `regex` crate lacks (`(?![a-zA-Z])`) are hand-written scans.

use std::sync::OnceLock;

use regex::{NoExpand, Regex};

use super::markdown::py_rstrip;

/// `_is_escaped`: an odd number of backslashes right before `pos`.
fn is_escaped(s: &[char], pos: usize) -> bool {
    s[..pos].iter().rev().take_while(|&&c| c == '\\').count() % 2 == 1
}

/// `_find_group_end`: index of the `}` closing the group at `depth`.
fn find_group_end(s: &[char], pos: usize, depth: i64) -> Option<usize> {
    let mut current = depth;
    for i in pos..s.len() {
        if s[i] == '{' && !is_escaped(s, i) {
            current += 1;
        } else if s[i] == '}' && !is_escaped(s, i) {
            current -= 1;
            if current < depth {
                return Some(i);
            }
        }
    }
    None
}

/// `fix_unbalanced_braces`: drop every unmatched, unescaped `{` / `}`.
pub fn fix_unbalanced_braces(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut stack = Vec::new();
    let mut unmatched = vec![false; chars.len()];
    for (i, &ch) in chars.iter().enumerate() {
        if (ch == '{' || ch == '}') && !is_escaped(&chars, i) {
            if ch == '{' {
                stack.push(i);
            } else if stack.pop().is_none() {
                unmatched[i] = true;
            }
        }
    }
    for i in stack {
        unmatched[i] = true;
    }
    chars
        .iter()
        .zip(&unmatched)
        .filter(|(_, u)| !**u)
        .map(|(&c, _)| c)
        .collect()
}

/// Python `str.isalpha` for one character (Unicode letters).
fn py_isalpha(c: char) -> bool {
    c.is_alphabetic()
}

fn starts_with_at(s: &[char], i: usize, pat: &str) -> bool {
    pat.chars()
        .enumerate()
        .all(|(k, p)| s.get(i + k) == Some(&p))
}

/// `_fix_left_right_pairs`: move a `\right` that sits in another brace
/// group than its `\left` to just before that group's closing `}`.
fn fix_left_right_pairs(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut brace_depth: usize = 0;
    let mut left_stack: Vec<(usize, usize)> = Vec::new();
    let mut adjustments: Vec<(usize, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if i > 0 && is_escaped(&chars, i) {
            i += 1;
            continue;
        }
        if starts_with_at(&chars, i, "\\left") && i + 5 < n && !py_isalpha(chars[i + 5]) {
            left_stack.push((i, brace_depth));
            i += 6;
            continue;
        }
        if starts_with_at(&chars, i, "\\right") && i + 6 < n && !py_isalpha(chars[i + 6]) {
            if let Some((left_pos, left_depth)) = left_stack.pop()
                && left_depth != brace_depth
                && let Some(target) = find_group_end(&chars, left_pos, left_depth as i64)
            {
                adjustments.push((i, i + 7, target));
            }
            i += 7;
            continue;
        }
        if chars[i] == '{' && !is_escaped(&chars, i) {
            brace_depth += 1;
        } else if chars[i] == '}' && !is_escaped(&chars, i) {
            brace_depth = brace_depth.saturating_sub(1);
        }
        i += 1;
    }
    if adjustments.is_empty() {
        return s.to_string();
    }
    // Python edits a list of one-character strings in place; an inserted
    // `\right` becomes one multi-character element, which shifts the
    // element indices of the later (smaller-start) edits by one.
    let mut result: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    adjustments.sort_by_key(|a| std::cmp::Reverse(a.0));
    for (start, end, target) in adjustments {
        let start = start.min(result.len());
        let end = end.min(result.len()).max(start);
        let part: String = result.drain(start..end).collect();
        let target = target.min(result.len());
        result.insert(target, part);
    }
    result.concat()
}

const DELIM_NAMES: [&str; 11] = [
    "lceil",
    "rceil",
    "lfloor",
    "rfloor",
    "backslash",
    "uparrow",
    "downarrow",
    "Uparrow",
    "Downarrow",
    "langle",
    "rangle",
];

/// Length in bytes of the optional delimiter group of `_LEFT_PATTERN` /
/// `_RIGHT_PATTERN` at `rest` (0 = no delimiter).
fn delimiter_len(rest: &[u8]) -> usize {
    if rest.first() == Some(&b'\\') {
        for name in DELIM_NAMES {
            if rest[1..].starts_with(name.as_bytes()) {
                return 1 + name.len();
            }
        }
        if matches!(rest.get(1), Some(b'{' | b'}' | b'|' | b'.')) {
            return 2;
        }
        return 0;
    }
    match rest.first() {
        Some(b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'/' | b'|' | b'.') => 1,
        _ => 0,
    }
}

/// `cmd` at byte `i`, not followed by an ASCII letter (`cmd(?![a-zA-Z])`).
fn command_at(b: &[u8], i: usize, cmd: &str) -> bool {
    b[i..].starts_with(cmd.as_bytes()) && !b.get(i + cmd.len()).is_some_and(u8::is_ascii_alphabetic)
}

/// `_LEFT_PATTERN.sub(_fix_delim, s)` (or `_RIGHT_PATTERN`): append `.` to
/// a bare `\left` / `\right` without a recognised delimiter.
fn fix_delimiters(s: &str, cmd: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        if command_at(b, i, cmd) {
            let dl = delimiter_len(&b[i + cmd.len()..]);
            if dl == 0 {
                out.push_str(&s[last..i + cmd.len()]);
                out.push('.');
                last = i + cmd.len();
            }
            i += cmd.len() + dl;
        } else {
            i += 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

fn count_command(s: &str, cmd: &str) -> usize {
    let b = s.as_bytes();
    let mut count = 0;
    let mut i = 0;
    while i < b.len() {
        if command_at(b, i, cmd) {
            count += 1;
            i += cmd.len();
        } else {
            i += 1;
        }
    }
    count
}

/// `_LEFT_RIGHT_REMOVE_PATTERN.sub("", s)`.
fn remove_left_right(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        let cmd = ["\\left", "\\right"]
            .into_iter()
            .find(|cmd| command_at(b, i, cmd));
        if let Some(cmd) = cmd {
            out.push_str(&s[last..i]);
            let mut end = i + cmd.len();
            if b.get(end) == Some(&b'.') {
                end += 1;
            }
            last = end;
            i = end;
        } else {
            i += 1;
        }
    }
    out.push_str(&s[last..]);
    out
}

/// `fix_latex_left_right`.
pub fn fix_latex_left_right(s: &str, fix_delimiter: bool) -> String {
    let mut s = s.to_string();
    if fix_delimiter {
        s = fix_delimiters(&s, "\\left");
        s = fix_delimiters(&s, "\\right");
    }
    if count_command(&s, "\\left") == count_command(&s, "\\right") {
        return fix_left_right_pairs(&s);
    }
    remove_left_right(&s)
}

const ENV_TYPES: [&str; 12] = [
    "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix", "cases", "aligned",
    "gathered", "align", "align*",
];

/// `fix_latex_environments`: balance `\begin{env}` / `\end{env}` counts.
pub fn fix_latex_environments(s: &str) -> String {
    let mut s = s.to_string();
    for env in ENV_TYPES {
        let begin = format!("\\begin{{{env}}}");
        let end = format!("\\end{{{env}}}");
        let begin_count = s.matches(&begin).count();
        let end_count = s.matches(&end).count();
        if begin_count == end_count {
            continue;
        }
        if end_count > begin_count {
            // `\begin\{env\}\{([^}]*)\}`: the first `\begin{env}{` whose
            // argument closes.
            let open = format!("{begin}{{");
            let fmt = s.match_indices(&open).find_map(|(at, _)| {
                let arg_start = at + open.len();
                s[arg_start..]
                    .find('}')
                    .map(|close| format!("{{{}}}", &s[arg_start..arg_start + close]))
            });
            let fmt = fmt.unwrap_or_else(|| {
                if env == "array" {
                    "{c}".to_string()
                } else {
                    String::new()
                }
            });
            let prefix = format!("{begin}{fmt} ").repeat(end_count - begin_count);
            s = prefix + &s;
        } else {
            s += &format!(" {end}").repeat(begin_count - end_count);
        }
    }
    s
}

fn dup_array_close() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(\\end\{array\}[\s\x1C-\x1F]*\\right\.[\s\x1C-\x1F]*){2,}").expect("regex")
    })
}

fn dup_bracket_close() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(\\\][\s\x1C-\x1F]*){2,}").expect("regex"))
}

/// `remove_duplicate_closures`.
pub fn remove_duplicate_closures(s: &str) -> String {
    let s = dup_array_close().replace_all(s, NoExpand("\\end{array} \\right."));
    dup_bracket_close()
        .replace_all(&s, NoExpand("\\]"))
        .into_owned()
}

/// `remove_trailing_backslashes`: `re.sub(r"\\+$", "", s).rstrip()`, where
/// `$` also matches before a final `\n`.
pub fn remove_trailing_backslashes(s: &str) -> String {
    let body_end = if s.ends_with('\n') {
        s.len() - 1
    } else {
        s.len()
    };
    let run_start = s[..body_end].trim_end_matches('\\').len();
    let out = format!("{}{}", &s[..run_start], &s[body_end..]);
    py_rstrip(&out).to_string()
}

/// `repair_latex_formula(s, label)`.
pub fn repair_latex_formula(s: &str, label: &str) -> String {
    if super::markdown::py_strip(s).is_empty() {
        return s.to_string();
    }
    let s = fix_unbalanced_braces(s);
    let is_display = label == "display_formula" || label.is_empty();
    let s = fix_latex_left_right(&s, is_display);
    let s = fix_latex_environments(&s);
    let s = remove_duplicate_closures(&s);
    remove_trailing_backslashes(&s)
}
