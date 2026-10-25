//! Validate the tracked spec tree (port of the former `scripts/check_specs.py`).
//!
//! Exit code 0 with "Specs check passed." when every invariant holds; exit code
//! 1 with "Specs check failed:" followed by one line per problem otherwise.
//! All problems are collected and printed together.
//!
//! Frontmatter is deliberately restricted to flat, scalar `key: value` data, so
//! it is parsed ad hoc and structured YAML is rejected.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const REQUIRED_KEYS: [&str; 6] = ["id", "title", "status", "created", "updated", "coordinator"];
const REQUIRED_HEADINGS: [&str; 9] = [
    "Context",
    "Goal and success criteria",
    "Scope and non-goals",
    "Decisions",
    "Workstreams",
    "Verification",
    "Documentation sync",
    "Progress and handoff",
    "Completion summary",
];
const ARCHIVE_STATUS: &str = "Completed";
const TEMPLATE_NAME: &str = "000-template.md";
const README_NAME: &str = "README.md";

// ---------------------------------------------------------------- py helpers

/// Python `str.isspace` (approximation: Unicode White_Space plus \x1c-\x1f).
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// Python `str.splitlines()`.
fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        lines.push(&text[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r'
            && let Some(&(j, '\n')) = iter.peek()
        {
            iter.next();
            next = j + 1;
        }
        start = next;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Python `repr(str)`.
fn repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() || (c.is_whitespace() && c != ' ') => {
                let n = c as u32;
                if n < 0x100 {
                    let _ = write!(out, "\\x{n:02x}");
                } else if n < 0x10000 {
                    let _ = write!(out, "\\u{n:04x}");
                } else {
                    let _ = write!(out, "\\U{n:08x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn repr_opt(s: Option<&str>) -> String {
    s.map_or_else(|| "None".to_string(), repr)
}

/// Python `int(str)` restricted to what can matter here (ASCII digits).
fn py_int(s: &str) -> Option<i128> {
    let s = strip(s);
    let (neg, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'_')
    {
        return None;
    }
    let cleaned: String = digits.chars().filter(|&c| c != '_').collect();
    let n: i128 = cleaned.parse().ok()?;
    Some(if neg { -n } else { n })
}

// ---------------------------------------------------------------------- dates

type Date = (i32, u32, u32);

fn is_leap(y: i32) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
    }
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(if m <= 2 { y - 1 } else { y });
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> Date {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = (yoe + era * 400) as i32 + i32::from(m <= 2);
    (y, m, d)
}

/// 0 = Monday.
fn weekday(y: i32, m: u32, d: u32) -> i64 {
    (days_from_civil(y, m, d) + 3).rem_euclid(7)
}

fn valid_ymd(y: i32, m: u32, d: u32) -> Option<Date> {
    ((1..=9999).contains(&y) && (1..=12).contains(&m) && d >= 1 && d <= days_in_month(y, m))
        .then_some((y, m, d))
}

fn iso_week_to_date(y: i32, week: u32, day: u32) -> Option<Date> {
    if !(1..=9999).contains(&y) || !(1..=7).contains(&day) {
        return None;
    }
    if !(1..53).contains(&week) {
        let jan1 = weekday(y, 1, 1);
        // Week 53 exists when Jan 1 is a Thursday, or a Wednesday in a leap year.
        if !(week == 53 && (jan1 == 3 || (jan1 == 2 && is_leap(y)))) {
            return None;
        }
    }
    let jan4 = days_from_civil(y, 1, 4);
    let monday1 = jan4 - weekday(y, 1, 4);
    let (ry, rm, rd) = civil_from_days(monday1 + i64::from(week - 1) * 7 + i64::from(day - 1));
    valid_ymd(ry, rm, rd)
}

fn digits(s: &str) -> Option<u32> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok())?
}

/// Python 3.11+ `date.fromisoformat` (calendar, compact and ISO week forms).
fn from_isoformat(s: &str) -> Option<Date> {
    if !s.is_ascii() || !matches!(s.len(), 7 | 8 | 10) {
        return None;
    }
    let year = digits(&s[0..4])? as i32;
    let sep = s.as_bytes()[4] == b'-';
    let mut pos = if sep { 5 } else { 4 };
    if s[pos..].starts_with('W') {
        pos += 1;
        let week = digits(s.get(pos..pos + 2)?)?;
        pos += 2;
        let mut day = 1;
        if s.len() > pos {
            if (s.as_bytes()[pos] == b'-') != sep {
                return None;
            }
            pos += usize::from(sep);
            day = digits(s.get(pos..pos + 1)?)?;
            pos += 1;
        }
        if s.len() != pos {
            return None;
        }
        iso_week_to_date(year, week, day)
    } else {
        let month = digits(s.get(pos..pos + 2)?)?;
        pos += 2;
        if (s.as_bytes().get(pos) == Some(&b'-')) != sep {
            return None;
        }
        pos += usize::from(sep);
        let day = digits(s.get(pos..pos + 2)?)?;
        if s.len() != pos + 2 {
            return None;
        }
        valid_ymd(year, month, day)
    }
}

fn parse_date(value: &str) -> Option<Date> {
    from_isoformat(strip(value).trim_matches(|c| c == '"' || c == '\''))
}

fn fmt_date((y, m, d): Date) -> String {
    format!("{y:04}-{m:02}-{d:02}")
}

// ------------------------------------------------------------ pattern helpers

/// `^(?P<num>\d{3})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$` -> (num, slug).
fn formal_id(name: &str) -> Option<(u32, &str)> {
    let bytes = name.as_bytes();
    if bytes.len() < 4 || !bytes[..3].iter().all(u8::is_ascii_digit) || bytes[3] != b'-' {
        return None;
    }
    let slug = name[4..].strip_suffix(".md")?;
    let ok = !slug.is_empty()
        && slug.split('-').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        });
    ok.then(|| (name[..3].parse().expect("three ASCII digits"), slug))
}

/// All matches of `\[[^\]]*\]\((?P<target>[^)]+)\)` in `line`, as targets.
fn md_links(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'[' {
            let close = b[i + 1..]
                .iter()
                .position(|&c| c == b']')
                .map(|p| i + 1 + p);
            if let Some(close) = close
                && b.get(close + 1) == Some(&b'(')
            {
                let start = close + 2;
                if let Some(p) = b[start..].iter().position(|&c| c == b')')
                    && p > 0
                {
                    out.push(&line[start..start + p]);
                    i = start + p + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// `^##\s+` on an already-stripped line; returns the remainder after whitespace.
fn h2_rest(stripped: &str) -> Option<&str> {
    let rest = stripped.strip_prefix("##")?;
    let trimmed = rest.trim_start_matches(is_space);
    (trimmed.len() < rest.len()).then_some(trimmed)
}

/// Python `Path(target).name`.
fn path_name(target: &str) -> &str {
    target
        .split('/')
        .rfind(|part| !part.is_empty() && *part != ".")
        .unwrap_or("")
}

/// `:?-+:?` full match.
fn is_separator_cell(cell: &str) -> bool {
    let c = cell.strip_prefix(':').unwrap_or(cell);
    let c = c.strip_suffix(':').unwrap_or(c);
    !c.is_empty() && c.bytes().all(|b| b == b'-')
}

// ------------------------------------------------------------------- parsing

/// Level-2 headings outside fenced code blocks.
fn parsing_headings(text: &str) -> Vec<String> {
    let mut headings = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for line in splitlines(text) {
        let stripped = strip(line);
        let first = stripped.chars().next();
        if let Some(mark_char @ ('`' | '~')) = first {
            let run = stripped.chars().take_while(|&c| c == mark_char).count();
            if run >= 3 {
                match fence {
                    None => fence = Some((mark_char, run)),
                    Some((fc, fl)) => {
                        if stripped.starts_with(fc) && stripped.chars().count() >= fl {
                            fence = None;
                        }
                    }
                }
                continue;
            }
        }
        if fence.is_none()
            && let Some(rest) = h2_rest(stripped)
            && !rest.is_empty()
        {
            headings.push(rest.to_string());
        }
    }
    headings
}

fn parse_frontmatter(text: &str) -> (HashMap<String, String>, Vec<String>) {
    let mut errors = Vec::new();
    let lines = splitlines(text);
    if lines.is_empty() || strip(lines[0]) != "---" {
        return (
            HashMap::new(),
            vec!["frontmatter must start with a '---' delimiter line".to_string()],
        );
    }
    let Some(end) = (1..lines.len()).find(|&i| strip(lines[i]) == "---") else {
        return (
            HashMap::new(),
            vec!["frontmatter is missing the closing '---' delimiter line".to_string()],
        );
    };
    let mut fm = HashMap::new();
    for (i, raw) in lines.iter().enumerate().take(end).skip(1) {
        let stripped = strip(raw);
        if stripped.is_empty() {
            continue;
        }
        let n = i + 1;
        if stripped.starts_with('#') {
            errors.push(format!(
                "frontmatter line {n} must be flat 'key: value', got comment: {}",
                repr(stripped)
            ));
            continue;
        }
        let Some((key, value)) = raw.split_once(':') else {
            errors.push(format!(
                "frontmatter line {n} is not flat 'key: value': {}",
                repr(stripped)
            ));
            continue;
        };
        let key = strip(key);
        let mut value = strip(value);
        if key.is_empty()
            || !key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            errors.push(format!(
                "frontmatter line {n} has an invalid key {}",
                repr(key)
            ));
            continue;
        }
        if value.is_empty() {
            errors.push(format!("frontmatter key {} has an empty value", repr(key)));
            continue;
        }
        if value.starts_with(['|', '>', '&', '*', '!', '{', '[']) {
            errors.push(format!(
                "frontmatter key {} uses structured YAML ({}); only flat scalar values are allowed",
                repr(key),
                repr(value)
            ));
            continue;
        }
        let vc: Vec<char> = value.chars().collect();
        if vc.len() >= 2 && vc[0] == vc[vc.len() - 1] && (vc[0] == '\'' || vc[0] == '"') {
            value = &value[1..value.len() - 1];
        }
        if fm.contains_key(key) {
            errors.push(format!("duplicate frontmatter key {}", repr(key)));
        }
        fm.insert(key.to_string(), value.to_string());
    }
    (fm, errors)
}

// -------------------------------------------------------------------- checks

fn rel_posix(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Mimic Python's `OSError` text for the common missing-file case.
fn read_error(path: &Path, err: &io::Error) -> String {
    if err.kind() == io::ErrorKind::NotFound {
        format!(
            "[Errno 2] No such file or directory: {}",
            repr(&path.to_string_lossy())
        )
    } else {
        err.to_string()
    }
}

fn check_spec_file(root: &Path, path: &Path, is_template: bool) -> Vec<String> {
    let rel = rel_posix(root, path);
    let mut errors = Vec::new();
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return vec![format!("{rel}: cannot read file: {}", read_error(path, &e))],
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let (fm, fm_errors) = parse_frontmatter(&text);
    errors.extend(fm_errors.into_iter().map(|m| format!("{rel}: {m}")));

    if !is_template {
        match formal_id(&name) {
            None => errors.push(format!("{rel}: filename does not match NNN-kebab-case.md")),
            Some((0, _)) => errors.push(format!(
                "{rel}: ID 000 is reserved for specs/000-template.md"
            )),
            Some((file_id, _)) => {
                let fm_id = fm
                    .get("id")
                    .and_then(|v| py_int(v.trim_matches(|c| c == '"' || c == '\'')))
                    .unwrap_or(-1);
                if fm_id != i128::from(file_id) {
                    errors.push(format!(
                        "{rel}: frontmatter id {} does not match filename ID {file_id:03}",
                        repr_opt(fm.get("id").map(String::as_str))
                    ));
                }
            }
        }
    }

    for key in REQUIRED_KEYS {
        if !fm.contains_key(key) {
            errors.push(format!(
                "{rel}: frontmatter is missing required key {}",
                repr(key)
            ));
        }
    }

    if !is_template {
        let created = fm.get("created").and_then(|v| parse_date(v));
        let updated = fm.get("updated").and_then(|v| parse_date(v));
        if let (Some(raw), None) = (fm.get("created"), created) {
            errors.push(format!(
                "{rel}: created {} is not a real YYYY-MM-DD date",
                repr(raw)
            ));
        }
        if let (Some(raw), None) = (fm.get("updated"), updated) {
            errors.push(format!(
                "{rel}: updated {} is not a real YYYY-MM-DD date",
                repr(raw)
            ));
        }
        if let (Some(c), Some(u)) = (created, updated)
            && u < c
        {
            errors.push(format!(
                "{rel}: updated ({}) is earlier than created ({})",
                fmt_date(u),
                fmt_date(c)
            ));
        }
    }

    let in_archive = path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|n| n == "archive");
    let status = fm.get("status").map(String::as_str);
    if in_archive {
        if status != Some(ARCHIVE_STATUS) {
            errors.push(format!(
                "{rel}: status {} is not allowed in specs/archive/ (expected {ARCHIVE_STATUS})",
                repr_opt(status)
            ));
        }
    } else if !matches!(status, Some("Draft" | "Active")) {
        errors.push(format!(
            "{rel}: status {} is not allowed in specs/ (expected Draft or Active)",
            repr_opt(status)
        ));
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for heading in parsing_headings(&text) {
        *counts.entry(heading).or_default() += 1;
    }
    for heading in REQUIRED_HEADINGS {
        match counts.get(heading).copied().unwrap_or(0) {
            0 => errors.push(format!("{rel}: missing required heading '## {heading}'")),
            1 => {}
            n => errors.push(format!(
                "{rel}: duplicate required heading '## {heading}' ({n} occurrences)"
            )),
        }
    }
    errors
}

type Index = Vec<(u32, String)>; // sorted by ID

fn check_readme(root: &Path, readme: &Path, active: &Index, archived: &Index) -> Vec<String> {
    let rel = rel_posix(root, readme);
    let mut errors = Vec::new();
    let text = match fs::read_to_string(readme) {
        Ok(t) => t,
        Err(e) => {
            return vec![format!(
                "{rel}: cannot read file: {}",
                read_error(readme, &e)
            )];
        }
    };
    let lines = splitlines(&text);

    let template_links = lines
        .iter()
        .flat_map(|l| md_links(l))
        .filter(|t| *t == TEMPLATE_NAME)
        .count();
    if template_links != 1 {
        errors.push(format!(
            "{rel}: must link the template ({TEMPLATE_NAME}) exactly once"
        ));
    }

    let h2: Vec<usize> = (0..lines.len())
        .filter(|&i| h2_rest(strip(lines[i])).is_some())
        .collect();
    // Insertion-ordered map: a repeated section replaces the earlier range in place.
    let mut sections: Vec<(&str, usize, usize)> = Vec::new();
    for (idx, &i) in h2.iter().enumerate() {
        let name = h2_rest(strip(lines[i])).map(strip).and_then(|r| {
            ["Active specs", "Archived specs"]
                .into_iter()
                .find(|n| *n == r)
        });
        let Some(name) = name else { continue };
        let end = h2.get(idx + 1).copied().unwrap_or(lines.len());
        match sections.iter_mut().find(|s| s.0 == name) {
            Some(s) => (s.1, s.2) = (i + 1, end),
            None => sections.push((name, i + 1, end)),
        }
    }

    for (section, start, end) in sections {
        let is_active = section == "Active specs";
        let expected = if is_active { active } else { archived };
        let mut linked: Vec<(u32, String)> = Vec::new();
        let mut prev: Option<u32> = None;
        for line in &lines[start..end] {
            let stripped = strip(line);
            if !stripped.starts_with('|') {
                continue;
            }
            let cells: Vec<&str> = stripped.trim_matches('|').split('|').map(strip).collect();
            if cells.iter().all(|c| is_separator_cell(c)) || cells[0] == "Spec" {
                continue;
            }
            let links = md_links(stripped);
            if links.is_empty() {
                errors.push(format!(
                    "{rel}: {section} row has no spec link: {}",
                    repr(stripped)
                ));
                continue;
            }
            if links.len() > 1 {
                errors.push(format!(
                    "{rel}: {section} row has multiple spec links: {}",
                    repr(stripped)
                ));
            }
            let target = links[0];
            if target == TEMPLATE_NAME {
                errors.push(format!("{rel}: {section} table must not link the template"));
                continue;
            }
            let parsed = formal_id(path_name(target)).filter(|(n, _)| *n != 0);
            let Some((link_id, slug)) = parsed else {
                errors.push(format!(
                    "{rel}: {section} table links a non-spec file {}",
                    repr(target)
                ));
                continue;
            };
            let mut expected_target = format!("{link_id:03}-{slug}.md");
            if !is_active {
                expected_target = format!("archive/{expected_target}");
            }
            if target != expected_target {
                errors.push(format!(
                    "{rel}: {section} table links {}; expected {} (wrong section or path)",
                    repr(target),
                    repr(&expected_target)
                ));
                continue;
            }
            if !root.join("specs").join(target).is_file() {
                errors.push(format!(
                    "{rel}: link {} does not resolve to a spec file",
                    repr(target)
                ));
                continue;
            }
            if linked.iter().any(|(id, _)| *id == link_id) {
                errors.push(format!(
                    "{rel}: {section} table links spec {link_id:03} ({}) more than once",
                    repr(target)
                ));
                continue;
            }
            linked.push((link_id, target.to_string()));
            if let Some(p) = prev
                && link_id <= p
            {
                errors.push(format!(
                    "{rel}: {section} table is not in ascending ID order ({p:03} before {link_id:03})"
                ));
            }
            prev = Some(link_id);
        }

        for (id, target) in expected {
            if !linked.iter().any(|(l, _)| l == id) {
                errors.push(format!(
                    "{rel}: does not index specs/{target} in the {section} table"
                ));
            }
        }
        for (id, _) in &linked {
            if !expected.iter().any(|(e, _)| e == id) {
                let kind = if is_active { "active" } else { "archived" };
                errors.push(format!(
                    "{rel}: {section} table indexes spec {id:03}, which is not a {kind} spec"
                ));
            }
        }
    }
    errors
}

fn sorted_md(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".md"))
        })
        .collect();
    paths.sort();
    paths
}

fn add_index(index: &mut Index, path: &Path, label: &str, errors: &mut Vec<String>) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let Some((num, _)) = formal_id(&name) else {
        return;
    };
    if num == 0 {
        return;
    }
    if let Some((_, existing)) = index.iter().find(|(n, _)| *n == num) {
        errors.push(format!(
            "duplicate spec ID {num:03} used by {} and {} in {label}",
            repr(existing),
            repr(&name)
        ));
        return;
    }
    index.push((num, name));
    index.sort();
}

/// Validate the spec tree under `root`; returns every problem found.
pub fn check_tree(root: &Path) -> Vec<String> {
    let mut errors = Vec::new();
    let specs_dir = root.join("specs");
    let archive_dir = specs_dir.join("archive");
    let readme = specs_dir.join(README_NAME);
    let template = specs_dir.join(TEMPLATE_NAME);

    if !specs_dir.is_dir() {
        errors.push("specs/ directory is missing".to_string());
    }
    if !archive_dir.is_dir() {
        errors.push("specs/archive/ directory is missing".to_string());
    }
    if !readme.is_file() {
        errors.push("specs/README.md is missing".to_string());
    }
    if !template.is_file() {
        errors.push("specs/000-template.md is missing".to_string());
    }
    if !specs_dir.is_dir() {
        return errors;
    }

    let mut active = Index::new();
    let mut archived = Index::new();
    for path in sorted_md(&specs_dir) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name == TEMPLATE_NAME {
            errors.extend(check_spec_file(root, &path, true));
            continue;
        }
        if name == README_NAME {
            continue;
        }
        errors.extend(check_spec_file(root, &path, false));
        add_index(&mut active, &path, "specs/", &mut errors);
    }
    for path in sorted_md(&archive_dir) {
        errors.extend(check_spec_file(root, &path, false));
        add_index(&mut archived, &path, "specs/archive/", &mut errors);
    }

    let max_id = active.iter().chain(archived.iter()).map(|(n, _)| *n).max();
    if let Some(max_id) = max_id {
        let has = |n: u32| active.iter().chain(archived.iter()).any(|(i, _)| *i == n);
        let missing: Vec<String> = (1..=max_id)
            .filter(|n| !has(*n))
            .map(|n| format!("{n:03}"))
            .collect();
        if !missing.is_empty() {
            errors.push(format!(
                "spec IDs are not continuous from 001; missing: {}",
                missing.join(", ")
            ));
        }
        for (n, a) in &active {
            if let Some((_, b)) = archived.iter().find(|(i, _)| i == n) {
                errors.push(format!(
                    "spec ID {n:03} is used by both specs/{a} and specs/archive/{b}"
                ));
            }
        }
    }

    errors.extend(check_readme(root, &readme, &active, &archived));
    errors
}

/// Render the validator output and exit code for a list of problems.
pub fn report(problems: &[String]) -> (u8, String) {
    if problems.is_empty() {
        return (0, "Specs check passed.\n".to_string());
    }
    let mut out = String::from("Specs check failed:\n");
    for p in problems {
        let _ = writeln!(out, "  - {p}");
    }
    (1, out)
}

#[cfg(test)]
mod tests;
