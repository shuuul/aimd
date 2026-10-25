//! Small Python-compatibility helpers so ported messages and string
//! handling stay byte-identical to the Python implementation.

/// Python `str.isspace()` for one character (Unicode `White_Space` plus the
/// ASCII information separators `\x1c`..`\x1f`, which Python also treats as
/// whitespace).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// Python `repr()` of a `str` (quote selection and common escapes), used in
/// error messages that the Python code formats with `{value!r}`.
pub fn repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a `list[str]` (`['a', 'b']`).
pub fn repr_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python_quoting() {
        assert_eq!(repr("fp8"), "'fp8'");
        assert_eq!(repr("it's"), "\"it's\"");
        assert_eq!(repr("a'\"b"), "'a\\'\"b'");
        assert_eq!(repr("x\ny"), "'x\\ny'");
        assert_eq!(repr_list(&["zh", "en"]), "['zh', 'en']");
    }

    #[test]
    fn strip_matches_python_whitespace() {
        assert_eq!(strip("\u{1c} a \u{3000}"), "a");
    }
}
