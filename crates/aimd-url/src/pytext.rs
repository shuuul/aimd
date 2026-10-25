//! Python `str` semantics the ported cleanup code depends on.
//!
//! The Python URL plugin uses `str.strip()`, `str.split()`, `str.splitlines()`,
//! `re` `\s`/`\d` and code-point slicing. Rust's `trim`/`split_whitespace`/
//! `lines` differ on a handful of code points (`\x1c`-`\x1f`, `\x85`,
//! ` `, ...), so the golden-parity paths use these helpers instead.

/// `str.isspace()` / `re` `\s` (Unicode).
pub(crate) fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\x0b'
            | '\x0c'
            | '\r'
            | '\x1c'..='\x1f'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// Unicode `Nd` ranges (`str.isdecimal()` / `re` `\d`), Unicode 17.
const DECIMAL_RANGES: &[(u32, u32)] = &[
    (0x30, 0x39),
    (0x660, 0x669),
    (0x6F0, 0x6F9),
    (0x7C0, 0x7C9),
    (0x966, 0x96F),
    (0x9E6, 0x9EF),
    (0xA66, 0xA6F),
    (0xAE6, 0xAEF),
    (0xB66, 0xB6F),
    (0xBE6, 0xBEF),
    (0xC66, 0xC6F),
    (0xCE6, 0xCEF),
    (0xD66, 0xD6F),
    (0xDE6, 0xDEF),
    (0xE50, 0xE59),
    (0xED0, 0xED9),
    (0xF20, 0xF29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17E0, 0x17E9),
    (0x1810, 0x1819),
    (0x1946, 0x194F),
    (0x19D0, 0x19D9),
    (0x1A80, 0x1A89),
    (0x1A90, 0x1A99),
    (0x1B50, 0x1B59),
    (0x1BB0, 0x1BB9),
    (0x1C40, 0x1C49),
    (0x1C50, 0x1C59),
    (0xA620, 0xA629),
    (0xA8D0, 0xA8D9),
    (0xA900, 0xA909),
    (0xA9D0, 0xA9D9),
    (0xA9F0, 0xA9F9),
    (0xAA50, 0xAA59),
    (0xABF0, 0xABF9),
    (0xFF10, 0xFF19),
    (0x104A0, 0x104A9),
    (0x10D30, 0x10D39),
    (0x10D40, 0x10D49),
    (0x11066, 0x1106F),
    (0x110F0, 0x110F9),
    (0x11136, 0x1113F),
    (0x111D0, 0x111D9),
    (0x112F0, 0x112F9),
    (0x11450, 0x11459),
    (0x114D0, 0x114D9),
    (0x11650, 0x11659),
    (0x116C0, 0x116C9),
    (0x116D0, 0x116E3),
    (0x11730, 0x11739),
    (0x118E0, 0x118E9),
    (0x11950, 0x11959),
    (0x11BF0, 0x11BF9),
    (0x11C50, 0x11C59),
    (0x11D50, 0x11D59),
    (0x11DA0, 0x11DA9),
    (0x11DE0, 0x11DE9),
    (0x11F50, 0x11F59),
    (0x16130, 0x16139),
    (0x16A60, 0x16A69),
    (0x16AC0, 0x16AC9),
    (0x16B50, 0x16B59),
    (0x16D70, 0x16D79),
    (0x1CCF0, 0x1CCF9),
    (0x1D7CE, 0x1D7FF),
    (0x1E140, 0x1E149),
    (0x1E2F0, 0x1E2F9),
    (0x1E4F0, 0x1E4F9),
    (0x1E5F1, 0x1E5FA),
    (0x1E950, 0x1E959),
    (0x1FBF0, 0x1FBF9),
];

/// `re` `\d` (Unicode decimal digit).
pub(crate) fn is_decimal(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return c.is_ascii_digit();
    }
    DECIMAL_RANGES
        .iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&cp))
}

/// `str.strip()`.
pub(crate) fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.split()` with no separator.
pub(crate) fn split_ws(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|t| !t.is_empty()).collect()
}

/// `str.splitlines()` (no `keepends`).
pub(crate) fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut iter = s.char_indices().peekable();
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
        out.push(&s[start..i]);
        let mut end = i + c.len_utf8();
        if c == '\r'
            && let Some(&(j, '\n')) = iter.peek()
        {
            iter.next();
            end = j + 1;
        }
        start = end;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `s[:n]` in code points.
pub(crate) fn prefix_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `str.lower()`.
pub(crate) fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// Python `repr(float)` (shortest round-trip, `1e+16` style exponents).
pub(crate) fn float_repr(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    // Rust's Debug uses the same shortest digits and the same switch to
    // exponent notation (exp < -4 or exp >= 16) as Python's repr.
    let s = format!("{v:?}");
    match s.find('e') {
        Some(pos) => {
            let (mantissa, exp) = s.split_at(pos);
            let exp = &exp[1..];
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            let digits = if digits.len() < 2 {
                format!("0{digits}")
            } else {
                digits.to_string()
            };
            format!("{mantissa}e{sign}{digits}")
        }
        None => s,
    }
}

/// Python `format(v, ",")` for ints.
pub(crate) fn group_thousands_int(v: i128) -> String {
    let digits = v.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if v < 0 { format!("-{out}") } else { out }
}

/// Python `format(v, ",")` for floats (`repr` digits with grouped integer part).
pub(crate) fn group_thousands_float(v: f64) -> String {
    let repr = float_repr(v);
    if repr.contains('e') || !v.is_finite() {
        return repr;
    }
    let (neg, body) = match repr.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, repr.as_str()),
    };
    let (int_part, frac) = match body.find('.') {
        Some(p) => body.split_at(p),
        None => (body, ""),
    };
    let grouped = group_thousands_int(int_part.parse::<i128>().unwrap_or(0));
    format!("{}{grouped}{frac}", if neg { "-" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines("a\r\nb\rc\nd"), vec!["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n\nb\n"), vec!["a", "", "b"]);
        assert_eq!(splitlines("a\u{2028}b"), vec!["a", "b"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn strip_and_split_match_python() {
        assert_eq!(strip("\x1c a \u{3000}"), "a");
        assert_eq!(split_ws(" a\x1fb  c "), vec!["a", "b", "c"]);
        assert!(is_decimal('５'));
        assert!(!is_decimal('a'));
    }

    #[test]
    fn number_formatting_matches_python() {
        assert_eq!(float_repr(12.0), "12.0");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1.5e-7), "1.5e-07");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(group_thousands_int(1_234_567), "1,234,567");
        assert_eq!(group_thousands_int(-1000), "-1,000");
        assert_eq!(group_thousands_int(999), "999");
        assert_eq!(group_thousands_float(1234.5), "1,234.5");
        assert_eq!(prefix_chars("你好abc", 3), "你好a");
    }
}
