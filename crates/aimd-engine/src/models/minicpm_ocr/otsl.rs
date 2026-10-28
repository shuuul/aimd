//! OTSL -> HTML table conversion, a port of the MiniCPM-OCR SDK's
//! `postprocess/otsl/{converter,types}.py` (itself ported from PaddleX
//! `paddleocr_vl/uilts.py`).
//!
//! The port keeps every quirk of the Python code (row padding by minimum
//! total cost, text re-alignment against the token rows, a cell whose text
//! slot holds the next tag, grid overwrites by later cells), because the
//! output is compared byte for byte with the SDK
//! (`tests/golden/minicpm_regions/text_cases.json`).

use super::markdown::py_strip;

const NL: &str = "<nl>";
const FCEL: &str = "<fcel>";
const ECEL: &str = "<ecel>";
const LCEL: &str = "<lcel>";
const UCEL: &str = "<ucel>";
const XCEL: &str = "<xcel>";
/// Every OTSL tag. None is a prefix of another, so the alternation order of
/// the SDK's regexes does not matter.
const TAGS: [&str; 6] = [NL, FCEL, ECEL, LCEL, UCEL, XCEL];

/// The tag starting at byte `i` of `s`, if any.
fn tag_at(s: &str, i: usize) -> Option<&'static str> {
    let rest = &s.as_bytes()[i..];
    TAGS.into_iter().find(|t| rest.starts_with(t.as_bytes()))
}

/// Byte offset of the first tag at or after `from`.
fn next_tag(s: &str, from: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    (from..bytes.len()).find(|&i| bytes[i] == b'<' && tag_at(s, i).is_some())
}

fn is_tag(s: &str) -> bool {
    TAGS.contains(&s)
}

/// `_OTSL_FIND_PATTERN.findall(line)`: `(?:tags).*?(?=(?:tags)|$)` with
/// `re.DOTALL`. Each cell runs from a tag to the next tag start or to the
/// end of the line (`$` also matches before a final `\n`).
fn find_cells(line: &str) -> Vec<&str> {
    let len = line.len();
    let ends_nl = line.ends_with('\n');
    let mut cells = Vec::new();
    let mut pos = 0;
    while let Some(start) = next_tag(line, pos) {
        let body = start + tag_at(line, start).map_or(0, str::len);
        let mut end = next_tag(line, body).unwrap_or(len);
        if ends_nl && len > body {
            end = end.min(len - 1);
        }
        cells.push(&line[start..end]);
        pos = end;
    }
    cells
}

/// `otsl_pad_to_sqr_v2`: pad (or cut) every row to the width with the
/// minimum total length difference.
pub fn otsl_pad_to_sqr_v2(otsl: &str) -> String {
    let otsl = py_strip(otsl);
    if !otsl.contains(NL) {
        return format!("{otsl}{NL}");
    }
    struct Row<'a> {
        cells: Vec<&'a str>,
        min_len: usize,
    }
    let mut rows: Vec<Row> = Vec::new();
    for line in otsl.split(NL) {
        if line.is_empty() {
            continue;
        }
        let cells = find_cells(line);
        if cells.is_empty() {
            continue;
        }
        let min_len = cells
            .iter()
            .rposition(|c| c.starts_with(FCEL))
            .map_or(0, |i| i + 1);
        rows.push(Row { cells, min_len });
    }
    if rows.is_empty() {
        return NL.to_string();
    }
    let global_min = rows.iter().map(|r| r.min_len).max().unwrap_or(0);
    let max_total = rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
    let mut best_cost = usize::MAX;
    let mut width = global_min.max(max_total);
    for w in global_min..=global_min.max(max_total) {
        let cost: usize = rows.iter().map(|r| r.cells.len().abs_diff(w)).sum();
        if cost < best_cost {
            best_cost = cost;
            width = w;
        }
    }
    let lines: Vec<String> = rows
        .iter()
        .map(|r| {
            let mut out: String = r.cells.iter().take(width).copied().collect();
            for _ in r.cells.len()..width {
                out.push_str(ECEL);
            }
            out
        })
        .collect();
    format!("{}{NL}", lines.join(NL))
}

/// `otsl_extract_tokens_and_text`: the flat tag list, and the
/// `re.split`-with-capture list with whitespace-only parts dropped.
fn extract_tokens_and_text(s: &str) -> (Vec<&str>, Vec<&str>) {
    let mut tokens = Vec::new();
    let mut parts = Vec::new();
    let mut last = 0;
    let mut i = 0;
    while let Some(start) = next_tag(s, i) {
        let tag = tag_at(s, start).unwrap_or_default();
        parts.push(&s[last..start]);
        parts.push(&s[start..start + tag.len()]);
        tokens.push(&s[start..start + tag.len()]);
        last = start + tag.len();
        i = last;
    }
    parts.push(&s[last..]);
    parts.retain(|p| !py_strip(p).is_empty());
    (tokens, parts)
}

/// One table cell (`TableCell`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Cell {
    text: String,
    row_span: usize,
    col_span: usize,
    start_row: usize,
    end_row: usize,
    start_col: usize,
    end_col: usize,
}

fn count_run(
    rows: &[Vec<&str>],
    mut c: usize,
    mut r: usize,
    right: bool,
    which: [&str; 2],
) -> usize {
    let mut span = 0;
    while let Some(t) = rows.get(r).and_then(|row| row.get(c)) {
        if !which.contains(t) {
            break;
        }
        span += 1;
        if right {
            c += 1;
        } else {
            r += 1;
        }
    }
    span
}

/// `otsl_parse_texts`.
fn parse_texts<'a>(texts: Vec<&'a str>, tokens: &[&'a str]) -> (Vec<Cell>, Vec<Vec<&'a str>>) {
    let mut rows: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for &t in tokens {
        if t == NL {
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
            }
        } else {
            current.push(t);
        }
    }
    if !current.is_empty() {
        rows.push(current);
    }

    let mut texts = texts;
    if !rows.is_empty() {
        let max_cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        for row in &mut rows {
            row.resize(max_cols, ECEL);
        }
        let mut new_texts = Vec::new();
        let mut idx = 0;
        for row in &rows {
            for &token in row {
                new_texts.push(token);
                if idx < texts.len() && texts[idx] == token {
                    idx += 1;
                    if idx < texts.len() && !is_tag(texts[idx]) {
                        new_texts.push(texts[idx]);
                        idx += 1;
                    }
                }
            }
            new_texts.push(NL);
            if idx < texts.len() && texts[idx] == NL {
                idx += 1;
            }
        }
        texts = new_texts;
    }

    let mut cells = Vec::new();
    let (mut r, mut c) = (0usize, 0usize);
    for (i, &text) in texts.iter().enumerate() {
        if text == FCEL || text == ECEL {
            let mut row_span = 1;
            let mut col_span = 1;
            let mut right_offset = 1;
            let mut cell_text = "";
            if text != ECEL {
                cell_text = texts.get(i + 1).copied().unwrap_or("");
                right_offset = 2;
            }
            let next_right = texts.get(i + right_offset).copied().unwrap_or("");
            let next_bottom = rows
                .get(r + 1)
                .and_then(|row| row.get(c))
                .copied()
                .unwrap_or("");
            if next_right == LCEL || next_right == XCEL {
                col_span += count_run(&rows, c + 1, r, true, [LCEL, XCEL]);
            }
            if next_bottom == UCEL || next_bottom == XCEL {
                row_span += count_run(&rows, c, r + 1, false, [UCEL, XCEL]);
            }
            cells.push(Cell {
                text: py_strip(cell_text).to_string(),
                row_span,
                col_span,
                start_row: r,
                end_row: r + row_span,
                start_col: c,
                end_col: c + col_span,
            });
        }
        if [FCEL, ECEL, LCEL, UCEL, XCEL].contains(&text) {
            c += 1;
        }
        if text == NL {
            r += 1;
            c = 0;
        }
    }
    (cells, rows)
}

/// Python's `html.escape(s, quote=True)`.
pub(crate) fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// `export_to_html(TableData)`.
fn export_to_html(cells: &[Cell], nrows: usize, ncols: usize) -> String {
    if cells.is_empty() {
        return String::new();
    }
    // `None` = the default 1x1 empty cell at that position.
    let mut grid: Vec<Vec<Option<usize>>> = vec![vec![None; ncols]; nrows];
    for (k, cell) in cells.iter().enumerate() {
        for row in grid
            .iter_mut()
            .take(cell.end_row.min(nrows))
            .skip(cell.start_row.min(nrows))
        {
            for slot in row
                .iter_mut()
                .take(cell.end_col.min(ncols))
                .skip(cell.start_col.min(ncols))
            {
                *slot = Some(k);
            }
        }
    }
    let mut body = String::new();
    for (i, row) in grid.iter().enumerate() {
        body.push_str("<tr>");
        for (j, slot) in row.iter().enumerate() {
            let Some(k) = *slot else {
                body.push_str("<td></td>");
                continue;
            };
            let cell = &cells[k];
            if cell.start_row != i || cell.start_col != j {
                continue;
            }
            let mut opening = String::from("td");
            if cell.row_span > 1 {
                opening.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                opening.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            body.push_str(&format!(
                "<{opening}>{}</td>",
                html_escape(py_strip(&cell.text))
            ));
        }
        body.push_str("</tr>");
    }
    format!("<table>{body}</table>")
}

/// `convert_otsl_to_html`: an HTML `<table>`, or `""` when the text holds no
/// `<fcel>` / `<ecel>` cell.
pub fn convert_otsl_to_html(otsl: &str) -> String {
    let padded = otsl_pad_to_sqr_v2(&html_unescape(otsl));
    let (tokens, texts) = extract_tokens_and_text(&padded);
    let (cells, rows) = parse_texts(texts, &tokens);
    let ncols = rows.iter().map(Vec::len).max().unwrap_or(0);
    export_to_html(&cells, rows.len(), ncols)
}

// --- html.unescape -------------------------------------------------------------

/// Named references `html.unescape` resolves here: the 252 HTML 4 names
/// (HTML5 values, so `lang` / `rang` are U+27E8 / U+27E9) plus `apos` and
/// the upper-case legacy aliases. Python knows all 2231 HTML5 names; the
/// rest are left as written (recorded in spec 009).
const NAMED: &[(&str, u32)] = &[
    ("AElig", 0xc6),
    ("Aacute", 0xc1),
    ("Acirc", 0xc2),
    ("Agrave", 0xc0),
    ("Alpha", 0x391),
    ("Aring", 0xc5),
    ("Atilde", 0xc3),
    ("Auml", 0xc4),
    ("Beta", 0x392),
    ("Ccedil", 0xc7),
    ("Chi", 0x3a7),
    ("Dagger", 0x2021),
    ("Delta", 0x394),
    ("ETH", 0xd0),
    ("Eacute", 0xc9),
    ("Ecirc", 0xca),
    ("Egrave", 0xc8),
    ("Epsilon", 0x395),
    ("Eta", 0x397),
    ("Euml", 0xcb),
    ("Gamma", 0x393),
    ("Iacute", 0xcd),
    ("Icirc", 0xce),
    ("Igrave", 0xcc),
    ("Iota", 0x399),
    ("Iuml", 0xcf),
    ("Kappa", 0x39a),
    ("Lambda", 0x39b),
    ("Mu", 0x39c),
    ("Ntilde", 0xd1),
    ("Nu", 0x39d),
    ("OElig", 0x152),
    ("Oacute", 0xd3),
    ("Ocirc", 0xd4),
    ("Ograve", 0xd2),
    ("Omega", 0x3a9),
    ("Omicron", 0x39f),
    ("Oslash", 0xd8),
    ("Otilde", 0xd5),
    ("Ouml", 0xd6),
    ("Phi", 0x3a6),
    ("Pi", 0x3a0),
    ("Prime", 0x2033),
    ("Psi", 0x3a8),
    ("Rho", 0x3a1),
    ("Scaron", 0x160),
    ("Sigma", 0x3a3),
    ("THORN", 0xde),
    ("Tau", 0x3a4),
    ("Theta", 0x398),
    ("Uacute", 0xda),
    ("Ucirc", 0xdb),
    ("Ugrave", 0xd9),
    ("Upsilon", 0x3a5),
    ("Uuml", 0xdc),
    ("Xi", 0x39e),
    ("Yacute", 0xdd),
    ("Yuml", 0x178),
    ("Zeta", 0x396),
    ("aacute", 0xe1),
    ("acirc", 0xe2),
    ("acute", 0xb4),
    ("aelig", 0xe6),
    ("agrave", 0xe0),
    ("alefsym", 0x2135),
    ("alpha", 0x3b1),
    ("amp", 0x26),
    ("and", 0x2227),
    ("ang", 0x2220),
    ("aring", 0xe5),
    ("asymp", 0x2248),
    ("atilde", 0xe3),
    ("auml", 0xe4),
    ("bdquo", 0x201e),
    ("beta", 0x3b2),
    ("brvbar", 0xa6),
    ("bull", 0x2022),
    ("cap", 0x2229),
    ("ccedil", 0xe7),
    ("cedil", 0xb8),
    ("cent", 0xa2),
    ("chi", 0x3c7),
    ("circ", 0x2c6),
    ("clubs", 0x2663),
    ("cong", 0x2245),
    ("copy", 0xa9),
    ("crarr", 0x21b5),
    ("cup", 0x222a),
    ("curren", 0xa4),
    ("dArr", 0x21d3),
    ("dagger", 0x2020),
    ("darr", 0x2193),
    ("deg", 0xb0),
    ("delta", 0x3b4),
    ("diams", 0x2666),
    ("divide", 0xf7),
    ("eacute", 0xe9),
    ("ecirc", 0xea),
    ("egrave", 0xe8),
    ("empty", 0x2205),
    ("emsp", 0x2003),
    ("ensp", 0x2002),
    ("epsilon", 0x3b5),
    ("equiv", 0x2261),
    ("eta", 0x3b7),
    ("eth", 0xf0),
    ("euml", 0xeb),
    ("euro", 0x20ac),
    ("exist", 0x2203),
    ("fnof", 0x192),
    ("forall", 0x2200),
    ("frac12", 0xbd),
    ("frac14", 0xbc),
    ("frac34", 0xbe),
    ("frasl", 0x2044),
    ("gamma", 0x3b3),
    ("ge", 0x2265),
    ("gt", 0x3e),
    ("hArr", 0x21d4),
    ("harr", 0x2194),
    ("hearts", 0x2665),
    ("hellip", 0x2026),
    ("iacute", 0xed),
    ("icirc", 0xee),
    ("iexcl", 0xa1),
    ("igrave", 0xec),
    ("image", 0x2111),
    ("infin", 0x221e),
    ("int", 0x222b),
    ("iota", 0x3b9),
    ("iquest", 0xbf),
    ("isin", 0x2208),
    ("iuml", 0xef),
    ("kappa", 0x3ba),
    ("lArr", 0x21d0),
    ("lambda", 0x3bb),
    ("lang", 0x27e8),
    ("laquo", 0xab),
    ("larr", 0x2190),
    ("lceil", 0x2308),
    ("ldquo", 0x201c),
    ("le", 0x2264),
    ("lfloor", 0x230a),
    ("lowast", 0x2217),
    ("loz", 0x25ca),
    ("lrm", 0x200e),
    ("lsaquo", 0x2039),
    ("lsquo", 0x2018),
    ("lt", 0x3c),
    ("macr", 0xaf),
    ("mdash", 0x2014),
    ("micro", 0xb5),
    ("middot", 0xb7),
    ("minus", 0x2212),
    ("mu", 0x3bc),
    ("nabla", 0x2207),
    ("nbsp", 0xa0),
    ("ndash", 0x2013),
    ("ne", 0x2260),
    ("ni", 0x220b),
    ("not", 0xac),
    ("notin", 0x2209),
    ("nsub", 0x2284),
    ("ntilde", 0xf1),
    ("nu", 0x3bd),
    ("oacute", 0xf3),
    ("ocirc", 0xf4),
    ("oelig", 0x153),
    ("ograve", 0xf2),
    ("oline", 0x203e),
    ("omega", 0x3c9),
    ("omicron", 0x3bf),
    ("oplus", 0x2295),
    ("or", 0x2228),
    ("ordf", 0xaa),
    ("ordm", 0xba),
    ("oslash", 0xf8),
    ("otilde", 0xf5),
    ("otimes", 0x2297),
    ("ouml", 0xf6),
    ("para", 0xb6),
    ("part", 0x2202),
    ("permil", 0x2030),
    ("perp", 0x22a5),
    ("phi", 0x3c6),
    ("pi", 0x3c0),
    ("piv", 0x3d6),
    ("plusmn", 0xb1),
    ("pound", 0xa3),
    ("prime", 0x2032),
    ("prod", 0x220f),
    ("prop", 0x221d),
    ("psi", 0x3c8),
    ("quot", 0x22),
    ("rArr", 0x21d2),
    ("radic", 0x221a),
    ("rang", 0x27e9),
    ("raquo", 0xbb),
    ("rarr", 0x2192),
    ("rceil", 0x2309),
    ("rdquo", 0x201d),
    ("real", 0x211c),
    ("reg", 0xae),
    ("rfloor", 0x230b),
    ("rho", 0x3c1),
    ("rlm", 0x200f),
    ("rsaquo", 0x203a),
    ("rsquo", 0x2019),
    ("sbquo", 0x201a),
    ("scaron", 0x161),
    ("sdot", 0x22c5),
    ("sect", 0xa7),
    ("shy", 0xad),
    ("sigma", 0x3c3),
    ("sigmaf", 0x3c2),
    ("sim", 0x223c),
    ("spades", 0x2660),
    ("sub", 0x2282),
    ("sube", 0x2286),
    ("sum", 0x2211),
    ("sup", 0x2283),
    ("sup1", 0xb9),
    ("sup2", 0xb2),
    ("sup3", 0xb3),
    ("supe", 0x2287),
    ("szlig", 0xdf),
    ("tau", 0x3c4),
    ("there4", 0x2234),
    ("theta", 0x3b8),
    ("thetasym", 0x3d1),
    ("thinsp", 0x2009),
    ("thorn", 0xfe),
    ("tilde", 0x2dc),
    ("times", 0xd7),
    ("trade", 0x2122),
    ("uArr", 0x21d1),
    ("uacute", 0xfa),
    ("uarr", 0x2191),
    ("ucirc", 0xfb),
    ("ugrave", 0xf9),
    ("uml", 0xa8),
    ("upsih", 0x3d2),
    ("upsilon", 0x3c5),
    ("uuml", 0xfc),
    ("weierp", 0x2118),
    ("xi", 0x3be),
    ("yacute", 0xfd),
    ("yen", 0xa5),
    ("yuml", 0xff),
    ("zeta", 0x3b6),
    ("zwj", 0x200d),
    ("zwnj", 0x200c),
    ("apos", 0x27),
    ("AMP", 0x26),
    ("GT", 0x3e),
    ("LT", 0x3c),
    ("QUOT", 0x22),
    ("COPY", 0xa9),
    ("REG", 0xae),
];

/// Names HTML5 also resolves without the trailing `;` (Latin-1 + the
/// basic five upper/lower-case aliases).
fn legacy_name(name: &str) -> bool {
    matches!(
        name,
        "AMP" | "amp" | "GT" | "gt" | "LT" | "lt" | "QUOT" | "quot" | "COPY" | "REG"
    ) || NAMED
        .iter()
        .any(|&(n, cp)| n == name && (0xa0..=0xff).contains(&cp) && n != "apos")
}

fn named_value(name: &str) -> Option<char> {
    NAMED
        .iter()
        .find(|&&(n, _)| n == name)
        .and_then(|&(_, cp)| char::from_u32(cp))
}

/// `html._invalid_charrefs` (Windows-1252 remapping of C1 controls etc.).
fn invalid_charref(num: u64) -> Option<char> {
    const C1: [u32; 32] = [
        0x20ac, 0x81, 0x201a, 0x192, 0x201e, 0x2026, 0x2020, 0x2021, 0x2c6, 0x2030, 0x160, 0x2039,
        0x152, 0x8d, 0x17d, 0x8f, 0x90, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
        0x2dc, 0x2122, 0x161, 0x203a, 0x153, 0x9d, 0x17e, 0x178,
    ];
    match num {
        0x00 => Some('\u{fffd}'),
        0x0d => Some('\r'),
        0x80..=0x9f => char::from_u32(C1[(num - 0x80) as usize]),
        _ => None,
    }
}

/// `html._invalid_codepoints` (dropped from the output).
fn invalid_codepoint(num: u64) -> bool {
    matches!(num, 0x1..=0x8 | 0xb | 0xe..=0x1f | 0x7f..=0x9f | 0xfdd0..=0xfdef)
        || (num & 0xfffe == 0xfffe && num <= 0x10ffff)
}

/// Python's `html.unescape` (numeric references exactly; named references
/// from [`NAMED`]).
pub(crate) fn html_unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut last = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            i += 1;
            continue;
        }
        let Some((end, rep)) = charref_at(s, i) else {
            i += 1;
            continue;
        };
        out.push_str(&s[last..i]);
        out.push_str(&rep);
        last = end;
        i = end;
    }
    out.push_str(&s[last..]);
    out
}

/// Matches `&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)` at byte
/// `amp` and returns the match end plus its replacement.
fn charref_at(s: &str, amp: usize) -> Option<(usize, String)> {
    let bytes = s.as_bytes();
    let start = amp + 1;
    if bytes.get(start) == Some(&b'#') {
        let (hex, digits_at) = match bytes.get(start + 1) {
            Some(b'x' | b'X') => (true, start + 2),
            _ => (false, start + 1),
        };
        let mut j = digits_at;
        while j < bytes.len()
            && (if hex {
                bytes[j].is_ascii_hexdigit()
            } else {
                bytes[j].is_ascii_digit()
            })
        {
            j += 1;
        }
        if j == digits_at {
            return None;
        }
        let digits = &s[digits_at..j];
        let end = if bytes.get(j) == Some(&b';') {
            j + 1
        } else {
            j
        };
        let num = u64::from_str_radix(digits, if hex { 16 } else { 10 }).unwrap_or(u64::MAX);
        let rep = if let Some(c) = invalid_charref(num) {
            c.to_string()
        } else if (0xd800..=0xdfff).contains(&num) || num > 0x10ffff {
            "\u{fffd}".to_string()
        } else if invalid_codepoint(num) {
            String::new()
        } else {
            char::from_u32(num as u32)
                .map(String::from)
                .unwrap_or_default()
        };
        return Some((end, rep));
    }
    // Named: up to 32 characters outside `\t\n\f <&#;`, then an optional `;`.
    let mut end = start;
    for (n, ch) in s[start..].chars().enumerate() {
        if n == 32 || matches!(ch, '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';') {
            break;
        }
        end += ch.len_utf8();
    }
    if end == start {
        return None;
    }
    let name = &s[start..end];
    let semi = bytes.get(end) == Some(&b';');
    let full_end = if semi { end + 1 } else { end };
    if let Some(c) = named_value(name).filter(|_| semi || legacy_name(name)) {
        return Some((full_end, c.to_string()));
    }
    // Longest legacy prefix of at least two characters (prefixes never end
    // in `;`, so only the no-semicolon names can match).
    let full = &s[start..full_end];
    let cuts: Vec<usize> = full.char_indices().map(|(k, _)| k).skip(2).collect();
    for &cut in cuts.iter().rev() {
        let prefix = &full[..cut];
        if legacy_name(prefix)
            && let Some(c) = named_value(prefix)
        {
            return Some((full_end, format!("{c}{}", &full[cut..])));
        }
    }
    Some((full_end, format!("&{full}")))
}
