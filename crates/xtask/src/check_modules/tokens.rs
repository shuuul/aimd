//! The tokens needed for use trees, with comments and literal contents skipped.

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Token<'a> {
    pub text: &'a str,
    pub line: usize,
}

impl Token<'_> {
    pub fn name(&self) -> &str {
        self.text.strip_prefix("r#").unwrap_or(self.text)
    }

    pub fn is_name(&self) -> bool {
        self.name()
            .as_bytes()
            .first()
            .is_some_and(|&b| ident_start(b))
    }
}

pub(super) fn tokenize(mut source: &str) -> Vec<Token<'_>> {
    let mut line = 1;
    let mut tokens = Vec::new();
    while let Some(&first) = source.as_bytes().first() {
        let (len, token) = if first.is_ascii_whitespace() {
            (1, None)
        } else if source.starts_with("//") {
            (source.find('\n').unwrap_or(source.len()), None)
        } else if source.starts_with("/*") {
            (block_comment_len(source), None)
        } else if let Some(len) = literal_len(source) {
            // Preserve a token boundary even though the literal is opaque.
            (len, Some(""))
        } else {
            let len = if ident_start(first) {
                let raw = usize::from(source.starts_with("r#")) * 2;
                raw + source.as_bytes()[raw..]
                    .iter()
                    .take_while(|&&b| ident_start(b) || b.is_ascii_digit())
                    .count()
            } else if source.starts_with("::") {
                2
            } else {
                1
            };
            (len, Some(&source[..len]))
        };
        if let Some(text) = token {
            tokens.push(Token { text, line });
        }
        let (consumed, rest) = source.split_at(len);
        line += consumed.bytes().filter(|&b| b == b'\n').count();
        source = rest;
    }
    tokens
}

fn ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_' || !byte.is_ascii()
}

fn block_comment_len(source: &str) -> usize {
    let mut depth = 1;
    let mut index = 2;
    let bytes = source.as_bytes();
    while index + 1 < bytes.len() {
        match &bytes[index..index + 2] {
            b"/*" => depth += 1,
            b"*/" => {
                depth -= 1;
                if depth == 0 {
                    return index + 2;
                }
            }
            _ => {
                index += 1;
                continue;
            }
        }
        index += 2;
    }
    bytes.len()
}

fn literal_len(source: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let prefix = usize::from(matches!(bytes[0], b'b' | b'c'));
    match bytes.get(prefix) {
        Some(b'r') => {
            let hashes = bytes[prefix + 1..]
                .iter()
                .take_while(|&&b| b == b'#')
                .count();
            let open = prefix + 1 + hashes;
            if bytes.get(open) != Some(&b'"') {
                return None;
            }
            let delimiter = &source[prefix + 1..=open];
            let close = format!("\"{}", &delimiter[..hashes]);
            Some(
                source[open + 1..]
                    .find(&close)
                    .map_or(bytes.len(), |end| open + 1 + end + close.len()),
            )
        }
        Some(b'"') => {
            let mut index = prefix + 1;
            while index < bytes.len() {
                match bytes[index] {
                    b'"' => return Some(index + 1),
                    b'\\' => index += 2,
                    _ => index += 1,
                }
            }
            Some(bytes.len())
        }
        Some(b'\'') => {
            let content = &source[prefix + 1..];
            let width = if let Some(escaped) = content.strip_prefix('\\') {
                1 + match escaped.as_bytes().first()? {
                    b'x' => 3,
                    b'u' => escaped.find('}')? + 1,
                    _ => escaped.chars().next()?.len_utf8(),
                }
            } else {
                // An apostrophe starting a lifetime or label has no closing quote.
                content.chars().next()?.len_utf8()
            };
            (content.as_bytes().get(width) == Some(&b'\'')).then_some(prefix + 2 + width)
        }
        _ => None,
    }
}
