//! Minimal XML tree parser for subtitle payloads (TTML, YouTube srv1/srv3).
//!
//! Mirrors what the Python code needs from `xml.etree.ElementTree.fromstring`:
//! a single root element, attributes, document-order element iteration and
//! `itertext()`. Malformed input (mismatched tags, unknown entities, junk
//! after the root) is an error, as it is for expat.

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    Element(Element),
    Text(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Element {
    /// Raw qualified name (`tt:p`, `p`).
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

impl Element {
    /// Local name without a namespace prefix (`tt:p` -> `p`).
    pub fn local_name(&self) -> &str {
        self.name.rsplit(':').next().unwrap_or(&self.name)
    }

    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// `ElementTree.iter()`: this element and all descendants, document order.
    pub fn iter(&self) -> Vec<&Element> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a Element>) {
        out.push(self);
        for child in &self.children {
            if let Node::Element(e) = child {
                e.collect(out);
            }
        }
    }

    /// `"".join(element.itertext())`.
    pub fn text(&self) -> String {
        let mut out = String::new();
        self.push_text(&mut out);
        out
    }

    fn push_text(&self, out: &mut String) {
        for child in &self.children {
            match child {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => e.push_text(out),
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct XmlError(#[allow(dead_code)] pub String);

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

/// Parse a document into its root element.
pub(crate) fn parse(input: &str) -> Result<Element, XmlError> {
    // XML end-of-line normalization (expat does this for str input too).
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut p = Parser {
        src: &normalized,
        pos: 0,
    };
    p.skip_misc()?;
    if !p.rest().starts_with('<') {
        return Err(XmlError("syntax error".into()));
    }
    let root = p.element()?;
    p.skip_misc()?;
    if p.pos < p.src.len() {
        return Err(XmlError("junk after document element".into()));
    }
    Ok(root)
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':') || (c as u32) >= 0x80
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn err<T>(&self, msg: &str) -> Result<T, XmlError> {
        Err(XmlError(format!("{msg} at byte {}", self.pos)))
    }

    fn skip_ws(&mut self) {
        let rest = self.rest();
        let trimmed = rest.trim_start_matches([' ', '\t', '\n']);
        self.pos += rest.len() - trimmed.len();
    }

    /// Skip whitespace, comments, processing instructions and DOCTYPE outside
    /// the root element.
    fn skip_misc(&mut self) -> Result<(), XmlError> {
        loop {
            self.skip_ws();
            let rest = self.rest();
            if rest.starts_with("<?") {
                self.skip_until("?>")?;
            } else if rest.starts_with("<!--") {
                self.skip_until("-->")?;
            } else if rest.starts_with("<!DOCTYPE") {
                self.skip_doctype()?;
            } else {
                return Ok(());
            }
        }
    }

    fn skip_until(&mut self, end: &str) -> Result<(), XmlError> {
        match self.rest().find(end) {
            Some(i) => {
                self.pos += i + end.len();
                Ok(())
            }
            None => self.err("unclosed token"),
        }
    }

    fn skip_doctype(&mut self) -> Result<(), XmlError> {
        let mut depth = 0usize;
        for (i, c) in self.rest().char_indices() {
            match c {
                '[' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                '>' if depth == 0 => {
                    self.pos += i + 1;
                    return Ok(());
                }
                _ => {}
            }
        }
        self.err("unclosed doctype")
    }

    fn name(&mut self) -> Result<String, XmlError> {
        let rest = self.rest();
        let end = rest
            .char_indices()
            .find(|&(_, c)| !is_name_char(c))
            .map_or(rest.len(), |(i, _)| i);
        if end == 0 {
            return self.err("expected name");
        }
        self.pos += end;
        Ok(rest[..end].to_string())
    }

    fn element(&mut self) -> Result<Element, XmlError> {
        self.pos += 1; // '<'
        let name = self.name()?;
        let mut attrs = Vec::new();
        loop {
            let before = self.pos;
            self.skip_ws();
            let rest = self.rest();
            if rest.starts_with("/>") {
                self.pos += 2;
                return Ok(Element {
                    name,
                    attrs,
                    children: Vec::new(),
                });
            }
            if rest.starts_with('>') {
                self.pos += 1;
                break;
            }
            if self.pos == before {
                return self.err("expected whitespace before attribute");
            }
            let key = self.name()?;
            self.skip_ws();
            if !self.rest().starts_with('=') {
                return self.err("expected '='");
            }
            self.pos += 1;
            self.skip_ws();
            let quote = match self.rest().chars().next() {
                Some(q @ ('"' | '\'')) => q,
                _ => return self.err("expected quoted attribute value"),
            };
            self.pos += 1;
            let Some(end) = self.rest().find(quote) else {
                return self.err("unclosed attribute value");
            };
            let raw = &self.rest()[..end];
            if raw.contains('<') {
                return self.err("'<' in attribute value");
            }
            let value = decode_entities(raw)?
                .chars()
                .map(|c| if matches!(c, '\t' | '\n') { ' ' } else { c })
                .collect::<String>();
            self.pos += end + 1;
            if attrs.iter().any(|(k, _): &(String, String)| *k == key) {
                return self.err("duplicate attribute");
            }
            attrs.push((key, value));
        }

        let mut children = Vec::new();
        let mut text = String::new();
        loop {
            let rest = self.rest();
            if rest.is_empty() {
                return self.err("unclosed element");
            }
            if let Some(after) = rest.strip_prefix("</") {
                let close_end = after.find('>').unwrap_or(after.len());
                let close = after[..close_end].trim_end_matches([' ', '\t', '\n']);
                if close != name {
                    return self.err("mismatched tag");
                }
                if close_end == after.len() {
                    return self.err("unclosed end tag");
                }
                self.pos += 2 + close_end + 1;
                if !text.is_empty() {
                    children.push(Node::Text(std::mem::take(&mut text)));
                }
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if rest.starts_with("<!--") {
                self.skip_until("-->")?;
                continue;
            }
            if let Some(after) = rest.strip_prefix("<![CDATA[") {
                let Some(end) = after.find("]]>") else {
                    return self.err("unclosed CDATA");
                };
                text.push_str(&after[..end]);
                self.pos += 9 + end + 3;
                continue;
            }
            if rest.starts_with("<?") {
                self.skip_until("?>")?;
                continue;
            }
            if rest.starts_with('<') {
                if !text.is_empty() {
                    children.push(Node::Text(std::mem::take(&mut text)));
                }
                children.push(Node::Element(self.element()?));
                continue;
            }
            let end = rest.find('<').unwrap_or(rest.len());
            text.push_str(&decode_entities(&rest[..end])?);
            self.pos += end;
        }
    }
}

fn decode_entities(raw: &str) -> Result<String, XmlError> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let Some(semi) = after.find(';') else {
            return Err(XmlError("not well-formed (invalid token)".into()));
        };
        let entity = &after[..semi];
        let decoded = match entity {
            "lt" => '<',
            "gt" => '>',
            "amp" => '&',
            "quot" => '"',
            "apos" => '\'',
            _ => {
                let code = if let Some(hex) = entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(dec) = entity.strip_prefix('#') {
                    dec.parse::<u32>().ok()
                } else {
                    None
                };
                match code.and_then(char::from_u32) {
                    Some(c) if c != '\0' => c,
                    _ => return Err(XmlError(format!("undefined entity &{entity};"))),
                }
            }
        };
        out.push(decoded);
        rest = &after[semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_text_and_entities() {
        let root = parse(
            "<?xml version=\"1.0\"?>\n<tt xmlns=\"x\"><body><p begin=\"1\">a &amp; <span>b</span> c&#39;</p></body></tt>",
        )
        .unwrap();
        let ps: Vec<_> = root
            .iter()
            .into_iter()
            .filter(|e| e.local_name() == "p")
            .collect();
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text(), "a & b c'");
        assert_eq!(ps[0].attr("begin"), Some("1"));
    }

    #[test]
    fn rejects_malformed_documents() {
        assert!(parse("<a><b></a>").is_err());
        assert!(parse("<a>&nbsp;</a>").is_err());
        assert!(parse("<a></a>junk").is_err());
        assert!(parse("ttml caption").is_err());
    }
}
