//! URL helpers shared by routing and the remote backends: a tiny subset of
//! Python `urllib.parse.urlparse`, `is_url`, and `detect_platform`.

/// `urlparse` components used by aimd (`scheme` is lowercased like Python).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedUrl<'a> {
    pub scheme: String,
    pub netloc: &'a str,
    pub path: &'a str,
}

/// Parse `scheme://netloc/path?query#fragment` the way `urlparse` splits
/// the first three components. Returns `None` when there is no
/// `scheme://` prefix (Python yields an empty scheme/netloc then).
pub fn parse_url(value: &str) -> Option<ParsedUrl<'_>> {
    let colon = value.find(':')?;
    let scheme = &value[..colon];
    let valid_scheme = scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !valid_scheme {
        return None;
    }
    let rest = value[colon + 1..].strip_prefix("//")?;
    let netloc_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let netloc = &rest[..netloc_end];
    let after = &rest[netloc_end..];
    let path_end = after.find(['?', '#']).unwrap_or(after.len());
    Some(ParsedUrl {
        scheme: scheme.to_ascii_lowercase(),
        netloc,
        path: &after[..path_end],
    })
}

/// Whether a string is an HTTP(S) URL (`aimd.plugins.url.is_url`).
pub fn is_url(value: &str) -> bool {
    parse_url(value)
        .is_some_and(|u| (u.scheme == "http" || u.scheme == "https") && !u.netloc.is_empty())
}

/// Transcript platform for a URL (`aimd.plugins.url.audio.detect_platform`).
pub fn detect_platform(url: &str) -> &'static str {
    let lower = url.to_lowercase();
    if lower.contains("youtube.com") || lower.contains("youtu.be") {
        "youtube"
    } else if lower.contains("bilibili.com") {
        "bilibili"
    } else if lower.contains("xiaohongshu.com") || lower.contains("xhslink.com") {
        "xiaohongshu"
    } else if lower.contains("xiaoyuzhoufm.com") {
        "xiaoyuzhoufm"
    } else {
        "unknown"
    }
}

/// Absolute `file://` URI for a path (Python `Path.as_uri()` quoting:
/// everything except `A-Za-z0-9_.-~/` is percent-encoded).
pub fn path_to_file_uri(path: &std::path::Path) -> String {
    use std::os::unix::ffi::OsStrExt;
    let mut out = String::from("file://");
    for &b in path.as_os_str().as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_url_requires_http_scheme_and_netloc() {
        assert!(is_url("https://example.com/video"));
        assert!(is_url("HTTP://example.com"));
        assert!(!is_url("ftp://example.com"));
        assert!(!is_url("https:///path"));
        assert!(!is_url("/tmp/audio.mp3"));
        assert!(!is_url("C:/file.wav"));
    }

    #[test]
    fn parse_url_splits_path() {
        let u = parse_url("http://host:8000/v1?x=1").unwrap();
        assert_eq!(u.netloc, "host:8000");
        assert_eq!(u.path, "/v1");
        assert_eq!(parse_url("http://host").unwrap().path, "");
    }

    #[test]
    fn platforms() {
        assert_eq!(detect_platform("https://youtu.be/x"), "youtube");
        assert_eq!(
            detect_platform("https://www.bilibili.com/video/BV1"),
            "bilibili"
        );
        assert_eq!(detect_platform("https://example.com"), "unknown");
    }

    #[test]
    fn file_uri_quotes_like_pathlib() {
        assert_eq!(
            path_to_file_uri(std::path::Path::new("/tmp/a b/文")),
            "file:///tmp/a%20b/%E6%96%87"
        );
    }
}
