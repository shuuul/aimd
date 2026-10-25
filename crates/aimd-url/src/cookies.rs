//! Cookie-source parsing, Netscape cookie files, and auth-related error
//! classification (ports `aimd.plugins.url.cookies`).

use std::path::{Path, PathBuf};

use aimd_core::{AimdError, Result};

/// Platforms whose media needs authenticated cookies; no `no-cookie` attempt.
pub const AUTH_REQUIRED_PLATFORMS: &[&str] = &["bilibili", "xiaohongshu"];

/// Automatic browser-cookie probe order when no cookie option is supplied.
pub const DEFAULT_BROWSER_SPECS: &[&str] = &[
    "chrome:default",
    "chromium",
    "brave",
    "edge",
    "vivaldi",
    "opera",
    "firefox",
    "safari",
    "whale",
];

/// `BROWSER[+KEYRING][:PROFILE][::CONTAINER]`, parsed like
/// `parse_cookies_from_browser`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserSpec {
    pub browser: String,
    pub profile: Option<String>,
    pub keyring: Option<String>,
    pub container: Option<String>,
}

impl BrowserSpec {
    /// Re-serialize for `--cookies-from-browser`.
    pub fn to_yt_dlp_arg(&self) -> String {
        let mut out = self.browser.clone();
        if let Some(k) = &self.keyring {
            out.push('+');
            out.push_str(k);
        }
        if let Some(p) = &self.profile {
            out.push(':');
            out.push_str(p);
        }
        if let Some(c) = &self.container {
            out.push_str("::");
            out.push_str(c);
        }
        out
    }
}

fn partition<'a>(s: &'a str, sep: &str) -> (&'a str, bool, &'a str) {
    match s.find(sep) {
        Some(i) => (&s[..i], true, &s[i + sep.len()..]),
        None => (s, false, ""),
    }
}

fn non_empty(has: bool, value: &str) -> Option<String> {
    let v = crate::pytext::strip(value);
    (has && !v.is_empty()).then(|| v.to_string())
}

/// Parse a browser cookie spec. Errors carry the Python `ValueError` text.
pub fn parse_cookies_from_browser(spec: &str) -> std::result::Result<BrowserSpec, String> {
    let raw = crate::pytext::strip(spec);
    if raw.is_empty() {
        return Err("cookies_from_browser cannot be empty".into());
    }
    let (browser_profile, has_container, container) = partition(raw, "::");
    let (browser_keyring, has_profile, profile) = partition(browser_profile, ":");
    let (browser, has_keyring, keyring) = partition(browser_keyring, "+");
    let browser = crate::pytext::lower(crate::pytext::strip(browser));
    if browser.is_empty() {
        return Err(format!("Invalid cookies_from_browser value: {spec}"));
    }
    Ok(BrowserSpec {
        browser,
        profile: non_empty(has_profile, profile),
        keyring: non_empty(has_keyring, keyring),
        container: non_empty(has_container, container),
    })
}

/// One ordered cookie attempt for a yt-dlp run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CookieSource {
    /// Explicit `--cookies` Netscape file.
    File(PathBuf),
    /// Browser cookies; `explicit` is false for the automatic probe.
    Browser {
        raw: String,
        spec: BrowserSpec,
        explicit: bool,
    },
    NoCookie,
}

impl CookieSource {
    /// Python source `name` (`cookiefile`, `cookiesfrombrowser:<spec>`, `no-cookie`).
    pub fn name(&self) -> String {
        match self {
            CookieSource::File(_) => "cookiefile".into(),
            CookieSource::Browser { raw, .. } => format!("cookiesfrombrowser:{raw}"),
            CookieSource::NoCookie => "no-cookie".into(),
        }
    }

    /// User-supplied (explicit) sources fail fast on invalid-option errors.
    pub fn is_explicit(&self) -> bool {
        match self {
            CookieSource::File(_) => true,
            CookieSource::Browser { explicit, .. } => *explicit,
            CookieSource::NoCookie => false,
        }
    }
}

/// `build_cookie_sources`: explicit file, explicit browser, else the automatic
/// browser probe; `no-cookie` last unless the platform requires auth.
///
/// Explicit options are validated up front (fail fast): an empty/invalid
/// browser spec, a missing cookies file, or a file that is not in Netscape
/// format is an `UnsupportedInput` error.
pub fn build_cookie_sources(
    platform: &str,
    cookies_file: Option<&Path>,
    cookies_from_browser: Option<&str>,
) -> Result<Vec<CookieSource>> {
    let mut sources = Vec::new();
    let cookies_file = cookies_file.filter(|p| !p.as_os_str().is_empty());
    let cookies_from_browser = cookies_from_browser.filter(|s| !s.is_empty());

    if let Some(path) = cookies_file {
        validate_cookie_file(path)?;
        sources.push(CookieSource::File(path.to_path_buf()));
    }
    if let Some(raw) = cookies_from_browser {
        let spec = parse_cookies_from_browser(raw).map_err(AimdError::unsupported)?;
        sources.push(CookieSource::Browser {
            raw: raw.to_string(),
            spec,
            explicit: true,
        });
    }
    if cookies_file.is_none() && cookies_from_browser.is_none() {
        for raw in DEFAULT_BROWSER_SPECS {
            sources.push(CookieSource::Browser {
                raw: (*raw).to_string(),
                spec: parse_cookies_from_browser(raw).expect("built-in spec"),
                explicit: false,
            });
        }
    }
    if !AUTH_REQUIRED_PLATFORMS.contains(&platform) {
        sources.push(CookieSource::NoCookie);
    }
    Ok(sources)
}

fn contains_any(message: &str, needles: &[&str]) -> bool {
    let lower = message.to_lowercase();
    needles.iter().any(|n| lower.contains(n))
}

/// Keyring/cookie decryption problems.
pub fn is_keyring_error(message: &str) -> bool {
    contains_any(
        message,
        &[
            "keyring",
            "secretservice",
            "secret service",
            "secret-service",
            "failed to decrypt",
            "could not decrypt",
            "dbus",
            "org.freedesktop.secret",
            "gnome-keyring",
            "kwallet",
        ],
    )
}

/// The browser-cookie source does not exist on this machine.
pub fn is_cookie_source_unavailable_error(message: &str) -> bool {
    contains_any(
        message,
        &[
            "could not find firefox cookies database",
            "could not find chrome cookies database",
            "could not find chromium cookies database",
            "could not find edge cookies database",
            "could not find opera cookies database",
            "cookies database",
            "cookie database",
            "browser profile not found",
            "profile directory does not exist",
        ],
    )
}

/// yt-dlp has no extractor for the URL.
pub fn is_unsupported_url_error(message: &str) -> bool {
    contains_any(message, &["unsupported url", "no suitable extractor"])
}

/// Login-required/private-content errors.
pub fn is_auth_required_error(message: &str) -> bool {
    contains_any(
        message,
        &[
            "http error 412",
            "precondition failed",
            "login required",
            "sign in",
            "private",
            "members only",
            "premium",
            "watchlater",
            "supporter-only",
            "cookies are required",
            "authentication",
            "403",
            "-403",
            "-101",
        ],
    )
}

/// yt-dlp rejected an explicit cookie option itself (CLI argument validation).
pub fn is_invalid_cookie_option_error(message: &str) -> bool {
    contains_any(
        message,
        &[
            "unsupported browser specified for cookies",
            "unsupported keyring specified for cookies",
            "does not look like a netscape format cookies file",
        ],
    )
}

/// One entry of a Netscape `cookies.txt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetscapeCookie {
    pub domain: String,
    pub include_subdomains: bool,
    pub path: String,
    pub secure: bool,
    /// Unix seconds; `None` for session cookies.
    pub expires: Option<u64>,
    pub name: String,
    pub value: String,
    pub http_only: bool,
}

/// Parse a Netscape cookie file the way yt-dlp's `YoutubeDLCookieJar` does:
/// the first line must be the `# [Netscape ]HTTP Cookie File` magic, comments
/// and blank lines are skipped, `#HttpOnly_` prefixes are honored, and
/// malformed entries are skipped (yt-dlp warns and continues).
pub fn parse_netscape_cookies(text: &str) -> std::result::Result<Vec<NetscapeCookie>, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines();
    let magic = lines.next().unwrap_or("");
    if !(magic.contains("# Netscape HTTP Cookie File") || magic.contains("# HTTP Cookie File")) {
        return Err("does not look like a Netscape format cookies file".into());
    }
    let mut cookies = Vec::new();
    for line in lines {
        let (http_only, line) = match line.strip_prefix("#HttpOnly_") {
            Some(rest) => (true, rest),
            None => (false, line),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() || line.starts_with('#') || line.starts_with('$') {
            continue;
        }
        let fields: Vec<&str> = line.trim_end_matches(['\r', '\n']).split('\t').collect();
        if fields.len() != 7 {
            continue;
        }
        let flag = |s: &str| s.eq_ignore_ascii_case("TRUE");
        let expires = match fields[4].trim() {
            "" | "0" => None,
            digits if digits.bytes().all(|b| b.is_ascii_digit()) => digits.parse().ok(),
            _ => continue,
        };
        cookies.push(NetscapeCookie {
            domain: fields[0].to_string(),
            include_subdomains: flag(fields[1]),
            path: fields[2].to_string(),
            secure: flag(fields[3]),
            expires,
            name: fields[5].to_string(),
            value: fields[6].to_string(),
            http_only,
        });
    }
    Ok(cookies)
}

/// Fail fast on an explicit cookies file yt-dlp would reject or ignore.
pub fn validate_cookie_file(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|err| {
        AimdError::unsupported(format!(
            "Cookies file cannot be read: {} ({err})",
            path.display()
        ))
    })?;
    let text = String::from_utf8_lossy(&bytes);
    parse_netscape_cookies(&text).map(|_| ()).map_err(|reason| {
        AimdError::unsupported(format!(
            "Invalid cookies file {}: {reason}. Export cookies in Netscape (cookies.txt) format.",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(b: &str, p: Option<&str>, k: Option<&str>, c: Option<&str>) -> BrowserSpec {
        BrowserSpec {
            browser: b.into(),
            profile: p.map(Into::into),
            keyring: k.map(Into::into),
            container: c.map(Into::into),
        }
    }

    #[test]
    fn parse_cookies_from_browser_variants() {
        assert_eq!(
            parse_cookies_from_browser("chrome").unwrap(),
            spec("chrome", None, None, None)
        );
        assert_eq!(
            parse_cookies_from_browser("chrome:default").unwrap(),
            spec("chrome", Some("default"), None, None)
        );
        assert_eq!(
            parse_cookies_from_browser("chrome+gnomekeyring:default").unwrap(),
            spec("chrome", Some("default"), Some("gnomekeyring"), None)
        );
        assert_eq!(
            parse_cookies_from_browser("firefox:default::container-1").unwrap(),
            spec("firefox", Some("default"), None, Some("container-1"))
        );
        assert_eq!(
            parse_cookies_from_browser("   ").unwrap_err(),
            "cookies_from_browser cannot be empty"
        );
        assert_eq!(
            parse_cookies_from_browser("+kwallet").unwrap_err(),
            "Invalid cookies_from_browser value: +kwallet"
        );
    }

    #[test]
    fn browser_spec_round_trips_to_cli_arg() {
        for raw in [
            "chrome",
            "chrome:default",
            "chrome+gnomekeyring:default",
            "firefox:default::c1",
            "firefox::c1",
        ] {
            let parsed = parse_cookies_from_browser(raw).unwrap();
            let again = parse_cookies_from_browser(&parsed.to_yt_dlp_arg()).unwrap();
            assert_eq!(parsed, again, "{raw}");
        }
    }

    #[test]
    fn build_cookie_sources_order_and_fallback() {
        let dir = std::env::temp_dir().join(format!("aimd-url-cookies-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("cookies.txt");
        std::fs::write(
            &file,
            "# Netscape HTTP Cookie File\n.bilibili.com\tTRUE\t/\tFALSE\t0\tSESSDATA\tx\n",
        )
        .unwrap();
        let names: Vec<String> =
            build_cookie_sources("bilibili", Some(&file), Some("chrome:default"))
                .unwrap()
                .iter()
                .map(CookieSource::name)
                .collect();
        assert_eq!(names, ["cookiefile", "cookiesfrombrowser:chrome:default"]);

        let names: Vec<String> = build_cookie_sources("youtube", None, None)
            .unwrap()
            .iter()
            .map(CookieSource::name)
            .collect();
        assert_eq!(
            names,
            [
                "cookiesfrombrowser:chrome:default",
                "cookiesfrombrowser:chromium",
                "cookiesfrombrowser:brave",
                "cookiesfrombrowser:edge",
                "cookiesfrombrowser:vivaldi",
                "cookiesfrombrowser:opera",
                "cookiesfrombrowser:firefox",
                "cookiesfrombrowser:safari",
                "cookiesfrombrowser:whale",
                "no-cookie",
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn explicit_invalid_cookie_options_fail_fast() {
        let err = build_cookie_sources("youtube", None, Some("   ")).unwrap_err();
        assert_eq!(err.kind, aimd_core::ErrorKind::UnsupportedInput);
        assert!(err.message.contains("cookies_from_browser cannot be empty"));

        let missing = build_cookie_sources(
            "youtube",
            Some(Path::new("/nonexistent/aimd/cookies.txt")),
            None,
        )
        .unwrap_err();
        assert_eq!(missing.kind, aimd_core::ErrorKind::UnsupportedInput);

        let dir = std::env::temp_dir().join(format!("aimd-url-badcookies-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("cookies.json");
        std::fs::write(&bad, "[{\"name\": \"x\"}]").unwrap();
        let err = build_cookie_sources("youtube", Some(&bad), None).unwrap_err();
        assert!(err.message.contains("Netscape"), "{}", err.message);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn netscape_parser_handles_http_only_and_skips_bad_lines() {
        let text = "# HTTP Cookie File\n# comment\n\n#HttpOnly_.example.com\tTRUE\t/\tTRUE\t1893456000\tsid\tabc\nbad line\n.example.com\tFALSE\t/p\tFALSE\t\tname\tvalue with spaces\n";
        let cookies = parse_netscape_cookies(text).unwrap();
        assert_eq!(cookies.len(), 2);
        assert!(cookies[0].http_only && cookies[0].secure && cookies[0].include_subdomains);
        assert_eq!(cookies[0].expires, Some(1_893_456_000));
        assert_eq!(cookies[1].expires, None);
        assert_eq!(cookies[1].value, "value with spaces");
        assert!(parse_netscape_cookies("").is_err());
    }

    #[test]
    fn auth_required_error_patterns() {
        assert!(is_auth_required_error("login required for this content"));
        assert!(is_auth_required_error("private playlist -403"));
        assert!(is_auth_required_error(
            "HTTP Error 412: Precondition Failed"
        ));
        assert!(!is_auth_required_error("temporary network timeout"));
        assert!(is_cookie_source_unavailable_error(
            "could not find firefox cookies database in x"
        ));
        assert!(is_keyring_error("failed to decrypt with DPAPI"));
        assert!(is_unsupported_url_error(
            "ERROR: Unsupported URL: https://x"
        ));
    }
}
