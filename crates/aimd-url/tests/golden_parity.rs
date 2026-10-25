//! Golden parity against the Python URL plugin.
//!
//! Fixtures under `tests/golden/subtitles/` are frozen: they were last
//! generated from the Python implementation (`aimd.plugins.url`) at commit
//! 3a6cc6d. The Python oracle and its exporter script have been removed, so
//! the fixtures are now the source of truth; never hand-edit or regenerate them.

use std::path::PathBuf;

use serde_json::Value;

use aimd_url::{
    build_metadata_context, detect_content_language, detect_platform, format_content,
    format_linear_transcript, get_preferred_languages, is_url, normalize_subtitle_payload,
    parse_cookies_from_browser, resolve_subtitle_language, strip_subtitle_formatting,
};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/subtitles")
}

fn load(name: &str) -> Value {
    let path = golden_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn opt_str(v: &Value) -> Option<&str> {
    v.as_str()
}

fn str_list(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

/// Python `str.split()` (no-argument whitespace split).
fn py_split(text: &str) -> Vec<&str> {
    text.split(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
        .filter(|s| !s.is_empty())
        .collect()
}

#[test]
fn subtitle_cleanup_matches_python() {
    let cases = load("cases.json");
    let mut failures = Vec::new();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let raw = std::fs::read_to_string(golden_dir().join("inputs").join(name)).unwrap();
        let normalized = normalize_subtitle_payload(&raw);
        if normalized.as_deref() != opt_str(&case["normalized"]) {
            failures.push(format!(
                "{name}: normalized\n  rust: {normalized:?}\n  py:   {:?}",
                case["normalized"]
            ));
        }
        let stripped = normalized.as_deref().map(strip_subtitle_formatting);
        if stripped.as_deref() != opt_str(&case["stripped"]) {
            failures.push(format!(
                "{name}: stripped\n  rust: {stripped:?}\n  py:   {:?}",
                case["stripped"]
            ));
        }
        let stripped_raw = strip_subtitle_formatting(&raw);
        if Some(stripped_raw.as_str()) != opt_str(&case["stripped_raw"]) {
            failures.push(format!(
                "{name}: stripped_raw\n  rust: {stripped_raw:?}\n  py:   {:?}",
                case["stripped_raw"]
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn text_helpers_match_python() {
    let text = load("text.json");
    let mut failures: Vec<String> = Vec::new();
    let mut check = |section: &str, i: usize, ok: bool, detail: String| {
        if !ok {
            failures.push(format!("{section}[{i}]: {detail}"));
        }
    };

    for (i, c) in text["preferred_languages"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let got = get_preferred_languages(opt_str(&c["language"]), &str_list(&c["available"]));
        let want = str_list(&c["result"]);
        check(
            "preferred_languages",
            i,
            got == want,
            format!("{got:?} != {want:?}"),
        );
    }

    for (i, c) in text["detect_content_language"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let texts: Vec<Option<&str>> = c["texts"].as_array().unwrap().iter().map(opt_str).collect();
        let got = detect_content_language(&texts);
        let want = opt_str(&c["result"]);
        check(
            "detect_content_language",
            i,
            got == want,
            format!("{got:?} != {want:?}"),
        );
    }

    for (i, c) in text["resolve_subtitle_language"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let a = c["args"].as_array().unwrap();
        let got = resolve_subtitle_language(
            opt_str(&a[0]),
            opt_str(&a[1]),
            opt_str(&a[2]),
            opt_str(&a[3]),
        );
        let want = opt_str(&c["result"]);
        check(
            "resolve_subtitle_language",
            i,
            got.as_deref() == want,
            format!("{got:?} != {want:?}"),
        );
    }

    for (i, c) in text["metadata_context"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let got = build_metadata_context(&c["info"]);
        let want = opt_str(&c["result"]);
        check(
            "metadata_context",
            i,
            got.as_deref() == want,
            format!("{got:?}\n != {want:?}"),
        );
    }

    for (i, c) in text["format_content"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let got = format_content(&c["info"], opt_str(&c["content"]), opt_str(&c["platform"]));
        let want = c["result"].as_str().unwrap();
        check(
            "format_content",
            i,
            got == want,
            format!("{got:?}\n != {want:?}"),
        );
    }

    for (i, c) in text["cookies_from_browser"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let spec = c["spec"].as_str().unwrap();
        match (parse_cookies_from_browser(spec), c.get("error")) {
            (Err(err), Some(want)) => check(
                "cookies_from_browser",
                i,
                Some(err.as_str()) == want.as_str(),
                format!("{err:?} != {want}"),
            ),
            (Ok(got), None) => {
                let ok = got.browser == c["browser"].as_str().unwrap()
                    && got.profile.as_deref() == opt_str(&c["profile"])
                    && got.keyring.as_deref() == opt_str(&c["keyring"])
                    && got.container.as_deref() == opt_str(&c["container"]);
                check(
                    "cookies_from_browser",
                    i,
                    ok,
                    format!("{spec:?}: {got:?} != {c}"),
                );
            }
            (got, _) => check(
                "cookies_from_browser",
                i,
                false,
                format!("{spec:?}: {got:?} vs {c}"),
            ),
        }
    }

    for (i, c) in text["urls"].as_array().unwrap().iter().enumerate() {
        let url = c["url"].as_str().unwrap();
        let want_url = c["is_url"].as_bool().unwrap();
        check(
            "urls.is_url",
            i,
            is_url(url) == want_url,
            format!("{url:?}"),
        );
        let platform = detect_platform(url);
        let want = c["platform"].as_str().unwrap();
        check(
            "urls.platform",
            i,
            platform == want,
            format!("{url:?}: {platform} != {want}"),
        );
    }

    for (i, c) in text["linear_transcript"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let got = format_linear_transcript(&py_split(c["text"].as_str().unwrap()));
        let want = c["result"].as_str().unwrap();
        check(
            "linear_transcript",
            i,
            got == want,
            format!("{got:?}\n != {want:?}"),
        );
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
