//! Tier 3 (`AIMD_E2E=1`): real network and external tools.
//!
//! - `youtube_subtitle_smoke`: real yt-dlp on a short public YouTube video
//!   with manual subtitles, compared with `uv run aimd <url>` (Python). The
//!   view-count line is masked since it changes between fetches. Override
//!   the URL with `AIMD_E2E_URL`.
//! - `readable_similarity_vs_defuddle`: dom_smoothie vs `npx defuddle parse`
//!   word-multiset F1 on `tests/fixtures/readable/*.html` plus any `*.html`
//!   in `AIMD_READABLE_CORPUS`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use aimd_core::{ProcessHooks, ProcessInput, Result};
use aimd_url::{
    AudioTranscriber, TranscribeRequest, UrlConfig, convert_url_with, readable_from_html,
};

fn e2e() -> bool {
    std::env::var("AIMD_E2E").is_ok_and(|v| v == "1")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct NoAsr;

impl AudioTranscriber for NoAsr {
    fn transcribe(&self, _: &Path, _: &TranscribeRequest, _: &dyn ProcessHooks) -> Result<String> {
        panic!("subtitle smoke must not fall back to audio")
    }
}

struct NoHooks;
impl ProcessHooks for NoHooks {}

fn mask_volatile(markdown: &str) -> String {
    markdown
        .lines()
        .map(|l| {
            if l.starts_with("**View Count:**") {
                "**View Count:** <masked>"
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim_end()
        .to_string()
}

#[test]
fn youtube_subtitle_smoke() {
    if !e2e() {
        return;
    }
    let url = std::env::var("AIMD_E2E_URL")
        .unwrap_or_else(|_| "https://www.youtube.com/watch?v=jNQXAC9IVRw".to_string());
    let input = ProcessInput {
        input_source: url.clone(),
        ..ProcessInput::default()
    };
    let rust = convert_url_with(&input, &UrlConfig::from_env(), &NoAsr, &NoHooks)
        .expect("rust URL conversion");
    assert!(rust.markdown.contains("## Content"), "{}", rust.markdown);

    let out_dir = std::env::temp_dir().join(format!("aimd-url-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).unwrap();
    let py_path = out_dir.join("python.md");
    let status = Command::new("uv")
        .current_dir(repo_root())
        .args(["run", "aimd", &url, "-o"])
        .arg(&py_path)
        .status()
        .expect("uv run aimd");
    assert!(status.success(), "python aimd failed");
    let python = std::fs::read_to_string(&py_path).unwrap();
    std::fs::write(out_dir.join("rust.md"), &rust.markdown).unwrap();

    let (r, p) = (mask_volatile(&rust.markdown), mask_volatile(&python));
    if r != p {
        let rl: Vec<&str> = r.lines().collect();
        let pl: Vec<&str> = p.lines().collect();
        for i in 0..rl.len().max(pl.len()) {
            let (a, b) = (rl.get(i), pl.get(i));
            if a != b {
                eprintln!("line {}: rust={a:?} python={b:?}", i + 1);
            }
        }
        panic!(
            "Rust and Python markdown differ; outputs in {}",
            out_dir.display()
        );
    }
    eprintln!(
        "youtube smoke: Rust == Python ({} bytes)",
        rust.markdown.len()
    );
    let _ = std::fs::remove_dir_all(&out_dir);
}

fn words(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for w in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        *counts.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    counts
}

/// Word-multiset F1 between two texts.
fn f1(a: &str, b: &str) -> f64 {
    let (wa, wb) = (words(a), words(b));
    let overlap: usize = wa
        .iter()
        .map(|(w, n)| (*n).min(*wb.get(w).unwrap_or(&0)))
        .sum();
    let (na, nb): (usize, usize) = (wa.values().sum(), wb.values().sum());
    if na == 0 || nb == 0 {
        return if na == nb { 1.0 } else { 0.0 };
    }
    let (p, r) = (overlap as f64 / na as f64, overlap as f64 / nb as f64);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

fn defuddle_markdown(path: &Path) -> String {
    let npx = std::env::var("AIMD_NPX").unwrap_or_else(|_| "npx".into());
    let out = Command::new(npx)
        .args(["-y", "defuddle", "parse"])
        .arg(path)
        .args(["--json", "--markdown"])
        .output()
        .expect("npx defuddle");
    assert!(
        out.status.success(),
        "defuddle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).expect("defuddle json");
    v.get("contentMarkdown")
        .or_else(|| v.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

#[test]
fn readable_similarity_vs_defuddle() {
    if !e2e() {
        return;
    }
    let mut pages: Vec<PathBuf> = Vec::new();
    let mut dirs = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/readable")];
    if let Ok(corpus) = std::env::var("AIMD_READABLE_CORPUS") {
        dirs.push(PathBuf::from(corpus));
    }
    for dir in dirs {
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "html" || e == "htm"))
            .collect();
        found.sort();
        pages.extend(found);
    }
    let mut scores = Vec::new();
    for page in &pages {
        let html = String::from_utf8_lossy(&std::fs::read(page).unwrap()).into_owned();
        let ours = readable_from_html(&html, None)
            .expect("dom_smoothie")
            .markdown;
        let theirs = defuddle_markdown(page);
        let score = f1(&ours, &theirs);
        eprintln!(
            "readable F1 {score:.3}  {}  (dom_smoothie {} words, defuddle {} words)",
            page.file_name().unwrap().to_string_lossy(),
            words(&ours).values().sum::<usize>(),
            words(&theirs).values().sum::<usize>()
        );
        scores.push(score);
    }
    let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
    eprintln!("readable F1 mean {mean:.3} over {} pages", scores.len());
    assert!(
        mean >= 0.85,
        "dom_smoothie diverges from defuddle: mean F1 {mean:.3}"
    );
}
