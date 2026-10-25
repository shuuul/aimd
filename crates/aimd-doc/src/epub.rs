//! EPUB pipeline: zip extraction, spine order, per-chapter Pandoc + cleanup,
//! flat image extraction.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use aimd_core::{AimdError, ErrorKind, ProcessHooks};
use regex::Regex;

use crate::clean::clean_markdown;
use crate::pandoc::{Runner, file_name};
use crate::util::{TempDir, extract_title, io_err, walk_sorted};
use crate::zip::ZipArchive;

const IMAGE_SUFFIXES: &[&str] = &["jpg", "jpeg", "png", "gif", "svg", "webp"];

static FULL_PATH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"full-path="([^"]+)""#).unwrap());
static ITEM_ID_HREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)<item\s+[^>]*id="([^"]+)"[^>]*href="([^"]+)""#).unwrap());
static ITEM_HREF_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)<item\s+[^>]*href="([^"]+)"[^>]*id="([^"]+)""#).unwrap());
static SPINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<spine[^>]*>(.*?)</spine>").unwrap());
static ITEMREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)<itemref\s+[^>]*idref="([^"]+)""#).unwrap());

pub(crate) struct EpubOutput {
    pub(crate) markdown: String,
    pub(crate) title: String,
}

fn ext_lower(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && let (Some(h), Some(l)) = (
                b.get(i + 1).and_then(|c| hex(*c)),
                b.get(i + 2).and_then(|c| hex(*c)),
            )
        {
            out.push(h * 16 + l);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn find_oebps_dir(temp: &Path) -> PathBuf {
    let mut all = Vec::new();
    walk_sorted(temp, &mut all);
    all.into_iter()
        .find(|p| {
            p.is_dir()
                && matches!(
                    p.file_name().and_then(|n| n.to_str()),
                    Some("OEBPS") | Some("OPS")
                )
        })
        .unwrap_or_else(|| temp.to_path_buf())
}

fn read_lossy(p: &Path) -> Option<String> {
    crate::util::read_text_ignore(p).ok()
}

/// Join `rel` onto `base` only when it stays relative (no absolute override).
fn join_relative(base: &Path, rel: &str) -> Option<PathBuf> {
    if Path::new(rel).is_absolute() {
        None
    } else {
        Some(base.join(rel))
    }
}

fn read_spine_order(temp: &Path) -> Vec<PathBuf> {
    let container = temp.join("META-INF").join("container.xml");
    let Some(c_text) = read_lossy(&container) else {
        return vec![];
    };
    let Some(m) = FULL_PATH.captures(&c_text) else {
        return vec![];
    };
    let Some(opf_path) = join_relative(temp, &m[1]) else {
        return vec![];
    };
    let Some(opf_text) = read_lossy(&opf_path) else {
        return vec![];
    };
    let opf_dir = opf_path.parent().unwrap_or(temp).to_path_buf();
    let canon_root = temp.canonicalize().unwrap_or_else(|_| temp.to_path_buf());

    let mut manifest = std::collections::HashMap::new();
    for c in ITEM_ID_HREF.captures_iter(&opf_text) {
        manifest.insert(c[1].to_string(), c[2].to_string());
    }
    for c in ITEM_HREF_ID.captures_iter(&opf_text) {
        manifest.insert(c[2].to_string(), c[1].to_string());
    }
    let Some(spine) = SPINE.captures(&opf_text) else {
        return vec![];
    };
    let mut files = Vec::new();
    for c in ITEMREF.captures_iter(&spine[1]) {
        let Some(href) = manifest.get(&c[1]) else {
            continue;
        };
        let decoded = percent_decode(href);
        let href = decoded.split('#').next().unwrap_or("");
        let Some(joined) = join_relative(&opf_dir, href) else {
            continue;
        };
        // `Path.resolve()`: canonicalize; a missing file is skipped anyway.
        let Ok(resolved) = joined.canonicalize() else {
            continue;
        };
        if !resolved.starts_with(&canon_root) {
            continue; // hardening: never read outside the extracted archive
        }
        if matches!(ext_lower(&resolved).as_str(), "html" | "xhtml" | "htm") {
            files.push(resolved);
        }
    }
    files
}

fn extract_images(oebps: &Path, images_dir: &Path) -> aimd_core::Result<()> {
    let mut all = Vec::new();
    walk_sorted(oebps, &mut all);
    for p in all {
        if p.is_file()
            && IMAGE_SUFFIXES.contains(&ext_lower(&p).as_str())
            && let Some(name) = p.file_name()
        {
            std::fs::copy(&p, images_dir.join(name)).map_err(io_err)?;
        }
    }
    Ok(())
}

pub(crate) fn process_epub(
    runner: &dyn Runner,
    file_path: &Path,
    output_dir: Option<&Path>,
    temp_dir: Option<&Path>,
    hooks: &dyn ProcessHooks,
) -> aimd_core::Result<EpubOutput> {
    let stem = file_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let output_dir = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| file_path.parent().unwrap_or(Path::new("")).join(&stem));
    let chapters_dir = output_dir.join("chapters");
    let images_dir = output_dir.join("images");
    std::fs::create_dir_all(&output_dir).map_err(io_err)?;
    std::fs::create_dir_all(&chapters_dir).map_err(io_err)?;
    std::fs::create_dir_all(&images_dir).map_err(io_err)?;

    let tmp = TempDir::new_in(temp_dir)?;
    let temp_path = tmp.path();

    let zip_err =
        |e: String| AimdError::failed(format!("Invalid EPUB file (not a valid ZIP): {e}"));
    let archive = ZipArchive::open(file_path).map_err(zip_err)?;
    archive.extract_all(temp_path).map_err(zip_err)?;

    let oebps = find_oebps_dir(temp_path);
    extract_images(&oebps, &images_dir)?;

    let mut spine = read_spine_order(temp_path);
    if spine.is_empty() {
        let mut all = Vec::new();
        walk_sorted(&oebps, &mut all);
        for ext in ["xhtml", "html", "htm"] {
            spine.extend(
                all.iter()
                    .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == ext))
                    .cloned(),
            );
        }
        spine.sort();
    }
    if spine.is_empty() {
        return Err(AimdError::failed(
            "No HTML/XHTML chapter files found in EPUB",
        ));
    }

    let total = spine.len() as u64;
    let mut chapters: Vec<String> = Vec::new();
    let mut first_chapter: Option<String> = None;
    for (idx, html) in spine.iter().enumerate() {
        if hooks.is_cancelled() {
            return Err(AimdError::cancelled(
                "Document conversion cancelled between chapters",
            ));
        }
        let n = idx as u64 + 1;
        hooks.progress(
            "converting",
            Some(n),
            Some(total),
            Some(&format!("Converting chapter {n} of {total}")),
        );
        if !html.exists() {
            continue;
        }
        let base = html
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let out_md = chapters_dir.join(format!("{base}.md"));
        let result = (|| -> aimd_core::Result<String> {
            runner.run(html, "html", &out_md, None)?;
            let raw = read_text_lossy_file(&out_md)?;
            let cleaned = clean_markdown(&raw);
            std::fs::write(&out_md, &cleaned).map_err(io_err)?;
            Ok(cleaned)
        })();
        match result {
            Ok(content) => {
                if first_chapter.is_none() {
                    first_chapter = Some(content.clone());
                }
                chapters.push(content);
            }
            Err(e) if matches!(e.kind, ErrorKind::BackendUnavailable | ErrorKind::Cancelled) => {
                return Err(e);
            }
            Err(e) => eprintln!(
                "WARNING Failed to convert {}: {}",
                file_name(html),
                e.message
            ),
        }
    }

    let Some(first) = first_chapter else {
        return Err(AimdError::failed(
            "Failed to convert any HTML files to markdown",
        ));
    };
    let combined = chapters
        .iter()
        .map(|c| c.trim())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let markdown = format!("{}\n", combined.trim());
    std::fs::write(output_dir.join(format!("{stem}.md")), &markdown).map_err(io_err)?;
    let image_count = std::fs::read_dir(&images_dir)
        .map(|d| d.count())
        .unwrap_or(0);
    aimd_core::log_info!(
        "EPUB extracted to {}: {} chapters, {} images",
        output_dir.display(),
        chapters.len(),
        image_count
    );
    Ok(EpubOutput {
        title: extract_title(&first, &stem),
        markdown,
    })
}

/// Python reads the cleaned chapter back with strict UTF-8.
fn read_text_lossy_file(p: &Path) -> aimd_core::Result<String> {
    let bytes = std::fs::read(p).map_err(io_err)?;
    String::from_utf8(bytes)
        .map(|s| crate::util::universal_newlines(&s))
        .map_err(|e| AimdError::failed(format!("Invalid UTF-8 in {}: {e}", file_name(p))))
}
