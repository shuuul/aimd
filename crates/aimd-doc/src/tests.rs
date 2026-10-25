//! Hermetic tests: fake Pandoc runner + generated fixtures, real-Pandoc
//! checks that skip when `pandoc` is absent.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use aimd_core::{AimdError, ErrorKind, NoHooks, ProcessHooks};

use super::*;
use crate::clean::clean_markdown;
use crate::pandoc::Runner;
use crate::util::TempDir;
use crate::zip::{ZipArchive, crc32};

/// Build a zip (entries deflated when `deflate`, else stored).
fn make_zip(entries: &[(&str, &[u8])], deflate: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = out.len() as u32;
        let (method, body) = if deflate && !name.ends_with('/') {
            (8u16, miniz_oxide::deflate::compress_to_vec(data, 6))
        } else {
            (0u16, data.to_vec())
        };
        let crc = crc32(data);
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&body);

        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes());
        central.extend_from_slice(&method.to_le_bytes());
        central.extend_from_slice(&[0; 4]);
        central.extend_from_slice(&crc.to_le_bytes());
        central.extend_from_slice(&(body.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(name.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0; 12]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let cd_offset = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn write_zip(path: &Path, entries: &[(&str, &[u8])]) {
    std::fs::write(path, make_zip(entries, true)).unwrap();
}

fn scratch() -> TempDir {
    TempDir::new_in(None).unwrap()
}

/// Fake Pandoc: writes a fixed or per-input body and records calls.
#[derive(Default)]
struct Fake {
    seen: Mutex<Vec<(String, String, bool)>>,
    body: Option<String>,
    fail_on: Option<String>,
    calls: std::sync::Arc<AtomicUsize>,
}

impl Runner for Fake {
    fn run(
        &self,
        input: &Path,
        fmt: &str,
        output: &Path,
        media: Option<&Path>,
    ) -> aimd_core::Result<()> {
        let stem = input.file_stem().unwrap().to_string_lossy().into_owned();
        self.seen
            .lock()
            .unwrap()
            .push((stem.clone(), fmt.to_string(), media.is_some()));
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_on.as_deref() == Some(stem.as_str()) {
            return Err(AimdError::failed("boom"));
        }
        let body = self
            .body
            .clone()
            .unwrap_or_else(|| format!("## {stem}\n\ncontent\n"));
        std::fs::write(output, body).unwrap();
        Ok(())
    }
}

fn order(fake: &Fake) -> Vec<String> {
    fake.seen
        .lock()
        .unwrap()
        .iter()
        .map(|s| s.0.clone())
        .collect()
}

#[test]
fn extension_map_covers_document_families() {
    for e in [
        ".adoc", ".bib", ".csv", ".docbook", ".docx", ".epub", ".fb2", ".html", ".ipynb", ".jats",
        ".json", ".md", ".odt", ".opml", ".org", ".ris", ".rst", ".rtf", ".tex", ".textile",
        ".tsv", ".typ", ".vimwiki", ".pptx", ".xlsx", ".txt", ".pdf",
    ] {
        assert!(is_document_extension(e), "{e}");
    }
    assert!(is_document_extension(".DOCX"));
    for e in [".mobi", ".azw3", ".doc", ".ppt", ".xls", ".mp3", ""] {
        assert!(!is_document_extension(e), "{e}");
    }
    for e in DOCUMENT_EXTENSIONS {
        assert!(
            matches!(*e, ".pdf" | ".txt" | ".md") || pandoc::input_format(e).is_some(),
            "{e} has no reader"
        );
    }
}

#[test]
fn zip_roundtrip_stored_and_deflate_and_sanitizes_names() {
    for deflate in [false, true] {
        let data = vec![b'a'; 5000];
        let bytes = make_zip(
            &[
                ("d/", b""),
                ("d/f.txt", &data),
                ("../evil.txt", b"x"),
                ("/abs.txt", b"y"),
            ],
            deflate,
        );
        let z = ZipArchive::parse(bytes).unwrap();
        let t = scratch();
        z.extract_all(t.path()).unwrap();
        assert_eq!(std::fs::read(t.path().join("d/f.txt")).unwrap(), data);
        assert!(t.path().join("evil.txt").exists());
        assert!(t.path().join("abs.txt").exists());
        assert!(!t.path().parent().unwrap().join("evil.txt").exists());
    }
    assert!(ZipArchive::parse(b"not a zip at all, definitely".to_vec()).is_err());
    let mut bad = make_zip(&[("a", b"hello")], false);
    bad[30 + 1] ^= 0xFF; // corrupt the stored payload -> CRC mismatch
    let z = ZipArchive::parse(bad).unwrap();
    assert!(z.extract_all(scratch().path()).unwrap_err().contains("CRC"));
}

#[test]
fn epub_large_content_returns_markdown() {
    let t = scratch();
    let epub = t.path().join("document.epub");
    write_zip(
        &epub,
        &[
            ("OEBPS/ch1.html", b"<html><body>c1</body></html>"),
            ("OEBPS/ch2.html", b"<html><body>c2</body></html>"),
        ],
    );
    let fake = Fake {
        body: Some(format!("# Chapter\n\n{}", "lorem ipsum ".repeat(5000))),
        ..Default::default()
    };
    let out = convert_with(&fake, &epub, None, None, &NoHooks).unwrap();
    assert!(out.markdown.len() > 40000);
    assert!(t.path().join("document/document.md").exists());
}

const OPF_ORDERED: &str = r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"><manifest><item id="ch_b" href="b.xhtml" media-type="application/xhtml+xml"/><item id="ch_a" href="a%20x.xhtml#frag" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="ch_b"/><itemref idref="ch_a"/></spine></package>"#;

#[test]
fn epub_spine_order_images_and_output_files() {
    let t = scratch();
    let epub = t.path().join("ordered.epub");
    write_zip(
        &epub,
        &[
            (
                "META-INF/container.xml",
                br#"<?xml version="1.0"?><container><rootfiles><rootfile full-path="content.opf"/></rootfiles></container>"#,
            ),
            ("content.opf", OPF_ORDERED.as_bytes()),
            ("b.xhtml", b"<html><body><p>Beta</p></body></html>"),
            ("a x.xhtml", b"<html><body><p>Alpha</p></body></html>"),
            ("OEBPS/img/pic.PNG", b"png"),
            ("OEBPS/notes.txt", b"n"),
        ],
    );
    let out_dir = t.path().join("out");
    let fake = Fake::default();
    let out = convert_with(&fake, &epub, Some(&out_dir), Some(t.path()), &NoHooks).unwrap();
    assert_eq!(order(&fake), ["b", "a x"], "spine order, not alphabetical");
    assert!(
        fake.seen
            .lock()
            .unwrap()
            .iter()
            .all(|s| s.1 == "html" && !s.2)
    );
    let combined = std::fs::read_to_string(out_dir.join("ordered.md")).unwrap();
    assert_eq!(combined, out.markdown);
    // quirk preserved from Python: `## b` normalizes to `# b`, demotes to `## b`
    assert_eq!(
        out.markdown,
        "## b\n\ncontent\n\n---\n\n## a x\n\ncontent\n"
    );
    assert_eq!(out.title.as_deref(), Some("## b"));
    assert!(out_dir.join("images/pic.PNG").exists());
    assert!(!out_dir.join("images/notes.txt").exists());
    assert!(out_dir.join("chapters/b.md").exists());
}

#[test]
fn epub_without_spine_falls_back_to_sorted_html() {
    let t = scratch();
    let epub = t.path().join("nospine.epub");
    write_zip(
        &epub,
        &[
            ("OEBPS/z.xhtml", b"z"),
            ("OEBPS/a.html", b"a"),
            ("OEBPS/m.htm", b"m"),
        ],
    );
    let fake = Fake::default();
    convert_with(&fake, &epub, None, None, &NoHooks).unwrap();
    assert_eq!(order(&fake), ["a", "m", "z"]);
    assert!(t.path().join("nospine/nospine.md").exists());
}

#[test]
fn epub_error_cases() {
    let t = scratch();
    let bad = t.path().join("bad.epub");
    std::fs::write(&bad, b"not a zip, just some text that is long enough").unwrap();
    let e = convert_with(&Fake::default(), &bad, None, None, &NoHooks).unwrap_err();
    assert_eq!(e.kind, ErrorKind::ProcessingFailed);
    assert!(e.message.starts_with("Invalid EPUB file (not a valid ZIP)"));

    let empty = t.path().join("empty.epub");
    write_zip(&empty, &[("OEBPS/readme.txt", b"x")]);
    let e = convert_with(&Fake::default(), &empty, None, None, &NoHooks).unwrap_err();
    assert_eq!(e.message, "No HTML/XHTML chapter files found in EPUB");

    let two = t.path().join("two.epub");
    write_zip(&two, &[("a.html", b"a"), ("b.html", b"b")]);
    let fake = Fake {
        fail_on: Some("a".into()),
        ..Default::default()
    };
    let out = convert_with(&fake, &two, None, None, &NoHooks).unwrap();
    assert_eq!(out.markdown, "## b\n\ncontent\n");

    let one = t.path().join("one.epub");
    write_zip(&one, &[("a.html", b"a")]);
    let e = convert_with(&fake, &one, None, None, &NoHooks).unwrap_err();
    assert_eq!(e.message, "Failed to convert any HTML files to markdown");
}

type ProgressRecord = (String, Option<u64>, Option<u64>, Option<String>);

struct CancelAfterFirst {
    calls: std::sync::Arc<AtomicUsize>,
    progress: Mutex<Vec<ProgressRecord>>,
}

impl ProcessHooks for CancelAfterFirst {
    fn is_cancelled(&self) -> bool {
        self.calls.load(Ordering::SeqCst) == 1
    }
    fn progress(&self, stage: &str, c: Option<u64>, t: Option<u64>, m: Option<&str>) {
        self.progress
            .lock()
            .unwrap()
            .push((stage.into(), c, t, m.map(str::to_string)));
    }
}

#[test]
fn epub_reports_progress_and_cancels_between_chapters() {
    let t = scratch();
    let epub = t.path().join("cancel.epub");
    write_zip(
        &epub,
        &[("OEBPS/one.html", b"one"), ("OEBPS/two.html", b"two")],
    );
    let fake = Fake {
        body: Some("chapter".into()),
        ..Default::default()
    };
    let hooks = CancelAfterFirst {
        calls: fake.calls.clone(),
        progress: Mutex::new(vec![]),
    };
    let e = convert_with(&fake, &epub, None, None, &hooks).unwrap_err();
    assert_eq!(e.kind, ErrorKind::Cancelled);
    assert!(e.message.contains("cancelled between chapters"));
    assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *hooks.progress.lock().unwrap(),
        [(
            "converting".to_string(),
            Some(1),
            Some(2),
            Some("Converting chapter 1 of 2".to_string())
        )]
    );
}

#[test]
fn pandoc_document_uses_detected_reader_and_media_dir() {
    let t = scratch();
    let src = t.path().join("paper.rst");
    std::fs::write(&src, "Title\n=====\n").unwrap();
    let fake = Fake {
        body: Some("# Paper\n\nBody".into()),
        ..Default::default()
    };
    let out = convert_with(&fake, &src, None, None, &NoHooks).unwrap();
    assert_eq!(
        fake.seen.lock().unwrap()[0],
        ("paper".to_string(), "rst".to_string(), false)
    );
    assert_eq!(out.title.as_deref(), Some("Paper"));
    assert_eq!(out.markdown, "# Paper\n\nBody");

    let od = t.path().join("o");
    let fake = Fake {
        body: Some("# Paper\n\nBody".into()),
        ..Default::default()
    };
    convert_with(&fake, &src, Some(&od), None, &NoHooks).unwrap();
    assert!(
        fake.seen.lock().unwrap()[0].2,
        "media dir when output_dir set"
    );
    assert_eq!(
        std::fs::read_to_string(od.join("paper.md")).unwrap(),
        "# Paper\n\nBody"
    );
}

#[test]
fn missing_and_unsupported_inputs() {
    let t = scratch();
    let e = convert_with(
        &Fake::default(),
        &t.path().join("missing.epub"),
        None,
        None,
        &NoHooks,
    )
    .unwrap_err();
    assert_eq!(e.kind, ErrorKind::InputNotFound);
    assert!(e.message.contains("Document file not found"));
    for name in ["x.doc", "x.mobi"] {
        let p = t.path().join(name);
        std::fs::write(&p, "x").unwrap();
        let e = convert_with(&Fake::default(), &p, None, None, &NoHooks).unwrap_err();
        assert_eq!(e.kind, ErrorKind::UnsupportedInput);
    }
}

#[test]
fn txt_and_md_pass_through() {
    let t = scratch();
    let p = t.path().join("note.txt");
    std::fs::write(&p, "Hello there\r\nsecond\n").unwrap();
    let out = convert_with(&Fake::default(), &p, None, None, &NoHooks).unwrap();
    assert_eq!(out.markdown, "Hello there\nsecond\n");
    assert_eq!(out.title.as_deref(), Some("Hello there"));
    let m = t.path().join("doc.md");
    std::fs::write(&m, "# T\n\nbody\n").unwrap();
    let od = t.path().join("md_out");
    let out = convert_with(&Fake::default(), &m, Some(&od), None, &NoHooks).unwrap();
    assert_eq!(out.title.as_deref(), Some("T"));
    assert_eq!(
        std::fs::read_to_string(od.join("doc.md")).unwrap(),
        "# T\n\nbody\n"
    );
}

#[test]
fn pandoc_missing_binary_and_nonzero_exit() {
    let t = scratch();
    let src = t.path().join("paper.rst");
    std::fs::write(&src, "x").unwrap();
    let out = t.path().join("out.md");
    let missing = pandoc::PandocCli {
        program: "definitely-not-pandoc-xyz".into(),
    };
    let e = missing.run(&src, "rst", &out, None).unwrap_err();
    assert_eq!(e.kind, ErrorKind::BackendUnavailable);
    assert!(e.message.contains("Pandoc CLI is required"));

    let sh = t.path().join("fail.sh");
    std::fs::write(&sh, "#!/bin/sh\necho boom >&2\nexit 1\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&sh, std::fs::Permissions::from_mode(0o755)).unwrap();
    let failing = pandoc::PandocCli { program: sh.into() };
    let e = failing.run(&src, "rst", &out, None).unwrap_err();
    assert_eq!(e.kind, ErrorKind::ProcessingFailed);
    assert!(
        e.message
            .starts_with("Pandoc conversion failed for paper.rst: boom")
    );
}

#[test]
fn cleanup_pipeline_basics() {
    let input = "<span id=\"x\"></span>\n# 1\n## Chapter 3\n\n#Tight\n## Dup\n## Dup\nbody ![](../Images/p.png) and <img src=\"a/b/c.jpg\" alt=\"\" /> <a href=\"#n1\">note</a>\ntext\n-----\ntrailing   \n";
    let out = clean_markdown(input);
    // Expected string produced by the Python `clean_markdown` on the same input.
    let expected = "## 1: Chapter 3\n#Tight\n\n## Dup: Dup\n\nbody ![](images/p.png) and ![Image](images/c.jpg) note\ntext\n\n---\n\ntrailing\n";
    assert_eq!(out, expected, "got:\n{out}");
}

#[test]
fn splitlines_matches_python() {
    use crate::util::splitlines;
    assert_eq!(splitlines("a\nb\r\nc\rd\x0be"), ["a", "b", "c", "d", "e"]);
    assert_eq!(splitlines("a\n"), ["a"]);
    assert_eq!(splitlines("a\n\nb"), ["a", "", "b"]);
    assert!(splitlines("").is_empty());
}

fn minimal_pdf(text: &str) -> Vec<u8> {
    let stream = format!("BT /F1 18 Tf 72 720 Td ({text}) Tj ET");
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
        format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![];
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objs.len() + 1
        )
        .as_bytes(),
    );
    out
}

#[test]
fn text_layer_pdf_converts_and_bad_pdfs_fail() {
    let t = scratch();
    let p = t.path().join("doc.pdf");
    std::fs::write(&p, minimal_pdf("Hello aimd native document world")).unwrap();
    let out = convert_document(&p, None, None, &NoHooks).unwrap();
    assert!(
        out.markdown.contains("Hello aimd native document world"),
        "{:?}",
        out.markdown
    );

    let blank = t.path().join("blank.pdf");
    std::fs::write(&blank, minimal_pdf("")).unwrap();
    let e = convert_document(&blank, None, None, &NoHooks).unwrap_err();
    assert_eq!(e.kind, ErrorKind::ProcessingFailed);

    let junk = t.path().join("junk.pdf");
    std::fs::write(&junk, b"nope").unwrap();
    let e = convert_document(&junk, None, None, &NoHooks).unwrap_err();
    assert_eq!(e.kind, ErrorKind::ProcessingFailed);
    assert!(
        e.message
            .starts_with("pdf-inspector failed to process PDF: junk.pdf")
    );
}

fn have_pandoc() -> bool {
    std::process::Command::new("pandoc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn real_pandoc_converts_rst_and_epub_fixture() {
    if !have_pandoc() {
        eprintln!("skip: pandoc not installed");
        return;
    }
    let t = scratch();
    let src = t.path().join("paper.rst");
    std::fs::write(&src, "Title\n=====\n\nBody text.\n").unwrap();
    let out = convert_document(&src, None, None, &NoHooks).unwrap();
    assert_eq!(out.title.as_deref(), Some("Title"));
    assert!(out.markdown.contains("Body text."));

    let epub = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal.epub");
    if epub.exists() {
        let od = t.path().join("epub_out");
        let out = convert_document(&epub, Some(&od), None, &NoHooks).unwrap();
        assert!(!out.markdown.trim().is_empty());
        assert!(od.join("minimal.md").exists());
    }
}

const PNG_1X1: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0x0d, 0x49, 0x48, 0x44, 0x52, 0, 0, 0,
    1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 0x1f, 0x15, 0xc4, 0x89, 0, 0, 0, 0x0d, 0x49, 0x44, 0x41, 0x54,
    0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0, 5, 0, 1, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0, 0,
    0, 0, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

#[test]
fn real_pandoc_docx_and_pptx_extract_media() {
    if !have_pandoc() {
        eprintln!("skip: pandoc not installed");
        return;
    }
    let t = scratch();
    std::fs::write(t.path().join("p.png"), PNG_1X1).unwrap();
    std::fs::write(
        t.path().join("src.md"),
        "# Deck\n\nHello docx.\n\n![pic](p.png)\n",
    )
    .unwrap();
    for ext in ["docx", "odt", "pptx"] {
        let doc = t.path().join(format!("sample.{ext}"));
        let st = std::process::Command::new("pandoc")
            .current_dir(t.path())
            .args(["src.md", "-o"])
            .arg(&doc)
            .status()
            .unwrap();
        assert!(st.success());
        let od = t.path().join(format!("out_{ext}"));
        let out = convert_document(&doc, Some(&od), None, &NoHooks).unwrap();
        assert!(
            out.markdown.contains("Hello") || ext == "pptx",
            "{ext}: {}",
            out.markdown
        );
        assert!(od.join("sample.md").exists());
        let media: Vec<_> = walkdir(&od.join("media"));
        assert!(!media.is_empty(), "{ext}: media extracted");
        assert!(
            out.markdown.contains(&format!("{}/media/", od.display())),
            "{ext}: link shape {}",
            out.markdown
        );
        // without output_dir nothing is written next to the input
        let plain = convert_document(&doc, None, None, &NoHooks).unwrap();
        assert!(!plain.markdown.is_empty());
    }
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut v = vec![];
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                v.extend(walkdir(&p));
            } else {
                v.push(p);
            }
        }
    }
    v
}
