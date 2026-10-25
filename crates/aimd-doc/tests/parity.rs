//! Tier 3 (`AIMD_E2E=1`): run the Python doc plugin and the Rust converter on
//! the same inputs and compare Markdown, titles, and extracted image sets.
//! Shells out to `uv run python`, so it is skipped by default.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use aimd_core::NoHooks;

const SCRIPT: &str = r##"
import json, sys, zipfile
from pathlib import Path
from aimd.plugins.doc.conversion import clean_markdown, process_doc_with_assets

work = Path(sys.argv[1])
repo = Path(sys.argv[2])
corpus = json.loads((work / "corpus.json").read_text())

def png(n):
    return b"\x89PNG-fake-" + str(n).encode()

rich = work / "rich.epub"
with zipfile.ZipFile(rich, "w") as z:
    z.writestr("mimetype", "application/epub+zip")
    z.writestr("META-INF/container.xml",
        '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    z.writestr("OEBPS/content.opf",
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata/><manifest>'
        '<item id="c2" href="text/ch%202.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="c1" href="text/ch1.xhtml#top" media-type="application/xhtml+xml"/>'
        '<item id="toc" href="toc.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="img" href="images/a.png" media-type="image/png"/>'
        '</manifest><spine><itemref idref="toc"/><itemref idref="c1"/><itemref idref="c2"/><itemref idref="gone"/></spine></package>')
    z.writestr("OEBPS/toc.xhtml",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>'
        '<li><a href="text/ch1.xhtml">Chapter 1</a></li><li><a href="text/ch%202.xhtml">Second</a></li></ol></nav></body></html>')
    z.writestr("OEBPS/text/ch1.xhtml",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        '<h1>Chapter 1</h1><h2>The Beginning</h2><p>Hello <span id="x"></span>world<a href="#fn1">1</a>.</p>'
        '<p><img src="../images/a.png" alt="First"/></p><p><img src="../images/b.jpg" alt=""/></p>'
        '<hr/><h2>The Beginning</h2><h2>The Beginning</h2><p>Tail</p>'
        '<div id="fn1"><sup><a href="#r1">1</a></sup> A footnote.</div></body></html>')
    z.writestr("OEBPS/text/ch 2.xhtml",
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Part 2</h1><h1>Intro</h1><p>Body &amp; more.</p><pre>  code line\n    indented</pre></body></html>')
    z.writestr("OEBPS/images/a.png", png(1))
    z.writestr("OEBPS/images/b.jpg", png(2))
    z.writestr("OEBPS/images/sub/c.svg", b"<svg/>")
    z.writestr("OEBPS/images/readme.txt", b"no")

out = {"clean": [clean_markdown_str for clean_markdown_str in []], "docs": {}}
import tempfile
for text in corpus:
    p = Path(tempfile.mkdtemp()) / "c.md"
    p.write_text(text, encoding="utf-8")
    clean_markdown(p)
    out["clean"].append(p.read_text(encoding="utf-8"))

for name, src in [("rich", rich), ("minimal", repo / "tests/test_files/minimal.epub")]:
    od = work / f"py_{name}"
    r = process_doc_with_assets(src, output_dir=od)
    out["docs"][name] = {"markdown": r.markdown, "title": r.title}
rst = work / "paper.rst"
rst.write_text("Title\n=====\n\nBody *text* with `code`.\n\n- a\n- b\n", encoding="utf-8")
r = process_doc_with_assets(rst)
out["docs"]["rst"] = {"markdown": r.markdown, "title": r.title}
print("@@JSON@@" + json.dumps(out))
"##;

fn corpus() -> Vec<String> {
    let base = [
        "# 1\n## Chapter 3\n\n#Tight\n## Dup\n## Dup\n",
        "# Chapter 1\n# The Title\n\ntext\n# A.\n# Sub\n",
        "# Part\n\n## Heading\n## Heading\n\n\n## Heading\ntext",
        "###### deep\n####### too deep\n##\n#\n# # nested\n#  # spaced\n#####x\n",
        "[One](a.xhtml)[Two](b.html)\n\n[Three](c.xhtml)\ntext after\n\n[Four](d.html)\n",
        "^[1](#fn1)^ first note\n^<a href=\"#x\">2</a>^ second\n^<a href=\"#y\">iv</a>^ third\ninline[^1](#z) ref\n",
        "<span class=\"image placeholder\"></span>\n![alt](../x/Images/pic.png)\n<img class=\"c\" src=\"OEBPS/images/q.JPG\" alt=\"Q\"/>\n![ ](dir/p.gif) ![z](p.svg) ![k](images/n.webp)\n",
        "a\n------\nb\n---\nc\n",
        "text\n# Heading right after\nmore\n## Another\n\n\n\n\nlots    \ttrailing   \n",
        "   indented\n\t\ttabbed\n  # not trimmed heading  \n```\n  code\n```\n",
        "Chapter 3\n\n# Chapter 3\n## The Beginning\n# Appendix B\n## Notes\n",
        "# Straße ÄÖ ünïcode\n\n[Ünï](ch.xhtml) <a href=\"#a\">ünï</a>\n",
        "line1\r\nline2\r\n# H\r\n",
        "\u{85}weird\u{2028}seps\u{b}here\n# H\n",
        "",
    ];
    base.iter().map(|s| s.to_string()).collect()
}

fn json_escape(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn images(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut m = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(dir.join("images")) {
        for e in rd.flatten() {
            m.insert(
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            );
        }
    }
    m
}

fn tree(dir: &Path, sub: &str) -> BTreeMap<String, Vec<u8>> {
    let mut m = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(dir.join(sub)) {
        for e in rd.flatten() {
            m.insert(
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            );
        }
    }
    m
}

#[test]
fn python_and_rust_converters_agree() {
    if std::env::var("AIMD_E2E").ok().as_deref() != Some("1") {
        eprintln!("skip: set AIMD_E2E=1 to run the Python parity check");
        return;
    }
    assert!(
        Command::new("pandoc").arg("--version").output().is_ok(),
        "pandoc required"
    );
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let work = std::env::temp_dir().join(format!("aimd-doc-parity-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();

    let cases = corpus();
    let json = format!(
        "[{}]",
        cases
            .iter()
            .map(|c| json_escape(c))
            .collect::<Vec<_>>()
            .join(",")
    );
    std::fs::write(work.join("corpus.json"), json).unwrap();

    let mut child = Command::new("uv")
        .args([
            "run",
            "python",
            "-",
            work.to_str().unwrap(),
            repo.to_str().unwrap(),
        ])
        .current_dir(&repo)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("uv");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(SCRIPT.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "python reference failed");
    let stdout = String::from_utf8(out.stdout).unwrap();
    let payload = stdout.split("@@JSON@@").nth(1).expect("json marker");
    let v: serde_json::Value = serde_json::from_str(payload.trim()).unwrap();

    for (i, expected) in v["clean"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            aimd_doc::__clean_markdown(&cases[i]),
            expected.as_str().unwrap(),
            "cleaner mismatch for corpus[{i}]: {:?}",
            cases[i]
        );
    }

    for name in ["rich", "minimal"] {
        let src = if name == "rich" {
            work.join("rich.epub")
        } else {
            repo.join("tests/test_files/minimal.epub")
        };
        let od = work.join(format!("rs_{name}"));
        let got = aimd_doc::convert_document(&src, Some(&od), None, &NoHooks).unwrap();
        let py = &v["docs"][name];
        assert_eq!(
            got.markdown,
            py["markdown"].as_str().unwrap(),
            "{name} markdown"
        );
        assert_eq!(got.title.as_deref(), py["title"].as_str(), "{name} title");
        let py_dir = work.join(format!("py_{name}"));
        assert_eq!(images(&od), images(&py_dir), "{name} images");
        assert_eq!(
            tree(&od, "chapters"),
            tree(&py_dir, "chapters"),
            "{name} chapters"
        );
        assert_eq!(
            std::fs::read(od.join(format!(
                "{}.md",
                if name == "rich" { "rich" } else { "minimal" }
            )))
            .unwrap(),
            std::fs::read(py_dir.join(format!(
                "{}.md",
                if name == "rich" { "rich" } else { "minimal" }
            )))
            .unwrap()
        );
        if name == "rich" {
            assert_eq!(images(&od).len(), 3, "a.png, b.jpg, c.svg");
        }
    }

    let rst = work.join("paper.rst");
    let got = aimd_doc::convert_document(&rst, None, None, &NoHooks).unwrap();
    assert_eq!(got.markdown, v["docs"]["rst"]["markdown"].as_str().unwrap());
    assert_eq!(got.title.as_deref(), v["docs"]["rst"]["title"].as_str());

    let _ = std::fs::remove_dir_all(&work);
}
