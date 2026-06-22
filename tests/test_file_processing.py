import zipfile
from pathlib import Path
from unittest.mock import patch

from aimd.core.process import _split_markdown_by_headers
from aimd.plugins.doc.processor import PANDOC_INPUT_FORMAT_BY_EXTENSION, process_doc_with_assets


def test_split_markdown_by_headers_fallback_without_headers() -> None:
    markdown = "Paragraph " * 6000
    sections, header_level = _split_markdown_by_headers(markdown, max_chunk_size=4000)

    assert header_level is None
    assert len(sections) > 1
    assert all(len(content) <= 4000 for _, content in sections)


def _fake_convert(html_file: Path, output_file: Path) -> None:
    """Write large markdown content to *output_file*, simulating pandoc."""
    output_file.write_text("# Chapter\n\n" + ("lorem ipsum " * 5000), encoding="utf-8")


def test_process_epub_large_content_returns_markdown(tmp_path: Path) -> None:
    epub_path = tmp_path / "document.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("OEBPS/ch1.html", "<html><body>c1</body></html>")
        zf.writestr("OEBPS/ch2.html", "<html><body>c2</body></html>")

    with patch(
        "aimd.plugins.doc.processor._convert_html_to_markdown",
        side_effect=_fake_convert,
    ):
        result = process_doc_with_assets(epub_path)

    assert result.markdown
    assert len(result.markdown) > 40000


def test_process_epub_spine_ordering(tmp_path: Path) -> None:
    """Chapters should follow spine order, not alphabetical."""
    epub_path = tmp_path / "ordered.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="content.opf"/></rootfiles>'
            "</container>",
        )
        zf.writestr(
            "content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf">'
            "<manifest>"
            '<item id="ch_b" href="b.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="ch_a" href="a.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="ch_b"/><itemref idref="ch_a"/></spine>'
            "</package>",
        )
        zf.writestr("b.xhtml", "<html><body><p>Beta</p></body></html>")
        zf.writestr("a.xhtml", "<html><body><p>Alpha</p></body></html>")

    written_order: list[str] = []

    def _track_convert(html_file: Path, output_file: Path) -> None:
        written_order.append(html_file.stem)
        output_file.write_text(f"## {html_file.stem}\n\ncontent\n", encoding="utf-8")

    with patch(
        "aimd.plugins.doc.processor._convert_html_to_markdown",
        side_effect=_track_convert,
    ):
        result = process_doc_with_assets(epub_path)

    assert written_order == ["b", "a"], "Spine order should be b then a"
    combined = (result.output_dir / "ordered.md").read_text(encoding="utf-8")
    assert combined.index("b") < combined.index("a")


def test_pandoc_extension_map_includes_supported_document_families() -> None:
    expected_extensions = {
        ".adoc",
        ".bib",
        ".csv",
        ".docbook",
        ".docx",
        ".epub",
        ".fb2",
        ".html",
        ".ipynb",
        ".jats",
        ".json",
        ".md",
        ".odt",
        ".opml",
        ".org",
        ".ris",
        ".rst",
        ".rtf",
        ".tex",
        ".textile",
        ".tsv",
        ".typ",
        ".vimwiki",
    }

    assert expected_extensions <= set(PANDOC_INPUT_FORMAT_BY_EXTENSION)
    assert ".mobi" not in PANDOC_INPUT_FORMAT_BY_EXTENSION
    assert ".azw3" not in PANDOC_INPUT_FORMAT_BY_EXTENSION


def test_process_pandoc_document_uses_detected_reader(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "paper.rst"
    source.write_text("Title\n=====\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_run_pandoc(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        kwargs["output_file"].write_text("# Paper\n\nBody", encoding="utf-8")

    monkeypatch.setattr("aimd.plugins.doc.processor._run_pandoc", _fake_run_pandoc)

    result = process_doc_with_assets(source)

    assert seen["input_file"] == source
    assert seen["input_format"] == "rst"
    assert result.title == "Paper"
    assert result.markdown == "# Paper\n\nBody"
