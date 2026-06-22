import zipfile
from pathlib import Path
from unittest.mock import patch

from aimd.core.infrastructure.documents.chunking import split_markdown_by_headers
from aimd.book.processor import process_book_with_images


def test_split_markdown_by_headers_fallback_without_headers() -> None:
    markdown = "Paragraph " * 6000
    sections, header_level = split_markdown_by_headers(markdown, max_chunk_size=4000)

    assert header_level is None
    assert len(sections) > 1
    assert all(len(content) <= 4000 for _, content in sections)


def _fake_convert(html_file: Path, output_file: Path) -> None:
    """Write large markdown content to *output_file*, simulating pandoc."""
    output_file.write_text("# Chapter\n\n" + ("lorem ipsum " * 5000), encoding="utf-8")


def test_process_epub_large_content_returns_markdown(tmp_path: Path) -> None:
    epub_path = tmp_path / "aimd.book.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("OEBPS/ch1.html", "<html><body>c1</body></html>")
        zf.writestr("OEBPS/ch2.html", "<html><body>c2</body></html>")

    with patch(
        "aimd.book.processor._convert_html_to_markdown",
        side_effect=_fake_convert,
    ):
        result = process_book_with_images(epub_path)

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
        "aimd.book.processor._convert_html_to_markdown",
        side_effect=_track_convert,
    ):
        result = process_book_with_images(epub_path)

    assert written_order == ["b", "a"], "Spine order should be b then a"
    combined = (result.output_dir / "ordered.md").read_text(encoding="utf-8")
    assert combined.index("b") < combined.index("a")
