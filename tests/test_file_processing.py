import zipfile
from pathlib import Path

import pytest

from aimd.tool.file import _split_markdown_by_headers, process_epub_with_images


def test_split_markdown_by_headers_fallback_without_headers() -> None:
    markdown = "Paragraph " * 6000
    sections, header_level = _split_markdown_by_headers(markdown, max_chunk_size=4000)

    assert header_level is None
    assert len(sections) > 1
    assert all(len(content) <= 4000 for _, content in sections)


@pytest.mark.asyncio
async def test_process_epub_large_content_keeps_chunk_list(
    monkeypatch, tmp_path: Path
) -> None:
    epub_path = tmp_path / "book.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("OEBPS/ch1.html", "<html><body>c1</body></html>")
        zf.writestr("OEBPS/ch2.html", "<html><body>c2</body></html>")

    monkeypatch.setattr("aimd.tool.file.pandoc.read", lambda **kwargs: object())
    monkeypatch.setattr(
        "aimd.tool.file.pandoc.write",
        lambda doc, format: "# Chapter\n\n" + ("lorem ipsum " * 5000),
    )

    text_context, _ = await process_epub_with_images(epub_path)
    assert text_context.chunk_list
    assert len("".join(text_context.chunk_list)) > 40000
