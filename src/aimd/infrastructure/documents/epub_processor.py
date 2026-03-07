"""EPUB extraction and conversion pipeline.

Aligns with the standalone epub-to-markdown shell script:
  - Spine-based chapter ordering (container.xml -> OPF -> manifest + spine)
  - Pandoc conversion via subprocess: ``-f html -t markdown_mmd-raw_html --wrap=none``
  - Post-processing via :func:`epub_cleaner.clean_markdown`
  - Flat ``images/`` directory (no subdirectory nesting)
  - Chapter files named after the original HTML stem
  - Combined book file uses ``---`` separators between chapters
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

from logly import logger

from ...errors import InputNotFoundError, ProcessingFailedError
from ...types import TextContext
from .chunking import split_text_by_paragraphs
from .epub_cleaner import clean_markdown
from .title_extractor import extract_title_from_content

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}


def _find_oebps_dir(temp_path: Path) -> Path:
    """Recursively search for OEBPS/OPS directory; fall back to *temp_path*."""
    for dirpath in temp_path.rglob("*"):
        if dirpath.is_dir() and dirpath.name in ("OEBPS", "OPS"):
            return dirpath
    return temp_path


def _read_spine_order(temp_path: Path) -> list[Path]:
    """Parse EPUB spine for correct chapter reading order."""
    container = temp_path / "META-INF" / "container.xml"
    if not container.exists():
        return []

    c_text = container.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'full-path="([^"]+)"', c_text)
    if not m:
        return []

    opf_path = temp_path / m.group(1)
    if not opf_path.exists():
        return []

    opf_dir = opf_path.parent
    opf_text = opf_path.read_text(encoding="utf-8", errors="ignore")

    manifest: dict[str, str] = {}
    for match in re.finditer(
        r'<item\s+[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf_text, re.I
    ):
        manifest[match.group(1)] = match.group(2)
    for match in re.finditer(
        r'<item\s+[^>]*href="([^"]+)"[^>]*id="([^"]+)"', opf_text, re.I
    ):
        manifest[match.group(2)] = match.group(1)

    spine_match = re.search(r"<spine[^>]*>(.*?)</spine>", opf_text, re.I | re.S)
    if not spine_match:
        return []

    spine_files: list[Path] = []
    for itemref in re.finditer(
        r'<itemref\s+[^>]*idref="([^"]+)"', spine_match.group(1), re.I
    ):
        idref = itemref.group(1)
        if idref not in manifest:
            continue
        href = urllib.parse.unquote(manifest[idref]).split("#")[0]
        file_path = (opf_dir / href).resolve()
        if file_path.exists() and file_path.suffix.lower() in {
            ".html",
            ".xhtml",
            ".htm",
        }:
            spine_files.append(file_path)

    return spine_files


def _extract_images(oebps_dir: Path, images_dir: Path) -> None:
    """Copy all images under *oebps_dir* into a flat *images_dir*."""
    for img_path in oebps_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in _IMAGE_SUFFIXES:
            shutil.copy(img_path, images_dir / img_path.name)


def _convert_html_to_markdown(html_file: Path, output_file: Path) -> None:
    """Convert a single HTML/XHTML file to markdown via the pandoc CLI."""
    result = subprocess.run(
        [
            "pandoc",
            str(html_file),
            "-f",
            "html",
            "-t",
            "markdown_mmd-raw_html",
            "--wrap=none",
            "-o",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProcessingFailedError(
            f"Pandoc conversion failed for {html_file.name}: {result.stderr.strip()}"
        )


async def process_epub_with_images(
    file_path: str | Path,
    output_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> tuple[TextContext, Path]:
    """Process EPUB file with image extraction and spine-ordered chapters."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"EPUB file not found: {file_path}")

    if output_dir is None:
        output_dir = file_path.parent / file_path.stem

    chapters_dir = output_dir / "chapters"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
        temp_path = Path(tmp)

        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile as e:
            raise ProcessingFailedError(
                f"Invalid EPUB file (not a valid ZIP): {e}"
            ) from e

        oebps_dir = _find_oebps_dir(temp_path)
        _extract_images(oebps_dir, images_dir)

        spine_files = _read_spine_order(temp_path)
        if not spine_files:
            html_files: list[Path] = []
            for ext in ("*.xhtml", "*.html", "*.htm"):
                html_files.extend(oebps_dir.rglob(ext))
            spine_files = sorted(html_files)

        if not spine_files:
            raise ProcessingFailedError("No HTML/XHTML chapter files found in EPUB")

        chapter_files: list[tuple[str, str]] = []
        for html_file in spine_files:
            if not html_file.exists():
                continue

            basename = html_file.stem
            out_md = chapters_dir / f"{basename}.md"

            try:
                _convert_html_to_markdown(html_file, out_md)
                clean_markdown(out_md)
                content = out_md.read_text(encoding="utf-8")
                chapter_files.append((basename, content))
            except Exception as e:
                logger.warning(f"Failed to convert {html_file.name}: {e}")
                continue

        if not chapter_files:
            raise ProcessingFailedError("Failed to convert any HTML files to markdown")

        combined_content = "\n\n---\n\n".join(
            content.strip() for _, content in chapter_files
        )

        full_md_path = output_dir / f"{file_path.stem}.md"
        full_md_path.write_text(combined_content.strip() + "\n", encoding="utf-8")

        title = extract_title_from_content(chapter_files[0][1], file_path.stem)

        logger.info(
            f"EPUB extracted to {output_dir}: {len(chapter_files)} chapters, "
            f"{sum(1 for _ in images_dir.iterdir())} images"
        )

        chunk_list = (
            [combined_content]
            if len(combined_content) <= 40000
            else split_text_by_paragraphs(combined_content, 40000)
        )
        return TextContext(title=title, chunk_list=chunk_list), output_dir
