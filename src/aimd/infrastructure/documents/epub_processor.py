"""EPUB extraction and conversion pipeline."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import pandoc
from logly import logger

from ...errors import InputNotFoundError, ProcessingFailedError
from ...types import TextContext
from .chunking import split_text_by_paragraphs
from .title_extractor import extract_title_from_content


async def process_epub_with_images(
    file_path: str | Path,
    output_dir: Path | None = None,
) -> tuple[TextContext, Path]:
    """Process EPUB file with image extraction."""
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

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        zip_path = temp_path / "epub.zip"
        shutil.copy(file_path, zip_path)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile as e:
            raise ProcessingFailedError(f"Invalid EPUB file (not a valid ZIP): {e}") from e

        oebps_dir = None
        for item in temp_path.iterdir():
            if item.is_dir() and item.name in ("OEBPS", "OPS"):
                oebps_dir = item
                break

        if oebps_dir is None:
            for item in temp_path.rglob("*.xhtml"):
                oebps_dir = item.parent
                break
            for item in temp_path.rglob("*.html"):
                oebps_dir = item.parent
                break

        if oebps_dir is None:
            raise ProcessingFailedError("Could not find OEBPS/OPS folder in EPUB")

        html_files = []
        for ext in ("*.xhtml", "*.html"):
            html_files.extend(oebps_dir.glob(ext))

        if not html_files:
            raise ProcessingFailedError("No HTML/XHTML files found in EPUB")

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"}
        for ext in image_extensions:
            for img_path in oebps_dir.rglob(ext):
                rel_path = img_path.relative_to(oebps_dir)
                dest_path = images_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, dest_path)

        chapter_files = []
        for html_file in sorted(html_files):
            try:
                doc = pandoc.read(file=str(html_file), format="html")
                markdown_content = pandoc.write(doc, format="markdown")
                chapter_idx = len(chapter_files) + 1
                chapter_name = f"chapter_{chapter_idx:03d}"
                md_path = chapters_dir / f"{chapter_name}.md"
                md_path.write_text(markdown_content, encoding="utf-8")
                chapter_files.append((chapter_name, markdown_content))
            except Exception as e:
                logger.warning(f"Failed to convert {html_file}: {e}")
                continue

        if not chapter_files:
            raise ProcessingFailedError("Failed to convert any HTML files to markdown")

        first_chapter_content = chapter_files[0][1]
        title = extract_title_from_content(first_chapter_content, file_path.stem)

        combined_content = "\n\n".join(
            f"# {name}\n\n{content}" for name, content in chapter_files
        )

        full_md_path = output_dir / f"{file_path.stem}.md"
        full_md_path.write_text(combined_content, encoding="utf-8")

        logger.info(
            f"EPUB extracted to {output_dir}: {len(chapter_files)} chapters, "
            f"{len(list(images_dir.rglob('*')))} images"
        )

        chunk_list = [combined_content] if len(combined_content) <= 40000 else split_text_by_paragraphs(combined_content, 40000)
        return TextContext(title=title, chunk_list=chunk_list), output_dir
