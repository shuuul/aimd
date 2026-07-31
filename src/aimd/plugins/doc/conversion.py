"""Pandoc-backed document conversion helpers.

EPUB keeps a small custom ZIP/spine pipeline so images and reading order remain
stable. Other Pandoc-supported local document formats go through the Pandoc CLI
directly and are normalized to Markdown.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path

from logly import logger

from aimd.core.errors import (
    BackendUnavailableError,
    InputNotFoundError,
    ProcessingFailedError,
    UnsupportedInputError,
)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}
_PANDOC_MARKDOWN_FORMAT = "markdown_mmd-raw_html"

# Practical filename-extension coverage for Pandoc input formats. Some Pandoc
# readers (for example commonmark_x, markdown_strict, or csljson) are format
# variants rather than distinct file extensions, so they share the closest
# conventional extension here.
PANDOC_INPUT_FORMAT_BY_EXTENSION = {
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".bib": "bibtex",
    ".biblatex": "biblatex",
    ".bits": "bits",
    ".commonmark": "commonmark",
    ".creole": "creole",
    ".csv": "csv",
    ".dbk": "docbook",
    ".docbook": "docbook",
    ".docx": "docx",
    ".dokuwiki": "dokuwiki",
    ".endnote": "endnotexml",
    ".enw": "endnotexml",
    ".epub": "epub",
    ".fb2": "fb2",
    ".gfm": "gfm",
    ".haddock": "haddock",
    ".htm": "html",
    ".html": "html",
    ".ipynb": "ipynb",
    ".jats": "jats",
    ".json": "json",
    ".latex": "latex",
    ".ltx": "latex",
    ".man": "man",
    ".markdown": "markdown",
    ".md": "markdown",
    ".mdoc": "mdoc",
    ".mediawiki": "mediawiki",
    ".muse": "muse",
    ".native": "native",
    ".odt": "odt",
    ".opml": "opml",
    ".org": "org",
    ".ris": "ris",
    ".rst": "rst",
    ".rtf": "rtf",
    ".t2t": "t2t",
    ".tex": "latex",
    ".textile": "textile",
    ".tsv": "tsv",
    ".typ": "typst",
    ".typst": "typst",
    ".vimwiki": "vimwiki",
    ".wiki": "mediawiki",
}
PANDOC_DOCUMENT_EXTENSIONS = frozenset(PANDOC_INPUT_FORMAT_BY_EXTENSION)


# --- Inlined cleaner.py (EPUB markdown post-processing) ---
_IMAGE_EXTS = r"jpg|jpeg|png|gif|webp|svg"


def clean_markdown(file_path: Path) -> None:
    """Read *file_path*, apply all EPUB-specific fixups, write back."""
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = _clean_spans(text)
    text = _fix_image_refs(text)
    text = _normalize_separators(text)
    text = _convert_footnotes(text)
    text = _strip_remaining_html(text)
    text = _flatten_toc(text)
    text = _normalize_headings(text)
    text = _merge_consecutive_headings(text)
    text = _demote_headings(text)
    text = _dedup_headings(text)
    text = _ensure_heading_spacing(text)
    text = _final_whitespace(text)
    file_path.write_text(text, encoding="utf-8")


def _clean_spans(text: str) -> str:
    text = re.sub(
        r'<span\b[^>]*class="[^"]*\bimage placeholder\b[^"]*"[^>]*>\s*</span>\n*',
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r'<span\b[^>]*id="[^"]*"[^>]*>\s*</span>\n*',
        "",
        text,
        flags=re.I,
    )
    return text


def _fix_image_refs(text: str) -> str:
    text = re.sub(
        rf'<img\b[^>]*src="[^"]*/images/([^"/]+\.(?:{_IMAGE_EXTS}))"[^>]*alt="([^"]*)"[^>]*/?>',
        lambda m: f"![{m.group(2) or 'Image'}](images/{m.group(1)})",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf'<img\b[^>]*src="([^"]*?/)?([^"/]+\.(?:{_IMAGE_EXTS}))"[^>]*alt="([^"]*)"[^>]*/?>',
        lambda m: f"![{m.group(3) or 'Image'}](images/{m.group(2)})",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"!\[([^\]]*)\]\((?:[^)\"]*/)?" r"images/([^)\"/]+)\)",
        r"![\1](images/\2)",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"!\[([^\]]*)\]\((?:[^)\"]*/)?" rf"([^)/\"]+\.(?:{_IMAGE_EXTS}))\)",
        r"![\1](images/\2)",
        text,
        flags=re.I,
    )
    return text


def _normalize_separators(text: str) -> str:
    return re.sub(r"\n[-]{5,}\n", "\n\n---\n\n", text)


def _convert_footnotes(text: str) -> str:
    text = re.sub(
        r"^\s*\^\[([^\]]+)\]\(#.+?\)\^\s*",
        lambda m: f"[^{m.group(1).strip()}]: ",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^\s*\^<a\b[^>]*>(\d+)</a>\^\s*",
        r"[^\1]: ",
        text,
        flags=re.M | re.I,
    )
    text = re.sub(
        r"^\s*\^<a\b[^>]*>([^<]+)</a>\^\s*",
        lambda m: f"[^{m.group(1).strip()}]: ",
        text,
        flags=re.M | re.I,
    )
    text = re.sub(r"\[\^([^\]]+)\]\(#.+?\)", r"[^\1]", text)
    return text


def _strip_remaining_html(text: str) -> str:
    text = re.sub(
        r'<a\b[^>]*href="#[^"]+"[^>]*>([^<]+)</a>',
        r"\1",
        text,
        flags=re.I,
    )
    text = re.sub(r"</?span\b[^>]*>", "", text, flags=re.I)
    return text


def _flatten_toc(text: str) -> str:
    text = re.sub(
        r"\[([^\]]+)\]\([^)]*(?:\.html|\.xhtml)[^)]*\)",
        r"@@TOC@@ \1",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*@@TOC@@\s*", r"\n@@TOC@@ ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.splitlines()
    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("@@TOC@@ "):
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("@@TOC@@ "):
                title = lines[i].strip()[len("@@TOC@@ ") :].strip()
                if title:
                    block.append(title)
                i += 1
            if block:
                new_lines.append(f"## {': '.join(block)}")
                new_lines.append("")
            continue
        new_lines.append(line)
        i += 1

    text = "\n".join(new_lines)
    return re.sub(r"@@TOC@@\s*", "", text)


def _normalize_heading_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    if re.fullmatch(r"#{1,6}", s):
        return ""

    while True:
        new_s = re.sub(
            r"^(#{1,6})\s+(#{1,6})(\s+.*)$",
            lambda m: "#" * (len(m.group(1)) + len(m.group(2))) + m.group(3),
            s,
        )
        if new_s == s:
            break
        s = new_s

    s = re.sub(r"^(#{1,6})(\S)", r"\1 \2", s)
    s = re.sub(r"^(#{1,6})\s+(#{1,6})\s+", r"\1 ", s)
    s = re.sub(r"^(#{1,6})\s+", r"\1 ", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


_LABEL_RE = re.compile(
    r"^(Chapter|Part|Section|Book|Volume|Appendix)\s+[A-Za-z0-9\.]+$", re.I
)
_SHORT_LABEL_RE = re.compile(r"^[A-Z0-9]+[\.\)]?$", re.I)


def _normalize_headings(text: str) -> str:
    lines = text.splitlines()
    fixed: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}(\s|#|$)", stripped):
            normalized = _normalize_heading_line(line)
            if normalized:
                fixed.append(normalized)
        else:
            fixed.append(line.rstrip())
    return "\n".join(fixed)


def _merge_consecutive_headings(text: str) -> str:
    fixed = text.splitlines()
    merged: list[str] = []
    i = 0
    while i < len(fixed):
        line = fixed[i]
        m_cur = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m_cur:
            merged.append(line)
            i += 1
            continue

        current_level = m_cur.group(1)
        parts = [m_cur.group(2).strip()]
        j = i + 1
        while j < len(fixed):
            lookahead = fixed[j]
            if not lookahead.strip():
                j += 1
                continue
            m_next = re.match(r"^(#{1,6})\s+(.*)", lookahead)
            if m_next:
                last = parts[-1]
                if _LABEL_RE.match(last) or _SHORT_LABEL_RE.match(last):
                    parts.append(m_next.group(2).strip())
                    j += 1
                else:
                    break
            else:
                break

        merged.append(f"{current_level} {': '.join(parts)}")
        i = j

    return "\n".join(merged)


def _demote_headings(text: str) -> str:
    lines = text.splitlines()
    adjusted: list[str] = []
    for line in lines:
        s = line.strip()
        if re.match(r"^#\s+", s):
            s = re.sub(r"^#\s+", "## ", s)
        elif re.match(r"^##\s+", s):
            s = re.sub(r"^##\s+", "### ", s)
        adjusted.append(s if s else "")
    return "\n".join(adjusted)


def _dedup_headings(text: str) -> str:
    return re.sub(r"^(#{1,6}\s+.+)\n+\1$", r"\1", text, flags=re.M)


def _ensure_heading_spacing(text: str) -> str:
    text = re.sub(r"([^\n])\n(#{1,6}\s)", r"\1\n\n\2", text)
    text = re.sub(r"(#{1,6}\s[^\n]+)\n([^\n#])", r"\1\n\n\2", text)
    return text


def _final_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text.strip() + "\n"


# --- end inlined ---


@dataclass(slots=True, frozen=True)
class DocConversion:
    """Document conversion result used by the MarkItDown plugin."""

    title: str
    markdown: str
    output_dir: Path


def _extract_title_from_markdown(content: str, fallback_title: str) -> str:
    """Extract a simple title from generated markdown."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback_title
        if stripped and not stripped.startswith(("![", "<", ":::")):
            return stripped[:100]
    return fallback_title


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


def _run_pandoc(
    *,
    input_file: Path,
    input_format: str,
    output_file: Path,
    extract_media_dir: Path | None = None,
) -> None:
    """Run Pandoc for a local file, raising a concise runtime error on failure."""
    command = [
        "pandoc",
        str(input_file),
        "-f",
        input_format,
        "-t",
        _PANDOC_MARKDOWN_FORMAT,
        "--wrap=none",
        "-o",
        str(output_file),
    ]
    if extract_media_dir is not None:
        command.extend(["--extract-media", str(extract_media_dir)])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise BackendUnavailableError(
            "Pandoc CLI is required for this document format"
        ) from exc

    if result.returncode != 0:
        raise ProcessingFailedError(
            f"Pandoc conversion failed for {input_file.name}: {result.stderr.strip()}"
        )


def _process_pandoc_document(
    file_path: Path,
    *,
    input_format: str,
    output_dir: Path | None,
    temp_dir: Path | None,
) -> DocConversion:
    """Convert a non-EPUB Pandoc-supported document to Markdown."""
    if output_dir is None:
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmp:
            temp_output = Path(tmp) / f"{file_path.stem}.md"
            _run_pandoc(
                input_file=file_path,
                input_format=input_format,
                output_file=temp_output,
            )
            markdown = temp_output.read_text(encoding="utf-8", errors="ignore")
            return DocConversion(
                title=_extract_title_from_markdown(markdown, file_path.stem),
                markdown=markdown,
                output_dir=file_path.parent,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_path.stem}.md"
    media_dir = output_dir / "media"
    _run_pandoc(
        input_file=file_path,
        input_format=input_format,
        output_file=output_file,
        extract_media_dir=media_dir,
    )
    markdown = output_file.read_text(encoding="utf-8", errors="ignore")
    return DocConversion(
        title=_extract_title_from_markdown(markdown, file_path.stem),
        markdown=markdown,
        output_dir=output_dir,
    )


def _process_epub_with_assets(
    file_path: Path,
    output_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> DocConversion:
    """Process an EPUB file with image extraction and spine-ordered chapters."""
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
                _run_pandoc(
                    input_file=html_file,
                    input_format="html",
                    output_file=out_md,
                )
                clean_markdown(out_md)
                content = out_md.read_text(encoding="utf-8")
                chapter_files.append((basename, content))
            except BackendUnavailableError:
                raise
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

        title = _extract_title_from_markdown(chapter_files[0][1], file_path.stem)

        logger.info(
            f"EPUB extracted to {output_dir}: {len(chapter_files)} chapters, "
            f"{sum(1 for _ in images_dir.iterdir())} images"
        )

        return DocConversion(
            title=title,
            markdown=combined_content,
            output_dir=output_dir,
        )


def process_doc_with_assets(
    file_path: str | Path,
    output_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> DocConversion:
    """Process a Pandoc-supported document into Markdown."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"Document file not found: {file_path}")

    input_format = PANDOC_INPUT_FORMAT_BY_EXTENSION.get(file_path.suffix.lower())
    if input_format is None:
        raise UnsupportedInputError(
            f"Unsupported Pandoc document format: {file_path.suffix}"
        )

    if input_format == "epub":
        return _process_epub_with_assets(
            file_path,
            output_dir=output_dir,
            temp_dir=temp_dir,
        )

    return _process_pandoc_document(
        file_path,
        input_format=input_format,
        output_dir=output_dir,
        temp_dir=temp_dir,
    )
