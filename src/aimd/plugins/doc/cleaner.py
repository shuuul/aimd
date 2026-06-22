"""Post-processing cleanup for pandoc-generated markdown from EPUB chapters.

Ported from the standalone epub-to-markdown shell script.  Handles image path
normalisation, EPUB-style footnote conversion, TOC hierarchy flattening,
heading normalisation / merging / demotion, and whitespace tidying.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    # ^[1](#ch1_fn1)^ Footnote text  ->  [^1]: Footnote text
    text = re.sub(
        r"^\s*\^\[([^\]]+)\]\(#.+?\)\^\s*",
        lambda m: f"[^{m.group(1).strip()}]: ",
        text,
        flags=re.M,
    )
    # ^<a ...>1</a>^ Footnote text  ->  [^1]: Footnote text
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
    # Inline footnote refs: [^1](#id) -> [^1]
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
    """Convert .html/.xhtml links into TOC markers, then collapse them."""
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
    """Merge consecutive headings when the first looks like a structural label."""
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
    """Shift # -> ## and ## -> ### so chapters start at ##."""
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
