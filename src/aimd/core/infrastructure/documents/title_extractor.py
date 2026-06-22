"""Title extraction utilities for markdown content."""

import re


def extract_title_from_content(
    content: str, fallback_title: str = "Untitled", for_filename: bool = False
) -> str:
    """Extract and clean title from content with unified logic."""
    if not content or not content.strip():
        return fallback_title

    lines = content.strip().split("\n")
    extracted_title = None

    for line in lines:
        line = line.strip()
        if line.startswith("# "):
            extracted_title = line[2:].strip()
            break

    if not extracted_title and content.strip().startswith("---"):
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter and line.strip().startswith("title:"):
                title_match = re.match(r'title:\s*["\']?([^"\']+)["\']?', line.strip())
                if title_match:
                    extracted_title = title_match.group(1).strip()
                    break

    if not extracted_title:
        for i, line in enumerate(lines):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (
                    all(c == "=" for c in next_line) or all(c == "-" for c in next_line)
                ):
                    extracted_title = line.strip()
                    break

    if not extracted_title:
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            if (
                line.startswith("![")
                or line.startswith("[]{")
                or line.startswith(":::")
                or line.startswith("<div")
                or line.startswith("</div")
                or "calibre" in line.lower()
                or "kindle-cn" in line.lower()
            ):
                continue
            if (
                len(line) >= 2
                and len(line) <= 100
                and not line.lower().startswith("http")
            ):
                extracted_title = line
                break

    if not extracted_title:
        return fallback_title

    clean_text = re.sub(r"^#+\s*", "", extracted_title)
    clean_text = re.sub(r"\*+([^*]+)\*+", r"\1", clean_text)
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
    clean_text = re.sub(r"\{[^}]*\}", "", clean_text)
    clean_text = re.sub(r"\[\^[^\]]*\](?:\([^)]*\))?", "", clean_text)
    clean_text = re.sub(r"\^[^\]]*\]", "", clean_text)
    clean_text = re.sub(r"#[a-zA-Z0-9_.-]+", "", clean_text)
    clean_text = re.sub(r"\([^)]*\)", "", clean_text)
    clean_text = re.sub(r'^["""\'\']+|["""\'\']+$', "", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = re.sub(r"[。，、；：！？]+$", "", clean_text)

    if for_filename:
        clean_text = re.sub(r'[<>:"/\\|?*]', "", clean_text)
        clean_text = re.sub(r"\s+", "_", clean_text.strip())
        if len(clean_text) > 50:
            clean_text = clean_text[:50].rstrip("_")

    return clean_text or fallback_title
