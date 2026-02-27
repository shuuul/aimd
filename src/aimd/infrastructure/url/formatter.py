"""Content formatting and platform detection for URL processing."""

import re


def strip_subtitle_formatting(text: str) -> str:
    """Strip SRT/VTT/TTML formatting from subtitle text, returning plain text.

    Handles SRT (sequence numbers + timestamps), WebVTT (WEBVTT header + timestamps),
    and TTML (XML tags). Returns the original text unchanged if no subtitle formatting
    is detected.
    """
    lines = text.strip().splitlines()
    if not lines:
        return ""

    timestamp_re = re.compile(
        r"\d{1,2}:\d{2}:\d{2}[.,]\d{2,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{2,3}"
    )
    sequence_re = re.compile(r"^\d+$")
    vtt_header_re = re.compile(r"^WEBVTT", re.IGNORECASE)
    vtt_meta_re = re.compile(r"^(Kind:|Language:)", re.IGNORECASE)
    ttml_tag_re = re.compile(r"<[^>]+>")

    has_timestamps = any(timestamp_re.search(line) for line in lines[:30])
    has_ttml = text.strip().startswith("<?xml") or "<tt " in text[:500]

    if has_ttml:
        text_content = ttml_tag_re.sub("", text)
        segments = [s.strip() for s in text_content.split("\n") if s.strip()]
        return " ".join(segments)

    if not has_timestamps:
        return text

    text_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if vtt_header_re.match(stripped):
            continue
        if vtt_meta_re.match(stripped):
            continue
        if timestamp_re.search(stripped):
            continue
        if sequence_re.match(stripped):
            continue
        if stripped.startswith("NOTE "):
            continue
        text_lines.append(stripped)

    return " ".join(text_lines)


def detect_platform(url: str) -> str:
    """Detect the platform from URL."""
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "bilibili.com" in url_lower:
        return "bilibili"
    if "xiaohongshu.com" in url_lower or "xhslink.com" in url_lower:
        return "xiaohongshu"
    if "xiaoyuzhoufm.com" in url_lower:
        return "xiaoyuzhoufm"
    return "unknown"


def format_content(info_dict: dict[str, object], content: str | None) -> str:
    """Format video metadata and extracted content into markdown."""
    title = info_dict.get("title", "Unknown Title")
    description = info_dict.get("description", "No description available")
    uploader = info_dict.get("uploader", info_dict.get("channel", "Unknown Uploader"))
    duration = info_dict.get("duration", 0)
    view_count = info_dict.get("view_count", 0)
    upload_date = info_dict.get("upload_date", "")
    webpage_url = info_dict.get("webpage_url", "")

    formatted_content = f"""# {title}

**Uploader:** {uploader}
**Duration:** {duration} seconds
**Upload Date:** {upload_date}
**View Count:** {view_count:,} views
**URL:** {webpage_url}

## Description

{description}

## Content

"""

    if content and content.strip():
        formatted_content += content
    else:
        formatted_content += "*No subtitles or transcription available for this video.*"

    formatted_content += """

---

*This content was extracted using yt-dlp via aimd*"""
    return formatted_content
