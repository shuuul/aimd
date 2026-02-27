"""Content formatting and platform detection for URL processing."""


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
