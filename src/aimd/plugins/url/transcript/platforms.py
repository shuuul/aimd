"""URL platform detection for transcript extraction."""


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
