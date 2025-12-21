import re
from pathlib import Path
from urllib.parse import urlparse
from logly import logger
import yt_dlp


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Sanitize title for use as filename.

    Args:
        title: Title to sanitize
        max_length: Maximum length for filename

    Returns:
        Sanitized filename safe for filesystem
    """

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\|?*]', "_", title)
    # Remove extra whitespace and replace with underscores
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    # Remove leading/trailing dots and underscores
    sanitized = sanitized.strip("._")
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("._")
    # Ensure we have a valid filename
    if not sanitized:
        sanitized = "output"
    return sanitized


def create_output_path_from_title(
    title: str, template_name: str, current_dir: Path = None
) -> Path:
    """Create output path using title and template name.

    Args:
        title: Title from TextContext
        template_name: Template name for suffix
        current_dir: Directory to save file (defaults to current working directory)

    Returns:
        Output path with sanitized title and template suffix
    """
    if current_dir is None:
        current_dir = Path.cwd()

    sanitized_title = sanitize_filename(title)
    filename = f"{sanitized_title}_{template_name}.md"
    return current_dir / filename


def save_result(result: str, output_path: Path) -> None:
    """Save result to file.

    Args:
        result: Text content to save
        output_path: Path to save the result
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Output saved to: {output_path}")


def is_url(s: str) -> bool:
    """Check if a string is a URL using basic URL parsing.

    Args:
        s: String to check

    Returns:
        True if string appears to be a URL
    """
    try:
        result = urlparse(s)
        # A non-empty scheme and netloc are strong indicators of a URL.
        # We check for scheme presence, and also for netloc to catch schemeless URLs like "www.google.com".
        return all([result.scheme, result.netloc]) or (
            result.scheme in ["http", "https"] and not result.netloc
        )
    except ValueError:
        return False


def is_valid_url(url: str) -> bool:
    """Check if URL has valid HTTP/HTTPS scheme.

    Args:
        url: URL to check

    Returns:
        True if URL has valid scheme
    """
    return url.startswith(("http://", "https://"))


def is_supported_url(url: str) -> bool:
    """Check if URL is supported by yt-dlp.

    Args:
        url: URL to check

    Returns:
        True if URL is supported
    """
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # This will raise an exception if URL is not supported
            info = ydl.extract_info(url, download=False)
            return info is not None
    except Exception:
        return False
