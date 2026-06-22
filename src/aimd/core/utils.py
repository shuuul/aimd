import re
from pathlib import Path
from urllib.parse import urlparse


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
