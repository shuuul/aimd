"""URL classification helpers."""

from urllib.parse import urlparse


def is_url(value: str) -> bool:
    """Return whether a string is an HTTP(S) URL."""
    try:
        result = urlparse(value)
    except ValueError:
        return False
    return result.scheme in {"http", "https"} and bool(result.netloc)
