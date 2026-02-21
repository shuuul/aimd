"""Cookie-source parsing and auth-related URL extraction helpers."""

from typing import Any

from logly import logger

AUTH_REQUIRED_PLATFORMS = {"bilibili", "xiaohongshu"}


def is_keyring_error(error: Exception) -> bool:
    """Check if an error is related to keyring/cookie decryption issues."""
    error_str = str(error).lower()
    keyring_indicators = [
        "keyring",
        "secretservice",
        "secret service",
        "secret-service",
        "failed to decrypt",
        "could not decrypt",
        "dbus",
        "org.freedesktop.secret",
        "gnome-keyring",
        "kwallet",
    ]
    return any(indicator in error_str for indicator in keyring_indicators)


def is_unsupported_url_error(error: Exception) -> bool:
    """Best-effort check for yt-dlp unsupported URL errors."""
    message = str(error).lower()
    return "unsupported url" in message or "no suitable extractor" in message


def is_auth_required_error(error: Exception) -> bool:
    """Best-effort check for login-required/private-content errors."""
    message = str(error).lower()
    indicators = [
        "login required",
        "sign in",
        "private",
        "members only",
        "premium",
        "watchlater",
        "supporter-only",
        "cookies are required",
        "authentication",
        "403",
        "-403",
        "-101",
    ]
    return any(indicator in message for indicator in indicators)


def parse_cookies_from_browser(
    spec: str,
) -> tuple[str, str | None, str | None, str | None]:
    """Parse browser cookie source spec to yt-dlp tuple."""
    raw = spec.strip()
    if not raw:
        raise ValueError("cookies_from_browser cannot be empty")

    browser_profile, sep, container = raw.partition("::")
    container_name = container.strip() if sep and container.strip() else None

    browser_keyring, has_profile, profile = browser_profile.partition(":")
    profile_name = profile.strip() if has_profile and profile.strip() else None

    browser_name, has_keyring, keyring = browser_keyring.partition("+")
    keyring_name = keyring.strip() if has_keyring and keyring.strip() else None
    browser_name = browser_name.strip().lower()

    if not browser_name:
        raise ValueError(f"Invalid cookies_from_browser value: {spec}")

    return browser_name, profile_name, keyring_name, container_name


def build_cookie_sources(
    *,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
) -> list[dict[str, Any]]:
    """Build ordered cookie source attempts for yt-dlp operations."""
    sources: list[dict[str, Any]] = []

    if cookies_file:
        sources.append(
            {
                "name": "cookiefile",
                "use_cookies": True,
                "cookiefile": cookies_file,
                "cookiesfrombrowser": None,
            }
        )

    if cookies_from_browser:
        try:
            browser_tuple = parse_cookies_from_browser(cookies_from_browser)
            sources.append(
                {
                    "name": f"cookiesfrombrowser:{cookies_from_browser}",
                    "use_cookies": True,
                    "cookiefile": None,
                    "cookiesfrombrowser": browser_tuple,
                }
            )
        except ValueError as exc:
            logger.warning(str(exc))

    if not cookies_file and not cookies_from_browser:
        default_browser_specs = ("chrome:default", "firefox")
        for spec in default_browser_specs:
            sources.append(
                {
                    "name": f"cookiesfrombrowser:{spec}",
                    "use_cookies": True,
                    "cookiefile": None,
                    "cookiesfrombrowser": parse_cookies_from_browser(spec),
                }
            )

    if platform not in AUTH_REQUIRED_PLATFORMS:
        sources.append(
            {
                "name": "no-cookie",
                "use_cookies": False,
                "cookiefile": None,
                "cookiesfrombrowser": None,
            }
        )

    return sources
