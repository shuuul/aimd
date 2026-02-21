import asyncio

import pytest

from aimd.errors import ProcessingFailedError
from aimd.tool.url import (
    _build_cookie_sources,
    _is_auth_required_error,
    _parse_cookies_from_browser,
    get_text_from_url,
)


@pytest.mark.asyncio
async def test_get_text_from_url_extracts_info_once(monkeypatch) -> None:
    call_count = {"count": 0}

    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        call_count["count"] += 1
        return {"title": "Example", "webpage_url": url}

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        return "subtitle text"

    monkeypatch.setattr("aimd.tool.url._extract_video_info", _mock_extract_video_info)
    monkeypatch.setattr("aimd.tool.url._extract_subtitles", _mock_extract_subtitles)

    result = await get_text_from_url("https://example.com/video")
    assert result.title == "Example"
    assert call_count["count"] == 1


@pytest.mark.asyncio
async def test_get_text_from_url_cookie_isolation(monkeypatch) -> None:
    seen_cookies: list[str | None] = []

    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        await asyncio.sleep(0.01)
        seen_cookies.append(cookies_file)
        return {"title": url, "webpage_url": url}

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        return "subtitle text"

    monkeypatch.setattr("aimd.tool.url._extract_video_info", _mock_extract_video_info)
    monkeypatch.setattr("aimd.tool.url._extract_subtitles", _mock_extract_subtitles)

    await asyncio.gather(
        get_text_from_url("https://example.com/video-a", cookies_file="a.txt"),
        get_text_from_url("https://example.com/video-b", cookies_file="b.txt"),
    )

    assert sorted(seen_cookies) == ["a.txt", "b.txt"]


def test_parse_cookies_from_browser_variants() -> None:
    assert _parse_cookies_from_browser("chrome") == ("chrome", None, None, None)
    assert _parse_cookies_from_browser("chrome:default") == (
        "chrome",
        "default",
        None,
        None,
    )
    assert _parse_cookies_from_browser("chrome+gnomekeyring:default") == (
        "chrome",
        "default",
        "gnomekeyring",
        None,
    )
    assert _parse_cookies_from_browser("firefox:default::container-1") == (
        "firefox",
        "default",
        None,
        "container-1",
    )


def test_build_cookie_sources_order_and_fallback() -> None:
    sources = _build_cookie_sources(
        platform="bilibili",
        cookies_file="cookies.txt",
        cookies_from_browser="chrome:default",
    )
    assert [s["name"] for s in sources] == [
        "cookiefile",
        "cookiesfrombrowser:chrome:default",
    ]

    default_sources = _build_cookie_sources(
        platform="youtube",
        cookies_file=None,
        cookies_from_browser=None,
    )
    assert [s["name"] for s in default_sources] == [
        "cookiesfrombrowser:chrome:default",
        "cookiesfrombrowser:firefox",
        "no-cookie",
    ]


def test_is_auth_required_error_patterns() -> None:
    assert _is_auth_required_error(RuntimeError("login required for this content"))
    assert _is_auth_required_error(RuntimeError("private playlist -403"))
    assert not _is_auth_required_error(RuntimeError("temporary network timeout"))


@pytest.mark.asyncio
async def test_auth_required_failure_surfaces_cookie_hint(monkeypatch) -> None:
    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        raise ProcessingFailedError(
            "Authenticated cookies are required for this URL. "
            "Provide --cookies (Netscape file) or --cookies-from-browser."
        )

    monkeypatch.setattr("aimd.tool.url._extract_video_info", _mock_extract_video_info)

    with pytest.raises(
        ProcessingFailedError, match="Authenticated cookies are required"
    ):
        await get_text_from_url("https://www.bilibili.com/video/BV1")
