import asyncio

import pytest

from aimd.tool.url import get_text_from_url


@pytest.mark.asyncio
async def test_get_text_from_url_extracts_info_once(monkeypatch) -> None:
    call_count = {"count": 0}

    async def _mock_extract_video_info(
        url: str, platform: str, cookies_file: str | None
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
        url: str, platform: str, cookies_file: str | None
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
