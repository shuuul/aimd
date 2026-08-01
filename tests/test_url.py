import asyncio
from pathlib import Path

import pytest

from aimd.core.errors import (
    BackendUnavailableError,
    ProcessingFailedError,
    UnsupportedInputError,
)
from aimd.plugins.url.audio import (
    _download_and_transcribe_audio,
    _try_download_with_format,
    download_audio,
    extract_content_from_audio,
)
from aimd.plugins.url.cookies import (
    build_cookie_sources,
    is_auth_required_error,
    parse_cookies_from_browser,
)
from aimd.plugins.url.metadata import extract_video_info
from aimd.plugins.url._plugin import get_text_from_url
from aimd.plugins.url.subtitles import extract_subtitles, get_preferred_languages
from aimd.plugins.url.ydl import create_info_ydl


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

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_subtitles",
        _mock_extract_subtitles,
    )

    result = await get_text_from_url("https://example.com/video")
    assert result.title == "Example"
    assert result.platform == "unknown"
    assert "**Platform:** unknown" in result.markdown
    assert call_count["count"] == 1


@pytest.mark.asyncio
async def test_get_text_from_url_includes_detected_platform_in_output(
    monkeypatch,
) -> None:
    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        return {
            "title": "Example",
            "channel": "Demo Channel",
            "webpage_url": url,
            "duration": 12,
            "upload_date": "20260307",
            "view_count": 34,
            "description": "desc",
        }

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        return "subtitle text"

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_subtitles",
        _mock_extract_subtitles,
    )

    result = await get_text_from_url("https://www.youtube.com/watch?v=test")

    assert result.platform == "youtube"
    assert "**Platform:** youtube" in result.markdown


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

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_subtitles",
        _mock_extract_subtitles,
    )

    await asyncio.gather(
        get_text_from_url("https://example.com/video-a", cookies_file="a.txt"),
        get_text_from_url("https://example.com/video-b", cookies_file="b.txt"),
    )

    assert sorted(seen_cookies) == ["a.txt", "b.txt"]


def test_parse_cookies_from_browser_variants() -> None:
    assert parse_cookies_from_browser("chrome") == ("chrome", None, None, None)
    assert parse_cookies_from_browser("chrome:default") == (
        "chrome",
        "default",
        None,
        None,
    )
    assert parse_cookies_from_browser("chrome+gnomekeyring:default") == (
        "chrome",
        "default",
        "gnomekeyring",
        None,
    )
    assert parse_cookies_from_browser("firefox:default::container-1") == (
        "firefox",
        "default",
        None,
        "container-1",
    )


def test_build_cookie_sources_order_and_fallback() -> None:
    sources = build_cookie_sources(
        platform="bilibili",
        cookies_file="cookies.txt",
        cookies_from_browser="chrome:default",
    )
    assert [s["name"] for s in sources] == [
        "cookiefile",
        "cookiesfrombrowser:chrome:default",
    ]

    default_sources = build_cookie_sources(
        platform="youtube",
        cookies_file=None,
        cookies_from_browser=None,
    )
    assert [s["name"] for s in default_sources] == [
        "cookiesfrombrowser:chrome:default",
        "cookiesfrombrowser:chromium",
        "cookiesfrombrowser:brave",
        "cookiesfrombrowser:edge",
        "cookiesfrombrowser:vivaldi",
        "cookiesfrombrowser:opera",
        "cookiesfrombrowser:firefox",
        "cookiesfrombrowser:safari",
        "cookiesfrombrowser:whale",
        "no-cookie",
    ]


def test_create_info_ydl_does_not_print_subtitle_listing(monkeypatch) -> None:
    seen_opts: list[dict] = []

    class _FakeYDL:
        def __init__(self, opts):
            seen_opts.append(opts)

    monkeypatch.setattr("aimd.plugins.url.ydl.impersonation_available", lambda: False)
    monkeypatch.setattr("aimd.plugins.url.ydl.yt_dlp.YoutubeDL", _FakeYDL)

    create_info_ydl(
        platform="youtube",
        cookie_source={"name": "no-cookie", "use_cookies": False},
    )

    assert seen_opts[-1]["writeautomaticsub"] is True
    assert seen_opts[-1]["writesubtitles"] is True
    assert seen_opts[-1]["skip_download"] is True
    assert seen_opts[-1]["ignoreconfig"] is True
    assert "logger" in seen_opts[-1]
    assert "listsubtitles" not in seen_opts[-1]


@pytest.mark.asyncio
async def test_extract_subtitles_uses_metadata_cookie_source(monkeypatch) -> None:
    seen_sources: list[dict] = []

    class _FakeResponse:
        def read(self):
            return b"subtitle text"

    class _FakeYDL:
        def __init__(self, *, cookie_source):
            seen_sources.append(cookie_source)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def urlopen(self, url):  # noqa: ARG002
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )
    cookie_source = {
        "name": "cookiesfrombrowser:chrome:default",
        "use_cookies": True,
        "cookiefile": None,
        "cookiesfrombrowser": ("chrome", "default", None, None),
    }

    content = await extract_subtitles(
        {
            "automatic_captions": {
                "en-orig": [{"ext": "srt", "url": "https://example.com/subs"}]
            },
            "_aimd_cookie_source": cookie_source,
        },
        "youtube",
        None,
    )

    assert content == "subtitle text"
    assert seen_sources == [cookie_source]


def test_explicit_invalid_cookies_from_browser_fails_fast() -> None:
    with pytest.raises(
        UnsupportedInputError, match="cookies_from_browser cannot be empty"
    ):
        build_cookie_sources(
            platform="youtube",
            cookies_file=None,
            cookies_from_browser="   ",
        )


def test_is_auth_required_error_patterns() -> None:
    assert is_auth_required_error(RuntimeError("login required for this content"))
    assert is_auth_required_error(RuntimeError("private playlist -403"))
    assert is_auth_required_error(RuntimeError("HTTP Error 412: Precondition Failed"))
    assert not is_auth_required_error(RuntimeError("temporary network timeout"))


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

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )

    with pytest.raises(
        ProcessingFailedError, match="Authenticated cookies are required"
    ):
        await get_text_from_url("https://www.bilibili.com/video/BV1")


@pytest.mark.asyncio
async def test_extract_video_info_surfaces_cookie_hint_after_bilibili_412(
    monkeypatch,
) -> None:
    class _FakeYDL:
        def __init__(self, source_name: str) -> None:
            self.source_name = source_name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url: str, download: bool = False):  # noqa: ARG002
            if self.source_name == "cookiesfrombrowser:chrome:default":
                raise RuntimeError("HTTP Error 412: Precondition Failed")
            raise RuntimeError("could not find firefox cookies database")

    monkeypatch.setattr(
        "aimd.plugins.url.metadata.create_info_ydl",
        lambda *, platform, cookie_source: _FakeYDL(  # noqa: ARG005
            cookie_source["name"]
        ),
    )

    with pytest.raises(
        ProcessingFailedError, match="Authenticated cookies are required"
    ):
        await extract_video_info(
            url="https://www.bilibili.com/video/BV1iVoVBgERD/",
            platform="bilibili",
            cookies_file=None,
            cookies_from_browser=None,
        )


@pytest.mark.asyncio
async def test_download_audio_prefers_audio_only_for_youtube(
    monkeypatch, tmp_path: Path
) -> None:
    attempts: list[tuple[str, str | None]] = []

    async def _mock_try_download_with_format(**kwargs):
        attempts.append((kwargs["format_selector"], kwargs["preferred_codec"]))
        out = tmp_path / "audio.webm"
        out.write_text("x", encoding="utf-8")
        return out

    monkeypatch.setattr(
        "aimd.plugins.url.audio._try_download_with_format",
        _mock_try_download_with_format,
    )

    out = await download_audio(
        info_dict={"title": "t", "id": "id"},
        url="https://www.youtube.com/watch?v=test",
        download_path=tmp_path,
    )

    assert out is not None
    assert attempts
    assert attempts[0] == (
        "bestaudio[acodec^=opus][abr<=128]/bestaudio[abr<=128]/bestaudio",
        None,
    )


@pytest.mark.asyncio
async def test_download_audio_prefers_audio_only_for_bilibili(
    monkeypatch, tmp_path: Path
) -> None:
    attempts: list[tuple[str, str | None]] = []

    async def _mock_try_download_with_format(**kwargs):
        attempts.append((kwargs["format_selector"], kwargs["preferred_codec"]))
        out = tmp_path / "audio.m4a"
        out.write_text("x", encoding="utf-8")
        return out

    monkeypatch.setattr(
        "aimd.plugins.url.audio._try_download_with_format",
        _mock_try_download_with_format,
    )

    out = await download_audio(
        info_dict={"title": "t", "id": "id"},
        url="https://www.bilibili.com/video/BV1BDcKziEqy",
        download_path=tmp_path,
    )

    assert out is not None
    assert attempts
    assert attempts[0] == ("bestaudio", None)


@pytest.mark.asyncio
async def test_download_with_format_only_adds_postprocessor_when_codec_requested(
    monkeypatch, tmp_path: Path
) -> None:
    seen_opts: list[dict] = []

    class _FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            seen_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):  # noqa: ARG002
            Path(f"{self.opts['outtmpl']}.webm").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "aimd.plugins.url.audio.build_cookie_sources",
        lambda **kwargs: [{"name": "no-cookie", "use_cookies": False}],
    )
    monkeypatch.setattr(
        "aimd.plugins.url.ydl.impersonation_available",
        lambda: False,
    )
    monkeypatch.setattr("aimd.plugins.url.audio.yt_dlp.YoutubeDL", _FakeYDL)

    result_no_codec = await _try_download_with_format(
        url="https://example.com/video",
        download_path=tmp_path,
        audio_filename="sample_audio",
        format_selector="bestaudio",
        preferred_codec=None,
        platform="youtube",
        cookies_file=None,
        cookies_from_browser=None,
    )
    assert result_no_codec is not None
    assert "postprocessors" not in seen_opts[-1]
    assert seen_opts[-1]["ignoreconfig"] is True
    assert "logger" in seen_opts[-1]

    result_with_codec = await _try_download_with_format(
        url="https://example.com/video",
        download_path=tmp_path,
        audio_filename="sample_audio_2",
        format_selector="bestaudio[ext=m4a]/bestaudio",
        preferred_codec="m4a",
        platform="youtube",
        cookies_file=None,
        cookies_from_browser=None,
    )
    assert result_with_codec is not None
    assert seen_opts[-1]["postprocessors"][0]["preferredcodec"] == "m4a"


@pytest.mark.asyncio
async def test_download_with_format_applies_youtube_cookies(
    monkeypatch, tmp_path: Path
) -> None:
    seen_opts: list[dict] = []

    class _FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            seen_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):  # noqa: ARG002
            Path(f"{self.opts['outtmpl']}.webm").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "aimd.plugins.url.audio.build_cookie_sources",
        lambda **kwargs: [  # noqa: ARG005
            {
                "name": "cookiesfrombrowser:chrome:default",
                "use_cookies": True,
                "cookiefile": None,
                "cookiesfrombrowser": ("chrome", "default", None, None),
            }
        ],
    )
    monkeypatch.setattr(
        "aimd.plugins.url.ydl.impersonation_available",
        lambda: False,
    )
    monkeypatch.setattr("aimd.plugins.url.audio.yt_dlp.YoutubeDL", _FakeYDL)

    result = await _try_download_with_format(
        url="https://www.youtube.com/watch?v=test",
        download_path=tmp_path,
        audio_filename="sample_audio",
        format_selector="bestaudio",
        preferred_codec=None,
        platform="youtube",
        cookies_file=None,
        cookies_from_browser=None,
    )

    assert result is not None
    assert seen_opts[-1]["cookiesfrombrowser"] == ("chrome", "default", None, None)


@pytest.mark.asyncio
async def test_download_with_format_surfaces_cookie_hint_after_bilibili_412(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):  # noqa: ARG002
            browser_name = self.opts["cookiesfrombrowser"][0]
            if browser_name == "chrome":
                raise RuntimeError("HTTP Error 412: Precondition Failed")
            raise RuntimeError("could not find firefox cookies database")

    monkeypatch.setattr(
        "aimd.plugins.url.audio.build_cookie_sources",
        lambda **kwargs: [  # noqa: ARG005
            {
                "name": "cookiesfrombrowser:chrome:default",
                "use_cookies": True,
                "cookiefile": None,
                "cookiesfrombrowser": ("chrome", "default", None, None),
            },
            {
                "name": "cookiesfrombrowser:firefox",
                "use_cookies": True,
                "cookiefile": None,
                "cookiesfrombrowser": ("firefox", None, None, None),
            },
        ],
    )
    monkeypatch.setattr(
        "aimd.plugins.url.ydl.impersonation_available",
        lambda: False,
    )
    monkeypatch.setattr("aimd.plugins.url.audio.yt_dlp.YoutubeDL", _FakeYDL)

    with pytest.raises(
        ProcessingFailedError, match="Authenticated cookies are required"
    ):
        await _try_download_with_format(
            url="https://www.bilibili.com/video/BV1iVoVBgERD/",
            download_path=tmp_path,
            audio_filename="sample_audio",
            format_selector="bestaudio",
            preferred_codec=None,
            platform="bilibili",
            cookies_file=None,
            cookies_from_browser=None,
        )


@pytest.mark.asyncio
async def test_url_audio_preserves_processing_failed_error(monkeypatch) -> None:
    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        return {"title": "Example", "webpage_url": url}

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        return None

    async def _mock_extract_content_from_audio(**kwargs):
        raise ProcessingFailedError(
            "Authenticated cookies are required for this download."
        )

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_subtitles",
        _mock_extract_subtitles,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_content_from_audio",
        _mock_extract_content_from_audio,
    )

    with pytest.raises(
        ProcessingFailedError, match="Authenticated cookies are required"
    ):
        await get_text_from_url("https://www.bilibili.com/video/BV1")


@pytest.mark.asyncio
async def test_url_audio_path_preserves_backend_unavailable_error(
    monkeypatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_text("x", encoding="utf-8")

    async def _mock_download_audio(**kwargs):  # noqa: ARG001
        return audio_path

    async def _mock_transcribe_file(*args, **kwargs):  # noqa: ANN002, ARG001
        raise BackendUnavailableError("no usable ASR backend")

    monkeypatch.setattr("aimd.plugins.url.audio.download_audio", _mock_download_audio)
    monkeypatch.setattr("aimd.plugins.url.audio.transcribe_file", _mock_transcribe_file)

    with pytest.raises(BackendUnavailableError, match="no usable ASR backend"):
        await _download_and_transcribe_audio(
            info_dict={"title": "Example", "id": "id1"},
            url="https://www.youtube.com/watch?v=test",
            download_path=tmp_path,
            language=None,
            model=None,
            save_original_path=None,
            cookies_file=None,
            cookies_from_browser=None,
            temp_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_extract_content_from_audio_soft_falls_back_on_generic_error(
    monkeypatch, tmp_path: Path
) -> None:
    async def _mock_download_audio(**kwargs):  # noqa: ARG001
        raise RuntimeError("network blip")

    monkeypatch.setattr("aimd.plugins.url.audio.download_audio", _mock_download_audio)

    result = await extract_content_from_audio(
        info_dict={"title": "Example", "id": "id1"},
        url="https://www.youtube.com/watch?v=test",
        language=None,
        temp_dir=tmp_path,
    )
    assert result is None


@pytest.mark.asyncio
async def test_download_audio_preserves_domain_error_from_format_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    async def _raise_domain_error(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ProcessingFailedError("authenticated cookies are required")

    monkeypatch.setattr(
        "aimd.plugins.url.audio._try_download_with_format", _raise_domain_error
    )

    with pytest.raises(
        ProcessingFailedError, match="authenticated cookies are required"
    ):
        await download_audio(
            info_dict={"title": "Example", "id": "id1"},
            url="https://www.bilibili.com/video/BV1",
            download_path=tmp_path,
        )


def test_get_preferred_languages_prefers_original_when_language_unspecified() -> None:
    preferred = get_preferred_languages(
        None, ["zh-Hans", "en-orig", "fr-orig", "en", "zh"]
    )

    assert preferred[:5] == ["en-orig", "fr-orig", "en", "zh-Hans", "zh"]


def test_get_preferred_languages_prefers_original_for_orig_alias() -> None:
    preferred = get_preferred_languages("orig", ["zh-Hans", "en-orig", "en"])

    assert preferred[:3] == ["en-orig", "en", "zh-Hans"]


@pytest.mark.asyncio
async def test_get_text_from_url_forwards_precision_to_audio_fallback(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        return {"title": "Example", "webpage_url": url}

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        return None

    async def _mock_extract_content_from_audio(**kwargs):
        captured.update(kwargs)
        return "audio transcript body"

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_video_info",
        _mock_extract_video_info,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_subtitles",
        _mock_extract_subtitles,
    )
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_content_from_audio",
        _mock_extract_content_from_audio,
    )

    result = await get_text_from_url(
        "https://example.com/video",
        model="qwen3-asr-1.7b",
        precision="8bit",
    )

    assert "audio transcript body" in result.markdown
    assert captured["model"] == "qwen3-asr-1.7b"
    assert captured["precision"] == "8bit"


@pytest.mark.asyncio
async def test_extract_content_from_audio_forwards_precision_to_transcribe(
    monkeypatch, tmp_path: Path
) -> None:
    audio_file = tmp_path / "audio_id1.m4a"
    audio_file.write_text("x", encoding="utf-8")
    captured: dict[str, object] = {}

    async def _mock_download_audio(**kwargs):  # noqa: ARG001
        return audio_file

    async def _mock_transcribe_file(file_path, **kwargs):
        captured["file_path"] = file_path
        captured.update(kwargs)
        return "transcribed audio text"

    monkeypatch.setattr("aimd.plugins.url.audio.download_audio", _mock_download_audio)
    monkeypatch.setattr("aimd.plugins.url.audio.transcribe_file", _mock_transcribe_file)

    result = await extract_content_from_audio(
        info_dict={"title": "Example", "id": "id1"},
        url="https://www.youtube.com/watch?v=test",
        language=None,
        model="qwen3-asr-0.6b",
        temp_dir=tmp_path,
        precision="6bit",
    )

    assert result == "transcribed audio text"
    assert captured["file_path"] == audio_file
    assert captured["model"] == "qwen3-asr-0.6b"
    assert captured["precision"] == "6bit"
