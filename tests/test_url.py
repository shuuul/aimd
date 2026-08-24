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
from aimd.plugins.url.markdown import (
    _format_linear_transcript,
    _merge_rolling_caption_lines,
    strip_subtitle_formatting,
)
from aimd.plugins.url.subtitles import (
    _SUBTITLE_DOWNLOAD_ATTEMPTS,
    _iter_subtitle_urls,
    _json3_to_srt,
    _normalize_subtitle_payload,
    _pick_subtitle_url,
    _srv1_to_srt,
    _srv3_to_srt,
    detect_content_language,
    download_subtitle,
    extract_subtitles,
    get_preferred_languages,
    normalize_metadata_language,
    resolve_subtitle_language,
)
from aimd.plugins.url.ydl import (
    SUBTITLE_SOCKET_TIMEOUT,
    _base_ydl_opts,
    create_info_ydl,
    create_subtitle_ydl,
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
    assert seen_opts[-1]["js_runtimes"] == {"deno": {}, "node": {}}
    assert "logger" in seen_opts[-1]
    assert "listsubtitles" not in seen_opts[-1]


def test_create_subtitle_ydl_uses_extended_socket_timeout(monkeypatch) -> None:
    seen_opts: list[dict] = []

    class _FakeYDL:
        def __init__(self, opts):
            seen_opts.append(opts)

    monkeypatch.setattr("aimd.plugins.url.ydl.impersonation_available", lambda: False)
    monkeypatch.setattr("aimd.plugins.url.ydl.yt_dlp.YoutubeDL", _FakeYDL)

    create_subtitle_ydl(
        platform="youtube",
        cookie_source={"name": "no-cookie", "use_cookies": False},
    )

    assert seen_opts[-1]["socket_timeout"] == SUBTITLE_SOCKET_TIMEOUT
    assert seen_opts[-1]["socket_timeout"] > 20


def test_base_ydl_opts_enables_youtube_js_runtimes(monkeypatch) -> None:
    monkeypatch.setattr("aimd.plugins.url.ydl.impersonation_available", lambda: False)

    youtube_opts = _base_ydl_opts("youtube")
    bilibili_opts = _base_ydl_opts("bilibili")

    assert youtube_opts["js_runtimes"] == {"deno": {}, "node": {}}
    assert "js_runtimes" not in bilibili_opts


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


def test_get_preferred_languages_for_english_puts_en_orig_before_chinese() -> None:
    """YouTube lists zh-Hans before en for English ASR videos such as fgzr3PhzIMk."""
    available = [
        "ab",
        "zh-Hans",
        "zh-Hant",
        "en-orig",
        "en",
        "fr",
    ]
    preferred = get_preferred_languages("en", available)

    assert preferred[:2] == ["en-orig", "en"]
    assert preferred.index("en-orig") < preferred.index("zh-Hans")
    assert preferred.index("en") < preferred.index("zh-Hans")


def test_resolve_subtitle_language_prefers_metadata_language() -> None:
    assert normalize_metadata_language("en-US") == "en"
    assert normalize_metadata_language("zh-CN") == "zh"
    assert (
        resolve_subtitle_language(
            None,
            title="未使用的中文标题足够汉字",
            description="未使用的中文描述足够汉字",
            metadata_language="en",
        )
        == "en"
    )


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("如何用 AIMD 把 YouTube 视频转成 Markdown",), "zh"),
        (("How to convert YouTube videos to Markdown with AIMD",), "en"),
        (
            ("【深度访谈】OpenAI CEO 谈 AGI 的未来", "English transcript available"),
            "zh",
        ),
        (("Deep Dive with OpenAI CEO on the Future of AGI",), "en"),
        (("B站 UP主 推荐：Python 异步编程实战教程",), "zh"),
        (("ChatGPT 使用指南：从入门到精通",), "zh"),
        (("使用 ChatGPT 写代码",), "zh"),
        (("WWDC 2024 Keynote Highlights",), "en"),
        (("", None), None),
        ((None, None), None),
        (("!!",), None),
    ],
)
def test_detect_content_language(
    texts: tuple[str | None, ...], expected: str | None
) -> None:
    assert detect_content_language(*texts) == expected


def test_resolve_subtitle_language_prefers_explicit_over_metadata() -> None:
    assert (
        resolve_subtitle_language(
            "en",
            title="中文标题带有足够汉字",
            description="这是中文描述",
        )
        == "en"
    )
    assert (
        resolve_subtitle_language(
            None,
            title="中文标题带有足够汉字",
            description="English description only as filler text",
        )
        == "zh"
    )


@pytest.mark.asyncio
async def test_extract_subtitles_default_prefers_orig_over_inferred_chinese_title(
    monkeypatch,
) -> None:
    """Default priority keeps *-orig even when title script looks Chinese."""
    seen_urls: list[str] = []

    class _FakeResponse:
        def read(self):
            return b"original english caption"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            seen_urls.append(url)
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "title": "【深度访谈】OpenAI CEO 谈 AGI 的未来",
            "description": "本期节目讨论人工智能。",
            "language": "en",
            "subtitles": {},
            "automatic_captions": {
                "zh-Hans": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=example"
                            "&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"
                        ),
                    }
                ],
                "en-orig": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=example"
                            "&kind=asr&lang=en&fmt=srt"
                        ),
                    }
                ],
            },
        },
        "youtube",
        None,
    )

    assert content == "original english caption"
    assert seen_urls == [
        "https://www.youtube.com/api/timedtext?v=example&kind=asr&lang=en&fmt=srt"
    ]


@pytest.mark.asyncio
async def test_extract_subtitles_explicit_zh_prefers_chinese_over_en_orig(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def read(self):
            return b"chinese translation"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            assert "tlang=zh-Hans" in url
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "title": "Why AI Moats Still Matter (And How They've Changed)",
            "language": "en",
            "automatic_captions": {
                "zh-Hans": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"
                        ),
                    }
                ],
                "en-orig": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&fmt=srt"
                        ),
                    }
                ],
            },
        },
        "youtube",
        "zh",
    )

    assert content == "chinese translation"


@pytest.mark.asyncio
async def test_extract_subtitles_prefers_english_when_title_is_english(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def read(self):
            return b"english subtitle"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            assert url == "https://example.com/en"
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "title": "Deep Dive with OpenAI CEO on the Future of AGI",
            "description": "A conversation about artificial intelligence.",
            "subtitles": {
                "zh-Hans": [{"ext": "srt", "url": "https://example.com/zh"}],
                "en": [{"ext": "srt", "url": "https://example.com/en"}],
            },
        },
        "youtube",
        None,
    )

    assert content == "english subtitle"


@pytest.mark.asyncio
async def test_extract_subtitles_skips_chinese_translations_for_english_youtube_video(
    monkeypatch,
) -> None:
    """Regression for https://www.youtube.com/watch?v=fgzr3PhzIMk.

    YouTube exposes zh-Hans/zh-Hant auto translations before en-orig/en. AIMD must
    still download English source captions for an English-language video.
    """
    seen_urls: list[str] = []

    class _FakeResponse:
        def read(self):
            return b"english source caption"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            seen_urls.append(url)
            assert "tlang=" not in url
            assert "lang=en" in url
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "id": "fgzr3PhzIMk",
            "title": "Why AI Moats Still Matter (And How They've Changed)",
            "description": (
                "a16z General Partners David Haber, Alex Rampell, and Erik "
                "Torenberg discuss why 19 out of 20 AI startups building the "
                "same thing will die."
            ),
            "language": "en",
            "subtitles": {},
            "automatic_captions": {
                "zh-Hans": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"
                        ),
                    }
                ],
                "zh-Hant": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&tlang=zh-Hant&fmt=srt"
                        ),
                    }
                ],
                "en-orig": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&fmt=srt"
                        ),
                    }
                ],
                "en": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&fmt=srt&src=en"
                        ),
                    }
                ],
            },
        },
        "youtube",
        None,
    )

    assert content == "english source caption"
    assert seen_urls == [
        "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk&kind=asr&lang=en&fmt=srt"
    ]


@pytest.mark.asyncio
async def test_extract_subtitles_retries_next_language_when_download_fails(
    monkeypatch,
) -> None:
    attempts: list[str] = []

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            attempts.append(url)
            if "lang=en&fmt=srt" in url and "tlang=" not in url and "src=en" not in url:
                raise TimeoutError("simulated en-orig timeout")
            return _FakeResponse(b"english fallback caption")

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "aimd.plugins.url.subtitles._SUBTITLE_RETRY_BACKOFF_SECONDS",
        (0, 0),
    )

    content = await extract_subtitles(
        {
            "title": "Why AI Moats Still Matter (And How They've Changed)",
            "language": "en",
            "automatic_captions": {
                "zh-Hans": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"
                        ),
                    }
                ],
                "en-orig": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&fmt=srt"
                        ),
                    }
                ],
                "en": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
                            "&kind=asr&lang=en&fmt=srt&src=en"
                        ),
                    }
                ],
            },
        },
        "youtube",
        None,
    )

    assert content == "english fallback caption"
    orig = (
        "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk&kind=asr&lang=en&fmt=srt"
    )
    fallback = (
        "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk"
        "&kind=asr&lang=en&fmt=srt&src=en"
    )
    assert attempts == [orig] * _SUBTITLE_DOWNLOAD_ATTEMPTS + [fallback]


def test_pick_subtitle_url_prefers_json3_over_converted_formats() -> None:
    """YouTube auto captions for Ppyt3MptX5k hang on fmt=srt; prefer json3."""
    entries = [
        {"ext": "json3", "url": "https://example.com/en?fmt=json3"},
        {"ext": "srv1", "url": "https://example.com/en?fmt=srv1"},
        {"ext": "srt", "url": "https://example.com/en?fmt=srt"},
        {"ext": "vtt", "url": "https://example.com/en?fmt=vtt"},
        {"ext": "ttml", "url": "https://example.com/en?fmt=ttml"},
        {"ext": "srv2", "url": "https://example.com/en?fmt=srv2"},
        {"ext": "srv3", "url": "https://example.com/en?fmt=srv3"},
    ]

    assert _pick_subtitle_url(entries) == "https://example.com/en?fmt=json3"
    assert _iter_subtitle_urls(entries) == [
        "https://example.com/en?fmt=json3",
        "https://example.com/en?fmt=ttml",
        "https://example.com/en?fmt=vtt",
        "https://example.com/en?fmt=srv1",
        "https://example.com/en?fmt=srt",
    ]


def test_json3_to_srt_converts_events_with_segs() -> None:
    payload = """
    {
      "wireMagic": "pb3",
      "events": [
        {"tStartMs": 0, "dDurationMs": 0, "id": 1},
        {
          "tStartMs": 80,
          "dDurationMs": 3200,
          "segs": [
            {"utf8": "you've "},
            {"utf8": "discovered", "tOffsetMs": 400}
          ]
        },
        {"tStartMs": 1360, "segs": [{"utf8": "\\n"}]},
        {
          "tStartMs": 3280,
          "dDurationMs": 2000,
          "segs": [{"utf8": "the brain"}]
        }
      ]
    }
    """
    srt = _json3_to_srt(payload)
    assert srt is not None
    assert "00:00:00,080 --> 00:00:03,280" in srt
    assert "you've discovered" in srt
    assert "00:00:03,280 --> 00:00:05,280" in srt
    assert "the brain" in srt
    assert srt.strip().startswith("1\n")
    assert strip_subtitle_formatting(srt) == "you've discovered the brain"


def test_srv1_to_srt_converts_transcript_text_nodes() -> None:
    payload = """<?xml version="1.0" encoding="utf-8" ?>
<transcript>
  <text start="0.08" dur="3.2">you've discovered</text>
  <text start="3.28" dur="2">the brain</text>
</transcript>
"""
    srt = _srv1_to_srt(payload)
    assert srt is not None
    assert "00:00:00,080 --> 00:00:03,280" in srt
    assert "you've discovered" in srt
    assert "the brain" in srt


def test_srv3_to_srt_converts_timedtext_paragraphs() -> None:
    payload = """<?xml version="1.0" encoding="utf-8" ?>
<timedtext format="3">
  <body>
    <p t="80" d="3200">you've discovered</p>
    <p t="3280" d="2000"><s>the </s><s t="400">brain</s></p>
  </body>
</timedtext>
"""
    srt = _srv3_to_srt(payload)
    assert srt is not None
    assert "00:00:00,080 --> 00:00:03,280" in srt
    assert "you've discovered" in srt
    assert "the brain" in srt


def test_normalize_subtitle_payload_leaves_srt_vtt_ttml() -> None:
    srt = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"
    ttml = (
        '<?xml version="1.0"?><tt xmlns="http://www.w3.org/ns/ttml"><p>hello</p></tt>'
    )
    assert _normalize_subtitle_payload(srt) == srt.strip()
    assert _normalize_subtitle_payload(vtt) == vtt.strip()
    assert _normalize_subtitle_payload(ttml) == ttml.strip()


def test_normalize_subtitle_payload_rejects_empty_json3() -> None:
    payload = '{"wireMagic": "pb3", "events": [{"tStartMs": 0}]}'
    assert _normalize_subtitle_payload(payload) is None


def test_merge_rolling_caption_lines_drops_youtube_asr_overlap() -> None:
    assert (
        _merge_rolling_caption_lines(
            [
                "you've discovered something about",
                "discovered something about the brain",
            ]
        )
        == "you've discovered something about the brain"
    )


def test_format_linear_transcript_splits_sentences_and_speaker_turns() -> None:
    tokens = (
        "you've discovered something about the brain I think is quite "
        "surprising. >> So we found that in fathers the brain is actually "
        "shrinking. Dr. Darby found that. >> Yeah, go for it."
    ).split()
    assert _format_linear_transcript(tokens) == (
        "you've discovered something about the brain I think is quite "
        "surprising.\n"
        ">> So we found that in fathers the brain is actually shrinking.\n"
        "Dr. Darby found that.\n"
        ">> Yeah, go for it."
    )


def test_strip_subtitle_formatting_uses_sentence_lines_for_interviews() -> None:
    srt = """1
00:00:00,080 --> 00:00:03,280
you've discovered something surprising.

2
00:00:03,280 --> 00:00:06,000
>> So we found that the brain is shrinking.
"""
    assert strip_subtitle_formatting(srt) == (
        "you've discovered something surprising.\n"
        ">> So we found that the brain is shrinking."
    )


def test_stripped_markdown_body_is_stable_across_subtitle_formats() -> None:
    """Default Markdown body should not depend on which timedtext format won."""
    expected = "you've discovered something about the brain"

    json3 = """
    {
      "wireMagic": "pb3",
      "events": [
        {
          "tStartMs": 80,
          "dDurationMs": 2000,
          "segs": [{"utf8": "you've "}, {"utf8": "discovered"}]
        },
        {
          "tStartMs": 480,
          "dDurationMs": 2500,
          "segs": [
            {"utf8": "you've "},
            {"utf8": "discovered "},
            {"utf8": "something"}
          ]
        },
        {
          "tStartMs": 880,
          "dDurationMs": 2500,
          "segs": [
            {"utf8": "discovered "},
            {"utf8": "something "},
            {"utf8": "about"}
          ]
        },
        {
          "tStartMs": 1680,
          "dDurationMs": 2500,
          "segs": [
            {"utf8": "something "},
            {"utf8": "about "},
            {"utf8": "the "},
            {"utf8": "brain"}
          ]
        }
      ]
    }
    """
    srt = """1
00:00:00,080 --> 00:00:03,280
you've discovered something about

2
00:00:01,360 --> 00:00:05,840
something about the brain
"""
    vtt = """WEBVTT

00:00:00.080 --> 00:00:03.280
you've discovered something about

00:00:01.360 --> 00:00:05.840
something about the brain
"""
    ttml = """<?xml version="1.0" encoding="utf-8" ?>
<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml">
  <body>
    <div>
      <p begin="00:00:00.080" end="00:00:03.280">you've discovered something about</p>
      <p begin="00:00:01.360" end="00:00:05.840">something about the brain</p>
    </div>
  </body>
</tt>
"""
    srv1 = """<?xml version="1.0" encoding="utf-8" ?>
<transcript>
  <text start="0.08" dur="3.2">you've discovered something about</text>
  <text start="1.36" dur="4.48">something about the brain</text>
</transcript>
"""
    srv3 = """<?xml version="1.0" encoding="utf-8" ?>
<timedtext format="3">
  <body>
    <p t="80" d="3200">you've discovered something about</p>
    <p t="1360" d="4480">something about the brain</p>
  </body>
</timedtext>
"""

    json3_srt = _normalize_subtitle_payload(json3)
    srv1_srt = _normalize_subtitle_payload(srv1)
    srv3_srt = _normalize_subtitle_payload(srv3)
    assert json3_srt is not None
    assert srv1_srt is not None
    assert srv3_srt is not None
    assert strip_subtitle_formatting(json3_srt) == expected
    assert strip_subtitle_formatting(srt) == expected
    assert strip_subtitle_formatting(vtt) == expected
    assert strip_subtitle_formatting(ttml) == expected
    assert strip_subtitle_formatting(srv1_srt) == expected
    assert strip_subtitle_formatting(srv3_srt) == expected


@pytest.mark.asyncio
async def test_extract_subtitles_prefers_json3_for_youtube_auto_captions(
    monkeypatch,
) -> None:
    """Regression for https://www.youtube.com/watch?v=Ppyt3MptX5k.

    Auto captions exist (en-orig) but YouTube's timedtext ``fmt=srt`` often
    times out. AIMD must download native json3 instead of walking translations.
    """
    attempts: list[str] = []

    class _FakeResponse:
        def read(self):
            return b"english original caption"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            attempts.append(url)
            if "fmt=srt" in url:
                raise TimeoutError("simulated youtube srt hang")
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "id": "Ppyt3MptX5k",
            "title": (
                "The Scientist Who Scans Fathers' Brains: "
                "Parenthood Shrinks Your Brain (And That's GOOD!)"
            ),
            "language": "en",
            "subtitles": {},
            "automatic_captions": {
                "zh-Hans": [
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k"
                            "&kind=asr&lang=en&tlang=zh-Hans&fmt=srt"
                        ),
                    }
                ],
                "en-orig": [
                    {
                        "ext": "json3",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k"
                            "&kind=asr&lang=en&fmt=json3"
                        ),
                    },
                    {
                        "ext": "srt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k"
                            "&kind=asr&lang=en&fmt=srt"
                        ),
                    },
                    {
                        "ext": "vtt",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k"
                            "&kind=asr&lang=en&fmt=vtt"
                        ),
                    },
                    {
                        "ext": "ttml",
                        "url": (
                            "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k"
                            "&kind=asr&lang=en&fmt=ttml"
                        ),
                    },
                ],
            },
        },
        "youtube",
        None,
    )

    assert content == "english original caption"
    assert attempts == [
        "https://www.youtube.com/api/timedtext?v=Ppyt3MptX5k&kind=asr&lang=en&fmt=json3"
    ]


@pytest.mark.asyncio
async def test_extract_subtitles_retries_next_format_when_ttml_download_fails(
    monkeypatch,
) -> None:
    attempts: list[str] = []

    class _FakeResponse:
        def read(self):
            return b"vtt caption"

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            attempts.append(url)
            if "fmt=ttml" in url:
                raise TimeoutError("simulated ttml timeout")
            if "fmt=vtt" in url:
                return _FakeResponse()
            raise AssertionError(f"unexpected subtitle url: {url}")

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "aimd.plugins.url.subtitles._SUBTITLE_RETRY_BACKOFF_SECONDS",
        (0, 0),
    )

    content = await extract_subtitles(
        {
            "title": "English talk",
            "language": "en",
            "automatic_captions": {
                "en-orig": [
                    {
                        "ext": "ttml",
                        "url": "https://example.com/en?fmt=ttml",
                    },
                    {
                        "ext": "vtt",
                        "url": "https://example.com/en?fmt=vtt",
                    },
                    {
                        "ext": "srt",
                        "url": "https://example.com/en?fmt=srt",
                    },
                ]
            },
        },
        "youtube",
        None,
    )

    assert content == "vtt caption"
    assert attempts == (
        ["https://example.com/en?fmt=ttml"] * _SUBTITLE_DOWNLOAD_ATTEMPTS
        + ["https://example.com/en?fmt=vtt"]
    )


@pytest.mark.asyncio
async def test_extract_subtitles_converts_json3_payload_to_srt(
    monkeypatch,
) -> None:
    class _FakeResponse:
        def read(self):
            return b"""{
              "wireMagic": "pb3",
              "events": [
                {
                  "tStartMs": 80,
                  "dDurationMs": 3200,
                  "segs": [{"utf8": "you've discovered"}]
                }
              ]
            }"""

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            assert "fmt=json3" in url
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "title": "English talk",
            "language": "en",
            "automatic_captions": {
                "en-orig": [
                    {
                        "ext": "json3",
                        "url": "https://example.com/en?fmt=json3",
                    }
                ]
            },
        },
        "youtube",
        None,
    )

    assert content is not None
    assert "00:00:00,080 --> 00:00:03,280" in content
    assert "you've discovered" in content


@pytest.mark.asyncio
async def test_extract_subtitles_skips_empty_json3_and_uses_ttml(
    monkeypatch,
) -> None:
    attempts: list[str] = []

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

    class _FakeYDL:
        def __init__(self, *, cookie_source):  # noqa: ANN001, ARG002
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ANN001
            attempts.append(url)
            if "fmt=json3" in url:
                return _FakeResponse(b'{"wireMagic": "pb3", "events": []}')
            if "fmt=ttml" in url:
                return _FakeResponse(b"ttml caption")
            raise AssertionError(f"unexpected subtitle url: {url}")

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(cookie_source=cookie_source),  # noqa: ARG005
    )

    content = await extract_subtitles(
        {
            "title": "English talk",
            "language": "en",
            "automatic_captions": {
                "en-orig": [
                    {
                        "ext": "json3",
                        "url": "https://example.com/en?fmt=json3",
                    },
                    {
                        "ext": "ttml",
                        "url": "https://example.com/en?fmt=ttml",
                    },
                ]
            },
        },
        "youtube",
        None,
    )

    assert content == "ttml caption"
    assert attempts == [
        "https://example.com/en?fmt=json3",
        "https://example.com/en?fmt=ttml",
    ]


@pytest.mark.asyncio
async def test_download_subtitle_retries_transient_timeout_then_succeeds(
    monkeypatch,
) -> None:
    attempts = {"count": 0}

    class _FakeResponse:
        def read(self):
            return b"recovered subtitle"

    class _FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ARG002
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise TimeoutError("simulated timedtext hang")
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "aimd.plugins.url.subtitles._SUBTITLE_RETRY_BACKOFF_SECONDS",
        (0, 0),
    )

    content = await download_subtitle(
        "https://example.com/en?fmt=srt",
        "youtube",
    )

    assert content == "recovered subtitle"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_download_subtitle_retries_http_502_then_succeeds(monkeypatch) -> None:
    from types import SimpleNamespace

    from yt_dlp.networking.exceptions import HTTPError

    attempts = {"count": 0}

    class _FakeResponse:
        def read(self):
            return b"recovered after 502"

    class _FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ARG002
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise HTTPError(
                    SimpleNamespace(
                        status=502, reason="Bad Gateway", close=lambda: None
                    )
                )
            return _FakeResponse()

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "aimd.plugins.url.subtitles._SUBTITLE_RETRY_BACKOFF_SECONDS",
        (0, 0),
    )

    content = await download_subtitle(
        "https://example.com/en?fmt=srt",
        "youtube",
    )

    assert content == "recovered after 502"
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_download_subtitle_gives_up_after_retryable_failures(
    monkeypatch,
) -> None:
    attempts = {"count": 0}

    class _FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ARG002
            attempts["count"] += 1
            raise TimeoutError("still hanging")

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "aimd.plugins.url.subtitles._SUBTITLE_RETRY_BACKOFF_SECONDS",
        (0, 0),
    )

    content = await download_subtitle(
        "https://example.com/en?fmt=srt",
        "youtube",
    )

    assert content is None
    assert attempts["count"] == _SUBTITLE_DOWNLOAD_ATTEMPTS


@pytest.mark.asyncio
async def test_download_subtitle_does_not_retry_http_404(monkeypatch) -> None:
    from types import SimpleNamespace

    from yt_dlp.networking.exceptions import HTTPError

    attempts = {"count": 0}

    class _FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ARG002
            return False

        def urlopen(self, url):  # noqa: ARG002
            attempts["count"] += 1
            raise HTTPError(
                SimpleNamespace(status=404, reason="Not Found", close=lambda: None)
            )

    monkeypatch.setattr(
        "aimd.plugins.url.subtitles.create_subtitle_ydl",
        lambda *, platform, cookie_source: _FakeYDL(),  # noqa: ARG005
    )

    content = await download_subtitle(
        "https://example.com/en?fmt=srt",
        "youtube",
    )

    assert content is None
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_get_text_from_url_passes_inferred_language_to_audio(
    monkeypatch,
) -> None:
    seen: dict[str, str | None] = {}

    async def _mock_extract_video_info(
        *,
        url: str,
        platform: str,
        cookies_file: str | None,
        cookies_from_browser: str | None,
    ):
        return {
            "title": "中文播客：聊聊创业与产品",
            "description": "今天和嘉宾讨论创业经验。",
            "webpage_url": url,
        }

    async def _mock_extract_subtitles(info_dict, platform: str, language: str | None):
        seen["subtitle_language"] = language
        return None

    async def _mock_extract_content_from_audio(**kwargs):  # noqa: ANN003
        seen["audio_language"] = kwargs["language"]
        return "transcribed chinese audio"

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

    result = await get_text_from_url("https://example.com/episode")

    assert seen["subtitle_language"] is None
    assert seen["audio_language"] == "zh"
    assert "transcribed chinese audio" in result.markdown


def test_get_preferred_languages_default_prioritizes_orig_tracks() -> None:
    preferred = get_preferred_languages(
        None, ["zh-Hans", "zh-Hant", "en-orig", "en", "fr-orig"]
    )

    assert preferred[0] == "en-orig"
    assert preferred[1] == "fr-orig"
    assert preferred.index("en-orig") < preferred.index("zh-Hans")
    assert preferred.index("en") < preferred.index("zh-Hans")


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
