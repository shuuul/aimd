from importlib.metadata import entry_points
import io
from pathlib import Path

from markitdown import MarkItDown, StreamInfo

from aimd.core.models import TextContext
from aimd.plugins.url.processor import UrlTextResult


def test_aimd_markitdown_plugins_are_discoverable() -> None:
    plugin_names = {
        entry_point.name for entry_point in entry_points(group="markitdown.plugin")
    }

    assert "asr" in plugin_names
    assert "doc" in plugin_names
    assert "url" in plugin_names
    assert "ocr" in plugin_names


def test_asr_plugin_converts_audio_with_markitdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake audio")

    def _transcribe_file_sync(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args[0] == audio
        assert kwargs["model"] == "tiny"
        assert kwargs["language"] == "zh"
        assert kwargs["temp_dir"] == tmp_path
        return "transcript text"

    monkeypatch.setattr(
        "aimd.plugins.asr._plugin.transcribe_file_sync", _transcribe_file_sync
    )

    result = MarkItDown(enable_plugins=True).convert(
        audio,
        model="tiny",
        language="zh",
        temp_dir=tmp_path,
    )

    assert result.title == "voice"
    assert result.markdown == "transcript text"


def test_ocr_plugin_converts_when_markitdown_requests_ocr(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"fake image")

    def _process_ocr_sync(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args[0] == image
        assert kwargs["model"] == "tiny"
        assert kwargs["language"] == "zh"
        assert kwargs["start"] == 0
        assert kwargs["end"] == 1
        assert kwargs["temp_dir"] == tmp_path
        return TextContext(title="page", chunk_list=["ocr text"])

    monkeypatch.setattr("aimd.plugins.ocr._plugin.process_ocr_sync", _process_ocr_sync)

    result = MarkItDown(enable_plugins=True).convert(
        image,
        task_type="ocr",
        model="tiny",
        language="zh",
        start=0,
        end=1,
        temp_dir=tmp_path,
    )

    assert result.title == "page"
    assert result.markdown == "ocr text"


def test_doc_plugin_converts_pandoc_document_with_markitdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    document = tmp_path / "paper.rst"
    document.write_text("Title\n=====\n", encoding="utf-8")

    class _Result:
        title = "Paper"
        markdown = "# Paper\n\nBody"

    def _process_doc_with_assets(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args[0] == document
        assert kwargs["temp_dir"] == tmp_path
        return _Result()

    monkeypatch.setattr(
        "aimd.plugins.doc._plugin.process_doc_with_assets",
        _process_doc_with_assets,
    )

    result = MarkItDown(enable_plugins=True).convert(document, temp_dir=tmp_path)

    assert result.title == "Paper"
    assert result.markdown == "# Paper\n\nBody"


def test_url_plugin_converts_transcript_url_with_markitdown(monkeypatch) -> None:
    async def _get_text_from_url(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args[0] == "https://example.com/video"
        assert kwargs["language"] == "zh"
        assert kwargs["model"] == "tiny"
        assert kwargs["raw_transcript"] is True
        return UrlTextResult(
            title="Video",
            markdown="# Video\n\ntranscript text",
            platform="unknown",
        )

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.get_text_from_url", _get_text_from_url
    )

    result = MarkItDown(enable_plugins=True).convert_stream(
        io.BytesIO(),
        stream_info=StreamInfo(url="https://example.com/video"),
        task_type="transcript",
        model="tiny",
        language="zh",
        raw_transcript=True,
    )

    assert result.title == "Video"
    assert result.markdown == "# Video\n\ntranscript text"


def test_url_plugin_converts_when_markitdown_requests_defuddle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    html = tmp_path / "article.html"
    html.write_text("<html><body>Hello</body></html>", encoding="utf-8")

    class _Result:
        title = "Article"
        content = "# Article\n\nReadable text"

    def _extract_html_with_defuddle(*args, **kwargs):  # noqa: ANN002, ANN003
        assert args[0] == html
        assert kwargs["markdown"] is True
        assert kwargs["npx_command"] == "npx"
        return _Result()

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_html_with_defuddle",
        _extract_html_with_defuddle,
    )

    result = MarkItDown(enable_plugins=True).convert(html, defuddle=True)

    assert result.title == "Article"
    assert result.markdown == "# Article\n\nReadable text"
