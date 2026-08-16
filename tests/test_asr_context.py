"""Tests for ASR context biasing (explicit context and URL metadata injection)."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from markitdown import MarkItDown, StreamInfo
import pytest
from typer.main import get_command
from typer.testing import CliRunner

import aimd.interfaces.mcp as mcp_app
from aimd.core.models import ProcessInput, ProcessResult, TextContext
from aimd.core.process import process_input
from aimd.interfaces.api.app import create_app
from aimd.plugins.url._plugin import get_text_from_url
from aimd.plugins.url.metadata import build_metadata_context

cli_app = import_module("aimd.interfaces.cli.app")
runner = CliRunner()


# --- build_metadata_context ---


def test_build_metadata_context_includes_core_fields() -> None:
    info = {
        "title": "聊聊 PostgreSQL 与 Redis",
        "uploader": "SomeChannel",
        "description": "本期嘉宾: 张三, 讨论 CUDA 优化。",
        "tags": ["数据库", "PostgreSQL", "Redis"],
        "chapters": [{"title": "开场"}, {"title": "深入 CUDA"}],
    }

    context = build_metadata_context(info)

    assert context is not None
    assert "Title: 聊聊 PostgreSQL 与 Redis" in context
    assert "Author: SomeChannel" in context
    assert "本期嘉宾: 张三" in context
    assert "Tags: 数据库, PostgreSQL, Redis" in context
    assert "Chapters: 开场; 深入 CUDA" in context


def test_build_metadata_context_returns_none_when_empty() -> None:
    assert build_metadata_context({}) is None
    assert build_metadata_context({"title": "  ", "tags": [], "chapters": None}) is None


def test_build_metadata_context_truncates_long_metadata() -> None:
    info = {"title": "t", "description": "x" * 5000}

    context = build_metadata_context(info)

    assert context is not None
    assert len(context) <= 2000


# --- get_text_from_url metadata context injection ---


async def _fake_video_info(
    *,
    url: str,
    platform: str,
    cookies_file: str | None,
    cookies_from_browser: str | None,
):
    return {
        "title": "Interview with Linus Torvalds",
        "description": "A conversation about the Linux kernel and Git.",
        "tags": ["linux", "git"],
        "webpage_url": url,
    }


async def _no_subtitles(info_dict, platform: str, language: str | None):
    return None


def _patch_url_pipeline(monkeypatch, captured: dict) -> None:
    async def _mock_extract_content_from_audio(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return "transcribed audio text"

    monkeypatch.setattr("aimd.plugins.url._plugin.extract_video_info", _fake_video_info)
    monkeypatch.setattr("aimd.plugins.url._plugin.extract_subtitles", _no_subtitles)
    monkeypatch.setattr(
        "aimd.plugins.url._plugin.extract_content_from_audio",
        _mock_extract_content_from_audio,
    )


@pytest.mark.asyncio
async def test_url_audio_fallback_injects_metadata_context_by_default(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_url_pipeline(monkeypatch, captured)

    result = await get_text_from_url("https://example.com/video")

    assert "transcribed audio text" in result.markdown
    context = captured["context"]
    assert isinstance(context, str)
    assert "Linus Torvalds" in context
    assert "Linux kernel" in context


@pytest.mark.asyncio
async def test_url_audio_fallback_metadata_context_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_url_pipeline(monkeypatch, captured)

    await get_text_from_url("https://example.com/video", metadata_context=False)

    assert captured["context"] is None


@pytest.mark.asyncio
async def test_url_audio_fallback_combines_explicit_and_metadata_context(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_url_pipeline(monkeypatch, captured)

    await get_text_from_url(
        "https://example.com/video", context="Vocabulary: Linus, Git."
    )

    context = captured["context"]
    assert isinstance(context, str)
    assert context.startswith("Vocabulary: Linus, Git.")
    assert "Linus Torvalds" in context


@pytest.mark.asyncio
async def test_url_audio_fallback_explicit_context_only(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _patch_url_pipeline(monkeypatch, captured)

    await get_text_from_url(
        "https://example.com/video",
        context="Vocabulary: Linus, Git.",
        metadata_context=False,
    )

    assert captured["context"] == "Vocabulary: Linus, Git."


# --- ASR plugin context propagation ---


@pytest.mark.asyncio
async def test_transcribe_segment_with_fallback_forwards_context() -> None:
    from aimd.plugins.asr._plugin import _transcribe_segment_with_fallback

    class _FakeModel:
        model_id = "fake-model"

        def __init__(self) -> None:
            self.seen: dict[str, object] = {}

        async def transcribe(
            self,
            file_path,
            *,
            language=None,
            temp_dir=None,
            context=None,
        ):
            self.seen["context"] = context
            return "hello world"

    model = _FakeModel()
    text = await _transcribe_segment_with_fallback(
        model,
        Path("segment.wav"),
        "mlx",
        language="zh",
        context="Vocabulary: Foo",
    )

    assert text == "hello world"
    assert model.seen["context"] == "Vocabulary: Foo"


def test_asr_converter_forwards_context_via_markitdown(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"fake audio")

    def _transcribe_file_sync(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["context"] == "Vocabulary: Foo"
        return "transcript text"

    monkeypatch.setattr(
        "aimd.plugins.asr._plugin._transcribe_file_sync", _transcribe_file_sync
    )

    result = MarkItDown(enable_plugins=True).convert(audio, context="Vocabulary: Foo")

    assert result.markdown == "transcript text"


def test_url_converter_forwards_context_via_markitdown(monkeypatch) -> None:
    import io

    from aimd.plugins.url._plugin import UrlTextResult

    async def _get_text_from_url(*args, **kwargs):  # noqa: ANN002, ANN003
        assert kwargs["context"] == "Vocabulary: Foo"
        assert kwargs["metadata_context"] is False
        return UrlTextResult(title="Video", markdown="text", platform="unknown")

    monkeypatch.setattr(
        "aimd.plugins.url._plugin.get_text_from_url", _get_text_from_url
    )

    result = MarkItDown(enable_plugins=True).convert_stream(
        io.BytesIO(),
        stream_info=StreamInfo(url="https://example.com/video"),
        task_type="transcript",
        context="Vocabulary: Foo",
        metadata_context=False,
    )

    assert result.markdown == "text"


# --- MLX adapter system_prompt injection ---


@pytest.mark.asyncio
async def test_mlx_adapter_injects_system_prompt(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("mlx_audio")
    import aimd.plugins.asr.models.mlx as mlx_mod

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"fake")
    captured: dict[str, object] = {}

    class _FakeSTT:
        def generate(self, path, language=None, system_prompt=None):
            captured["language"] = language
            captured["system_prompt"] = system_prompt
            return SimpleNamespace(text=" 转录文本 ")

    monkeypatch.setattr(mlx_mod, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlx_mod, "convert_to_wav_if_needed", lambda *a, **k: None)
    monkeypatch.setattr("mlx_audio.stt.load", lambda model_id: _FakeSTT())

    model = mlx_mod.MLXAudioASRModel("qwen3-asr-1.7b", precision="4bit")
    text = await model.transcribe(audio, language="zh", context="Vocabulary: Foo")

    assert text == "转录文本"
    assert captured["language"] == "Chinese"
    assert captured["system_prompt"] == "Vocabulary: Foo"


# --- Transformers chat-template context path ---


def test_transformers_inputs_without_context_use_transcription_request() -> None:
    from aimd.plugins.asr.models.transformers import _build_transcription_inputs

    class _FakeProcessor:
        def apply_transcription_request(self, audio, language):
            return {
                "via": "transcription_request",
                "audio": audio,
                "language": language,
            }

        def apply_chat_template(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("apply_chat_template should not be called")

    inputs = _build_transcription_inputs(
        _FakeProcessor(), Path("a.wav"), "Chinese", None
    )

    assert inputs["via"] == "transcription_request"
    assert inputs["language"] == "Chinese"


def test_transformers_inputs_with_context_use_chat_template() -> None:
    from aimd.plugins.asr.models.transformers import _build_transcription_inputs

    captured: dict[str, object] = {}

    class _FakeProcessor:
        def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN003
            captured["conversations"] = conversations
            captured["kwargs"] = kwargs
            return {"input_ids": "fake"}

    _build_transcription_inputs(
        _FakeProcessor(), Path("a.wav"), "Chinese", "Vocabulary: Foo"
    )

    messages = captured["conversations"][0]
    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Vocabulary: Foo"}],
    }
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "audio", "path": "a.wav"}
    assert messages[2]["content"][0]["text"] == "language Chinese<asr_text>"
    assert captured["kwargs"]["continue_final_message"] is True
    assert captured["kwargs"]["tokenize"] is True


def test_transformers_inputs_context_without_language_uses_generation_prompt() -> None:
    from aimd.plugins.asr.models.transformers import _build_transcription_inputs

    captured: dict[str, object] = {}

    class _FakeProcessor:
        def apply_chat_template(self, conversations, **kwargs):  # noqa: ANN003
            captured["conversations"] = conversations
            captured["kwargs"] = kwargs
            return {"input_ids": "fake"}

    _build_transcription_inputs(
        _FakeProcessor(), Path("a.wav"), None, "Vocabulary: Foo"
    )

    messages = captured["conversations"][0]
    assert len(messages) == 2
    assert captured["kwargs"]["add_generation_prompt"] is True


# --- core use-case forwarding ---


@pytest.mark.asyncio
async def test_use_case_url_forwards_context_and_metadata_flag() -> None:
    seen: dict[str, object] = {}

    async def _process_url(*args, context=None, metadata_context=True):  # noqa: ANN002
        seen["context"] = context
        seen["metadata_context"] = metadata_context
        return TextContext(title="a", chunk_list=["t"]), "raw", "youtube"

    async def _unexpected_process_file(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("should not process file")

    result = await process_input(
        ProcessInput(
            input_source="https://example.com/video",
            context="Vocabulary: Foo",
            metadata_context=False,
        ),
        process_url=_process_url,
        process_file=_unexpected_process_file,
        is_supported_file_fn=lambda _: True,
    )

    assert result.task_type == "transcript"
    assert seen == {"context": "Vocabulary: Foo", "metadata_context": False}


@pytest.mark.asyncio
async def test_use_case_local_file_forwards_context(tmp_path: Path) -> None:
    audio = tmp_path / "a.mp3"
    audio.write_text("x", encoding="utf-8")
    seen: dict[str, object] = {}

    async def _process_file(*args, context=None):  # noqa: ANN002
        seen["context"] = context
        return TextContext(title="a", chunk_list=["t"]), "raw audio", None

    async def _unexpected_process_url(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("should not process URL")

    await process_input(
        ProcessInput(input_source=str(audio), context="Vocabulary: Foo"),
        process_url=_unexpected_process_url,
        process_file=_process_file,
        is_supported_file_fn=lambda _: True,
    )

    assert seen["context"] == "Vocabulary: Foo"


# --- CLI ---


def test_cli_exposes_context_options() -> None:
    command = get_command(cli_app.app)
    context_option = next(param for param in command.params if param.name == "context")
    assert "--context" in context_option.opts
    no_context_option = next(
        param for param in command.params if param.name == "no_context"
    )
    assert "--no-context" in no_context_option.opts


def test_cli_context_options_are_forwarded(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["context"] = request.context
        seen["metadata_context"] = request.metadata_context
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="Demo", chunk_list=["hello"]),
            markdown="hello",
        )

    monkeypatch.setattr(cli_app, "process_input", _fake_process_input)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli_app.app,
        ["input.mp3", "--context", "Vocabulary: Foo", "--no-context"],
    )

    assert result.exit_code == 0
    assert seen == {"context": "Vocabulary: Foo", "metadata_context": False}


def test_cli_metadata_context_defaults_to_enabled(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["context"] = request.context
        seen["metadata_context"] = request.metadata_context
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="Demo", chunk_list=["hello"]),
            markdown="hello",
        )

    monkeypatch.setattr(cli_app, "process_input", _fake_process_input)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli_app.app, ["input.mp3"])

    assert result.exit_code == 0
    assert seen == {"context": None, "metadata_context": True}


# --- HTTP API ---


def test_api_forwards_context_fields(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["context"] = request.context
        seen["metadata_context"] = request.metadata_context
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="t", chunk_list=["x"]),
            markdown="x",
        )

    monkeypatch.setattr(
        "aimd.interfaces.api.app.process_core_input", _fake_process_input
    )
    client = TestClient(create_app())

    response = client.post(
        "/v1/process",
        json={
            "input_source": "input.mp3",
            "context": "Vocabulary: Foo",
            "metadata_context": False,
        },
    )

    assert response.status_code == 200
    assert seen == {"context": "Vocabulary: Foo", "metadata_context": False}


# --- MCP ---


@pytest.mark.asyncio
async def test_mcp_process_input_forwards_context(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["context"] = request.context
        seen["metadata_context"] = request.metadata_context
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="t", chunk_list=["x"]),
        )

    monkeypatch.setattr(
        "aimd.interfaces.mcp.app.process_core_input", _fake_process_input
    )

    await mcp_app.process_input(
        "input.mp3",
        context="Vocabulary: Foo",
        metadata_context=False,
    )

    assert seen == {"context": "Vocabulary: Foo", "metadata_context": False}


@pytest.mark.asyncio
async def test_mcp_process_input_schema_exposes_context() -> None:
    tools = await mcp_app.mcp.list_tools()
    process_tool = next(tool for tool in tools if tool.name == "process_input")
    assert "context" in process_tool.input_schema["properties"]
    assert "metadata_context" in process_tool.input_schema["properties"]
