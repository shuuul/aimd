import sys
from pathlib import Path
from types import ModuleType

import pytest

from aimd.const import FUNASR_DEFAULT_MODEL
from aimd.infrastructure.transcription.funasr_engine import transcribe_audio_funasr


def _install_fake_funasr(monkeypatch: pytest.MonkeyPatch) -> None:
    funasr_module = ModuleType("funasr")
    funasr_module.__path__ = []
    utils_module = ModuleType("funasr.utils")
    utils_module.__path__ = []
    postprocess_module = ModuleType("funasr.utils.postprocess_utils")
    postprocess_module.rich_transcription_postprocess = lambda text: text

    monkeypatch.setitem(sys.modules, "funasr", funasr_module)
    monkeypatch.setitem(sys.modules, "funasr.utils", utils_module)
    monkeypatch.setitem(
        sys.modules, "funasr.utils.postprocess_utils", postprocess_module
    )


class _FakeModel:
    def __init__(self, text: str | None = None, exc: Exception | None = None) -> None:
        self.text = text
        self.exc = exc

    def generate(self, **kwargs):  # noqa: ANN003, ARG002
        if self.exc is not None:
            raise self.exc
        return [{"text": self.text}]


@pytest.mark.asyncio
async def test_transcribe_audio_funasr_retries_with_sensevoice_on_control_token_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_funasr(monkeypatch)
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"wav")

    calls: list[str] = []
    models = {
        FUNASR_DEFAULT_MODEL: _FakeModel(
            exc=RuntimeError("tokenizer refused to decode control token <|no|>")
        ),
        "FunAudioLLM/SenseVoiceSmall": _FakeModel(text="handled by fallback"),
    }

    def _mock_get_model(model_name: str, device: str):  # noqa: ARG001
        calls.append(model_name)
        return models[model_name]

    monkeypatch.setattr(
        "aimd.infrastructure.transcription.funasr_engine._get_model", _mock_get_model
    )

    result = await transcribe_audio_funasr(audio_file, model=FUNASR_DEFAULT_MODEL)

    assert result == "handled by fallback"
    assert calls == [FUNASR_DEFAULT_MODEL, "FunAudioLLM/SenseVoiceSmall"]


@pytest.mark.asyncio
async def test_transcribe_audio_funasr_retries_when_output_is_only_no_speech_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_funasr(monkeypatch)
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"wav")

    calls: list[str] = []
    models = {
        FUNASR_DEFAULT_MODEL: _FakeModel(text="<|no|>"),
        "FunAudioLLM/SenseVoiceSmall": _FakeModel(text="speech recovered"),
    }

    def _mock_get_model(model_name: str, device: str):  # noqa: ARG001
        calls.append(model_name)
        return models[model_name]

    monkeypatch.setattr(
        "aimd.infrastructure.transcription.funasr_engine._get_model", _mock_get_model
    )

    result = await transcribe_audio_funasr(audio_file, model=FUNASR_DEFAULT_MODEL)

    assert result == "speech recovered"
    assert calls == [FUNASR_DEFAULT_MODEL, "FunAudioLLM/SenseVoiceSmall"]


@pytest.mark.asyncio
async def test_transcribe_audio_funasr_strips_no_speech_control_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_funasr(monkeypatch)
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"wav")

    def _mock_get_model(model_name: str, device: str):  # noqa: ARG001
        return _FakeModel(text="hello <|no|> world")

    monkeypatch.setattr(
        "aimd.infrastructure.transcription.funasr_engine._get_model", _mock_get_model
    )

    result = await transcribe_audio_funasr(audio_file, model=FUNASR_DEFAULT_MODEL)

    assert result == "hello world"
