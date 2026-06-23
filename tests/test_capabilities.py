import pytest

from aimd.core.errors import BackendUnavailableError
from aimd.plugins.asr.capabilities import select_transcription_backend


def test_select_prefers_mlx_on_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("aimd.plugins.asr.capabilities.is_apple_silicon", lambda: True)
    monkeypatch.setattr(
        "aimd.plugins.asr.capabilities._module_available",
        lambda name: name == "mlx_audio",
    )
    assert select_transcription_backend() == "mlx"


def test_select_prefers_transformers_on_linux_cuda(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")

    def fake_available(name):
        return name in ("torch", "torchaudio", "transformers")

    monkeypatch.setattr(
        "aimd.plugins.asr.capabilities._module_available", fake_available
    )
    monkeypatch.setattr("aimd.plugins.asr.capabilities._cuda_available", lambda: True)
    assert select_transcription_backend() == "transformers"


def test_select_prefers_transformers_on_windows_cuda(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "aimd.plugins.asr.capabilities._module_available",
        lambda name: name in ("torch", "torchaudio", "transformers"),
    )
    monkeypatch.setattr("aimd.plugins.asr.capabilities._cuda_available", lambda: True)
    assert select_transcription_backend() == "transformers"


def test_select_raises_when_no_backend(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "aimd.plugins.asr.capabilities._module_available", lambda name: False
    )
    with pytest.raises(BackendUnavailableError):
        select_transcription_backend()
