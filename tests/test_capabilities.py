import pytest

from aimd.infrastructure.capabilities.detector import (
    EngineCapability,
    resolve_engine_with_preflight,
)
from aimd.errors import EngineUnavailableError, UnsupportedEngineError


def test_resolve_engine_invalid_name() -> None:
    with pytest.raises(UnsupportedEngineError):
        resolve_engine_with_preflight("bad-engine")


def test_resolve_engine_explicit_unavailable(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", False, "missing yap", "install yap"),
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "qwen": EngineCapability("qwen", False, "no qwen", None),
        "whisper": EngineCapability("whisper", False, "no whisper", None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("yap")


def test_resolve_engine_auto_linux_prefers_qwen(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", False, "unsupported", None),
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "qwen": EngineCapability("qwen", True, None, None),
        "whisper": EngineCapability("whisper", True, None, None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "qwen"


def test_resolve_engine_auto_linux_falls_back_to_whisper(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", False, "unsupported", None),
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "qwen": EngineCapability("qwen", False, "no qwen-asr", None),
        "whisper": EngineCapability("whisper", True, None, None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "whisper"


def test_resolve_engine_auto_macos_prefers_mlx(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", True, None, None),
        "mlx": EngineCapability("mlx", True, None, None),
        "qwen": EngineCapability("qwen", False, "unsupported", None),
        "whisper": EngineCapability("whisper", True, None, None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "mlx"
