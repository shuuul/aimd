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
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "funasr": EngineCapability("funasr", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("mlx")


def test_resolve_engine_auto_linux_prefers_funasr(monkeypatch) -> None:
    mock_capabilities = {
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "funasr": EngineCapability("funasr", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "funasr"


def test_resolve_engine_auto_linux_no_engine(monkeypatch) -> None:
    mock_capabilities = {
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "funasr": EngineCapability("funasr", False, "no funasr", None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("auto")


def test_resolve_engine_auto_macos_prefers_mlx(monkeypatch) -> None:
    mock_capabilities = {
        "mlx": EngineCapability("mlx", True, None, None),
        "funasr": EngineCapability("funasr", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "mlx"


def test_resolve_engine_auto_macos_falls_back_to_funasr(monkeypatch) -> None:
    mock_capabilities = {
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "funasr": EngineCapability("funasr", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "funasr"
