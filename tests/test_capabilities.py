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
        "cuda": EngineCapability("cuda", False, "no cuda", None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("yap")


def test_resolve_engine_auto_non_macos_prefers_cuda(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", False, "unsupported", None),
        "mlx": EngineCapability("mlx", False, "unsupported", None),
        "cuda": EngineCapability("cuda", True, None, None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "cuda"


def test_resolve_engine_auto_macos_prefers_yap(monkeypatch) -> None:
    mock_capabilities = {
        "yap": EngineCapability("yap", True, None, None),
        "mlx": EngineCapability("mlx", True, None, None),
        "cuda": EngineCapability("cuda", True, None, None),
        "cpu": EngineCapability("cpu", True, None, None),
    }
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "aimd.infrastructure.capabilities.detector.get_engine_capabilities",
        lambda: mock_capabilities,
    )

    assert resolve_engine_with_preflight("auto") == "yap"
