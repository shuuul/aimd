import pytest

from aimd_media.capabilities import (
    EngineCapability,
    resolve_engine_with_preflight,
)
from aimd_media.errors import EngineUnavailableError, UnsupportedEngineError


def test_resolve_engine_invalid_name() -> None:
    with pytest.raises(UnsupportedEngineError):
        resolve_engine_with_preflight("bad-engine")


def _patch_capabilities(monkeypatch, capabilities: dict[str, EngineCapability]) -> None:
    monkeypatch.setattr(
        "aimd_media.capabilities.get_engine_capabilities",
        lambda: capabilities,
    )


def test_resolve_engine_explicit_unavailable(monkeypatch) -> None:
    _patch_capabilities(
        monkeypatch,
        {
            "mlx": EngineCapability("mlx", False, "unsupported", None),
            "qwen": EngineCapability("qwen", True, None, None),
        },
    )

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("mlx")


@pytest.mark.parametrize(
    ("system", "capabilities", "expected_engine"),
    [
        (
            "Linux",
            {
                "mlx": EngineCapability("mlx", False, "unsupported", None),
                "qwen": EngineCapability("qwen", True, None, None),
            },
            "qwen",
        ),
        (
            "Darwin",
            {
                "mlx": EngineCapability("mlx", True, None, None),
                "qwen": EngineCapability("qwen", False, "linux only", None),
            },
            "mlx",
        ),
    ],
)
def test_resolve_engine_auto_prefers_platform_engine(
    monkeypatch,
    system: str,
    capabilities: dict[str, EngineCapability],
    expected_engine: str,
) -> None:
    monkeypatch.setattr("aimd_media.capabilities.platform.system", lambda: system)
    _patch_capabilities(monkeypatch, capabilities)

    assert resolve_engine_with_preflight("auto") == expected_engine


@pytest.mark.parametrize(
    ("system", "capabilities"),
    [
        (
            "Linux",
            {
                "mlx": EngineCapability("mlx", False, "unsupported", None),
                "qwen": EngineCapability("qwen", False, "no qwen", None),
            },
        ),
        (
            "Darwin",
            {
                "mlx": EngineCapability("mlx", False, "unsupported", None),
                "qwen": EngineCapability("qwen", False, "linux only", None),
            },
        ),
    ],
)
def test_resolve_engine_auto_no_engine(
    monkeypatch,
    system: str,
    capabilities: dict[str, EngineCapability],
) -> None:
    monkeypatch.setattr("aimd_media.capabilities.platform.system", lambda: system)
    _patch_capabilities(monkeypatch, capabilities)

    with pytest.raises(EngineUnavailableError):
        resolve_engine_with_preflight("auto")
