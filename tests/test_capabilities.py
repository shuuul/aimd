import pytest

from aimd.plugins.asr.capabilities import (
    BackendCapability,
    select_transcription_backend,
)
from aimd.plugins.asr.errors import BackendUnavailableError


def _patch_capabilities(
    monkeypatch, capabilities: dict[str, BackendCapability]
) -> None:
    monkeypatch.setattr(
        "aimd.plugins.asr.capabilities.get_backend_capabilities",
        lambda: capabilities,
    )


@pytest.mark.parametrize(
    ("system", "capabilities", "expected_backend"),
    [
        (
            "Linux",
            {
                "mlx": BackendCapability("mlx", False, "unsupported", None),
                "transformers": BackendCapability("transformers", True, None, None),
            },
            "transformers",
        ),
        (
            "Darwin",
            {
                "mlx": BackendCapability("mlx", True, None, None),
                "transformers": BackendCapability("transformers", False, "linux only", None),
            },
            "mlx",
        ),
    ],
)
def test_select_transcription_backend_prefers_platform_backend(
    monkeypatch,
    system: str,
    capabilities: dict[str, BackendCapability],
    expected_backend: str,
) -> None:
    monkeypatch.setattr("aimd.plugins.asr.capabilities.platform.system", lambda: system)
    _patch_capabilities(monkeypatch, capabilities)

    assert select_transcription_backend() == expected_backend


@pytest.mark.parametrize(
    ("system", "capabilities"),
    [
        (
            "Linux",
            {
                "mlx": BackendCapability("mlx", False, "unsupported", None),
                "transformers": BackendCapability("transformers", False, "no transformers", None),
            },
        ),
        (
            "Darwin",
            {
                "mlx": BackendCapability("mlx", False, "unsupported", None),
                "transformers": BackendCapability("transformers", False, "linux only", None),
            },
        ),
    ],
)
def test_select_transcription_backend_no_backend(
    monkeypatch,
    system: str,
    capabilities: dict[str, BackendCapability],
) -> None:
    monkeypatch.setattr("aimd.plugins.asr.capabilities.platform.system", lambda: system)
    _patch_capabilities(monkeypatch, capabilities)

    with pytest.raises(BackendUnavailableError):
        select_transcription_backend()
