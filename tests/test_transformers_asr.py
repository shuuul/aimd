import os
from pathlib import Path
import shutil
import subprocess

import pytest
from transformers import AutoConfig

from aimd.plugins.asr._plugin import _select_backend_for_model
from aimd.plugins.asr.models.transformers import (
    _parse_qwen_output,
    _resolve_language,
)
from aimd.plugins.asr.qwen3_asr_transformers import (
    Qwen3ASRConfig,
    register_qwen3_asr_transformers,
)


def test_register_qwen3_asr_transformers_config() -> None:
    register_qwen3_asr_transformers()
    config = AutoConfig.for_model("qwen3_asr")
    assert config.model_type == Qwen3ASRConfig.model_type


def test_explicit_qwen3_asr_model_selects_transformers_backend() -> None:
    assert _select_backend_for_model("Qwen/Qwen3-ASR-0.6B") == "transformers"


def test_resolve_qwen_language_code() -> None:
    assert _resolve_language("zh") == "Chinese"
    assert _resolve_language("English") == "English"
    assert _resolve_language(None) is None


def test_parse_qwen_output_extracts_asr_text() -> None:
    assert (
        _parse_qwen_output("language English<asr_text>Hello world", None)
        == "Hello world"
    )


def test_parse_qwen_output_keeps_forced_language_text() -> None:
    assert _parse_qwen_output("Hello world", "English") == "Hello world"


def test_parse_qwen_output_falls_back_to_plain_text() -> None:
    assert _parse_qwen_output("Hello world", None) == "Hello world"


@pytest.mark.skipif(
    os.environ.get("AIMD_RUN_QWEN3_ASR_INTEGRATION") != "1",
    reason="set AIMD_RUN_QWEN3_ASR_INTEGRATION=1 to run real Qwen3-ASR inference",
)
@pytest.mark.asyncio
async def test_qwen3_asr_transformers_real_inference_smoke(tmp_path: Path) -> None:
    """Run a skipped-by-default real Qwen3-ASR inference smoke.

    This intentionally downloads/loads Qwen/Qwen3-ASR-0.6B if it is not already
    cached. It is meant for dependency-upgrade validation, not normal CI.
    """
    from aimd.plugins.asr._plugin import transcribe_file

    explicit_audio = os.environ.get("AIMD_QWEN3_ASR_TEST_AUDIO")
    if explicit_audio:
        audio_path = Path(explicit_audio)
    else:
        say = shutil.which("say")
        ffmpeg = shutil.which("ffmpeg")
        if say is None or ffmpeg is None:
            pytest.skip("set AIMD_QWEN3_ASR_TEST_AUDIO or install say and ffmpeg")
        aiff_path = tmp_path / "qwen3-asr-smoke.aiff"
        audio_path = tmp_path / "qwen3-asr-smoke.wav"
        subprocess.run(
            [say, "-v", "Tingting", "这是一个 AIMD 测试音频。", "-o", str(aiff_path)],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(aiff_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            check=True,
        )

    text = await transcribe_file(audio_path, language="zh", model="Qwen/Qwen3-ASR-0.6B")
    assert text.strip()
