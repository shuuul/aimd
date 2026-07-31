import os
from pathlib import Path
import shutil
import subprocess

import pytest
from transformers import AutoConfig

from aimd.plugins.asr._plugin import _select_backend_for_model
from aimd.plugins.asr.const import (
    QWEN_ASR_DEFAULT_MODEL,
    QWEN_ASR_MODELS,
    TRANSFORMERS_ASR_DEFAULT_MODEL,
    TRANSFORMERS_ASR_MODELS,
    resolve_transformers_asr_model,
)
from aimd.plugins.asr.models.transformers import (
    _parse_qwen_output,
    _resolve_language,
)


def test_native_transformers_registers_qwen3_asr_config() -> None:
    config = AutoConfig.for_model("qwen3_asr")
    assert config.model_type == "qwen3_asr"


def test_legacy_qwen_constants_remain_compatible_aliases() -> None:
    assert QWEN_ASR_DEFAULT_MODEL == TRANSFORMERS_ASR_DEFAULT_MODEL
    assert QWEN_ASR_MODELS is TRANSFORMERS_ASR_MODELS


def test_explicit_qwen3_asr_model_selects_transformers_backend() -> None:
    assert _select_backend_for_model("Qwen/Qwen3-ASR-0.6B-hf") == "transformers"
    assert _select_backend_for_model("Qwen/Qwen3-ASR-0.6B") == "transformers"
    assert _select_backend_for_model("Qwen/Qwen3-ASR-1.7B-hf") == "transformers"


def test_resolve_transformers_asr_model_maps_legacy_ids() -> None:
    assert (
        resolve_transformers_asr_model("Qwen/Qwen3-ASR-0.6B")
        == "Qwen/Qwen3-ASR-0.6B-hf"
    )
    assert (
        resolve_transformers_asr_model("Qwen/Qwen3-ASR-1.7B")
        == "Qwen/Qwen3-ASR-1.7B-hf"
    )
    assert (
        resolve_transformers_asr_model("Qwen/Qwen3-ASR-0.6B-hf")
        == "Qwen/Qwen3-ASR-0.6B-hf"
    )
    assert resolve_transformers_asr_model(None) == TRANSFORMERS_ASR_DEFAULT_MODEL
    assert TRANSFORMERS_ASR_DEFAULT_MODEL.endswith("-hf")


def test_resolve_qwen_language_code() -> None:
    assert _resolve_language("zh") == "Chinese"
    assert _resolve_language("English") == "English"
    assert _resolve_language(None) is None


def test_parse_qwen_output_extracts_asr_text() -> None:
    assert _parse_qwen_output("language English<asr_text>Hello world") == "Hello world"


def test_parse_qwen_output_falls_back_to_plain_text() -> None:
    assert _parse_qwen_output("Hello world") == "Hello world"


@pytest.mark.skipif(
    os.environ.get("AIMD_RUN_QWEN3_ASR_INTEGRATION") != "1",
    reason="set AIMD_RUN_QWEN3_ASR_INTEGRATION=1 to run real Qwen3-ASR inference",
)
@pytest.mark.asyncio
async def test_qwen3_asr_transformers_real_inference_smoke(tmp_path: Path) -> None:
    """Run a skipped-by-default real Qwen3-ASR inference smoke.

    This intentionally downloads/loads Qwen/Qwen3-ASR-0.6B-hf if it is not already
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

    text = await transcribe_file(
        audio_path, language="zh", model="Qwen/Qwen3-ASR-0.6B-hf"
    )
    assert text.strip()
