import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest
from transformers import AutoConfig

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError
from aimd.plugins.asr._plugin import _select_backend_for_model
from aimd.plugins.asr.const import (
    QWEN_ASR_DEFAULT_MODEL,
    QWEN_ASR_MODELS,
    TRANSFORMERS_ASR_DEFAULT_MODEL,
    TRANSFORMERS_ASR_MODELS,
    resolve_transformers_asr_model,
)
from aimd.plugins.asr.models.transformers import (
    TransformersASRModel,
    _get_model_and_processor,
    _parse_qwen_output,
    _resolve_language,
    _resolve_torch_dtype,
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


def test_resolve_transformers_asr_model_maps_kebab_aliases() -> None:
    assert resolve_transformers_asr_model("qwen3-asr-1.7b") == "Qwen/Qwen3-ASR-1.7B-hf"
    assert resolve_transformers_asr_model("qwen3-asr-0.6b") == "Qwen/Qwen3-ASR-0.6B-hf"
    # Legacy underscore aliases stay compatible.
    assert resolve_transformers_asr_model("qwen3_asr_1_7b") == "Qwen/Qwen3-ASR-1.7B-hf"
    assert resolve_transformers_asr_model("qwen3_asr_0_6b") == "Qwen/Qwen3-ASR-0.6B-hf"


def test_kebab_alias_uses_platform_backend_selection() -> None:
    # Kebab aliases are backend-neutral; full HF IDs still force Transformers.
    assert _select_backend_for_model("Qwen/Qwen3-ASR-1.7B-hf") == "transformers"


def test_transformers_asr_model_resolves_alias_and_precision() -> None:
    model = TransformersASRModel("qwen3-asr-0.6b", precision="bf16")
    assert model.model_id == "Qwen/Qwen3-ASR-0.6B-hf"
    assert model.precision == "bf16"

    default = TransformersASRModel(None)
    assert default.model_id == TRANSFORMERS_ASR_DEFAULT_MODEL
    assert default.precision is None


@pytest.mark.parametrize("precision", ["4bit", "6bit", "8bit"])
def test_transformers_asr_model_rejects_quantized_precision(precision: str) -> None:
    with pytest.raises(ProcessingFailedError, match="quantized precision"):
        TransformersASRModel("qwen3-asr-1.7b", precision=precision)


def test_transformers_asr_model_rejects_unknown_precision() -> None:
    with pytest.raises(ProcessingFailedError, match="Unsupported precision"):
        TransformersASRModel(None, precision="fp8")


def test_resolve_torch_dtype_auto_and_bf16(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    assert _resolve_torch_dtype("cpu", None) is torch.float32
    assert _resolve_torch_dtype("mps", None) is torch.float16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _resolve_torch_dtype("cuda", None) is torch.bfloat16
    assert _resolve_torch_dtype("cuda", "bf16") is torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _resolve_torch_dtype("cuda", None) is torch.float16
    with pytest.raises(BackendUnavailableError, match="bf16"):
        _resolve_torch_dtype("cuda", "bf16")
    with pytest.raises(BackendUnavailableError, match="bf16"):
        _resolve_torch_dtype("cpu", "bf16")
    with pytest.raises(ProcessingFailedError, match="does not support precision"):
        _resolve_torch_dtype("cuda", "4bit")


def test_get_model_and_processor_cache_key_includes_precision(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import transformers

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        "aimd.plugins.asr.models.transformers._select_device", lambda: "cuda"
    )
    monkeypatch.setattr(
        "aimd.plugins.asr.models.transformers._require_native_qwen3_asr",
        lambda: None,
    )

    load_calls: list[tuple[str, object]] = []

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    def _from_pretrained(model_name, dtype=None):  # noqa: ANN001
        load_calls.append((model_name, dtype))
        return fake_model

    monkeypatch.setattr(
        transformers.AutoModelForMultimodalLM, "from_pretrained", _from_pretrained
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda model_name: "processor",  # noqa: ARG005
    )
    monkeypatch.setattr("aimd.plugins.asr.models.transformers._cached_model", None)
    monkeypatch.setattr("aimd.plugins.asr.models.transformers._cached_processor", None)
    monkeypatch.setattr("aimd.plugins.asr.models.transformers._cached_model_name", None)
    monkeypatch.setattr("aimd.plugins.asr.models.transformers._cached_device", None)
    monkeypatch.setattr("aimd.plugins.asr.models.transformers._cached_precision", None)

    _get_model_and_processor("Qwen/Qwen3-ASR-1.7B-hf")
    _get_model_and_processor("Qwen/Qwen3-ASR-1.7B-hf")
    assert len(load_calls) == 1

    # A different precision must not reuse the cached dtype.
    _get_model_and_processor("Qwen/Qwen3-ASR-1.7B-hf", precision="bf16")
    assert len(load_calls) == 2
    assert all(dtype is torch.bfloat16 for _, dtype in load_calls)
