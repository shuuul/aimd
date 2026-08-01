"""Tests for audio transcoding support (.mp4a etc.)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aimd.plugins.asr.audio_utils import (
    SUPPORTED_AUDIO_FORMATS,
    convert_to_wav_if_needed,
    segment_audio,
)
from aimd.plugins.asr.const import (
    AUDIO_EXTENSIONS,
    MLX_AUDIO_DEFAULT_MODEL,
    MLX_AUDIO_MODELS,
    resolve_mlx_asr_model,
)
from aimd.core.errors import (
    InputNotFoundError,
    ProcessingFailedError,
    UnsupportedInputError,
)
from aimd.plugins.asr.models.mlx import MLXAudioASRModel, _resolve_language
from aimd.plugins.asr import transcribe_file


def _mock_ffmpeg_ok() -> MagicMock:
    """Return a MagicMock simulating a successful subprocess.run result."""
    result = MagicMock()
    result.returncode = 0
    result.stderr = b""
    return result


class TestAudioExtensions:
    """Validate extension set membership."""

    def test_mp4a_in_audio_extensions(self) -> None:
        assert ".mp4a" in AUDIO_EXTENSIONS

    def test_mp4a_not_in_supported_audio_formats(self) -> None:
        """mp4a is accepted as input but requires ffmpeg transcoding."""
        assert ".mp4a" not in SUPPORTED_AUDIO_FORMATS


class TestMlxAudioModels:
    """Validate mlx-audio model support metadata."""

    @pytest.mark.parametrize("model_id", MLX_AUDIO_MODELS)
    def test_qwen3_asr_models_are_allowed(self, model_id: str) -> None:
        assert MLXAudioASRModel(model_id).model_id == model_id

    @pytest.mark.parametrize(
        "model_id",
        [
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/VibeVoice-ASR-bf16",
            "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        ],
    )
    def test_removed_mlx_models_are_rejected(self, model_id: str) -> None:
        with pytest.raises(ProcessingFailedError, match="Unsupported MLX ASR model"):
            MLXAudioASRModel(model_id)

    def test_default_mlx_audio_model_is_supported(self) -> None:
        assert MLX_AUDIO_DEFAULT_MODEL == "mlx-community/Qwen3-ASR-1.7B-4bit"
        assert MLX_AUDIO_DEFAULT_MODEL in MLX_AUDIO_MODELS

    def test_default_language_only_applies_to_qwen3_asr(self) -> None:
        assert _resolve_language("mlx-community/Qwen3-ASR-1.7B-4bit", None) == "Chinese"


class TestMlxAsrAliasAndPrecision:
    """Validate kebab-case alias + precision resolution for the MLX backend."""

    @pytest.mark.parametrize("precision", ["4bit", "6bit", "8bit", "bf16"])
    def test_canonical_alias_with_each_precision(self, precision: str) -> None:
        assert (
            resolve_mlx_asr_model("qwen3-asr-1.7b", precision)
            == f"mlx-community/Qwen3-ASR-1.7B-{precision}"
        )
        assert (
            resolve_mlx_asr_model("qwen3-asr-0.6b", precision)
            == f"mlx-community/Qwen3-ASR-0.6B-{precision}"
        )

    def test_alias_defaults_to_4bit(self) -> None:
        assert (
            resolve_mlx_asr_model("qwen3-asr-1.7b")
            == "mlx-community/Qwen3-ASR-1.7B-4bit"
        )
        assert (
            resolve_mlx_asr_model("qwen3-asr-0.6b")
            == "mlx-community/Qwen3-ASR-0.6B-4bit"
        )
        assert resolve_mlx_asr_model(None) == MLX_AUDIO_DEFAULT_MODEL

    def test_legacy_underscore_alias(self) -> None:
        assert (
            resolve_mlx_asr_model("qwen3_asr_1_7b", "8bit")
            == "mlx-community/Qwen3-ASR-1.7B-8bit"
        )
        assert (
            resolve_mlx_asr_model("qwen3_asr_0_6b")
            == "mlx-community/Qwen3-ASR-0.6B-4bit"
        )

    def test_dash_precision_is_normalized(self) -> None:
        assert (
            resolve_mlx_asr_model("qwen3-asr-1.7b", "4-bit")
            == "mlx-community/Qwen3-ASR-1.7B-4bit"
        )
        assert (
            resolve_mlx_asr_model("qwen3-asr-1.7b", "BF16")
            == "mlx-community/Qwen3-ASR-1.7B-bf16"
        )

    def test_full_mlx_id_with_matching_precision(self) -> None:
        assert (
            resolve_mlx_asr_model("mlx-community/Qwen3-ASR-1.7B-8bit", "8bit")
            == "mlx-community/Qwen3-ASR-1.7B-8bit"
        )
        assert (
            resolve_mlx_asr_model("mlx-community/Qwen3-ASR-0.6B-bf16")
            == "mlx-community/Qwen3-ASR-0.6B-bf16"
        )

    def test_full_mlx_id_with_conflicting_precision_raises(self) -> None:
        with pytest.raises(ProcessingFailedError, match="conflicts"):
            resolve_mlx_asr_model("mlx-community/Qwen3-ASR-1.7B-8bit", "4bit")

    def test_unknown_precision_raises(self) -> None:
        with pytest.raises(ProcessingFailedError, match="Unsupported precision"):
            resolve_mlx_asr_model("qwen3-asr-1.7b", "fp8")

    def test_model_adapter_accepts_alias_and_precision(self) -> None:
        assert (
            MLXAudioASRModel("qwen3-asr-0.6b", precision="6bit").model_id
            == "mlx-community/Qwen3-ASR-0.6B-6bit"
        )
        assert (
            MLXAudioASRModel(None, precision="8bit").model_id
            == "mlx-community/Qwen3-ASR-1.7B-8bit"
        )


class TestConvertToWavIfNeeded:
    """Verify convert_to_wav_if_needed behaviour for unsupported extensions."""

    def test_returns_none_for_supported_format(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.m4a"
        src.write_text("fake", encoding="utf-8")
        assert convert_to_wav_if_needed(src) is None

    def test_returns_none_for_wav(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.wav"
        src.write_text("fake", encoding="utf-8")
        assert convert_to_wav_if_needed(src) is None

    def test_mp4a_triggers_ffmpeg_conversion(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.mp4a"
        src.write_text("fake", encoding="utf-8")
        with (
            patch(
                "aimd.plugins.asr.audio_utils.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "aimd.plugins.asr.audio_utils.subprocess.run",
                return_value=_mock_ffmpeg_ok(),
            ),
        ):
            result = convert_to_wav_if_needed(src, temp_dir=tmp_path)
        assert result is not None
        assert result.suffix == ".wav"
        assert result.parent == tmp_path

    def test_mp4a_without_temp_dir_uses_system_tmp(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.mp4a"
        src.write_text("fake", encoding="utf-8")
        with (
            patch(
                "aimd.plugins.asr.audio_utils.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "aimd.plugins.asr.audio_utils.subprocess.run",
                return_value=_mock_ffmpeg_ok(),
            ),
        ):
            result = convert_to_wav_if_needed(src)
        assert result is not None
        assert result.suffix == ".wav"

    def test_ffmpeg_missing_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.mp4a"
        src.write_text("fake", encoding="utf-8")
        with patch(
            "aimd.plugins.asr.audio_utils.shutil.which",
            return_value=None,
        ):
            with pytest.raises(ProcessingFailedError, match="ffmpeg not found"):
                convert_to_wav_if_needed(src)


class TestGetTextFromAudioAcceptsMp4a:
    """Verify transcription processor accepts .mp4a files."""

    @pytest.mark.asyncio
    async def test_mp4a_accepted_not_rejected(self, tmp_path: Path) -> None:
        """mp4a should not raise UnsupportedInputError."""
        src = tmp_path / "sample.mp4a"
        src.write_text("fake", encoding="utf-8")

        with (
            patch(
                "aimd.plugins.asr._plugin.select_transcription_backend",
                return_value="mlx",
            ),
            patch(
                "aimd.plugins.asr.models.mlx.MLXAudioASRModel.transcribe",
                new_callable=AsyncMock,
                return_value="transcribed text",
            ),
        ):
            result = await transcribe_file(src, temp_dir=tmp_path)
        assert result == "transcribed text"

    @pytest.mark.asyncio
    async def test_unknown_extension_still_rejected(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.xyz"
        src.write_text("fake", encoding="utf-8")
        with pytest.raises(UnsupportedInputError, match="Unsupported file format"):
            await transcribe_file(src, temp_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_missing_input_raises_input_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.mp3"
        with pytest.raises(InputNotFoundError, match="Audio/video file not found"):
            await transcribe_file(missing, temp_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_transcribe_file_forwards_precision_to_backend(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "sample.m4a"
        src.write_text("fake", encoding="utf-8")

        with (
            patch(
                "aimd.plugins.asr._plugin.select_transcription_backend",
                return_value="mlx",
            ),
            patch("aimd.plugins.asr._plugin.MLXAudioASRModel") as mock_model_class,
        ):
            mock_model = MagicMock()
            mock_model.model_id = "mlx-community/Qwen3-ASR-1.7B-8bit"
            mock_model.transcribe = AsyncMock(return_value="transcribed text")
            mock_model_class.return_value = mock_model

            result = await transcribe_file(
                src,
                model="qwen3-asr-1.7b",
                precision="8bit",
                temp_dir=tmp_path,
            )

        assert result == "transcribed text"
        mock_model_class.assert_called_once_with("qwen3-asr-1.7b", precision="8bit")


class TestSegmentAudioCleanup:
    """segment_audio removes partial files when ffmpeg fails."""

    def test_segment_audio_cleans_partials_on_ffmpeg_failure(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "long.wav"
        src.write_text("fake", encoding="utf-8")

        def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
            # Simulate ffmpeg writing a partial segment then failing.
            (tmp_path / "seg_long_000.wav").write_text("partial", encoding="utf-8")
            result = MagicMock()
            result.returncode = 1
            result.stderr = "ffmpeg boom"
            return result

        with (
            patch(
                "aimd.plugins.asr.audio_utils.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "aimd.plugins.asr.audio_utils.subprocess.run",
                side_effect=_fake_run,
            ),
        ):
            with pytest.raises(ProcessingFailedError, match="segmentation failed"):
                segment_audio(src, segment_time_secs=600.0, temp_dir=tmp_path)

        assert list(tmp_path.glob("seg_long_*.wav")) == []
