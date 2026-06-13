"""Tests for audio transcoding support (.mp4a etc.)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aimd.const import AUDIO_EXTENSIONS
from aimd.errors import ProcessingFailedError, UnsupportedInputError
from aimd.infrastructure.transcription.audio_utils import (
    SUPPORTED_AUDIO_FORMATS,
    convert_to_wav_if_needed,
)
from aimd.infrastructure.transcription.processor import get_text_from_audio


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
                "aimd.infrastructure.transcription.audio_utils.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "aimd.infrastructure.transcription.audio_utils.subprocess.run",
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
                "aimd.infrastructure.transcription.audio_utils.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "aimd.infrastructure.transcription.audio_utils.subprocess.run",
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
            "aimd.infrastructure.transcription.audio_utils.shutil.which",
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
                "aimd.infrastructure.transcription.processor.resolve_engine_with_preflight",
                return_value="mlx",
            ),
            patch(
                "aimd.infrastructure.transcription.processor.transcribe_audio_mlx",
                new_callable=AsyncMock,
                return_value="transcribed text",
            ),
        ):
            result = await get_text_from_audio(src, engine="mlx", temp_dir=tmp_path)
        assert result.chunk_list == ["transcribed text"]

    @pytest.mark.asyncio
    async def test_unknown_extension_still_rejected(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.xyz"
        src.write_text("fake", encoding="utf-8")
        with pytest.raises(UnsupportedInputError, match="Unsupported file format"):
            await get_text_from_audio(src, engine="mlx", temp_dir=tmp_path)
