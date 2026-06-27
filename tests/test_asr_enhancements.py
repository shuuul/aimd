import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aimd.plugins.asr.audio_utils import (
    detect_repetition_loop,
    get_8bit_fallback_model,
)
from aimd.plugins.asr._plugin import transcribe_file


def test_detect_repetition_loop() -> None:
    # Test non-repetitive text
    assert not detect_repetition_loop("这是一个正常的测试句子。")
    assert not detect_repetition_loop(
        "Hello this is a normal sentence for testing ASR outputs."
    )

    # Test single character repetition
    assert detect_repetition_loop("啊啊啊啊啊啊啊啊")  # 8 times
    assert not detect_repetition_loop("啊啊啊啊")  # 4 times

    # Test short phrase repetition
    assert detect_repetition_loop("我们我们我们我们我们")  # 5 times
    assert not detect_repetition_loop("我们我们")

    # Test space separated word repetition
    assert detect_repetition_loop("the the the the the the")  # 6 times
    assert not detect_repetition_loop("the the the")

    # Test space separated phrase repetition
    assert detect_repetition_loop(
        "in the end in the end in the end in the end"
    )  # 4 times
    assert not detect_repetition_loop("in the end in the end")


def test_get_8bit_fallback_model() -> None:
    assert (
        get_8bit_fallback_model("mlx-community/Qwen3-ASR-1.7B-4bit")
        == "mlx-community/Qwen3-ASR-1.7B-8bit"
    )
    assert (
        get_8bit_fallback_model("mlx-community/Qwen3-ASR-0.6B-4bit")
        == "mlx-community/Qwen3-ASR-0.6B-8bit"
    )
    assert get_8bit_fallback_model("Qwen3-ASR-1.7B-8bit") is None
    assert get_8bit_fallback_model(None) is None


@pytest.mark.asyncio
async def test_transcribe_file_fallback_on_repetition(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("fake wav data")

    # We mock duration to be short, so no segmentation occurs.
    # We mock MLXAudioASRModel to return repeating text first, then normal text.
    with (
        patch(
            "aimd.plugins.asr._plugin.select_transcription_backend",
            return_value="mlx",
        ),
        patch("aimd.plugins.asr.audio_utils.get_audio_duration", return_value=120.0),
        patch("aimd.plugins.asr._plugin.MLXAudioASRModel") as MockModelClass,
    ):
        mock_4bit_model = MagicMock()
        mock_4bit_model.model_id = "mlx-community/Qwen3-ASR-1.7B-4bit"
        mock_4bit_model.transcribe = AsyncMock(return_value="的 的 的 的 的 的 的")

        mock_8bit_model = MagicMock()
        mock_8bit_model.model_id = "mlx-community/Qwen3-ASR-1.7B-8bit"
        mock_8bit_model.transcribe = AsyncMock(return_value="这是一段正常的识别文本")

        MockModelClass.side_effect = [mock_4bit_model, mock_8bit_model]

        result = await transcribe_file(fake_audio)

        assert result == "这是一段正常的识别文本"
        mock_4bit_model.transcribe.assert_called_once()
        mock_8bit_model.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_file_segmentation(tmp_path: Path) -> None:
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("fake wav data")

    fake_seg1 = tmp_path / "seg_test_000.wav"
    fake_seg1.write_text("seg1")
    fake_seg2 = tmp_path / "seg_test_001.wav"
    fake_seg2.write_text("seg2")

    # We mock duration to be > 600.0, so segmentation occurs.
    with (
        patch(
            "aimd.plugins.asr._plugin.select_transcription_backend",
            return_value="mlx",
        ),
        patch("aimd.plugins.asr.audio_utils.get_audio_duration", return_value=1200.0),
        patch(
            "aimd.plugins.asr.audio_utils.segment_audio",
            return_value=[fake_seg1, fake_seg2],
        ) as mock_segment,
        patch("aimd.plugins.asr._plugin.MLXAudioASRModel") as MockModelClass,
    ):
        mock_model = MagicMock()
        mock_model.model_id = "mlx-community/Qwen3-ASR-1.7B-4bit"
        mock_model.transcribe = AsyncMock()
        mock_model.transcribe.side_effect = ["hello", "world"]

        MockModelClass.return_value = mock_model

        result = await transcribe_file(fake_audio)

        assert result == "hello world"
        mock_segment.assert_called_once_with(
            fake_audio, segment_time_secs=600.0, temp_dir=None
        )
        assert mock_model.transcribe.call_count == 2
        # Check files were deleted
        assert not fake_seg1.exists()
        assert not fake_seg2.exists()
