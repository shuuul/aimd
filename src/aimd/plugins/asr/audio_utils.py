"""Shared audio preprocessing helpers for transcription backends."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from logly import logger

from aimd.core.errors import ProcessingFailedError

# Container/codec formats reliably read by both mlx-audio and librosa+soundfile.
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


def convert_to_wav_if_needed(source: Path, temp_dir: Path | None = None) -> Path | None:
    """Convert an unsupported audio file to 16 kHz mono WAV via ffmpeg.

    Returns the path to a temporary WAV file, or None if the format is
    already in :data:`SUPPORTED_AUDIO_FORMATS`. The caller is responsible
    for unlinking the returned path.

    When *temp_dir* is provided, the temporary WAV is created inside that
    directory so callers can redirect temp I/O to a sandbox-safe location
    via ``AIMD_TEMP_DIR``.
    """
    if source.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
        return None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ProcessingFailedError(
            f"Cannot convert {source.suffix} to WAV: ffmpeg not found. "
            "Install ffmpeg or provide a supported format "
            f"({', '.join(sorted(SUPPORTED_AUDIO_FORMATS))})."
        )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=str(temp_dir) if temp_dir else None
    )
    tmp.close()
    wav_path = Path(tmp.name)

    logger.info(f"Converting {source.suffix} to WAV for transcription backend")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-ar", "16000", "-ac", "1", wav_path.name],
        capture_output=True,
        cwd=wav_path.parent,
    )
    if result.returncode != 0:
        wav_path.unlink(missing_ok=True)
        raise ProcessingFailedError(
            f"ffmpeg conversion to WAV failed: {result.stderr.decode()}"
        )

    return wav_path


def get_audio_duration(source: Path) -> float:
    """Get the duration of the audio/video file in seconds using ffprobe or ffmpeg."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    # Fallback to ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ProcessingFailedError(
            "ffmpeg/ffprobe not found. Cannot determine audio duration."
        )

    result = subprocess.run([ffmpeg, "-i", str(source)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)

    raise ProcessingFailedError(
        f"Could not determine audio duration using ffmpeg/ffprobe. stderr: {result.stderr}"
    )


def segment_audio(
    source: Path,
    segment_time_secs: float = 600.0,
    temp_dir: Path | None = None,
) -> list[Path]:
    """Segment an audio file into chunks of `segment_time_secs` seconds using ffmpeg.

    Converts and splits the source file into standard 16 kHz mono WAV chunks.
    Returns a list of Paths to the segment files in WAV format.
    The caller is responsible for deleting these segment files when done.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ProcessingFailedError("ffmpeg not found. Cannot segment audio.")

    prefix = f"seg_{source.stem}_"
    out_pattern = f"{prefix}%03d.wav"
    out_path_template = Path(temp_dir or tempfile.gettempdir()) / out_pattern

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-f",
        "segment",
        "-segment_time",
        str(segment_time_secs),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path_template),
    ]
    logger.info(f"Segmenting audio into {segment_time_secs}s chunks")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProcessingFailedError(
            f"ffmpeg audio segmentation failed: {result.stderr}"
        )

    parent = out_path_template.parent
    glob_pattern = f"{prefix}*.wav"
    generated_segments = sorted(parent.glob(glob_pattern))

    if not generated_segments:
        raise ProcessingFailedError(
            "ffmpeg segmentation succeeded but no segment files were created."
        )

    return generated_segments


def detect_repetition_loop(text: str) -> bool:
    """Detect repetition loops in generated text.

    Checks for:
    1. Single characters repeated consecutively 8+ times (excluding whitespace).
    2. Multi-character patterns (length 2-10) repeated consecutively 4+ times (5+ times total).
    3. Space-separated words/phrases repeated consecutively.
    """
    if not text:
        return False

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    # 1. Any single non-space character repeated 8+ times consecutively.
    if re.search(r"([^\s])\1{7,}", cleaned_text):
        return True

    # 2. Short phrases/subsequences (length 2-10 chars) repeated 4+ times.
    for length in range(2, 11):
        pattern = rf"([^\s]{{{length}}})\1{{3,}}"
        if re.search(pattern, cleaned_text):
            return True

    # 3. Space-separated word sequences repeated 5+ times.
    words = cleaned_text.split()
    if len(words) >= 5:
        # Check single word repeats (6+ times)
        count = 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                count += 1
                if count >= 6:
                    return True
            else:
                count = 1

        # Check phrase of 2-5 words repeats (4+ times total)
        for phrase_len in range(2, 6):
            if len(words) >= phrase_len * 4:
                for i in range(len(words) - phrase_len * 4 + 1):
                    sub = words[i : i + phrase_len]
                    if (
                        words[i + phrase_len : i + 2 * phrase_len] == sub
                        and words[i + 2 * phrase_len : i + 3 * phrase_len] == sub
                        and words[i + 3 * phrase_len : i + 4 * phrase_len] == sub
                    ):
                        return True

    return False


def get_8bit_fallback_model(model_id: str) -> str | None:
    """Return the 8-bit version of the model ID if it is a 4-bit model, else None."""
    if not model_id:
        return None
    if "-4bit" in model_id:
        return model_id.replace("-4bit", "-8bit")
    return None
