"""Shared audio preprocessing helpers for transcription engines."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from logly import logger

from ...errors import ProcessingFailedError

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

    logger.info(f"Converting {source.suffix} to WAV for transcription engine")
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
