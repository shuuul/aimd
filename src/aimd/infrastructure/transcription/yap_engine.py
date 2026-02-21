"""yap transcription engine implementation."""

import asyncio
import platform
import shutil
import tempfile
from pathlib import Path

from logly import logger

from ...const import YAP_SUPPORTED_LOCALES
from ...errors import ProcessingFailedError, UnsupportedInputError


async def transcribe_audio_yap(file_path: Path, locale: str = "zh_CN") -> str:
    """Transcribe audio using yap CLI (macOS only)."""
    if platform.system() != "Darwin":
        raise ProcessingFailedError("yap engine is only available on macOS")

    if not shutil.which("yap"):
        raise ProcessingFailedError(
            "yap CLI tool is not installed. Please install it from: "
            "https://github.com/finnvoor/yap"
        )

    if locale not in YAP_SUPPORTED_LOCALES:
        raise UnsupportedInputError(
            f"Unsupported locale: {locale}. Supported: {YAP_SUPPORTED_LOCALES}"
        )

    logger.info(f"Transcribing with yap: {file_path}, locale: {locale}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
        temp_output_path = Path(temp_file.name)

    try:
        cmd = [
            "yap",
            "transcribe",
            "--locale",
            locale,
            str(file_path),
            "--output-file",
            str(temp_output_path),
        ]

        logger.debug(f"Running command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise ProcessingFailedError(f"yap command failed: {error_msg}")

        if not temp_output_path.exists():
            raise ProcessingFailedError("yap did not create output file")

        try:
            transcribed_text = temp_output_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            try:
                transcribed_text = temp_output_path.read_text(encoding="gb2312").strip()
            except UnicodeDecodeError:
                transcribed_text = temp_output_path.read_text(encoding="latin-1").strip()

        if not transcribed_text:
            raise ProcessingFailedError("yap produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with yap"
        )
        return transcribed_text
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()
