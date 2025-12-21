"""Audio transcription tools using yap CLI, mlx-whisper, and faster-whisper."""

import asyncio
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import torch
from logly import logger

from ..types import TextContext
from ..const import (
    TRANSCRIPTION_ENGINES,
    YAP_SUPPORTED_LOCALES,
    WHISPER_MODEL_SIZES,
    MLX_MODEL_MAPPINGS,
)


async def get_text_from_audio(
    file_path: str | Path,
    engine: str = "auto",
    locale: str | None = None,
    model_size: str = "large-v3-turbo",
) -> TextContext:
    """Extract text from audio file using the specified transcription engine.

    Args:
        file_path: Path to the audio file
        engine: Transcription engine ("auto", "yap", "mlx", "cuda", "cpu")
        locale: Language locale for transcription
        model_size: Model size for whisper-based engines

    Returns:
        TextContext with title and transcribed text

    Raises:
        ValueError: If engine is invalid or unsupported
        RuntimeError: If transcription fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Determine the actual engine to use
    actual_engine = _resolve_engine(engine)

    logger.info(f"Transcribing audio file: {file_path} using engine: {actual_engine}")

    # Route to appropriate transcription function
    if actual_engine == "yap":
        transcribed_text = await transcribe_audio_yap(file_path, locale or "zh_CN")
    elif actual_engine == "mlx":
        transcribed_text = await transcribe_audio_mlx(file_path, model_size, locale)
    elif actual_engine in ("cuda", "cpu"):
        transcribed_text = await transcribe_audio_cuda(
            file_path, model_size, actual_engine, locale
        )
    else:
        raise ValueError(f"Unsupported engine: {actual_engine}")

    return TextContext(title=file_path.stem, chunk_list=[transcribed_text])


async def transcribe_audio_yap(
    file_path: Path, locale: Literal["zh_CN", "en_US"] = "zh_CN"
) -> str:
    """Transcribe audio using yap CLI (macOS only).

    Args:
        file_path: Path to the audio file
        locale: Language locale for transcription

    Returns:
        Transcribed text

    Raises:
        RuntimeError: If yap is not available or transcription fails
    """
    if platform.system() != "Darwin":
        raise RuntimeError("yap engine is only available on macOS")

    if not shutil.which("yap"):
        raise RuntimeError(
            "yap CLI tool is not installed. Please install it from: "
            "https://github.com/finnvoor/yap"
        )

    if locale not in YAP_SUPPORTED_LOCALES:
        raise ValueError(
            f"Unsupported locale: {locale}. Supported: {YAP_SUPPORTED_LOCALES}"
        )

    logger.info(f"Transcribing with yap: {file_path}, locale: {locale}")

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_file:
        temp_output_path = Path(temp_file.name)

    try:
        # Build yap command
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

        # Run yap command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"yap command failed: {error_msg}")

        # Read transcribed text from output file
        if not temp_output_path.exists():
            raise RuntimeError("yap did not create output file")

        # Try different encodings for yap output
        try:
            transcribed_text = temp_output_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            try:
                transcribed_text = temp_output_path.read_text(encoding="gb2312").strip()
            except UnicodeDecodeError:
                transcribed_text = temp_output_path.read_text(
                    encoding="latin-1"
                ).strip()

        if not transcribed_text:
            raise RuntimeError("yap produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with yap"
        )
        return transcribed_text

    finally:
        # Clean up temporary file
        if temp_output_path.exists():
            temp_output_path.unlink()


async def transcribe_audio_mlx(
    file_path: Path,
    model_size: str = "large-v3-turbo",
    language: str | None = None,
) -> str:
    """Transcribe audio using mlx-whisper (Apple Silicon only).

    Args:
        file_path: Path to the audio file
        model_size: Whisper model size
        language: Language code (None for auto-detection)

    Returns:
        Transcribed text

    Raises:
        RuntimeError: If mlx-whisper is not available or transcription fails
    """
    if platform.system() != "Darwin":
        raise RuntimeError("mlx engine is only available on macOS")

    if not _is_apple_silicon():
        raise RuntimeError("mlx engine requires Apple Silicon (M1/M2/M3/M4)")

    try:
        import mlx_whisper
    except ImportError:
        raise RuntimeError(
            "mlx-whisper library is not installed. Please install it: "
            "pip install mlx-whisper"
        )

    if model_size not in MLX_MODEL_MAPPINGS:
        raise ValueError(
            f"Unsupported model size: {model_size}. Supported: {list(MLX_MODEL_MAPPINGS.keys())}"
        )

    mlx_model_path = f"mlx-community/{MLX_MODEL_MAPPINGS[model_size]}"

    logger.info(f"Transcribing with MLX model: {mlx_model_path}")

    try:
        # Run transcription in thread pool to avoid blocking
        def _transcribe():
            return mlx_whisper.transcribe(
                str(file_path),
                path_or_hf_repo=mlx_model_path,
                language=language,
                word_timestamps=True,
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _transcribe)

        transcribed_text = result["text"].strip()

        if not transcribed_text:
            raise RuntimeError("MLX Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with MLX"
        )
        return transcribed_text

    except Exception as e:
        raise RuntimeError(f"MLX transcription failed: {e}") from e


async def transcribe_audio_cuda(
    file_path: Path,
    model_size: str = "large-v3-turbo",
    device: str = "cpu",
    language: str | None = None,
) -> str:
    """Transcribe audio using faster-whisper with CPU or CUDA.

    Args:
        file_path: Path to the audio file
        model_size: Whisper model size
        device: Device to use ("cpu" or "cuda")
        language: Language code (None for auto-detection)

    Returns:
        Transcribed text

    Raises:
        RuntimeError: If faster-whisper is not available or transcription fails
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper library is not installed. Please install it: "
            "pip install faster-whisper"
        )

    if model_size not in WHISPER_MODEL_SIZES:
        raise ValueError(
            f"Unsupported model size: {model_size}. Supported: {WHISPER_MODEL_SIZES}"
        )

    # Auto-detect CUDA availability if device is "cuda"
    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

    logger.info(
        f"Transcribing with faster-whisper: model={model_size}, device={device}"
    )

    try:
        # Determine compute type based on device
        compute_type = "float16" if device == "cuda" else "int8"

        # Run transcription in thread pool to avoid blocking
        def _transcribe():
            # Load model
            model = WhisperModel(model_size, device=device, compute_type=compute_type)

            # Transcribe with optional language specification
            segments, info = model.transcribe(
                str(file_path), language=language, beam_size=5
            )

            # Collect all segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + "\n"

            return transcribed_text.strip(), info

        loop = asyncio.get_event_loop()
        transcribed_text, info = await loop.run_in_executor(None, _transcribe)

        if not transcribed_text:
            raise RuntimeError("Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters, "
            f"detected language: {info.language} ({info.language_probability:.2f})"
        )
        return transcribed_text

    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}") from e


def _resolve_engine(engine: str) -> str:
    """Resolve engine preference to actual engine based on platform capabilities.

    Args:
        engine: Engine preference ("auto", "yap", "mlx", "cuda", "cpu")

    Returns:
        Actual engine to use

    Raises:
        ValueError: If engine is invalid
    """
    if engine not in TRANSCRIPTION_ENGINES:
        raise ValueError(
            f"Invalid engine '{engine}'. Valid options: {TRANSCRIPTION_ENGINES}"
        )

    if engine != "auto":
        return engine

    system = platform.system().lower()

    # Auto detection logic
    if system == "darwin":
        # On macOS, prefer yap if available, then mlx if on Apple Silicon, then cpu
        if shutil.which("yap"):
            return "yap"
        elif _is_apple_silicon():
            return "mlx"
        else:
            return "cpu"
    else:
        # On non-macOS, use cuda if available, otherwise cpu
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon.

    Returns:
        True if running on Apple Silicon (M1/M2/M3/M4) macOS
    """
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        cpu_info = result.stdout.strip().lower()
        return "apple" in cpu_info and any(
            m in cpu_info for m in ["m1", "m2", "m3", "m4"]
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
