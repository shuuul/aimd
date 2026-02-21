"""Audio transcription tools using yap CLI, mlx-whisper, and faster-whisper.

Supports transcription of:
- Audio files: mp3, wav, m4a, flac, ogg, aac, opus, wma, webm
- Video files: mp4, mkv, avi, mov, wmv, flv, ts, m4v (audio is extracted automatically)
"""

import asyncio
import platform
import shutil
import tempfile
from pathlib import Path

from logly import logger

from ..capabilities import resolve_engine_with_preflight
from ..errors import InputNotFoundError, ProcessingFailedError, UnsupportedInputError
from ..platform_utils import is_apple_silicon
from ..types import TextContext
from ..const import (
    YAP_SUPPORTED_LOCALES,
    LANGUAGE_TO_YAP_LOCALE,
    WHISPER_MODEL_SIZES,
    MLX_MODEL_MAPPINGS,
    AUDIO_EXTENSIONS,
)


async def get_text_from_audio(
    file_path: str | Path,
    engine: str = "auto",
    language: str | None = None,
    model_size: str = "large-v3-turbo",
) -> TextContext:
    """Extract text from audio or video file using the specified transcription engine.

    Supports both audio files (mp3, wav, m4a, etc.) and video files (mp4, mkv, etc.).
    For video files, the audio track is automatically extracted for transcription.

    Args:
        file_path: Path to the audio or video file
        engine: Transcription engine ("auto", "yap", "mlx", "cuda", "cpu")
        language: Whisper language code (e.g. "zh", "en", "ja"). None for auto-detection.
        model_size: Model size for whisper-based engines

    Returns:
        TextContext with title and transcribed text

    Raises:
        ValueError: If engine is invalid or unsupported, or file format not supported
        RuntimeError: If transcription fails
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise InputNotFoundError(f"Audio/video file not found: {file_path}")

    # Validate file extension
    file_ext = file_path.suffix.lower()
    if file_ext not in AUDIO_EXTENSIONS:
        raise UnsupportedInputError(
            f"Unsupported file format: {file_ext}. "
            f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    # Check if this is a video file (will need audio extraction)
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".m4v"}
    is_video = file_ext in video_extensions

    if is_video:
        logger.info(f"Processing video file: {file_path} (audio will be extracted)")
    else:
        logger.info(f"Processing audio file: {file_path}")

    # Determine the actual engine to use
    actual_engine = _resolve_engine(engine)

    logger.info(f"Using transcription engine: {actual_engine}")

    # Route to appropriate transcription function
    try:
        if actual_engine == "yap":
            yap_locale = _language_to_yap_locale(language)
            transcribed_text = await transcribe_audio_yap(file_path, yap_locale)
        elif actual_engine == "mlx":
            transcribed_text = await transcribe_audio_mlx(
                file_path, model_size, language
            )
        elif actual_engine in ("cuda", "cpu"):
            transcribed_text = await transcribe_audio_cuda(
                file_path, model_size, actual_engine, language
            )
        else:
            raise UnsupportedInputError(f"Unsupported engine: {actual_engine}")
    except Exception as e:
        # Add more context to the error
        if isinstance(e, (InputNotFoundError, UnsupportedInputError)):
            raise
        error_msg = str(e)
        if "format" in error_msg.lower() or "codec" in error_msg.lower():
            raise ProcessingFailedError(
                f"Transcription failed due to format/codec issue: {error_msg}. "
                f"Try converting the file to a standard format like mp3 or wav."
            ) from e
        raise ProcessingFailedError(f"Transcription failed: {error_msg}") from e

    return TextContext(title=file_path.stem, chunk_list=[transcribed_text])


async def transcribe_audio_yap(file_path: Path, locale: str = "zh_CN") -> str:
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
            raise ProcessingFailedError(f"yap command failed: {error_msg}")

        # Read transcribed text from output file
        if not temp_output_path.exists():
            raise ProcessingFailedError("yap did not create output file")

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
            raise ProcessingFailedError("yap produced empty transcription")

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
        raise ProcessingFailedError("mlx engine is only available on macOS")

    if not is_apple_silicon():
        raise ProcessingFailedError("mlx engine requires Apple Silicon (M1/M2/M3/M4)")

    try:
        import mlx_whisper
    except ImportError:
        raise ProcessingFailedError(
            "mlx-whisper library is not installed. Please install it: "
            "pip install mlx-whisper"
        )

    if model_size not in MLX_MODEL_MAPPINGS:
        raise UnsupportedInputError(
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
            raise ProcessingFailedError("MLX Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters with MLX"
        )
        return transcribed_text

    except Exception as e:
        raise ProcessingFailedError(f"MLX transcription failed: {e}") from e


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
        raise ProcessingFailedError(
            "faster-whisper library is not installed. Please install it: "
            "pip install faster-whisper"
        )

    if model_size not in WHISPER_MODEL_SIZES:
        raise UnsupportedInputError(
            f"Unsupported model size: {model_size}. Supported: {WHISPER_MODEL_SIZES}"
        )

    # Auto-detect CUDA availability if device is "cuda"
    if device == "cuda":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not installed, falling back to CPU")
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
            raise ProcessingFailedError("Whisper produced empty transcription")

        logger.info(
            f"Successfully transcribed {len(transcribed_text)} characters, "
            f"detected language: {info.language} ({info.language_probability:.2f})"
        )
        return transcribed_text

    except Exception as e:
        raise ProcessingFailedError(f"Whisper transcription failed: {e}") from e


def _resolve_engine(engine: str) -> str:
    """Resolve engine preference to actual engine based on platform capabilities.

    Args:
        engine: Engine preference ("auto", "yap", "mlx", "cuda", "cpu")

    Returns:
        Actual engine to use

    Raises:
        ValueError: If engine is invalid
    """
    return resolve_engine_with_preflight(engine)


def _language_to_yap_locale(language: str | None) -> str:
    """Convert a Whisper language code to a yap locale code.

    yap requires full locale codes (e.g. "zh_CN", "en_US"), but the public API
    uses Whisper-style short codes (e.g. "zh", "en"). This function maps between
    the two. Defaults to "zh_CN" when language is None.

    Args:
        language: Whisper language code (e.g. "zh", "en"), or None.

    Returns:
        yap-compatible locale string.

    Raises:
        ValueError: If the language code has no known yap locale mapping.
    """
    if language is None:
        return "zh_CN"

    lang = language.lower()
    if lang in LANGUAGE_TO_YAP_LOCALE:
        return LANGUAGE_TO_YAP_LOCALE[lang]

    raise ValueError(
        f"Unsupported language for yap engine: '{language}'. "
        f"Supported: {list(LANGUAGE_TO_YAP_LOCALE.keys())}"
    )
