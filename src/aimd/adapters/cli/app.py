"""CLI adapter for aimd."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from logly import logger

from ...application.bootstrap import build_container
from ...application.models import ProcessInput
from ...application.services.output_writer import build_output_text
from ...application.use_cases.process_input import ensure_supported_input
from ...errors import AimdError
from ...infrastructure.documents.pandoc_reader import is_supported_file
from ...utils import create_output_path_from_title

load_dotenv()

logger.remove_all()
logger.configure(color=True, auto_sink=False)
logger.add(
    "console",
    filter_min_level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function} - {message}",
)

app = typer.Typer(
    name="aimd",
    help="Context preparation tool for LLM workflows - Transcribe audio/video and convert documents",
    no_args_is_help=True,
)


def _configure_logging(log_level: str) -> None:
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if str(log_level).upper() not in valid_levels:
        typer.echo(
            f"Error: Invalid log level '{log_level}'. Valid levels: {', '.join(valid_levels)}",
            err=True,
        )
        raise typer.Exit(1)

    logger.remove_all()
    logger.configure(color=True, auto_sink=False)
    logger.add(
        "console",
        filter_min_level=str(log_level).upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function} - {message}",
    )

    if str(log_level).upper() == "DEBUG":
        logger.debug(f"Logging level configured to: {log_level}")


@app.command()
def process(
    input_source: str = typer.Argument(
        ...,
        help="Audio file, video file, video URL, or document to process",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If not specified, auto-generated from input",
    ),
    transcribe_engine: str = typer.Option(
        "auto",
        "--engine",
        "-e",
        help="Transcription engine: mlx (Apple Silicon), qwen (Linux/CUDA), funasr (CPU/CUDA), yap (macOS).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model for transcription. For mlx: mlx-community/Qwen3-ASR-1.7B-8bit (default), "
        "or mlx-community/Fun-ASR-Nano-2512-4bit (Fun-ASR-Nano via mlx-audio-plus). "
        "For qwen: Qwen/Qwen3-ASR-1.7B (default). "
        "For funasr: FunAudioLLM/Fun-ASR-Nano-2512 (default) or FunAudioLLM/SenseVoiceSmall.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code for transcription (e.g., zh, en, ja).",
    ),
    save_original: Optional[Path] = typer.Option(
        None,
        "--save-original",
        "-s",
        help="Save original downloaded audio/video file to a path or directory.",
    ),
    cookies: Optional[Path] = typer.Option(
        None,
        "--cookies",
        "-c",
        help="Path to Netscape-format cookies file for URL extraction.",
    ),
    cookies_from_browser: Optional[str] = typer.Option(
        None,
        "--cookies-from-browser",
        help="Browser cookie source for URL extraction.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
    raw_transcript: bool = typer.Option(
        False,
        "--raw-transcript",
        help="Preserve original subtitle formatting (SRT/VTT timestamps). "
        "By default, subtitles are simplified to plain text.",
    ),
    temp_dir: Optional[Path] = typer.Option(
        None,
        "--temp-dir",
        help="Custom temporary directory for intermediate files. "
        "Overrides AIMD_TEMP_DIR env var. Useful for sandboxed environments.",
        envvar="AIMD_TEMP_DIR",
    ),
) -> None:
    """Process audio/video/url/doc inputs to markdown."""
    _configure_logging(log_level)
    container = build_container()

    try:
        task_type = ensure_supported_input(input_source, is_supported_file)
    except AimdError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    logger.info(f"Input: {input_source}")
    logger.info(f"Task: {task_type}")
    if task_type == "transcript":
        logger.info(f"Transcription Engine: {transcribe_engine}")

    async def run_processing() -> None:
        try:
            resolved_temp_dir = temp_dir
            if resolved_temp_dir is not None:
                resolved_temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using custom temp directory: {resolved_temp_dir}")

            result = await container.process_input_use_case.execute(
                ProcessInput(
                    input_source=input_source,
                    output_file=output_file,
                    transcribe_engine=transcribe_engine,
                    model=model,
                    language=language,
                    save_original=save_original,
                    cookies=cookies,
                    cookies_from_browser=cookies_from_browser,
                    temp_dir=resolved_temp_dir,
                    raw_transcript=raw_transcript,
                )
            )

            if result.platform:
                logger.info(f"Platform: {result.platform}")

            if result.task_type == "convert" and result.output_dir is not None:
                input_path = Path(input_source)
                logger.info(f"EPUB converted with images to: {result.output_dir}")
                logger.info(f"Main file: {result.output_dir / f'{input_path.stem}.md'}")
                logger.info(f"Images extracted to: {result.output_dir / 'images'}")
                typer.echo("Successfully converted EPUB with images")
                typer.echo(f"Output saved to {result.output_dir}")
                return

            final_output_file = output_file
            if final_output_file is None:
                suffix = (
                    "transcript" if result.task_type == "transcript" else "converted"
                )
                default_dir = (
                    Path.cwd()
                    if result.task_type == "transcript"
                    else Path(input_source).parent
                )
                final_output_file = create_output_path_from_title(
                    result.text_context.title,
                    suffix,
                    default_dir,
                )

            logger.info(f"Output: {final_output_file}")
            text = build_output_text(result.task_type, result.text_context.chunk_list)
            final_output_file.parent.mkdir(parents=True, exist_ok=True)
            final_output_file.write_text(text, encoding="utf-8")

            if result.task_type == "transcript":
                logger.info(f"Transcript saved to: {final_output_file}")
                typer.echo("Successfully transcribed")
            else:
                logger.info(f"Converted file saved to: {final_output_file}")
                typer.echo("Successfully converted")
            typer.echo(f"Output saved to {final_output_file}")
        except AimdError as e:
            logger.error(str(e))
            raise typer.Exit(1)
        except Exception as e:
            task_name = "Transcription" if task_type == "transcript" else "Conversion"
            logger.error(f"{task_name} failed: {e}")
            raise typer.Exit(1)

    asyncio.run(run_processing())


def main() -> None:
    """Entry point for CLI app."""
    app()
