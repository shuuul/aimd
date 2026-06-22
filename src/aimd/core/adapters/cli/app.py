"""CLI adapter for aimd."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from logly import logger

from ...application.bootstrap import build_container
from ...application.models import ProcessInput
from ...application.services.output_writer import persist_output
from ...errors import AimdError
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
        help="Audio file, video file, video URL, document, image, or scanned PDF to process",
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
        help="Engine. Transcript: mlx (Apple Silicon) or qwen (Linux/CUDA). OCR: mlx4ocr (macOS) or transformers (Linux).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model for transcription, or OCR model. macOS OCR: paddleocr_v6 "
        "(default), glm_ocr, or paddleocr_vl. Linux/CUDA OCR: got_ocr "
        "(default), glm_ocr, paddleocr_vl, or a Hugging Face model ID. "
        "mlx defaults to mlx-community/Qwen3-ASR-1.7B-4bit "
        "and also supports other documented mlx-audio STT model IDs. "
        "qwen supports Qwen/Qwen3-ASR-1.7B (default) or Qwen/Qwen3-ASR-0.6B.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code/hint for transcription or OCR (e.g., zh, en, ja).",
    ),
    start: Optional[int] = typer.Option(
        None,
        "--start",
        help="0-based inclusive start page for OCR PDF inputs.",
    ),
    end: Optional[int] = typer.Option(
        None,
        "--end",
        help="0-based inclusive end page for OCR PDF inputs.",
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
        route = container.process_input_use_case.ensure_supported_input(input_source)
        task_type = route.task_type
    except AimdError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    logger.info(f"Input: {input_source}")
    logger.info(f"Source: {route.source_kind}")
    logger.info(f"Task: {task_type}")
    if task_type == "transcript":
        logger.info(f"Transcription Engine: {transcribe_engine}")
    elif task_type == "ocr":
        logger.info(f"OCR Engine: {transcribe_engine}")

    async def run_processing() -> None:
        try:
            resolved_temp_dir = temp_dir
            if resolved_temp_dir is not None:
                resolved_temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using custom temp directory: {resolved_temp_dir}")

            result = await container.process_input_use_case.execute(
                ProcessInput(
                    input_source=input_source,
                    transcribe_engine=transcribe_engine,
                    model=model,
                    language=language,
                    start=start,
                    end=end,
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
                logger.info(f"Book converted with images to: {result.output_dir}")
                logger.info(f"Main file: {result.output_dir / f'{input_path.stem}.md'}")
                logger.info(f"Images extracted to: {result.output_dir / 'images'}")
                typer.echo("Successfully converted book with images")
                typer.echo(f"Output saved to {result.output_dir}")
                return

            final_output_file = output_file
            if final_output_file is None:
                suffix = {
                    "transcript": "transcript",
                    "convert": "converted",
                    "ocr": "ocr",
                }[result.task_type]
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
            persist_output(
                final_output_file,
                result.task_type,
                result.text_context.chunk_list,
            )

            if result.task_type == "transcript":
                logger.info(f"Transcript saved to: {final_output_file}")
                typer.echo("Successfully transcribed")
            elif result.task_type == "ocr":
                logger.info(f"OCR output saved to: {final_output_file}")
                typer.echo("Successfully OCR processed")
            else:
                logger.info(f"Converted file saved to: {final_output_file}")
                typer.echo("Successfully converted")
            typer.echo(f"Output saved to {final_output_file}")
        except AimdError as e:
            logger.error(str(e))
            raise typer.Exit(1)
        except Exception as e:
            task_name = {
                "transcript": "Transcription",
                "convert": "Conversion",
                "ocr": "OCR",
            }.get(task_type, "Processing")
            logger.error(f"{task_name} failed: {e}")
            raise typer.Exit(1)

    asyncio.run(run_processing())


def main() -> None:
    """Entry point for CLI app."""
    app()
