"""CLI entrypoint for aimd."""

import asyncio
from pathlib import Path
import re
from typing import cast, Optional

import typer
from dotenv import load_dotenv
from logly import logger

from aimd.interfaces.output import (
    MODEL_HELP_TEXT,
    PRECISION_HELP_TEXT,
    persist_output,
)
from aimd.core.errors import AimdError
from aimd.core.models import ProcessInput, TaskType
from aimd.core.process import process_input

load_dotenv()

logger.remove()
logger.add(
    "stderr",
    level="INFO",
    colorize=True,
    format="{time:%Y-%m-%d %H:%M:%S} | {level: <8} | {module}:{function} - {message}",
)

app = typer.Typer(
    name="aimd",
    help="Context preparation tool for LLM workflows - Transcribe audio/video and convert documents",
    no_args_is_help=True,
)


def _create_output_path_from_title(
    title: str, template_name: str, current_dir: Path | None = None
) -> Path:
    """Create a markdown output path from a result title."""
    if current_dir is None:
        current_dir = Path.cwd()

    sanitized = re.sub(r'[<>:"/\\|?*]', "_", title)
    sanitized = re.sub(r"\s+", "_", sanitized.strip()).strip("._")
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip("._")
    if not sanitized:
        sanitized = "output"
    return current_dir / f"{sanitized}_{template_name}.md"


def _configure_logging(log_level: str) -> None:
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if str(log_level).upper() not in valid_levels:
        typer.echo(
            f"Error: Invalid log level '{log_level}'. Valid levels: {', '.join(valid_levels)}",
            err=True,
        )
        raise typer.Exit(1)

    logger.remove()
    logger.add(
        "stderr",
        level=str(log_level).upper(),
        colorize=True,
        format="{time:%Y-%m-%d %H:%M:%S} | {level: <8} | {module}:{function} - {message}",
    )

    if str(log_level).upper() == "DEBUG":
        logger.debug(f"Logging level configured to: {log_level}")


@app.command()
def process(
    input_source: str = typer.Argument(
        ...,
        help="Audio file, video file, video URL, document, image, or scanned PDF to process",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Optional explicit task: transcript, convert, or ocr. Defaults to auto-routing.",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path. If not specified, auto-generated from input",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=MODEL_HELP_TEXT,
    ),
    precision: Optional[str] = typer.Option(
        None,
        "--precision",
        help=PRECISION_HELP_TEXT,
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

    requested_task: TaskType | None = None
    if task is not None:
        normalized_task = task.strip().lower()
        if normalized_task not in {"transcript", "convert", "ocr"}:
            typer.echo(
                "Error: Unsupported task. Supported tasks: transcript, convert, ocr.",
                err=True,
            )
            raise typer.Exit(1)
        requested_task = cast(TaskType, normalized_task)

    logger.info(f"Input: {input_source}")

    async def run_processing() -> None:
        try:
            resolved_temp_dir = temp_dir
            if resolved_temp_dir is not None:
                resolved_temp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using custom temp directory: {resolved_temp_dir}")

            result = await process_input(
                ProcessInput(
                    input_source=input_source,
                    task_type=requested_task,
                    model=model,
                    language=language,
                    start=start,
                    end=end,
                    save_original=save_original,
                    cookies=cookies,
                    cookies_from_browser=cookies_from_browser,
                    temp_dir=resolved_temp_dir,
                    raw_transcript=raw_transcript,
                    precision=precision,
                )
            )
            logger.info(f"Task: {result.task_type}")

            if result.platform:
                logger.info(f"Platform: {result.platform}")

            if result.task_type == "convert" and result.output_dir is not None:
                input_path = Path(input_source)
                if output_file is not None:
                    logger.warning(
                        "Ignoring --output for document asset conversions; "
                        "output is a directory."
                    )
                logger.info(f"Document converted with assets to: {result.output_dir}")
                logger.info(f"Main file: {result.output_dir / f'{input_path.stem}.md'}")
                logger.info(f"Images extracted to: {result.output_dir / 'images'}")
                typer.echo("Successfully converted document with assets")
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
                final_output_file = _create_output_path_from_title(
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
            raise typer.Exit(1) from None
        except Exception as e:
            task_name = {
                "transcript": "Transcription",
                "convert": "Conversion",
                "ocr": "OCR",
            }.get(requested_task, "Processing")
            logger.error(f"{task_name} failed: {e}")
            raise typer.Exit(1) from None

    asyncio.run(run_processing())


def main() -> None:
    """Entry point for CLI app."""
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
