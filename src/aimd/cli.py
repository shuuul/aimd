"""Command-line interface for aimd - Context preparation tool for LLM workflows."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from logly import logger
from dotenv import load_dotenv

from .const import AUDIO_EXTENSIONS, EPUB_EXTENSIONS
from .utils import (
    is_url,
    create_output_path_from_title,
)
from .tool.file import (
    get_text_from_file,
    is_supported_file,
    process_epub_with_images,
)
from .tool.audio import get_text_from_audio
from .tool.url import get_text_from_url

load_dotenv()

# Configure logly
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


def _configure_logging(log_level: str):
    """Configure logly logging with the specified level."""
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


def _get_task_type(input_source: str) -> str:
    """Determine task type based on input source."""
    if is_url(input_source):
        return "transcript"
    try:
        file_path = Path(input_source)
        if file_path.exists():
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                return "transcript"
            if is_supported_file(file_path):
                return "convert"
    except (OSError, ValueError):
        pass
    return "unknown"


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
        help="Transcription engine: yap (macOS), mlx (Apple Silicon), cuda, cpu. Used for audio/video.",
    ),
    locale: Optional[str] = typer.Option(
        None,
        "--locale",
        "-l",
        help="Language locale for transcription (e.g., zh_CN, en_US). Used for audio/video.",
    ),
    save_original: Optional[Path] = typer.Option(
        None,
        "--save-original",
        "-s",
        help="Save the original downloaded audio/video file to specified path or directory.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
):
    """Process audio, video, video URLs, or documents to markdown.

    Automatically detects input type:
    - Audio/Video files: Transcribe to markdown
    - Video URLs: Extract subtitles/transcribe to markdown
    - Documents (epub, txt, etc.): Convert to markdown
    """
    _configure_logging(log_level)

    task_type = _get_task_type(input_source)

    if task_type == "unknown":
        typer.echo(f"Error: Unsupported input source: {input_source}", err=True)
        typer.echo(
            "Supported inputs: audio files, video files, video URLs, or documents (epub, txt, etc.)",
            err=True,
        )
        raise typer.Exit(1)

    logger.info(f"Input: {input_source}")
    logger.info(f"Task: {task_type}")
    if task_type == "transcript":
        logger.info(f"Transcription Engine: {transcribe_engine}")

    async def run_processing():
        try:
            if task_type == "transcript":
                await _process_transcript(
                    input_source, output_file, transcribe_engine, locale, save_original
                )
            else:
                await _process_convert(input_source, output_file)

        except Exception as e:
            task_name = "Transcription" if task_type == "transcript" else "Conversion"
            logger.error(f"{task_name} failed: {e}")
            raise typer.Exit(1)

    asyncio.run(run_processing())


async def _process_transcript(
    input_source: str,
    output_file: Optional[Path],
    engine: str,
    locale: str | None,
    save_original: Optional[Path] = None,
):
    """Process audio/video transcription."""
    if is_url(input_source):
        logger.info(f"Getting transcript from URL: {input_source}")
        text_context = await get_text_from_url(
            input_source, engine, locale, save_original
        )
    else:
        file_path = Path(input_source)
        logger.info(f"Getting transcript from: {file_path}")
        text_context = await get_text_from_audio(file_path, engine, locale)

    if output_file is None:
        output_file = create_output_path_from_title(
            text_context.title, "transcript", Path.cwd()
        )

    logger.info(f"Output: {output_file}")

    text = text_context.chunk_list[0] if text_context.chunk_list else ""
    if not text:
        raise ValueError(f"Failed to get transcript from: {input_source}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info(f"Transcript saved to: {output_file}")

    typer.echo("Successfully transcribed")
    typer.echo(f"Output saved to {output_file}")


async def _process_convert(input_file: str, output_file: Optional[Path]):
    """Process document conversion."""
    input_path = Path(input_file)
    file_extension = input_path.suffix.lower()

    # Special handling for EPUB files with image extraction
    if file_extension in EPUB_EXTENSIONS:
        logger.info(f"Processing EPUB with image extraction: {input_path}")
        text_context, output_dir = await process_epub_with_images(input_path)

        logger.info(f"EPUB converted with images to: {output_dir}")
        logger.info(f"Main file: {output_dir / f'{input_path.stem}.md'}")
        logger.info(f"Images extracted to: {output_dir / 'images'}")

        typer.echo("Successfully converted EPUB with images")
        typer.echo(f"Output saved to {output_dir}")
        return

    # Standard file conversion
    logger.info(f"Converting file: {input_path}")
    text_context = await get_text_from_file(input_path)

    if output_file is None:
        output_file = create_output_path_from_title(
            text_context.title, "converted", input_path.parent
        )

    logger.info(f"Output: {output_file}")

    text = "\n\n".join(text_context.chunk_list)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    logger.info(f"Converted file saved to: {output_file}")

    typer.echo("Successfully converted")
    typer.echo(f"Output saved to {output_file}")


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
