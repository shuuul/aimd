"""Command-line interface for aimd - Context preparation tool for LLM workflows."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from logly import logger

from .errors import AimdError
from .service import (
    ensure_supported_input,
    process_convert_input,
    process_transcript_input,
)
from .utils import create_output_path_from_title

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
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code for transcription (e.g., zh, en, ja). "
        "Uses Whisper language codes. None for auto-detection.",
    ),
    save_original: Optional[Path] = typer.Option(
        None,
        "--save-original",
        "-s",
        help="Save the original downloaded audio/video file to specified path or directory.",
    ),
    cookies: Optional[Path] = typer.Option(
        None,
        "--cookies",
        "-c",
        help="Path to cookies file in Netscape format for video URL extraction. "
        "Bypasses browser keyring entirely. Export with: "
        "yt-dlp --cookies-from-browser chrome --cookies cookies.txt",
    ),
    cookies_from_browser: Optional[str] = typer.Option(
        None,
        "--cookies-from-browser",
        help="Browser cookie source for URL extraction. "
        "Examples: chrome, chrome:default, chrome+gnomekeyring:default, firefox.",
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

    try:
        task_type = ensure_supported_input(input_source)
    except AimdError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    logger.info(f"Input: {input_source}")
    logger.info(f"Task: {task_type}")
    if task_type == "transcript":
        logger.info(f"Transcription Engine: {transcribe_engine}")

    async def run_processing():
        try:
            if task_type == "transcript":
                await _process_transcript(
                    input_source,
                    output_file,
                    transcribe_engine,
                    language,
                    save_original,
                    cookies,
                    cookies_from_browser,
                )
            else:
                await _process_convert(input_source, output_file)

        except AimdError as e:
            logger.error(str(e))
            raise typer.Exit(1)
        except Exception as e:
            task_name = "Transcription" if task_type == "transcript" else "Conversion"
            logger.error(f"{task_name} failed: {e}")
            raise typer.Exit(1)

    asyncio.run(run_processing())


async def _process_transcript(
    input_source: str,
    output_file: Optional[Path],
    engine: str,
    language: str | None,
    save_original: Optional[Path] = None,
    cookies: Optional[Path] = None,
    cookies_from_browser: str | None = None,
):
    """Process audio/video transcription."""
    text_context = await process_transcript_input(
        input_source=input_source,
        engine=engine,
        language=language,
        save_original=save_original,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
    )

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
    text_context, output_dir = await process_convert_input(input_file)
    input_path = Path(input_file)

    # EPUB-family files are exported as a directory tree.
    if output_dir is not None:
        logger.info(f"EPUB converted with images to: {output_dir}")
        logger.info(f"Main file: {output_dir / f'{input_path.stem}.md'}")
        logger.info(f"Images extracted to: {output_dir / 'images'}")

        typer.echo("Successfully converted EPUB with images")
        typer.echo(f"Output saved to {output_dir}")
        return

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
