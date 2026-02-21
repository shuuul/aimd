"""MCP server entrypoint for aimd capabilities."""

from pathlib import Path
from typing import Any, Literal

from logly import logger
from mcp.server.fastmcp import FastMCP

from .capabilities import get_engine_capabilities, resolve_engine_with_preflight
from .errors import AimdError, EngineUnavailableError
from .service import (
    ensure_supported_input,
    process_convert_input,
    process_transcript_input,
)

# Avoid stdout/stderr noise interfering with STDIO transport frames.
logger.remove_all()

mcp = FastMCP("aimd")


@mcp.tool()
async def healthz() -> dict[str, str]:
    """Health check for aimd MCP server."""
    return {"status": "ok"}


@mcp.tool()
async def list_engines() -> dict[str, Any]:
    """List transcription engine capabilities and the auto-selected engine."""
    capabilities = get_engine_capabilities()
    auto_selected_engine: str | None = None
    try:
        auto_selected_engine = resolve_engine_with_preflight("auto")
    except EngineUnavailableError:
        auto_selected_engine = None

    ordered_engines = ("yap", "mlx", "cuda", "cpu")
    return {
        "auto_selected_engine": auto_selected_engine,
        "engines": [
            {
                "name": engine,
                "available": capabilities[engine].available,
                "reason": capabilities[engine].reason,
                "fix_hint": capabilities[engine].fix_hint,
                "selected_by_auto": engine == auto_selected_engine,
            }
            for engine in ordered_engines
        ],
    }


def _persist_output(
    output_file: Path,
    task_type: Literal["transcript", "convert"],
    chunk_list: list[str],
) -> None:
    if task_type == "transcript":
        text = chunk_list[0] if chunk_list else ""
        if not text:
            raise ValueError("Transcription returned empty content")
    else:
        text = "\n\n".join(chunk_list)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")


@mcp.tool()
async def process_input(
    input_source: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    output_file: str | None = None,
    save_original: str | None = None,
    cookies: str | None = None,
) -> dict[str, Any]:
    """Process audio/video/url/documents and return markdown context.

    Args:
        input_source: Audio/video file path, video URL, or document file path.
        transcribe_engine: Transcription engine: auto, yap, mlx, cuda, cpu.
        language: Whisper language code, e.g. zh, en, ja.
        output_file: Optional path to write resulting markdown output.
        save_original: Optional path to persist downloaded audio for URL processing.
        cookies: Optional Netscape cookies file path for URL extraction.
    """
    try:
        task_type = ensure_supported_input(input_source)
        persisted_output_file: str | None = None
        output_dir: str | None = None

        if task_type == "transcript":
            text_context = await process_transcript_input(
                input_source=input_source,
                engine=transcribe_engine,
                language=language,
                save_original=Path(save_original) if save_original else None,
                cookies=Path(cookies) if cookies else None,
            )
        else:
            text_context, epub_output_dir = await process_convert_input(input_source)
            if epub_output_dir is not None:
                output_dir = str(epub_output_dir.resolve())

        if output_file and output_dir is None:
            output_path = Path(output_file)
            _persist_output(output_path, task_type, text_context.chunk_list)
            persisted_output_file = str(output_path.resolve())

        return {
            "task_type": task_type,
            "title": text_context.title,
            "chunk_list": text_context.chunk_list,
            "split_header_level": text_context.split_header_level,
            "output_file": persisted_output_file,
            "output_dir": output_dir,
        }
    except AimdError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
