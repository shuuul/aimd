"""MCP server module for aimd."""

import os
from pathlib import Path
from typing import Any

from logly import logger
from mcp.server.fastmcp import FastMCP

from aimd.interfaces.output import persist_result_output_if_requested
from aimd.core.models import ProcessInput, ProcessResult
from aimd.core.process import process_input as process_core_input
from aimd.core.errors import AimdError

logger.remove_all()

mcp = FastMCP("aimd")


def _get_request_temp_dir() -> Path | None:
    env_temp_dir = os.environ.get("AIMD_TEMP_DIR")
    if not env_temp_dir:
        return None

    temp_dir = Path(env_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _process_result_payload(
    result: ProcessResult,
    *,
    output_file: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    return {
        "task_type": result.task_type,
        "title": result.text_context.title,
        "chunk_list": result.text_context.chunk_list,
        "split_header_level": result.text_context.split_header_level,
        "platform": result.platform,
        "output_file": output_file,
        "output_dir": output_dir,
    }


@mcp.tool()
async def healthz() -> dict[str, str]:
    """Health check for aimd MCP server."""
    return {"status": "ok"}


@mcp.tool()
async def process_input(
    input_source: str,
    task_type: str | None = None,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    output_file: str | None = None,
    save_original: str | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    raw_transcript: bool = False,
) -> dict[str, Any]:
    """Process audio/video/url/documents and return markdown context."""
    try:
        temp_dir = _get_request_temp_dir()

        result = await process_core_input(
            ProcessInput(
                input_source=input_source,
                task_type=task_type,
                model=model,
                language=language,
                start=start,
                end=end,
                save_original=Path(save_original) if save_original else None,
                cookies=Path(cookies) if cookies else None,
                cookies_from_browser=cookies_from_browser,
                temp_dir=temp_dir,
                raw_transcript=raw_transcript,
            )
        )

        persisted = persist_result_output_if_requested(result, output_file)
        return _process_result_payload(
            result,
            output_file=persisted.output_file,
            output_dir=persisted.output_dir,
        )
    except AimdError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> None:
    """Run MCP server over stdio."""
    mcp.run(transport="stdio")
