"""MCP server module for aimd."""

from pathlib import Path
from typing import Any

from logly import logger
from mcp.server.mcpserver import MCPServer

from aimd.core.errors import AimdError
from aimd.core.models import ProcessInput, ProcessResult, TaskType
from aimd.core.process import process_input as process_core_input
from aimd.interfaces.output import (
    get_request_temp_dir,
    persist_result_output_if_requested,
)

logger.remove()

mcp = MCPServer("aimd")


def _process_result_payload(
    result: ProcessResult,
    *,
    output_file: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    return {
        "task_type": result.task_type,
        "title": result.text_context.title,
        "markdown": result.markdown,
        "asset_base_uri": result.asset_base_uri,
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
    task_type: TaskType | None = None,
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    output_file: str | None = None,
    save_original: str | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    raw_transcript: bool = False,
    precision: str | None = None,
    context: str | None = None,
    metadata_context: bool = True,
) -> dict[str, Any]:
    """Process audio/video/url/documents and return markdown context.

    precision selects model quantization: 4bit, 6bit, 8bit, or bf16 (dash
    variants like 4-bit are accepted). macOS MLX backends select the matching
    mlx-community checkpoint (default 4bit when omitted). CUDA Transformers
    backends only accept bf16 (requires CUDA bf16 support) and otherwise keep
    automatic dtype selection when precision is omitted.

    context is free-form ASR biasing text (proper nouns, names, domain
    vocabulary) that improves transcription accuracy. metadata_context
    (default True) additionally injects URL page metadata (title,
    description, tags) as ASR context for URL inputs.
    """
    try:
        temp_dir = get_request_temp_dir()

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
                precision=precision,
                context=context,
                metadata_context=metadata_context,
            )
        )

        persisted = persist_result_output_if_requested(result, output_file)
        if persisted.ignored_output_file:
            logger.warning(
                "Ignoring output_file for document asset conversions; output is a directory."
            )
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
