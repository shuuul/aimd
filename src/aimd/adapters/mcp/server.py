"""MCP adapter for aimd."""

import os
from pathlib import Path
from typing import Any

from logly import logger
from mcp.server.fastmcp import FastMCP

from ...application.bootstrap import build_container
from ...application.models import ProcessInput
from ...application.services.output_writer import persist_output
from ...errors import AimdError

logger.remove_all()

container = build_container()
mcp = FastMCP("aimd")


@mcp.tool()
async def healthz() -> dict[str, str]:
    """Health check for aimd MCP server."""
    return {"status": "ok"}


@mcp.tool()
async def list_engines() -> dict[str, Any]:
    """List transcription engine capabilities and auto-selected engine."""
    result = container.list_engines_use_case.execute()
    ordered_engines = ("yap", "mlx", "cuda", "cpu")
    return {
        "auto_selected_engine": result.auto_selected_engine,
        "engines": [
            {
                "name": engine,
                "available": result.engines[engine].available,
                "reason": result.engines[engine].reason,
                "fix_hint": result.engines[engine].fix_hint,
                "selected_by_auto": engine == result.auto_selected_engine,
            }
            for engine in ordered_engines
        ],
    }


@mcp.tool()
async def process_input(
    input_source: str,
    transcribe_engine: str = "auto",
    language: str | None = None,
    output_file: str | None = None,
    save_original: str | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    """Process audio/video/url/documents and return markdown context."""
    try:
        env_temp_dir = os.environ.get("AIMD_TEMP_DIR")
        temp_dir = Path(env_temp_dir) if env_temp_dir else None
        if temp_dir is not None:
            temp_dir.mkdir(parents=True, exist_ok=True)

        result = await container.process_input_use_case.execute(
            ProcessInput(
                input_source=input_source,
                output_file=Path(output_file) if output_file else None,
                transcribe_engine=transcribe_engine,
                language=language,
                save_original=Path(save_original) if save_original else None,
                cookies=Path(cookies) if cookies else None,
                cookies_from_browser=cookies_from_browser,
                temp_dir=temp_dir,
            )
        )

        persisted_output_file: str | None = None
        output_dir: str | None = None
        if result.output_dir is not None:
            output_dir = str(result.output_dir.resolve())

        if output_file and result.output_dir is None:
            resolved = persist_output(
                Path(output_file),
                result.task_type,
                result.text_context.chunk_list,
            )
            persisted_output_file = str(resolved)

        return {
            "task_type": result.task_type,
            "title": result.text_context.title,
            "chunk_list": result.text_context.chunk_list,
            "split_header_level": result.text_context.split_header_level,
            "output_file": persisted_output_file,
            "output_dir": output_dir,
        }
    except AimdError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def main() -> None:
    """Run MCP server over stdio."""
    mcp.run(transport="stdio")
