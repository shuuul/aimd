"""MCP server package for aimd."""

from typing import Any

from logly import logger
from mcp.server.fastmcp import FastMCP

from aimd.application.bootstrap import build_container
from aimd.application.services.interface_payloads import (
    build_process_input,
    engine_capabilities_payload,
    get_request_temp_dir,
    process_result_payload,
)
from aimd.application.services.output_writer import persist_result_output_if_requested
from aimd.errors import AimdError

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
    return engine_capabilities_payload(result)


@mcp.tool()
async def process_input(
    input_source: str,
    transcribe_engine: str = "auto",
    model: str | None = None,
    language: str | None = None,
    output_file: str | None = None,
    save_original: str | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    raw_transcript: bool = False,
) -> dict[str, Any]:
    """Process audio/video/url/documents and return markdown context."""
    try:
        temp_dir = get_request_temp_dir()

        result = await container.process_input_use_case.execute(
            build_process_input(
                input_source=input_source,
                transcribe_engine=transcribe_engine,
                model=model,
                language=language,
                save_original=save_original,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
                temp_dir=temp_dir,
                raw_transcript=raw_transcript,
            )
        )

        persisted = persist_result_output_if_requested(result, output_file)
        return process_result_payload(
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
