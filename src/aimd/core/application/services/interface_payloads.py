"""Shared adapter mapping helpers."""

import os
from pathlib import Path
from typing import Any

from ..models import ProcessInput, ProcessResult, TaskType
from ..use_cases.list_engines import ListEnginesResult


def get_request_temp_dir() -> Path | None:
    """Read and prepare AIMD_TEMP_DIR at request time."""
    env_temp_dir = os.environ.get("AIMD_TEMP_DIR")
    if not env_temp_dir:
        return None

    temp_dir = Path(env_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def build_process_input(
    *,
    input_source: str,
    task_type: TaskType | None = None,
    transcribe_engine: str = "auto",
    model: str | None = None,
    language: str | None = None,
    start: int | None = None,
    end: int | None = None,
    save_original: str | Path | None = None,
    cookies: str | Path | None = None,
    cookies_from_browser: str | None = None,
    raw_transcript: bool = False,
    temp_dir: Path | None = None,
) -> ProcessInput:
    """Build the canonical process request from adapter-level values."""
    return ProcessInput(
        input_source=input_source,
        task_type=task_type,
        transcribe_engine=transcribe_engine,
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


def engine_capabilities_payload(result: ListEnginesResult) -> dict[str, Any]:
    """Return a JSON-friendly engine capability payload."""
    ordered_engines = ("mlx", "qwen")
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


def process_result_payload(
    result: ProcessResult,
    *,
    output_file: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    """Return a JSON-friendly process result payload."""
    return {
        "task_type": result.task_type,
        "title": result.text_context.title,
        "chunk_list": result.text_context.chunk_list,
        "split_header_level": result.text_context.split_header_level,
        "platform": result.platform,
        "output_file": output_file,
        "output_dir": output_dir,
    }
