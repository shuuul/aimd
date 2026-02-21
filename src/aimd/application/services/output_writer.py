"""Shared output persistence helpers for adapters and use-cases."""

from pathlib import Path
from typing import Literal


def build_output_text(
    task_type: Literal["transcript", "convert"],
    chunk_list: list[str],
) -> str:
    """Build persisted markdown text for the given task output."""
    if task_type == "transcript":
        text = chunk_list[0] if chunk_list else ""
        if not text:
            raise ValueError("Transcription returned empty content")
        return text

    return "\n\n".join(chunk_list)


def persist_output(
    output_file: Path,
    task_type: Literal["transcript", "convert"],
    chunk_list: list[str],
) -> Path:
    """Write task output to disk and return resolved path."""
    text = build_output_text(task_type, chunk_list)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    return output_file.resolve()
