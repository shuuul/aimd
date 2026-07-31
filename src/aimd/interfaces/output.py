"""Output persistence helpers for aimd interfaces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aimd.core.errors import ProcessingFailedError
from aimd.core.models import ProcessResult


def build_output_text(
    task_type: Literal["transcript", "convert", "ocr"],
    chunk_list: list[str],
) -> str:
    """Build persisted markdown text for the given task output."""
    text = "\n\n".join(chunk_list)
    if task_type == "transcript" and not text:
        raise ProcessingFailedError("Transcription returned empty content")
    return text


def persist_output(
    output_file: Path,
    task_type: Literal["transcript", "convert", "ocr"],
    chunk_list: list[str],
) -> Path:
    """Write task output to disk and return resolved path."""
    text = build_output_text(task_type, chunk_list)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding="utf-8")
    return output_file.resolve()


@dataclass(slots=True, frozen=True)
class PersistedOutput:
    """Interface-facing output locations after optional persistence."""

    output_file: str | None
    output_dir: str | None
    ignored_output_file: bool = False


def persist_result_output_if_requested(
    result: ProcessResult,
    requested_output_file: str | Path | None,
) -> PersistedOutput:
    """Persist a result when an interface requested a file output."""
    output_dir = (
        str(result.output_dir.resolve()) if result.output_dir is not None else None
    )
    if requested_output_file is None:
        return PersistedOutput(output_file=None, output_dir=output_dir)

    if result.output_dir is not None:
        return PersistedOutput(
            output_file=None,
            output_dir=output_dir,
            ignored_output_file=True,
        )

    resolved = persist_output(
        Path(requested_output_file),
        result.task_type,
        result.text_context.chunk_list,
    )
    return PersistedOutput(output_file=str(resolved), output_dir=None)


MODEL_HELP_TEXT = (
    "Model for transcription, or OCR model. macOS OCR: glm_ocr "
    "(default) or an mlx-vlm compatible Hugging Face model ID. "
    "Linux/CUDA OCR: got_ocr (default), unlimited_ocr, glm_ocr, "
    "or a Hugging Face model ID. "
    "mlx defaults to mlx-community/Qwen3-ASR-1.7B-4bit "
    "and also supports other documented mlx-audio STT model IDs. "
    "CUDA Transformers ASR supports Qwen/Qwen3-ASR-1.7B-hf "
    "(default) or Qwen/Qwen3-ASR-0.6B-hf "
    "(legacy Qwen/Qwen3-ASR-* IDs still resolve to -hf)."
)


def get_request_temp_dir() -> Path | None:
    """Shared helper for API and MCP to resolve AIMD_TEMP_DIR with mkdir."""
    import os

    env_temp_dir = os.environ.get("AIMD_TEMP_DIR")
    if not env_temp_dir:
        return None

    temp_dir = Path(env_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir
