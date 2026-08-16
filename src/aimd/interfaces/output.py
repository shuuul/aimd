"""Output persistence helpers for aimd interfaces."""

from dataclasses import dataclass
from pathlib import Path

from aimd.core.errors import ProcessingFailedError
from aimd.core.models import ProcessResult, TaskType


def build_output_text(
    task_type: TaskType,
    markdown: str,
) -> str:
    """Build persisted markdown text for the given task output."""
    if task_type == "transcript" and not markdown:
        raise ProcessingFailedError("Transcription returned empty content")
    return markdown


def persist_output(
    output_file: Path,
    task_type: TaskType,
    markdown: str,
) -> Path:
    """Write exact task Markdown to disk and return the resolved path."""
    text = build_output_text(task_type, markdown)
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

    markdown = build_output_text(result.task_type, result.markdown)
    output_path = Path(requested_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    resolved = output_path.resolve()
    return PersistedOutput(output_file=str(resolved), output_dir=None)


MODEL_HELP_TEXT = (
    "Model for transcription or OCR. Default OCR is unlimited-ocr "
    "(mlx-community/Unlimited-OCR-4bit on macOS, baidu/Unlimited-OCR on "
    "Linux/CUDA). "
    "macOS OCR: unlimited-ocr (default, mlx-vlm>=0.6.4) or glm-ocr, or explicit "
    "mlx-community/Unlimited-OCR-{4bit,6bit,8bit,bf16} / "
    "mlx-community/GLM-OCR-{4bit,6bit,8bit,bf16} model IDs. "
    "Linux/CUDA OCR: unlimited-ocr (default) or glm-ocr "
    "(baidu/Unlimited-OCR, zai-org/GLM-OCR). "
    "mlx ASR: qwen3-asr-1.7b (default) or qwen3-asr-0.6b combined with "
    "--precision, or explicit mlx-community/Qwen3-ASR-* model IDs. "
    "CUDA Transformers ASR: qwen3-asr-1.7b (default) or qwen3-asr-0.6b, "
    "resolving to Qwen/Qwen3-ASR-*-hf "
    "(legacy underscore aliases and Qwen/Qwen3-ASR-* IDs still work)."
)

PRECISION_HELP_TEXT = (
    "Model precision/quantization: 4bit, 6bit, 8bit, or bf16 "
    "(dash variants like 4-bit are accepted). "
    "macOS MLX backends select the matching mlx-community checkpoint "
    "(default 4bit when omitted). "
    "CUDA Transformers backends only accept bf16, and require CUDA bf16 "
    "support; quantized values are rejected. When omitted, Transformers keeps "
    "automatic dtype selection (bf16 on supported CUDA, fp16 on CUDA/MPS, "
    "fp32 on CPU)."
)

CONTEXT_HELP_TEXT = (
    "ASR context/biasing text (proper nouns, names, domain vocabulary) that "
    "helps transcription accuracy. For URL inputs, page metadata (title, "
    "description, tags) is appended automatically unless --no-context is set."
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
