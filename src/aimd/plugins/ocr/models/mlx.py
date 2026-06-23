"""mlx-vlm OCR model adapter."""

from importlib import import_module
from pathlib import Path

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError

DEFAULT_OCR_MODEL = "glm_ocr"
MLX_VLM_MODEL_ALIASES = {
    "glm_ocr": "mlx-community/GLM-OCR-bf16",
    "glm-ocr": "mlx-community/GLM-OCR-bf16",
    "mlx-community/glm-ocr-bf16": "mlx-community/GLM-OCR-bf16",
}
MLX_VLM_DEFAULT_PROMPT = "Text Recognition:"

_cached_mlx_vlm_model = None
_cached_mlx_vlm_processor = None
_cached_mlx_vlm_model_id: str | None = None


def resolve_mlx_vlm_model(model: str | None) -> str:
    """Resolve aimd OCR model aliases to mlx-vlm model IDs."""
    requested = model or DEFAULT_OCR_MODEL
    normalized = requested.strip().lower().replace(" ", "_")
    if normalized in MLX_VLM_MODEL_ALIASES:
        return MLX_VLM_MODEL_ALIASES[normalized]
    if "/" in requested:
        return requested.strip()
    raise ProcessingFailedError(
        "Unsupported MLX VLM OCR model. Supported models: glm_ocr, "
        "or an explicit mlx-vlm compatible Hugging Face model ID."
    )


class MLXVLMOCRModel:
    """Run OCR through mlx-vlm on image paths."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = resolve_mlx_vlm_model(model_id)

    def recognize_image(self, image_path: Path) -> str:
        model, processor = _load_mlx_vlm_model(self.model_id)
        return _generate_mlx_vlm_text(model, processor, image_path)

    def recognize_images(self, image_paths: list[Path]) -> list[str]:
        model, processor = _load_mlx_vlm_model(self.model_id)
        return [
            _generate_mlx_vlm_text(model, processor, image_path)
            for image_path in image_paths
        ]


def _load_mlx_vlm_modules():
    try:
        return import_module("mlx_vlm"), import_module("mlx_vlm.prompt_utils")
    except ImportError as exc:
        raise BackendUnavailableError(
            "MLX VLM OCR requires mlx-vlm. Install OCR dependencies with `uv sync` "
            "on macOS/Python 3.12+, then retry."
        ) from exc


def _load_mlx_vlm_model(model_id: str) -> tuple[object, object]:
    """Load or return cached mlx-vlm model state."""
    global _cached_mlx_vlm_model, _cached_mlx_vlm_processor, _cached_mlx_vlm_model_id  # noqa: PLW0603
    if _cached_mlx_vlm_model is not None and _cached_mlx_vlm_model_id == model_id:
        return _cached_mlx_vlm_model, _cached_mlx_vlm_processor

    mlx_vlm, _prompt_utils = _load_mlx_vlm_modules()
    try:
        _cached_mlx_vlm_model, _cached_mlx_vlm_processor = mlx_vlm.load(model_id)
    except Exception as exc:  # noqa: BLE001 - upstream model load errors vary
        _cached_mlx_vlm_model = None
        _cached_mlx_vlm_processor = None
        _cached_mlx_vlm_model_id = None
        raise BackendUnavailableError(
            f"Unable to load MLX VLM OCR model {model_id!r}: {exc}"
        ) from exc
    _cached_mlx_vlm_model_id = model_id
    return _cached_mlx_vlm_model, _cached_mlx_vlm_processor


def _generate_mlx_vlm_text(
    model: object,
    processor: object,
    image_path: Path,
) -> str:
    """Generate OCR text for one image using an already-loaded mlx-vlm model."""
    mlx_vlm, prompt_utils = _load_mlx_vlm_modules()
    config = getattr(model, "config", None)
    formatted_prompt = prompt_utils.apply_chat_template(
        processor,
        config,
        MLX_VLM_DEFAULT_PROMPT,
        num_images=1,
    )
    result = mlx_vlm.generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path.as_posix()],
        max_tokens=4096,
    )
    text = str(getattr(result, "text", result)).strip()
    if not text:
        raise ProcessingFailedError("MLX VLM OCR produced empty content")
    return text
