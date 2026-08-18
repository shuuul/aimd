"""Transformers OCR model registry."""

from aimd.core.errors import ProcessingFailedError

from .base import TransformersOCRModel, clear_model_cache
from .glm import GLM_OCR_MODEL_ID, GLMOCRModel
from .unlimited import (
    UNLIMITED_OCR_MODEL_ID,
    UnlimitedOCRModel,
    normalize_unlimited_ocr_markdown,
    normalize_unlimited_ocr_output,
    read_unlimited_ocr_output_files,
)

DEFAULT_TRANSFORMERS_OCR_MODEL = "unlimited-ocr"
TRANSFORMERS_OCR_MODEL_ALIASES = {
    "unlimited-ocr": UNLIMITED_OCR_MODEL_ID,
    "baidu/unlimited-ocr": UNLIMITED_OCR_MODEL_ID,
    "glm-ocr": GLM_OCR_MODEL_ID,
    "zai-org/glm-ocr": GLM_OCR_MODEL_ID,
    # Legacy underscore aliases retained for backwards compatibility.
    "unlimited_ocr": UNLIMITED_OCR_MODEL_ID,
    "glm_ocr": GLM_OCR_MODEL_ID,
}


def resolve_transformers_ocr_model(model: str | None) -> str:
    """Resolve aimd OCR model aliases to Hugging Face model IDs."""
    requested = model or DEFAULT_TRANSFORMERS_OCR_MODEL
    normalized = requested.strip().lower().replace(" ", "_")
    if normalized in TRANSFORMERS_OCR_MODEL_ALIASES:
        return TRANSFORMERS_OCR_MODEL_ALIASES[normalized]
    raise ProcessingFailedError(
        "Unsupported Transformers OCR model. Supported models: unlimited-ocr, "
        "glm-ocr, or the explicit Hugging Face IDs baidu/Unlimited-OCR and "
        "zai-org/GLM-OCR."
    )


def create_transformers_ocr_model(
    model: str | None,
    precision: str | None = None,
) -> TransformersOCRModel:
    """Create a model-specific Transformers OCR adapter."""
    model_id = resolve_transformers_ocr_model(model)
    if model_id == UNLIMITED_OCR_MODEL_ID:
        return UnlimitedOCRModel(precision=precision)
    return GLMOCRModel(precision=precision)


__all__ = [
    "DEFAULT_TRANSFORMERS_OCR_MODEL",
    "GLM_OCR_MODEL_ID",
    "TRANSFORMERS_OCR_MODEL_ALIASES",
    "UNLIMITED_OCR_MODEL_ID",
    "GLMOCRModel",
    "UnlimitedOCRModel",
    "TransformersOCRModel",
    "clear_model_cache",
    "create_transformers_ocr_model",
    "normalize_unlimited_ocr_markdown",
    "normalize_unlimited_ocr_output",
    "read_unlimited_ocr_output_files",
    "resolve_transformers_ocr_model",
]
