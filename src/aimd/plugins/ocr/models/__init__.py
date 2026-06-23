"""Transformers OCR model registry."""

from aimd.core.errors import ProcessingFailedError

from .base import clear_model_cache
from .generic import GenericTransformersOCRModel
from .got import GOT_OCR_MODEL_ID, GOTOCRModel
from .unlimited import (
    UNLIMITED_OCR_MODEL_ID,
    UnlimitedOCRModel,
    normalize_unlimited_ocr_output,
    read_unlimited_ocr_output_files,
)

DEFAULT_TRANSFORMERS_OCR_MODEL = "got_ocr"
TRANSFORMERS_OCR_MODEL_ALIASES = {
    "got_ocr": GOT_OCR_MODEL_ID,
    "got-ocr": GOT_OCR_MODEL_ID,
    "got_ocr2": GOT_OCR_MODEL_ID,
    "got-ocr2": GOT_OCR_MODEL_ID,
    "stepfun-ai/got-ocr-2.0-hf": GOT_OCR_MODEL_ID,
    "unlimited_ocr": UNLIMITED_OCR_MODEL_ID,
    "unlimited-ocr": UNLIMITED_OCR_MODEL_ID,
    "baidu/unlimited-ocr": UNLIMITED_OCR_MODEL_ID,
    "glm_ocr": "zai-org/GLM-OCR",
    "glm-ocr": "zai-org/GLM-OCR",
    "zai-org/glm-ocr": "zai-org/GLM-OCR",
}


def resolve_transformers_ocr_model(model: str | None) -> str:
    """Resolve aimd OCR model aliases to Hugging Face model IDs."""
    requested = model or DEFAULT_TRANSFORMERS_OCR_MODEL
    normalized = requested.strip().lower().replace(" ", "_")
    if normalized in TRANSFORMERS_OCR_MODEL_ALIASES:
        return TRANSFORMERS_OCR_MODEL_ALIASES[normalized]
    if "/" in requested:
        return requested.strip()
    raise ProcessingFailedError(
        "Unsupported Transformers OCR model. Supported models: got_ocr, "
        "unlimited_ocr, glm_ocr, or an explicit Hugging Face model ID. "
        "PaddleOCR aliases are not supported."
    )


def create_transformers_ocr_model(
    model: str | None,
) -> GOTOCRModel | UnlimitedOCRModel | GenericTransformersOCRModel:
    """Create a model-specific Transformers OCR adapter."""
    model_id = resolve_transformers_ocr_model(model)
    if model_id == GOT_OCR_MODEL_ID:
        return GOTOCRModel()
    if model_id == UNLIMITED_OCR_MODEL_ID:
        return UnlimitedOCRModel()
    return GenericTransformersOCRModel(model_id)


__all__ = [
    "DEFAULT_TRANSFORMERS_OCR_MODEL",
    "GOT_OCR_MODEL_ID",
    "TRANSFORMERS_OCR_MODEL_ALIASES",
    "UNLIMITED_OCR_MODEL_ID",
    "GenericTransformersOCRModel",
    "GOTOCRModel",
    "UnlimitedOCRModel",
    "clear_model_cache",
    "create_transformers_ocr_model",
    "normalize_unlimited_ocr_output",
    "read_unlimited_ocr_output_files",
    "resolve_transformers_ocr_model",
]
