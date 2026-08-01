"""GLM-OCR Transformers model adapter."""

from pathlib import Path

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError

from .base import (
    get_cached_model_and_processor,
    get_cuda_dtype,
    inputs_to_model_device,
    validate_transformers_precision,
)

GLM_OCR_MODEL_ID = "zai-org/GLM-OCR"


def _build_messages(image_path: Path, language: str | None) -> list[dict[str, object]]:
    instruction = "Text Recognition:"
    if language:
        instruction = f"Text Recognition. Prefer {language} text when ambiguous:"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path.as_posix()},
                {"type": "text", "text": instruction},
            ],
        }
    ]


class GLMOCRModel:
    """OCR adapter for zai-org/GLM-OCR's processor chat template."""

    model_id = GLM_OCR_MODEL_ID

    def __init__(self, precision: str | None = None) -> None:
        self.precision = validate_transformers_precision(precision)

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,  # noqa: ARG002 - no temp files needed
    ) -> str:
        model, processor = get_cached_model_and_processor(
            self.model_id,
            lambda name: _load_glm_ocr_model(name, precision=self.precision),
            precision=self.precision,
        )
        try:
            inputs = processor.apply_chat_template(
                _build_messages(input_path, language),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:  # noqa: BLE001 - processor errors are model-specific
            raise ProcessingFailedError(
                f"Unable to prepare image for GLM-OCR: {exc}"
            ) from exc

        inputs.pop("token_type_ids", None)
        inputs = inputs_to_model_device(inputs, model)
        input_ids = inputs["input_ids"]

        import torch

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=8192)

        generated_ids = getattr(generated, "sequences", generated)
        decoded = processor.decode(
            generated_ids[0][input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = decoded.strip()
        if not text:
            raise ProcessingFailedError("GLM-OCR produced empty content")
        return text

    def recognize_images(
        self,
        image_paths: list[Path],
        *,
        language: str | None = None,
        temp_dir: Path | None = None,
    ) -> list[str]:
        return [
            self.recognize_image(
                image_path,
                language=language,
                temp_dir=temp_dir,
            )
            for image_path in image_paths
        ]


def _load_glm_ocr_model(
    model_name: str, precision: str | None = None
) -> tuple[object, object]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise BackendUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=get_cuda_dtype(precision),
        ).to("cuda")
    except Exception as exc:  # noqa: BLE001 - upstream model errors vary widely
        raise BackendUnavailableError(
            f"Unable to load Transformers OCR model {model_name!r}: {exc} "
            "GLM-OCR requires transformers>=5.14.1 with native glm_ocr support."
        ) from exc
    return model, processor
