"""Generic chat-template Transformers OCR model adapter."""

from pathlib import Path

from aimd.core.errors import EngineUnavailableError, ProcessingFailedError

from .base import (
    TransformersOCRModel,
    get_cached_model_and_processor,
    get_cuda_dtype,
    inputs_to_model_device,
)


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


class GenericTransformersOCRModel(TransformersOCRModel):
    """OCR adapter for image-text models with processor chat templates."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,
        temp_dir: Path | None = None,  # noqa: ARG002 - no temp files needed
    ) -> str:
        model, processor = get_cached_model_and_processor(
            self.model_id, _load_generic_model
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
                f"Unable to prepare image for Transformers OCR: {exc}"
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
            raise ProcessingFailedError("Transformers OCR produced empty content")
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


def _load_generic_model(model_name: str) -> tuple[object, object]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise EngineUnavailableError(
            "Transformers OCR requires torch and transformers. Install project "
            "dependencies with `uv sync`, then retry."
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=get_cuda_dtype(),
        ).to("cuda")
    except Exception as exc:  # noqa: BLE001 - upstream model errors vary widely
        extra = ""
        if model_name == "zai-org/GLM-OCR":
            extra = " GLM-OCR currently requires installing Transformers from GitHub."
        elif model_name == "PaddlePaddle/PaddleOCR-VL-1.5":
            extra = " PaddleOCR-VL may require torchvision with the current model code."
        raise EngineUnavailableError(
            f"Unable to load Transformers OCR model {model_name!r}: {exc}{extra}"
        ) from exc
    return model, processor
