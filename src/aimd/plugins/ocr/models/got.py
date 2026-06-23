"""GOT-OCR Transformers model adapter."""

from pathlib import Path

from aimd.core.errors import ProcessingFailedError

from .base import (
    get_cached_model_and_processor,
    inputs_to_model_device,
)
from .generic import _load_generic_model

GOT_OCR_MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


class GOTOCRModel:
    """Adapter for stepfun-ai/GOT-OCR-2.0-hf."""

    model_id = GOT_OCR_MODEL_ID

    def recognize_image(
        self,
        input_path: Path,
        *,
        language: str | None = None,  # noqa: ARG002 - GOT prompt is processor-owned
        temp_dir: Path | None = None,  # noqa: ARG002 - no temp files needed
    ) -> str:
        model, processor = get_cached_model_and_processor(
            self.model_id, _load_generic_model
        )
        try:
            inputs = processor(input_path.as_posix(), return_tensors="pt")
        except Exception as exc:  # noqa: BLE001 - processor errors are model-specific
            raise ProcessingFailedError(
                f"Unable to prepare image for GOT-OCR: {exc}"
            ) from exc

        inputs = inputs_to_model_device(inputs, model)
        input_ids = inputs["input_ids"]

        import torch

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )

        generated_ids = getattr(generated, "sequences", generated)
        text = processor.decode(
            generated_ids[0][input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text:
            raise ProcessingFailedError("GOT-OCR produced empty content")
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
