"""Local Qwen3-ASR Transformers registration.

This vendors the upstream Qwen3-ASR Transformers backend so AIMD can load
Qwen/Qwen3-ASR checkpoints without depending on the qwen-asr package.
"""

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor

_REGISTERED = False


def register_qwen3_asr_transformers() -> None:
    """Register Qwen3-ASR classes with Hugging Face Auto* loaders."""
    global _REGISTERED  # noqa: PLW0603
    if _REGISTERED:
        return
    try:
        AutoConfig.for_model("qwen3_asr")
    except ValueError:
        pass
    else:
        _REGISTERED = True
        return

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    _REGISTERED = True


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRProcessor",
    "register_qwen3_asr_transformers",
]
