"""Shared MLX audio-tower reference for the ANE spike (WS-12b).

Loads the real ``mlx_audio`` Qwen3-ASR audio tower (``AudioEncoder``) from the
local ``mlx-community/Qwen3-ASR-1.7B-4bit`` snapshot — the same loading path as
scripts/export_asr_encoder_reference.py (the WS-06a porting source of truth) —
and runs it on a [1, 128, T] mel + ones mask. The tower weights are the
unquantized bf16 ``audio_tower.*`` tensors; activations are float32 (MLX
promotes bf16 weights against the float32 mel).

Used by convert_tower.py (conversion sanity check) and bench.py (GPU reference
latency + parity reference output).
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

MLX_CHECKPOINT = "mlx-community/Qwen3-ASR-1.7B-4bit"


def find_snapshot(pattern: str = "models--mlx-community--Qwen3-ASR-1.7B-4bit") -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = (
        sorted((hub / pattern / "snapshots").glob("*"))
        if (hub / pattern).exists()
        else []
    )
    for cand in candidates:
        if cand.is_dir() and list(cand.glob("*.safetensors")):
            return cand
    raise FileNotFoundError(f"no local snapshot of {MLX_CHECKPOINT} under {hub}")


def load_mlx_tower(snapshot: Path | None = None):
    """Instantiate the mlx_audio AudioEncoder with the bf16 audio_tower tensors."""
    import mlx.core as mx

    from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import AudioEncoder, Qwen3ASRModel

    snapshot = snapshot or find_snapshot()
    config = ModelConfig.from_dict(json.loads((snapshot / "config.json").read_text()))

    all_weights = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        all_weights.update(mx.load(str(shard)))
    tower_weights = {
        k[len("audio_tower.") :]: v
        for k, v in Qwen3ASRModel.sanitize(all_weights).items()
        if k.startswith("audio_tower.")
    }
    if not tower_weights:
        raise RuntimeError("checkpoint has no audio_tower.* tensors")

    tower = AudioEncoder(config.audio_config)
    tower.load_weights(list(tower_weights.items()), strict=True)
    mx.eval(tower.parameters())
    return tower


def mlx_tower_forward(
    tower, input_features: np.ndarray, attention_mask: np.ndarray | None = None
):
    """One tower forward on [1, 128, T] f32 mel; returns float32 [pads, 2048]."""
    import mlx.core as mx

    if attention_mask is None:
        attention_mask = np.ones((1, input_features.shape[-1]), dtype=np.int32)
    out = tower(mx.array(input_features), mx.array(attention_mask))
    mx.eval(out)
    return np.asarray(out, dtype=np.float32)


def mlx_versions() -> dict[str, str]:
    result = {}
    for name in ("mlx", "mlx-audio"):
        try:
            result[name] = version(name)
        except Exception:
            result[name] = "unknown"
    return result
