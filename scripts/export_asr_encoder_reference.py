#!/usr/bin/env python
"""Export golden audio-tower (encoder) fixtures for the native engine (WS-06a).

Runs the real ``mlx_audio`` Qwen3-ASR audio tower
(``mlx_audio.stt.models.qwen3_asr.qwen3_asr.AudioEncoder``, the porting
source of truth) over deterministic synthetic mel inputs and writes, per
case, into ``crates/aimd-engine/tests/golden/asr_encoder/cases/<name>/``:

- ``mel.f32``                      the encoder input, little-endian float32,
                                   row-major ``[frames, 128]`` (time-major;
                                   the Rust tower consumes this layout), the
                                   true (unpadded) frame region of the
                                   processor's padded ``input_features``
- ``stage_conv_front_end.f32``     hidden states entering the encoder layers
                                   (post conv front-end + conv_out +
                                   positional embedding + per-chunk trim),
                                   ``[pads, 1024]``
- ``stage_layer_{00,01,12,23}.f32`` hidden states after encoder layers
                                   0/1/12/23, ``[pads, 1024]``
- ``stage_final_projection.f32``   the tower output after ``ln_post`` ->
                                   ``proj1`` -> ``proj2``, ``[pads, 2048]``
- ``meta.json``                    case metadata: shapes, the mel recipe,
                                   chunk/block geometry, audio pad count

Plus ``manifest.json``: versions, the tower weight-name mapping the Rust
port must implement, the frames -> audio-pad formula, tolerances, and
per-case sha256s.

The export is faithful to the inference path: mel inputs are produced by the
HF ``Qwen3ASRFeatureExtractor`` exactly like
``Qwen3ASRModel._preprocess_audio`` (``return_attention_mask=True,
truncation=False, padding=True``), the tower is the real checkpoint-loaded
module, and the instrumented staged forward is asserted bitwise-equal to
``audio_tower(input_features, feature_attention_mask)``.

Case selection:

- ``sine_1_5s``: 150 frames -> 20 pads, conv chunks ``[100, 50]`` (multi-chunk
  conv with a short final chunk); same recipe as the tokenizer fixture
  ``asr_no_context_no_language`` (440 Hz sine, 1.5 s).
- ``multi_tone_2_7s``: 270 frames -> 35 pads, chunks ``[100, 100, 70]``.
- ``sine_9s``: 900 frames -> 117 pads, nine full chunks, attention split into
  two blocks ``[104, 13]`` (exercises the ``n_window_infer`` windowing).

Only the audio tower runs; the 4-bit text decoder is never touched. All
tower activations are float32 (MLX promotes the bf16 tower weights against
the float32 mel input), so stages are dumped as f32.

Usage::

    uv run python scripts/export_asr_encoder_reference.py \
        [--out crates/aimd-engine/tests/golden/asr_encoder]

Deterministic: fixed recipes, no wall-clock; sha256s are byte-stable across
regeneration runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "crates" / "aimd-engine" / "tests" / "golden" / "asr_encoder"

# Element-wise parity gate for the Rust port (abs OR rel), matching the
# constants in encoder.rs tests. Tower weights are consumed bit-identically
# (raw bf16) on both sides, so the residual is f32 accumulation order.
# Observed across all cases (interior stages / final projection):
# max abs 7.8e-5 / 8.5e-7, max rel ~0.19 at near-zero elements (abs-covered).
GATE_ABS = 1e-3
GATE_REL = 1e-2

CHECKPOINT = "mlx-community/Qwen3-ASR-1.7B-4bit"
FEATURE_EXTRACTOR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"

STAGES = [
    "conv_front_end",
    "layer_00",
    "layer_01",
    "layer_12",
    "layer_23",
    "final_projection",
]

# MLX audio-tower tensor name (without the leading "audio_tower.") -> role in
# the Rust port. Shapes are from the checkpoint (all BF16, unquantized).
WEIGHT_MAP = {
    "conv2d1.weight": "conv2d1 weights [480, 3, 3, 1] (MLX NHWC [o, kh, kw, c])",
    "conv2d1.bias": "conv2d1 bias [480]",
    "conv2d2.weight": "conv2d2 weights [480, 3, 3, 480]",
    "conv2d2.bias": "conv2d2 bias [480]",
    "conv2d3.weight": "conv2d3 weights [480, 3, 3, 480]",
    "conv2d3.bias": "conv2d3 bias [480]",
    "conv_out.weight": "conv_out linear [1024, 7680], no bias; input order c*16 + f",
    "layers.N.self_attn.q_proj.{weight,bias}": "per-layer q projection [1024, 1024] + [1024]",
    "layers.N.self_attn.k_proj.{weight,bias}": "per-layer k projection",
    "layers.N.self_attn.v_proj.{weight,bias}": "per-layer v projection",
    "layers.N.self_attn.out_proj.{weight,bias}": "per-layer attention output projection",
    "layers.N.self_attn_layer_norm.{weight,bias}": "pre-attention LayerNorm (eps 1e-5, biased var)",
    "layers.N.fc1.{weight,bias}": "FFN up [4096, 1024], exact-erf GELU",
    "layers.N.fc2.{weight,bias}": "FFN down [1024, 4096]",
    "layers.N.final_layer_norm.{weight,bias}": "pre-FFN LayerNorm",
    "ln_post.{weight,bias}": "final LayerNorm before the projection head",
    "proj1.{weight,bias}": "projection head hidden [1024, 1024]",
    "proj2.{weight,bias}": "projection head output [2048, 1024]",
}

# Frames -> audio pad tokens, mirroring
# mlx_audio.stt.models.qwen3_asr.qwen3_asr._get_feat_extract_output_lengths:
#   rem  = frames % 100
#   a1   = floor((rem - 1) / 2) + 1          (conv2d1 time output)
#   a2   = floor((a1 - 1) / 2) + 1           (conv2d2)
#   a3   = floor((a2 - 1) / 2) + 1           (conv2d3)
#   pads = a3 + (frames // 100) * 13         (one full 100-frame chunk -> 13)
FRAMES_TO_PADS_FORMULA = (
    "pads(L) = floor((floor((floor((L%100 - 1)/2) + 1 - 1)/2) + 1 - 1)/2) + 1 "
    "+ (L // 100) * 13 (floor division, python semantics for rem == 0)"
)


# ---------------------------------------------------------------------------
# Signal generators (deterministic; sine_1_5s matches the tokenizer fixture
# recipe of scripts/export_tokenizer_reference.py exactly)
# ---------------------------------------------------------------------------


def gen_sine(
    sr: int = 16000, duration: float = 1.5, freq: float = 440.0, amp: float = 0.5
):
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32), {
        "type": "sine",
        "freq_hz": freq,
        "amplitude": amp,
        "duration_s": duration,
    }


def gen_multi_tone(sr: int = 16000, duration: float = 2.7):
    """Sum of five sines spread across the 0-8 kHz band (mel-exporter recipe)."""
    freqs = [220.0, 440.0, 1000.0, 3500.0, 7800.0]
    amps = [0.5, 0.3, 0.4, 0.25, 0.15]
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    wav = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps))
    return np.clip(wav, -1.0, 1.0).astype(np.float32), {
        "type": "multi_tone",
        "freqs_hz": freqs,
        "amplitudes": amps,
        "duration_s": duration,
    }


CASES = [
    ("sine_1_5s", lambda: gen_sine(duration=1.5)),
    ("multi_tone_2_7s", lambda: gen_multi_tone(duration=2.7)),
    ("sine_9s", lambda: gen_sine(duration=9.0)),
]


# ---------------------------------------------------------------------------
# Instrumented staged forward (mirrors AudioEncoder.__call__ line by line)
# ---------------------------------------------------------------------------


def staged_audio_tower_forward(tower, input_features, feature_attention_mask):
    """Run ``tower`` (an mlx_audio ``AudioEncoder``) stage by stage.

    Returns ``(captures, geometry)`` where ``captures`` maps stage names to
    mx arrays and ``geometry`` records the chunk/block bookkeeping. The final
    capture is asserted bitwise-equal to the real ``tower(...)`` output by
    the caller.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    if feature_attention_mask is not None:
        feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
    else:
        feature_lens = mx.array(
            [input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32
        )

    feature_lens_np = np.array(feature_lens)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    chunk_size = tower.n_window * 2
    chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)

    chunk_lengths = []
    for i in range(len(feature_lens_np)):
        num_chunks = int(chunk_num[i])
        feat_len = int(feature_lens_np[i])
        for j in range(num_chunks):
            if j == num_chunks - 1:
                remainder = feat_len % chunk_size
                chunk_lengths.append(chunk_size if remainder == 0 else remainder)
            else:
                chunk_lengths.append(chunk_size)
    chunk_lengths = np.array(chunk_lengths, dtype=np.int32)

    chunks = []
    for i in range(len(feature_lens_np)):
        feat = input_features[i]
        feat_len = int(feature_lens_np[i])
        num_chunks = int(chunk_num[i])
        pos = 0
        for j in range(num_chunks):
            if j == num_chunks - 1:
                remainder = feat_len % chunk_size
                clen = chunk_size if remainder == 0 else remainder
            else:
                clen = chunk_size
            chunk = feat[:, pos : pos + clen]
            chunks.append(chunk)
            pos += clen

    max_chunk_len = int(max(chunk_lengths))
    padded_chunks = []
    for i, chunk in enumerate(chunks):
        clen = int(chunk_lengths[i])
        if clen < max_chunk_len:
            pad_width = max_chunk_len - clen
            chunk = mx.pad(chunk, [(0, 0), (0, pad_width)])
        padded_chunks.append(chunk)

    padded_feature = mx.stack(padded_chunks, axis=0)

    feature_lens_after_cnn = _get_feat_extract_output_lengths(mx.array(chunk_lengths))
    feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
    max_len_after_cnn = int(feature_lens_after_cnn_np.max())

    x = padded_feature[:, :, :, None]
    x = nn.gelu(tower.conv2d1(x))
    x = nn.gelu(tower.conv2d2(x))
    x = nn.gelu(tower.conv2d3(x))

    b, f, t, c = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
    x = tower.conv_out(x)

    pos_emb = tower.positional_embedding(x.shape[1])
    x = x + pos_emb[None, :, :]

    hidden_list = []
    for i in range(x.shape[0]):
        valid_len = int(feature_lens_after_cnn_np[i])
        hidden_list.append(x[i, :valid_len])

    hidden_states = mx.concatenate(hidden_list, axis=0)
    captures = {"conv_front_end": hidden_states}

    aftercnn_lens_np = np.array(aftercnn_lens)
    window_aftercnn = max_len_after_cnn * (tower.n_window_infer // (tower.n_window * 2))

    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens_np:
        cnn_len = int(cnn_len)
        num_full_windows = cnn_len // window_aftercnn
        for _ in range(num_full_windows):
            cu_chunk_lens.append(window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens.append(remainder)

    cu_seqlens = np.cumsum(cu_chunk_lens).tolist()

    seq_len = hidden_states.shape[0]
    mask = mx.full((seq_len, seq_len), -1e9, dtype=hidden_states.dtype)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        mask[start:end, start:end] = 0.0
    attention_mask = mask[None, None, :, :]

    hidden_states = hidden_states[None, :, :]

    captured_layers = {0: "layer_00", 1: "layer_01", 12: "layer_12", 23: "layer_23"}
    for idx, layer in enumerate(tower.layers):
        hidden_states = layer(hidden_states, mask=attention_mask)
        if idx in captured_layers:
            captures[captured_layers[idx]] = hidden_states[0]

    hidden_states = hidden_states[0]
    hidden_states = tower.ln_post(hidden_states)
    hidden_states = nn.gelu(tower.proj1(hidden_states))
    hidden_states = tower.proj2(hidden_states)
    captures["final_projection"] = hidden_states

    for value in captures.values():
        mx.eval(value)

    geometry = {
        "feature_lens": [int(v) for v in feature_lens_np],
        "chunk_size": int(chunk_size),
        "chunk_lengths": [int(v) for v in chunk_lengths],
        "feature_lens_after_cnn": [int(v) for v in feature_lens_after_cnn_np],
        "max_chunk_len": max_chunk_len,
        "max_len_after_cnn": max_len_after_cnn,
        "window_aftercnn": int(window_aftercnn),
        "cu_seqlens": [int(v) for v in cu_seqlens],
        "aftercnn_lens": [int(v) for v in aftercnn_lens_np],
    }
    return captures, geometry


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_snapshot() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "models--mlx-community--Qwen3-ASR-1.7B-4bit"
    candidates = (
        sorted((hub / pattern / "snapshots").glob("*"))
        if (hub / pattern).exists()
        else []
    )
    for cand in candidates:
        if cand.is_dir() and list(cand.glob("*.safetensors")):
            return cand
    print(f"FATAL: no local snapshot of {CHECKPOINT} under {hub}")
    sys.exit(1)


def load_tower(snapshot: Path):
    """Instantiate the mlx_audio audio tower from the 4-bit checkpoint.

    Only ``AudioEncoder`` is built and its ``audio_tower.*`` tensors are
    loaded (prefix stripped, strict). The quantized text decoder tensors are
    never touched. ``Qwen3ASRModel.sanitize`` is applied for faithfulness;
    for this checkpoint it is the identity (keys carry no ``thinker.``
    prefix and the conv2d weights are already MLX NHWC ``[o, kh, kw, c]``).
    """
    import mlx.core as mx

    from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
        AudioEncoder,
        Qwen3ASRModel,
    )

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
        print("FATAL: checkpoint has no audio_tower.* tensors")
        sys.exit(1)

    tower = AudioEncoder(config.audio_config)
    tower.load_weights(list(tower_weights.items()), strict=True)
    mx.eval(tower.parameters())
    return tower


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import mlx.core as mx
    import transformers
    from transformers import AutoProcessor

    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    out_dir = args.out
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    snapshot = find_snapshot()
    tower = load_tower(snapshot)
    cfg = tower.config

    processor = AutoProcessor.from_pretrained(FEATURE_EXTRACTOR_REPO)
    fe = processor.feature_extractor

    manifest_cases = []
    total_bytes = 0
    for name, generator in CASES:
        wav, recipe = generator()
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # Exactly Qwen3ASRModel._preprocess_audio's extractor call.
        audio_inputs = fe(
            wav,
            sampling_rate=16000,
            return_attention_mask=True,
            truncation=False,
            padding=True,
            return_tensors="np",
        )
        input_features = audio_inputs["input_features"]  # [1, 128, T_padded] f32
        attention_mask = audio_inputs["attention_mask"]  # [1, T_padded] f32/bool
        frames = int(np.asarray(attention_mask).sum(axis=-1)[0])
        if input_features.shape[0] != 1:
            print(f"FATAL {name}: expected a single feature row")
            return 1
        if input_features.shape[1] != cfg.num_mel_bins:
            print(
                f"FATAL {name}: mel bins {input_features.shape[1]} != {cfg.num_mel_bins}"
            )
            return 1

        # The Rust tower consumes the true (mask-covered) frame region,
        # time-major [frames, 128]; the padded tail never enters the chunks.
        mel = np.ascontiguousarray(input_features[0][:, :frames].T, dtype="<f4")

        captures, geometry = staged_audio_tower_forward(
            tower, mx.array(input_features), mx.array(attention_mask)
        )

        pads = int(captures["final_projection"].shape[0])
        expected_pads = int(
            _get_feat_extract_output_lengths(mx.array([frames]))[0].item()
        )
        if pads != expected_pads:
            print(f"FATAL {name}: staged pads {pads} != formula {expected_pads}")
            return 1
        for stage in STAGES:
            if stage not in captures:
                print(f"FATAL {name}: missing stage {stage}")
                return 1

        # Faithfulness self-check: the staged path must reproduce the real
        # tower output exactly (same ops, same order, same kernels).
        reference = tower(mx.array(input_features), mx.array(attention_mask))
        reference_np = np.array(reference)
        diff = float(
            np.abs(reference_np - np.array(captures["final_projection"])).max()
        )
        if diff != 0.0:
            print(f"FATAL {name}: staged forward differs from real tower by {diff}")
            return 1

        case_dir = cases_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        mel_path = case_dir / "mel.f32"
        write_f32(mel_path, mel)
        files["mel.f32"] = {"shape": [frames, int(cfg.num_mel_bins)]}
        stage_shapes = {}
        for stage in STAGES:
            arr = np.asarray(captures[stage], dtype=np.float32)
            stage_path = case_dir / f"stage_{stage}.f32"
            write_f32(stage_path, arr)
            files[f"stage_{stage}.f32"] = {"shape": [int(v) for v in arr.shape]}
            stage_shapes[stage] = [int(v) for v in arr.shape]

        meta = {
            "name": name,
            "checkpoint": CHECKPOINT,
            "generator_recipe": recipe,
            "samples": int(wav.size),
            "frames": frames,
            "audio_pad_count": pads,
            "mel_layout": "row_major_[frames,128]_f32le (time-major, true frame region)",
            "stage_layout": "row_major_f32le; activations are float32 (bf16 weights promoted)",
            "stage_shapes": stage_shapes,
            "geometry": geometry,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

        sha = {
            fname: sha256_file(case_dir / fname)
            for fname in ["mel.f32", "meta.json"] + [f"stage_{s}.f32" for s in STAGES]
        }
        manifest_cases.append(
            {
                "name": name,
                "frames": frames,
                "audio_pad_count": pads,
                "geometry": geometry,
                "files": files,
                "sha256": sha,
            }
        )
        case_bytes = sum((case_dir / f).stat().st_size for f in sha)
        total_bytes += case_bytes
        print(
            f"{name:20s} frames={frames:5d} pads={pads:4d} chunks={geometry['chunk_lengths']} "
            f"blocks={geometry['cu_seqlens']} bytes={case_bytes}"
        )

    manifest = {
        "version": 1,
        "generator": "scripts/export_asr_encoder_reference.py",
        "checkpoint": CHECKPOINT,
        "feature_extractor": FEATURE_EXTRACTOR_REPO,
        "generated_with": {
            "mlx": _dist_version("mlx"),
            "mlx_audio": _dist_version("mlx-audio"),
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "tower_config": {
            "d_model": cfg.d_model,
            "encoder_layers": cfg.encoder_layers,
            "encoder_attention_heads": cfg.encoder_attention_heads,
            "encoder_ffn_dim": cfg.encoder_ffn_dim,
            "num_mel_bins": cfg.num_mel_bins,
            "n_window": cfg.n_window,
            "n_window_infer": cfg.n_window_infer,
            "conv_chunksize": cfg.conv_chunksize,
            "downsample_hidden_size": cfg.downsample_hidden_size,
            "output_dim": cfg.output_dim,
            "activation_function": cfg.activation_function,
            "max_source_positions": cfg.max_source_positions,
        },
        "weight_map": WEIGHT_MAP,
        "frames_to_pads_formula": FRAMES_TO_PADS_FORMULA,
        "notes": [
            "Tower weights are unquantized bf16 (audio_tower.*); the 4-bit text decoder is not part of these fixtures.",
            "MLX runs the tower with float32 activations (bf16 weights promoted per op); stages are dumped f32.",
            "The conv front-end runs per 100-frame chunk (n_window*2) with zero padding at chunk edges only; "
            "positional embedding restarts at 0 for every chunk.",
            "Attention is block-diagonal: additive -1e9 mask outside cu_seqlens blocks; block length is "
            "max_len_after_cnn * (n_window_infer // (n_window * 2)).",
            "conv_chunksize (500) and max_source_positions (1500) are loaded by mlx_audio but unused by the tower forward.",
        ],
        "tolerance": {
            "abs": GATE_ABS,
            "rel": GATE_REL,
            "note": (
                "element-wise pass if abs <= gate_abs or rel <= gate_rel; tower weights are "
                "bit-identical raw bf16 on both sides, so the residual is f32 accumulation order. "
                "Observed Rust parity: max abs 7.8e-5 (interior stages) / 8.5e-7 (final "
                "projection); max rel ~0.19 only at near-zero elements (abs-covered)."
            ),
        },
        "cases": manifest_cases,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {len(manifest_cases)} cases to {out_dir}")
    print(f"fixture bytes: {total_bytes} ({total_bytes / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
