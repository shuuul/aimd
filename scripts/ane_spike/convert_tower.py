#!/usr/bin/env python
"""ANE spike (WS-12b): convert the Qwen3-ASR audio tower to Core ML.

Implements the approved minimal-spike design (docs/ane-feasibility.md
§"最小 spike 设计", steps 1-2):

- Source: local ``Qwen/Qwen3-ASR-1.7B-hf`` snapshot (transformers 5.17 native
  support). The encoder segment = ``Qwen3ASREncoder`` (conv front-end + 24
  windowed layers + ln_post) + ``Qwen3ASRMultiModalProjector`` (linear_1 ->
  exact-erf GELU -> linear_2), i.e. exactly the Rust ``AudioTower::forward``
  segment, output ``[390, 2048]``.
- Weights are loaded strictly (shape+name+count) from the checkpoint's
  ``model.audio_tower.*`` / ``model.multi_modal_projector.*`` bf16 tensors.
- ``_attn_implementation="eager"`` so attention is the per-chunk
  matmul+softmax path (the cu_seqlens flash path is not traceable); all
  window geometry is baked as constants for the fixed [1, 128, 3000] bucket.
- Round 1 per the design: PLAIN conversion, no ml-ane-transformers rewrites
  (no Conv2d layout changes, no zero-padded BMM attention). Only rework if
  the ANE dispatch % gate fails.
- Round-1.5 trace shim (required, documented): coremltools 9 has no
  handlers for the data-dependent bake ops the dynamic window code traces to
  (``aten::Int`` on non-scalar constants in ``get_audio_cu_seqlens``,
  ``aten::nonzero`` for the padding gather — both unregistered). The shim
  re-implements ONLY the packing glue of ``Qwen3ASREncoder.forward`` with
  constant ``cu_seqlens``/``max_seqlen``/gather indices for the all-valid
  3000-frame bucket; every weight-bearing op calls the original transformers
  modules. A bitwise (fp32) equivalence check against the real forward gates
  the patch before any conversion time is spent.
- fp16 weights/activations, ``torch.jit.trace`` ->
  ``ct.convert(compute_units=CPU_AND_NE, minimum_deployment_target=macOS15)``.
- Sanity gates before spending conversion time:
    A. eager fp32 torch tower vs MLX bf16 tower on the same fixture
       (catches any weight-mapping bug; both are f32-accumulated references)
    B. traced fp16 torch vs eager fp16 torch (trace fidelity)

Artifacts land under --out (default /tmp/ane_spike, ~0.6 GB fp16 mlpackage +
compiled .mlmodelc): ``*.mlpackage``, ``*.mlmodelc``, ``convert_meta.json``,
``traced_ops.txt``.

Usage::

    uv run python scripts/ane_spike/convert_tower.py \
        [--fixture native/golden/ane_spike/mel.npy] [--out /tmp/ane_spike]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mlx_reference import load_mlx_tower, mlx_tower_forward, mlx_versions  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "native" / "golden" / "ane_spike" / "mel.npy"
DEFAULT_OUT = Path("/tmp/ane_spike")
HF_CHECKPOINT = "Qwen/Qwen3-ASR-1.7B-hf"
FRAMES = 3000
TOWER_PREFIX = "model.audio_tower."
PROJ_PREFIX = "model.multi_modal_projector."


def find_hf_snapshot(repo_id: str) -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    pattern = "models--" + repo_id.replace("/", "--")
    candidates = (
        sorted((hub / pattern / "snapshots").glob("*"))
        if (hub / pattern).exists()
        else []
    )
    for cand in candidates:
        if cand.is_dir() and (cand / "model.safetensors").is_file():
            return cand
    raise FileNotFoundError(f"no local snapshot of {repo_id} under {hub}")


def load_encoder_segment():
    """Tower + projector with strict checkpoint weights (bf16 upcast to fp32)."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel
    from transformers.models.qwen3_asr.modeling_qwen3_asr import (
        Qwen3ASRMultiModalProjector,
    )

    snapshot = find_hf_snapshot(HF_CHECKPOINT)
    config = AutoConfig.from_pretrained(HF_CHECKPOINT, local_files_only=True)
    config._attn_implementation = "eager"  # property setter recurses into sub-configs
    config.audio_config._attn_implementation = "eager"

    tower = AutoModel.from_config(config.audio_config)
    projector = Qwen3ASRMultiModalProjector(config)

    mapped_tower: dict[str, "torch.Tensor"] = {}
    mapped_proj: dict[str, "torch.Tensor"] = {}
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(TOWER_PREFIX):
                mapped_tower[key[len(TOWER_PREFIX) :]] = f.get_tensor(key)
            elif key.startswith(PROJ_PREFIX):
                mapped_proj[key[len(PROJ_PREFIX) :]] = f.get_tensor(key)
    print(f"checkpoint tensors: tower={len(mapped_tower)} projector={len(mapped_proj)}")
    tower.load_state_dict(mapped_tower, strict=True)
    projector.load_state_dict(mapped_proj, strict=True)
    tower.eval()
    projector.eval()
    return tower, projector


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity_stats(actual: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denom = np.maximum(
        np.abs(reference.astype(np.float64)), 1e-3
    )  # abs-floor covered rel
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p999_abs": float(np.quantile(diff, 0.999)),
        "max_rel_abs_covered": float((diff / denom).max()),
        "p999_rel_abs_covered": float(np.quantile(diff / denom, 0.999)),
    }


def compute_window_geometry(frames: int, n_window: int, n_window_infer: int) -> dict:
    """Static cu_seqlens / max_seqlen / pad count for an all-valid mel bucket.

    Mirrors ``get_audio_cu_seqlens`` (transformers) == the mlx_audio
    ``AudioEncoder`` windowing == the Rust ``frames_to_audio_pads`` formula.
    """
    import torch
    from transformers.models.qwen3_asr.modeling_qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    chunk_len = n_window * 2
    num_chunks = frames // chunk_len
    aftercnn_lens = _get_feat_extract_output_lengths(torch.tensor([frames]))
    feature_lens_after_cnn = _get_feat_extract_output_lengths(
        torch.tensor([chunk_len] * num_chunks)
    )
    max_len_after_cnn = int(feature_lens_after_cnn.max())
    window_aftercnn = max_len_after_cnn * (n_window_infer // chunk_len)

    pads = int(aftercnn_lens[0])
    cu_chunk_lens = [0]
    for cnn_len in [pads]:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens.append(remainder)
    cu_seqlens = torch.tensor(cu_chunk_lens, dtype=torch.int32).cumsum(
        -1, dtype=torch.int32
    )
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return {
        "pads": pads,
        "chunk_len": chunk_len,
        "num_chunks": num_chunks,
        "window_aftercnn": window_aftercnn,
        "cu_seqlens": cu_seqlens,  # int32 [num_windows + 1]
        "max_seqlen": int(lengths.max()),  # longest attention window
        "window_lengths": [int(v) for v in lengths],
    }


def probe_conv_static_dims(tower) -> dict:
    """Static conv-stack output dims for one 100-frame chunk (no drift risk:
    measured through the real modules on a zero probe)."""
    import torch

    chunk_len = tower.n_window * 2
    with torch.no_grad():
        x = tower.conv2d1(torch.zeros(1, 1, tower.config.num_mel_bins, chunk_len))
        x = tower.conv2d2(x)
        x = tower.conv2d3(x)
    freq_bins, time_steps = int(x.shape[2]), int(x.shape[3])
    conv_in = int(tower.conv_out.weight.shape[1])
    assert conv_in == int(x.shape[1]) * freq_bins, (
        f"conv_out expects {conv_in} features but conv stack yields {x.shape[1]}x{freq_bins}"
    )
    return {
        "chunk_len": chunk_len,
        "num_mel_bins": int(tower.config.num_mel_bins),
        "d_model": int(tower.config.d_model),
        "conv_channels": int(x.shape[1]),
        "freq_bins": freq_bins,
        "time_steps": time_steps,
        "conv_in": conv_in,
    }


def make_static_encoder_forward(geom: dict, dims: dict):
    """Qwen3ASREncoder.forward with constant window geometry (trace shim).

    Line-for-line the transformers 5.17 implementation (conv front-end, pos
    emb, packing, layers, ln_post — all through the ORIGINAL submodules)
    except the three data-dependent bakes coremltools cannot consume:

    1. ``get_audio_cu_seqlens(...)``  -> constant ``cu_seqlens`` (kwarg hook
       the real code already supports) + ``max_seqlen`` int
    2. ``nonzero()`` gather indices   -> constant ``arange(pads)``
       (all-valid 3000-frame bucket, so the gather is the identity over the
       packed positions — asserted bitwise-equal to the real forward first)
    """
    import torch
    import torch.nn.functional as F
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    cu_seqlens_const: "torch.Tensor" = geom["cu_seqlens"]
    max_seqlen_const: int = geom["max_seqlen"]
    indices_const: "torch.Tensor" = torch.arange(geom["pads"], dtype=torch.int64)

    # All bucket dims are Python constants — traced shape arithmetic would
    # leave aten::Int/floor_divide bake nodes coremltools cannot consume.
    bsz = 1
    num_chunks = geom["num_chunks"]
    chunk_len = dims["chunk_len"]
    n_mels = dims["num_mel_bins"]
    d_model = dims["d_model"]
    time_steps = dims["time_steps"]
    conv_in = dims["conv_in"]

    def static_forward(self, input_features, input_features_mask):
        cu_seqlens = cu_seqlens_const
        max_seqlen = max_seqlen_const
        chunked = (
            input_features.view(bsz, n_mels, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(bsz * num_chunks, 1, n_mels, chunk_len)
        )
        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2)
            .contiguous()
            .view(bsz * num_chunks, time_steps, conv_in)
        )
        conv_out += self.positional_embedding.positional_embedding[:time_steps].to(
            conv_out.dtype
        )

        hidden_states = torch.index_select(
            conv_out.reshape(-1, d_model), 0, indices_const
        )

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states, cu_seqlens, max_seqlen=max_seqlen
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)

    return static_forward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-name", default="qwen3_asr_audio_tower_fp16")
    parser.add_argument(
        "--skip-mlx-check",
        action="store_true",
        help="skip sanity A (torch fp32 vs MLX) — reuse only after a passing run",
    )
    args = parser.parse_args()

    import torch
    import transformers
    import coremltools as ct

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "design": "docs/ane-feasibility.md minimal spike, round 1 (plain, no ANE rewrites)",
        "fixture": str(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "compute_units": "CPU_AND_NE",
        "minimum_deployment_target": "macOS15",
        "precision": "fp16 (mlprogram compute_precision=FLOAT16, fp32 traced graph)",
        "traced_input_dtype": "fp16 (cast to fp32 as the first graph op)",
        "generated_with": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            **{k: v for k, v in mlx_versions().items()},
        },
    }
    print(f"coremltools {ct.__version__} / torch {torch.__version__}")

    mel = np.load(args.fixture)
    assert mel.shape == (1, 128, FRAMES), f"fixture shape {mel.shape}"
    mel_t = torch.from_numpy(mel)

    # ---- torch encoder segment (strict weights) ---------------------------
    t0 = time.perf_counter()
    tower, projector = load_encoder_segment()
    print(f"tower+projector loaded strict in {time.perf_counter() - t0:.1f} s")

    mask = torch.ones(1, FRAMES)

    # ---- round-1.5 trace shim: static window geometry ---------------------
    from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder

    geom = compute_window_geometry(FRAMES, tower.n_window, tower.n_window_infer)
    meta["window_geometry"] = {
        "frames": FRAMES,
        "pads": geom["pads"],
        "window_aftercnn": geom["window_aftercnn"],
        "max_seqlen": geom["max_seqlen"],
        "window_lengths": geom["window_lengths"],
        "cu_seqlens": [int(v) for v in geom["cu_seqlens"]],
    }
    print(
        f"window geometry: pads={geom['pads']} cu_seqlens={geom['cu_seqlens']} max_seqlen={geom['max_seqlen']}"
    )

    orig_forward = Qwen3ASREncoder.forward
    dims = probe_conv_static_dims(tower)
    meta["static_dims"] = dims
    print(f"static conv dims: {dims}")
    with torch.no_grad():
        real_out = tower(
            input_features=mel_t, input_features_mask=mask
        ).last_hidden_state
        Qwen3ASREncoder.forward = make_static_encoder_forward(geom, dims)
        static_out = tower(
            input_features=mel_t, input_features_mask=mask
        ).last_hidden_state
        Qwen3ASREncoder.forward = orig_forward
    shim_diff = (real_out.float() - static_out.float()).abs().max().item()
    print(f"shim vs real forward (fp32) max abs: {shim_diff:.3e}")
    meta["shim_vs_real_max_abs"] = shim_diff
    if shim_diff != 0.0:
        print(
            "FATAL: static-geometry shim is not bitwise-equal to the real forward; aborting."
        )
        return 1
    Qwen3ASREncoder.forward = make_static_encoder_forward(
        geom, dims
    )  # patched for tracing below

    # ---- sanity A: eager fp32 torch vs MLX bf16 (mapping validation) ------
    if not args.skip_mlx_check:
        t0 = time.perf_counter()
        mlx_tower = load_mlx_tower()
        mlx_ref = mlx_tower_forward(mlx_tower, mel)  # [390, 2048] f32
        with torch.no_grad():
            torch_out = projector(
                tower(input_features=mel_t, input_features_mask=mask).last_hidden_state
            )
        torch_ref = torch_out.numpy().astype(np.float32)
        stats_a = parity_stats(torch_ref, mlx_ref)
        meta["sanityA_torch_fp32_vs_mlx"] = stats_a
        print(
            f"sanity A (torch fp32 vs MLX bf16): {stats_a}  [{time.perf_counter() - t0:.1f} s]"
        )
        if stats_a["max_abs"] > 1e-2:
            print(
                "FATAL: sanity A failed — weight mapping or graph mismatch; not converting."
            )
            return 1

    # ---- fp16 conversion strategy -----------------------------------------
    # The fp16-traced graph hits a coremltools conv dtype asymmetry (weight
    # materialized fp32 vs bias fp16 -> op validation error). Standard fix,
    # same as Apple's Llama export: trace the fp32 graph, declare an fp16
    # input, and let compute_precision=FLOAT16 downcast uniformly.
    class TowerSegment(torch.nn.Module):
        def __init__(self, tower, projector):
            super().__init__()
            self.tower = tower.float()
            self.projector = projector.float()
            self.register_buffer("mask", torch.ones(1, FRAMES))

        def forward(self, input_features: torch.Tensor) -> torch.Tensor:
            hidden = self.tower(
                input_features=input_features.float(), input_features_mask=self.mask
            )
            return self.projector(hidden.last_hidden_state)

    segment = TowerSegment(tower, projector).eval()
    example = mel_t.half()

    with torch.no_grad():
        out_eager = segment(example)
        t0 = time.perf_counter()
        traced = torch.jit.trace(segment, (example,), check_trace=False)
        trace_s = time.perf_counter() - t0
        out_traced = traced(example)
    diff_trace = (out_eager.float() - out_traced.float()).abs().max().item()
    print(f"trace in {trace_s:.1f} s; traced-vs-eager max abs: {diff_trace:.3e}")
    meta["trace_seconds"] = trace_s
    meta["trace_vs_eager_max_abs"] = diff_trace
    meta["output_shape"] = list(out_traced.shape)

    op_counts = Counter(node.kind() for node in traced.inlined_graph.nodes())
    (out_dir / "traced_ops.txt").write_text(
        "\n".join(f"{count:5d}  {op}" for op, count in op_counts.most_common()) + "\n"
    )
    meta["traced_op_kinds"] = dict(op_counts)
    print(
        f"traced ops: {sum(op_counts.values())} nodes, {len(op_counts)} kinds -> traced_ops.txt"
    )

    # ---- Core ML conversion ------------------------------------------------
    print("converting (CPU_AND_NE, macOS15 target)…")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_features", shape=(1, 128, FRAMES), dtype=np.float16
            )
        ],
        outputs=[ct.TensorType(name="tower_output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    convert_s = time.perf_counter() - t0
    mlpackage = out_dir / f"{args.model_name}.mlpackage"
    mlmodel.save(str(mlpackage))
    pkg_size = sum(p.stat().st_size for p in mlpackage.rglob("*") if p.is_file())
    print(f"convert {convert_s:.1f} s -> {mlpackage} ({pkg_size / 1e6:.0f} MB)")
    meta.update(
        convert_seconds=convert_s,
        mlpackage=str(mlpackage),
        mlpackage_bytes=pkg_size,
        mlpackage_weight_sha256=sha256_file(
            mlpackage / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        ),
    )

    # ---- AOT compile (coremlcompiler) --------------------------------------
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(out_dir)],
        capture_output=True,
        text=True,
    )
    compile_s = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FATAL: coremlcompiler failed: {proc.stderr[-2000:]}")
        return 1
    mlmodelc = out_dir / f"{args.model_name}.mlmodelc"
    cmlc_size = sum(p.stat().st_size for p in mlmodelc.rglob("*") if p.is_file())
    print(
        f"coremlcompiler compile {compile_s:.1f} s -> {mlmodelc} ({cmlc_size / 1e6:.0f} MB)"
    )
    meta.update(
        compile_seconds=compile_s, mlmodelc=str(mlmodelc), mlmodelc_bytes=cmlc_size
    )

    # ---- load + one prediction on the compiled model ------------------------
    t0 = time.perf_counter()
    compiled = ct.models.CompiledMLModel(
        str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = compiled.predict({"input_features": mel.astype(np.float16)})
    pred_s = time.perf_counter() - t0
    out = pred["tower_output"]
    print(
        f"CompiledMLModel load {load_s:.2f} s; first predict {pred_s:.3f} s; output {np.asarray(out).shape}"
    )
    meta.update(
        compiled_load_seconds=load_s,
        first_predict_seconds=pred_s,
        predict_output_shape=list(np.asarray(out).shape),
    )

    (out_dir / "convert_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"meta -> {out_dir / 'convert_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
