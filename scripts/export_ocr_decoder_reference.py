#!/usr/bin/env python
"""Export golden decoder reference fixtures for the Unlimited-OCR native port (WS-07b).

Runs the mlx-vlm Unlimited-OCR DeepSeek-V2-lite LANGUAGE DECODER (checkpoint
``mlx-community/Unlimited-OCR-4bit``) on a deterministic synthetic input and
writes into ``native/golden/ocr_decoder/``:

- ``logits.bf16``     decoder logits through the quantized ``lm_head`` as
                      raw little-endian bf16 bits, row-major
                      ``[len(logits_kept_positions), vocab=129280]``: the
                      prompt positions ``logits_kept_positions`` (a subset
                      of ``dump_positions``; full-prompt window-128 run).
- ``ring_logits.bf16`` logits for a SECOND run whose caches use
                      ``RingSlidingKVCache(window_size=4)`` so the steady
                      ring-overwrite phase is reached within a few steps;
                      rows are decode steps ``ring.kept_steps`` (the
                      logits that produce forced tokens k+1), bf16 bits.
- ``ring_cache_keys.f32`` layer-0 key rows after the first ring write.

The image features fed to the decoder are not dumped: they are the vision
golden ``small_final_features.bf16u16`` (with its f16 snaps, see
``image_features_f16_snaps``), which the Rust test reloads. Committed
goldens stay <= 1 MB each, hence the bf16 row subsets.
- ``manifest.json``   prompt ids (locked to the tokenizer golden fixture
                      ``ocr_document_parsing``), forced greedy argmax
                      streams for both runs, per-step top-2 logit gaps,
                      MoE routing probe (layers 1 and 11: selected expert
                      ids, softmax scores, and the 7th score margin), eos
                      id, shapes, seeds, versions, and sha256s.

Reference chain (mirrors mlx_vlm ``generate_step`` for this model):

1. ``model.language_model.model.embed_tokens(input_ids)`` builds the prompt
   embeds; the ``<image>`` (128815) rows are replaced, in order, with the
   synthetic feature rows (bf16, like ``Model.get_input_embeddings`` does
   with projected vision features).
2. Prefill runs over the full prompt with ``model.make_cache()`` caches
   (``RingSlidingKVCache(128)`` per layer), i.e. the mlx-vlm unchunked
   forward; the reference runtime chunks the prompt but chunked and full
   prefill produce identical cache/logits (plain causal appends).
3. The first forced token is ``argmax(logits[:, -1, :])``; each next token
   is ``argmax`` of the previous token's logits (temperature 0 sampler).
   The stream is FORCED (EOS tokens kept in-stream and flagged) because on
   a synthetic noise image the model emits EOS at step 0 — same fixture
   strategy as ``export_asr_decoder_reference.py``. No logits processors
   are applied (the sliding-window no-repeat n-gram guard is unit-tested
   against the Python processor semantics instead).
4. The ring run replaces the caches with ``RingSlidingKVCache(4)`` and
   forces 12 steps: decode step 4 is the first written through the ring
   path (offset reaches ``prefill_length + window``), so steps 4..8 lock
   the ring-overwrite attention semantics end-to-end.
5. MoE routing probe: ``MoEGate`` of layers 1 and 11 is wrapped with a
   recording proxy for the window-128 run; the manifest records the
   greedy top-6 selected ids, their softmax scores, and the 7th-largest
   score (selection margin) for the last prompt position and forced steps
   0..2. Routing is softmax + greedy top-k per the runtime config
   (``scoring_func="softmax"``, ``routed_scaling_factor=1.0``).

Usage::

    uv run python scripts/export_ocr_decoder_reference.py \\
        [--out native/golden/ocr_decoder] [--checkpoint DIR]

Deterministic: fixed seed, fixed prompt fixture, no wall-clock inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "native" / "golden" / "ocr_decoder"
TOKENIZER_CASE = (
    REPO_ROOT
    / "native"
    / "golden"
    / "tokenizer"
    / "cases"
    / "ocr_document_parsing.json"
)
DEFAULT_REPO_DIRNAME = "models--mlx-community--Unlimited-OCR-4bit"

FEATURE_SEED = 20261025
DUMP_HEAD_POSITIONS = 4  # logits dumped for prompt positions 0..3 ...
MAX_GREEDY_TOKENS = 16  # ... and greedy tokens capped at 16
RING_WINDOW = 4  # small ring so the overwrite phase starts at step 4
RING_STEPS = 12
RING_DUMP_STEPS = 9  # ring logits computed for forced steps 0..8 ...
RING_KEPT_STEPS = [4, 5, 6]  # ... and committed for the first ring writes
PROBE_LAYERS = [1, 11]
PROBE_STEPS = 3  # probe positions: last prompt row + forced steps 0..2
# Routing stability threshold in 6th-vs-7th softmax-score space: margins
# below this flip between the bf16 reference and an f16 store port (see
# the WS-07b parity investigation: a 6e-4 margin flipped, 9e-4 held, and
# the flip permanently diverges both trajectories).
RING_STABLE_MARGIN = 0.003


def resolve_checkpoint(explicit: str | None) -> Path:
    """Local snapshot dir of the 4-bit checkpoint (explicit arg wins)."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise SystemExit(f"checkpoint {path} has no config.json")
        return path
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshots = hub / DEFAULT_REPO_DIRNAME / "snapshots"
    if not snapshots.is_dir():
        raise SystemExit(
            f"{DEFAULT_REPO_DIRNAME} not found under {hub}; pass --checkpoint"
        )
    for entry in sorted(snapshots.iterdir()):
        if (entry / "config.json").is_file() and (
            entry / "model.safetensors"
        ).is_file():
            return entry
    raise SystemExit(f"no snapshot with weights under {snapshots}; pass --checkpoint")


def f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """Bit patterns of bf16-exact f32 values (asserted lossless)."""
    u32 = np.ascontiguousarray(a, dtype="<f4").view(np.uint32)
    assert not (u32 & np.uint32(0xFFFF)).any(), "values are not bf16-exact"
    return (u32 >> np.uint32(16)).astype("<u2")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vision_features(rows: int, hidden: int) -> tuple[np.ndarray | None, str, int]:
    """Real WS-07a vision golden features when present and geometry-matched.

    Reads ``native/golden/ocr_vision/small_final_features.bf16u16`` (bf16 bit
    patterns, u16 LE, row-major) — the projector output for the same gundam
    geometry as the tokenizer fixture (273x1280 for
    ``ocr_document_parsing``). Values that would not survive the Rust f16
    fixture path (f16 subnormal edge) are snapped to their f16 round-trip
    and the model is fed the snapped values, so fixture and reference stay
    bit-consistent by construction. Returns ``(features, source, snaps)``;
    ``features is None`` means fall back to the synthetic draw.
    """
    path = (
        REPO_ROOT / "native" / "golden" / "ocr_vision" / "small_final_features.bf16u16"
    )
    if not path.is_file():
        return None, "synthetic", 0
    bits = np.frombuffer(path.read_bytes(), dtype="<u2")
    if bits.size != rows * hidden:
        return None, "synthetic", 0
    f32 = (bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, hidden)
    widened = f32.astype(np.float16).astype(np.float32)
    snaps = int(np.count_nonzero(widened != f32))
    if snaps:
        f32 = np.where(widened != f32, widened, f32)
    if not np.isfinite(f32).all():
        return None, "synthetic", 0
    return f32, str(path.relative_to(REPO_ROOT)), snaps


def synthetic_features(seed: int, rows: int, hidden: int) -> tuple[np.ndarray, int]:
    """bf16 [rows, hidden] whose values survive the Rust f16 fixture path.

    Draws standard normals, rounds to bf16 (the model embedding dtype —
    exactly what ``Model.get_input_embeddings`` stores), and verifies every
    bf16 value converts bf16 -> f32 -> f16 -> f32 exactly. bf16 has an 8-bit
    mantissa and f16 a 10-bit one, so rounding is the only concern at the
    range edges (f16 subnormals / overflow), where the seed advances
    deterministically. The returned array is the f32 widening of the bf16
    values; the model is fed ``mx.array(...).astype(mx.bfloat16)``.
    """
    global mx  # set in main() before this helper runs
    while True:
        draw = (
            np.random.RandomState(seed)
            .standard_normal((rows, hidden))
            .astype(np.float32)
        )
        bf = np.asarray(mx.array(draw).astype(mx.bfloat16).astype(mx.float32))
        feats = bf.astype(np.float16).astype(np.float32)
        if np.array_equal(feats, bf) and np.isfinite(feats).all():
            return bf, seed
        seed += 1


def top2_gap(logits_row: np.ndarray) -> float:
    values = np.sort(logits_row.astype(np.float32))
    return float(values[-1] - values[-2])


class GateProbe:
    """Records ``(inds, scores, full_softmax)`` from the wrapped MoEGate."""

    def __init__(self, inner, store: list):
        self.inner = inner
        self.store = store

    def __call__(self, x):
        inds, scores = self.inner(x)
        # Reproduce the gate's exact softmax over all experts for the
        # selection-margin probe (gates = x @ W.T, precise softmax).
        full = mx.softmax(x @ self.inner.weight.T, axis=-1, precise=True)
        self.store.append(
            (
                np.asarray(inds.astype(mx.int32)),
                np.asarray(scores.astype(mx.float32)),
                np.asarray(full.astype(mx.float32)),
            )
        )
        return inds, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    global mx
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    case = json.loads(TOKENIZER_CASE.read_text())
    prompt_ids = [int(i) for i in case["prompt_ids"]]
    image_pad_count = int(case["image_pad_count"])
    assert prompt_ids.count(128815) == image_pad_count, "fixture pad count mismatch"
    prompt_len = len(prompt_ids)

    snapshot = resolve_checkpoint(args.checkpoint)
    print(f"loading {snapshot}")
    model, _processor = load(str(snapshot))
    lm = model.language_model
    text_cfg = lm.config
    hidden = int(text_cfg.hidden_size)
    vocab = int(text_cfg.vocab_size)
    layers = list(lm.model.layers)
    eos_ids = [int(model.config.eos_token_id)]
    print(
        f"vocab={vocab} hidden={hidden} layers={len(layers)} "
        f"attn={type(layers[0].self_attn).__name__} "
        f"scoring={layers[1].mlp.gate.scoring_func} eos={eos_ids}"
    )
    assert len(layers) == 12
    assert (
        isinstance(layers[1].mlp, type(layers[11].mlp))
        and "MoE" in type(layers[1].mlp).__name__
    )

    # Vision-projector features replacing the <image> rows: the WS-07a
    # vision golden block when present, else the deterministic synthetic
    # draw (both bf16-exact on the Rust f16 fixture path).
    feats_f32, features_source, feature_snaps = load_vision_features(
        image_pad_count, hidden
    )
    feature_seed = None
    if feats_f32 is None:
        feats_f32, feature_seed = synthetic_features(
            FEATURE_SEED, image_pad_count, hidden
        )
    feats_ref = mx.array(feats_f32).astype(mx.bfloat16)
    print(
        f"features source={features_source} seed={feature_seed} "
        f"f16_snaps={feature_snaps} shape={feats_ref.shape} dtype={feats_ref.dtype}"
    )

    # 1) merged input embeds: feature rows replace <image> (128815) rows in order
    ids = mx.array([prompt_ids])
    embeds = lm.model.embed_tokens(ids)
    image_positions = [i for i, t in enumerate(prompt_ids) if t == 128815]
    for row, pos in enumerate(image_positions):
        embeds[0, pos] = feats_ref[row]
    mx.eval(embeds)
    assert embeds.shape == (1, prompt_len, hidden)

    dump_positions = list(range(DUMP_HEAD_POSITIONS)) + [prompt_len - 1]

    def forced_run(window: int, steps: int, cache_dump: tuple[int, int] | None = None):
        """Prefill + forced greedy argmax stream over RingSlidingKVCache(window).

        ``cache_dump=(after_step, from_row)`` additionally captures layer 0's
        key rows ``[from_row, prefill_length + window)`` right after that
        decode step completes — a routing-independent probe of the ring
        write semantics.
        """
        caches = [RingSlidingKVCache(window) for _ in layers]
        out = lm(inputs=ids, inputs_embeds=embeds, cache=caches)
        # Dump rows at the fixture positions straight from the prefill
        # logits (bf16 -> f32 widening).
        dump_rows = np.asarray(out.logits[0][dump_positions].astype(mx.float32))
        last_logits = out.logits[:, -1, :]
        prefill_logits = np.asarray(last_logits.astype(mx.float32))[0]
        mx.eval([c.state for c in caches])
        tokens: list[int] = []
        gaps: list[float] = [top2_gap(prefill_logits)]
        y = int(mx.argmax(last_logits, axis=-1))
        step_logits_rows: list[np.ndarray] = []
        cache_rows: np.ndarray | None = None
        for step in range(steps):
            tokens.append(y)
            out = lm(inputs=mx.array([[y]]), cache=caches)
            row = np.asarray(out.logits[:, -1, :].astype(mx.float32))[0]
            mx.eval([c.state for c in caches])
            gaps.append(top2_gap(row))
            if len(step_logits_rows) < RING_DUMP_STEPS:
                step_logits_rows.append(row)
            if cache_dump is not None and step == cache_dump[0]:
                from_row = cache_dump[1]
                end_row = caches[0].prefill_length + window
                keys = np.asarray(
                    caches[0].keys[0, :, from_row:end_row, :].astype(mx.float32)
                )
                cache_rows = keys.transpose(1, 0, 2).copy()
            y = int(np.argmax(row))
        return dump_rows, prefill_logits, tokens, gaps, step_logits_rows, cache_rows

    # 2+3) window-128 reference run: prefill logits + forced 16-step stream
    dump_rows, prefill_logits, tokens, gaps, _, _ = forced_run(128, MAX_GREEDY_TOKENS)
    first_eos_step = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
    print(
        f"forced greedy stream ({len(tokens)} tokens, first eos step={first_eos_step}): {tokens}"
    )
    print(f"min top-2 logit gap: {min(gaps):.6f}")

    # 4) MoE routing probe on the window-128 run (re-run for clean captures).
    stores = {layer: [] for layer in PROBE_LAYERS}
    originals = {}
    for layer in PROBE_LAYERS:
        mlp = layers[layer].mlp
        originals[layer] = mlp.gate
        mlp.gate = GateProbe(originals[layer], stores[layer])
    try:
        forced_run(128, PROBE_STEPS)
    finally:
        for layer in PROBE_LAYERS:
            layers[layer].mlp.gate = originals[layer]

    probe_entries = []
    for layer, store in stores.items():
        # call 0 = prefill (keep last row), calls 1.. = single-row decode steps
        assert len(store) >= 1 + PROBE_STEPS, f"probe layer {layer}: {len(store)} calls"
        picks = [(store[0][0][-1:], store[0][1][-1:], store[0][2][-1:])]
        for call in range(1, 1 + PROBE_STEPS):
            picks.append((store[call][0], store[call][1], store[call][2]))
        for position, (inds, scores, full) in zip(
            ["prompt_last", "step_0", "step_1", "step_2"], picks
        ):
            flat_scores = np.sort(full.reshape(-1))[::-1]
            probe_entries.append(
                {
                    "layer": layer,
                    "position": position,
                    "selected": [int(v) for v in inds.reshape(-1)],
                    "scores": [float(v) for v in scores.reshape(-1)],
                    "seventh_score": float(flat_scores[6]),
                }
            )

    # 5) ring run with window=4 (ring overwrite starts at forced step 4),
    # with a per-step MoE routing-margin probe: the greedy top-6 selection
    # is only drift-stable when the 6th-vs-7th softmax score margin is
    # comfortably above the f16-vs-bf16 store noise (~1e-3 score space);
    # thinner margins flip individual expert selections between the bf16
    # reference and any f16 port, after which the two runs follow equally
    # valid but different token branches.
    ring_margin_stores = {i: [] for i in range(1, 12)}
    ring_originals = {i: layers[i].mlp.gate for i in range(1, 12)}
    for i in range(1, 12):
        layers[i].mlp.gate = GateProbe(ring_originals[i], ring_margin_stores[i])
    RING_CACHE_FROM_ROW = prompt_len - 8  # prompt tail + the ring slots
    try:
        _, _, ring_tokens, ring_gaps, ring_rows, ring_cache_rows = forced_run(
            RING_WINDOW,
            RING_STEPS,
            cache_dump=(RING_WINDOW, RING_CACHE_FROM_ROW),
        )
    finally:
        for i in range(1, 12):
            layers[i].mlp.gate = ring_originals[i]
    ring_min_margins = []
    for step in range(RING_STEPS):
        min_margin = None
        for i in range(1, 12):
            sorted_scores = np.sort(ring_margin_stores[i][step + 1][2].reshape(-1))
            margin = float(sorted_scores[-6] - sorted_scores[-7])
            if min_margin is None or margin < min_margin:
                min_margin = margin
        ring_min_margins.append(min_margin)
    ring_flip_step = next(
        (s for s, m in enumerate(ring_min_margins) if m < RING_STABLE_MARGIN),
        None,
    )
    print(f"ring run (window={RING_WINDOW}) tokens: {ring_tokens}")
    print(f"ring min top-2 gap: {min(ring_gaps):.6f}")
    print(
        f"ring routing min margins per step: {['%.5f' % m for m in ring_min_margins]}"
    )
    print(f"ring routing flip step (margin < {RING_STABLE_MARGIN}): {ring_flip_step}")

    # write fixtures
    args.out.mkdir(parents=True, exist_ok=True)
    logits_path = args.out / "logits.bf16"
    ring_logits_path = args.out / "ring_logits.bf16"
    ring_cache_path = args.out / "ring_cache_keys.f32"
    LOGITS_KEPT_POSITIONS = [0, prompt_len - 1]  # noqa: N806
    kept_rows = [dump_positions.index(p) for p in LOGITS_KEPT_POSITIONS]
    logits_dump = f32_to_bf16_bits(dump_rows[kept_rows])
    ring_dump = f32_to_bf16_bits(np.stack(ring_rows)[RING_KEPT_STEPS])
    logits_path.write_bytes(logits_dump.tobytes())
    ring_logits_path.write_bytes(ring_dump.tobytes())
    ring_cache_rows.tofile(str(ring_cache_path))

    manifest = {
        "kind": "ocr_decoder",
        "checkpoint": "mlx-community/Unlimited-OCR-4bit",
        "checkpoint_snapshot": str(snapshot),
        "prompt_source": "native/golden/tokenizer/cases/ocr_document_parsing.json",
        "image_features_source": features_source,
        "image_features_f16_snaps": feature_snaps,
        "image_seed": feature_seed,
        "image_token_id": 128815,
        "image_pad_count": image_pad_count,
        "n_image": image_pad_count,
        "hidden": hidden,
        "vocab": vocab,
        "prompt_len": prompt_len,
        "prompt_ids": prompt_ids,
        "dump_positions": dump_positions,
        "logits_rows": len(dump_positions),
        "logits_kept_positions": LOGITS_KEPT_POSITIONS,
        "eos_ids": eos_ids,
        "max_greedy_tokens": MAX_GREEDY_TOKENS,
        "greedy_tokens": tokens,
        "greedy_is_eos": [t in eos_ids for t in tokens],
        "first_eos_step": first_eos_step,
        "greedy_top2_gap": gaps,
        "ring": {
            "window": RING_WINDOW,
            "steps": RING_STEPS,
            "forced_tokens": ring_tokens,
            "dump_steps": list(range(len(ring_rows))),
            "kept_steps": RING_KEPT_STEPS,
            "top2_gap": ring_gaps[: len(ring_rows) + 1],
            "routing_min_margins": ring_min_margins,
            "cache_dump": {
                "layer": 0,
                "from_row": RING_CACHE_FROM_ROW,
                "to_row": prompt_len + RING_WINDOW,
                "after_step": RING_WINDOW,
                "note": (
                    "ring_cache_keys.f32 holds layer 0's key rows "
                    "[from_row, prefill_length + window) right after decode "
                    "step 4 (the first ring-path write) as f32 "
                    "[rows, kv_heads=10, head_dim=128] - a "
                    "routing-independent lock on the ring slot arithmetic "
                    "and written content."
                ),
            },
            "note": (
                "ring_logits.bf16 row i is the step kept_steps[i] decode "
                "logits row (the distribution producing forced token "
                "step+1); decode step 4 is "
                "the first written through the ring path (offset reaches "
                "prefill_length + window), so rows 4..8 lock the "
                "ring-overwrite attention semantics. The greedy top-6 "
                "routing margins per step (min over MoE layers, 6th vs 7th "
                "softmax score) are recorded in routing_min_margins: at "
                "margins below ~1e-3 the expert selection is not stable "
                "under the bf16-vs-f16 store drift, and the f16 port "
                "reproduces the branch exactly only up to the first such "
                "flip (step 4 here; verified by a per-layer bisection - "
                "layer inputs match to 4e-4 through the attention, and the "
                "divergence starts inside layer 1's MoE). Past the flip the "
                "two runs follow equally valid but different branches, so "
                "the Rust parity test pins the verified prefix and compares "
                "logits rows only within it."
            ),
        },
        "moe_probe": {
            "layers": PROBE_LAYERS,
            "scoring_func": "softmax",
            "routed_scaling_factor": 1.0,
            "entries": probe_entries,
        },
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "notes": (
            "logits.bf16/ring_logits.bf16 hold the bf16 reference values "
            "bit-exactly; the Rust MLX port runs the same bf16 graph, so "
            "logits parity holds within a small kernel-order drift bound "
            "while greedy argmax tokens must match exactly. greedy_tokens is the forced 16-step argmax stream "
            "(EOS kept in-stream and flagged) because the production "
            "eos-aware loop emits EOS at step 0 on this synthetic noise "
            "input. Attention is LlamaAttention (qk_nope_head_dim=0 flips "
            "attn_type), MoE routing is softmax scoring with greedy top-6 "
            "and routed_scaling_factor 1.0, and logits go through the "
            "checkpoint's quantized lm_head (untied from embed_tokens)."
        ),
        "files": {
            "logits.bf16": {
                "shape": [len(LOGITS_KEPT_POSITIONS), vocab],
                "dtype": "bf16_bits_u16",
                "sha256": sha256_file(logits_path),
            },
            "ring_logits.bf16": {
                "shape": [len(RING_KEPT_STEPS), vocab],
                "dtype": "bf16_bits_u16",
                "sha256": sha256_file(ring_logits_path),
            },
            "ring_cache_keys.f32": {
                "shape": [prompt_len + RING_WINDOW - RING_CACHE_FROM_ROW, 10, 128],
                "dtype": "<f4",
                "sha256": sha256_file(ring_cache_path),
            },
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {logits_path} ({logits_path.stat().st_size} bytes)")
    print(f"wrote {ring_logits_path} ({ring_logits_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
