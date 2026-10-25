#!/usr/bin/env python
"""Export golden decoder reference fixtures for the Qwen3-ASR native port (WS-06b).

Runs the mlx-audio Qwen3-ASR TEXT DECODER (checkpoint
``mlx-community/Qwen3-ASR-1.7B-4bit``) on a deterministic synthetic input and
writes into ``native/golden/asr_decoder/``:

- ``features.f32``  synthetic audio-encoder output, little-endian float32,
                    row-major ``[n_audio, hidden=2048]``. Values are exact
                    widenings of the bf16 array actually fed to the model, so
                    the Rust f16 conversion is lossless.
- ``logits.f32``    decoder logits through the tied-embedding head
                    (``embed_tokens.as_linear``), little-endian float32,
                    row-major ``[len(dump_positions), vocab=151936]`` at the
                    prompt positions listed in ``dump_positions``.
- ``manifest.json`` prompt ids (locked to the tokenizer golden fixture
                    ``asr_no_context_language_chinese``), dump positions,
                    the forced 16-step greedy argmax stream (EOS tokens kept
                    in-stream and flagged, see notes), per-token top-2 logit
                    gaps, eos ids, shapes, seed, versions, and sha256s.

Reference chain (mirrors mlx-audio ``Qwen3ASRModel`` exactly):

1. ``_build_inputs_embeds(input_ids, audio_features)`` replaces the embedding
   row of every ``<|audio_pad|>`` prompt position with the corresponding
   audio-feature row (in order; audio features cast to the embedding dtype).
2. The 28-layer decoder prefill runs over the full prompt, filling a KV
   cache (``model.model(inputs_embeds=..., cache=...)``).
3. The first generated token is the argmax of the last prompt position's
   logits; each next token is the argmax of the previous token's logits
   (greedy; mlx ``generate_step`` uses ``mx.argmax`` as the sampler).
4. Production generation stops on an EOS token (``_eos_token_ids``:
   ``<|endoftext|>``=151643 / ``<|im_end|>``=151645) and drops it from the
   output. On this synthetic noise input the model emits EOS immediately,
   so the fixture instead records the FORCED 16-step argmax stream (EOS
   kept in-stream and flagged) to lock multi-step decoder parity; the
   eos-aware production loop is exercised by the Rust unit tests.

Usage::

    uv run python scripts/export_asr_decoder_reference.py \\
        [--out native/golden/asr_decoder] [--checkpoint DIR]

Deterministic: fixed seed, fixed prompt fixture, no wall-clock inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "native" / "golden" / "asr_decoder"
TOKENIZER_CASE = (
    REPO_ROOT / "native" / "golden" / "tokenizer" / "cases" / "asr_no_context_language_chinese.json"
)
DEFAULT_REPO_DIRNAME = "models--mlx-community--Qwen3-ASR-1.7B-4bit"

AUDIO_SEED = 20261025
DUMP_HEAD_POSITIONS = 4  # logits dumped for prompt positions 0..3 ...
MAX_GREEDY_TOKENS = 16  # ... and greedy tokens capped at 16


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
        if (entry / "config.json").is_file() and (entry / "model.safetensors").is_file():
            return entry
    raise SystemExit(f"no snapshot with weights under {snapshots}; pass --checkpoint")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_audio.stt import load

    case = json.loads(TOKENIZER_CASE.read_text())
    audio_pad_count = int(case["audio_pad_count"])
    language = case["language"]
    context = case.get("context")

    snapshot = resolve_checkpoint(args.checkpoint)
    print(f"loading {snapshot}")
    model = load(str(snapshot))

    text_cfg = model.config.text_config
    vocab = int(text_cfg.vocab_size)
    hidden = int(text_cfg.hidden_size)
    audio_token_id = int(model.config.audio_token_id)
    eos_ids = sorted(model._eos_token_ids())
    print(f"vocab={vocab} hidden={hidden} audio_token_id={audio_token_id} eos={eos_ids}")

    # Prompt: the exact chat string build_asr_prompt renders, re-encoded by
    # the checkpoint tokenizer and asserted equal to the committed fixture.
    system_content = f"{context}\n" if context else ""
    assistant_prefix = f"language {language}<asr_text>" if language else ""
    prompt_text = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * audio_pad_count}<|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )
    encoded = model._tokenizer.encode(prompt_text, add_special_tokens=False)
    if encoded != list(case["prompt_ids"]):
        raise SystemExit("re-encoded prompt ids diverge from the tokenizer golden fixture")
    prompt_ids = [int(i) for i in case["prompt_ids"]]
    prompt_len = len(prompt_ids)
    print(f"prompt_len={prompt_len} (audio pads: {audio_pad_count})")

    # Synthetic audio features: [n_audio, hidden], cast to the embedding dtype
    # (bf16) exactly like _build_inputs_embeds does; dumped as widened f32 so
    # the Rust fixture holds the identical values.
    rng = np.random.RandomState(AUDIO_SEED)
    feats_f32 = rng.standard_normal((audio_pad_count, hidden)).astype(np.float32)
    feats_ref = mx.array(feats_f32).astype(mx.bfloat16)
    features_dump = np.asarray(feats_ref.astype(mx.float32), dtype="<f4")
    assert features_dump.shape == (audio_pad_count, hidden)
    assert np.isfinite(features_dump).all()

    # 1) merged input embeds: audio features replace <|audio_pad|> rows
    input_ids = mx.array([prompt_ids])
    embeds = model._build_inputs_embeds(input_ids, feats_ref)

    # 2) decoder prefill over the whole prompt (fills the KV cache)
    cache = model.make_cache()
    hidden_states = model.model(inputs_embeds=embeds, cache=cache)  # [1, L, H]
    mx.eval(hidden_states)
    assert hidden_states.shape == (1, prompt_len, hidden)

    # 3) logits at the dumped positions (tied-embedding head)
    dump_positions = list(range(DUMP_HEAD_POSITIONS)) + [prompt_len - 1]
    last_logits = model.model.embed_tokens.as_linear(hidden_states[0][prompt_len - 1 : prompt_len])
    assert last_logits.shape == (1, vocab)
    mx.eval(last_logits)
    logits_dump = np.asarray(
        model.model.embed_tokens.as_linear(hidden_states[0][dump_positions]).astype(mx.float32),
        dtype="<f4",
    )
    assert logits_dump.shape == (len(dump_positions), vocab)
    assert np.isfinite(logits_dump).all()

    # 4) forced 16-step greedy argmax stream. The production loop stops at
    # EOS; on this synthetic (noise) input the model emits EOS immediately
    # with a decisive gap, which would leave a 0-token fixture. To lock
    # multi-step decoder parity the exporter instead records the raw 16-step
    # argmax stream, KEEPING any EOS token in-stream and flagging it
    # (greedy_is_eos, first_eos_step). The Rust fixture test drives the same
    # forced loop over the decoder step API.
    gaps: list[float] = []
    is_eos: list[bool] = []
    first_eos_step: int | None = None
    top2 = np.sort(np.asarray(last_logits.astype(mx.float32))[0])
    gaps.append(float(top2[-1] - top2[-2]))
    y = mx.argmax(last_logits, axis=-1)  # [1]
    tokens: list[int] = []
    for step in range(MAX_GREEDY_TOKENS):
        token = int(y)
        tokens.append(token)
        eos_hit = token in eos_ids
        is_eos.append(eos_hit)
        if eos_hit and first_eos_step is None:
            first_eos_step = step
        if len(tokens) >= MAX_GREEDY_TOKENS:
            break
        logits = model._forward_with_embeds(model.model.embed_tokens(y[:, None]), cache)
        step_logits = logits[:, -1, :]
        mx.eval(step_logits)
        t2 = np.sort(np.asarray(step_logits.astype(mx.float32))[0])
        gaps.append(float(t2[-1] - t2[-2]))
        y = mx.argmax(step_logits, axis=-1)
    print(f"forced greedy stream ({len(tokens)} tokens, first eos step={first_eos_step}): {tokens}")
    print(f"min top-2 logit gap: {min(gaps):.6f}")

    # write fixtures
    args.out.mkdir(parents=True, exist_ok=True)
    features_path = args.out / "features.f32"
    logits_path = args.out / "logits.f32"
    features_path.write_bytes(features_dump.tobytes())
    logits_path.write_bytes(logits_dump.tobytes())

    manifest = {
        "kind": "asr_decoder",
        "checkpoint": "mlx-community/Qwen3-ASR-1.7B-4bit",
        "checkpoint_snapshot": str(snapshot),
        "prompt_source": "native/golden/tokenizer/cases/asr_no_context_language_chinese.json",
        "language": language,
        "context": context,
        "audio_seed": AUDIO_SEED,
        "audio_token_id": audio_token_id,
        "audio_pad_count": audio_pad_count,
        "n_audio": audio_pad_count,
        "hidden": hidden,
        "vocab": vocab,
        "prompt_len": prompt_len,
        "prompt_ids": prompt_ids,
        "dump_positions": dump_positions,
        "logits_rows": len(dump_positions),
        "eos_ids": eos_ids,
        "max_greedy_tokens": MAX_GREEDY_TOKENS,
        "greedy_tokens": tokens,
        "greedy_is_eos": is_eos,
        "first_eos_step": first_eos_step,
        "greedy_top2_gap": gaps,
        "mlx_version": __import__("importlib").metadata.version("mlx"),
        "mlx_audio_version": __import__("importlib").metadata.version("mlx-audio"),
        "notes": (
            "features.f32/logits.f32 are exact widenings of bf16 reference "
            "values; the Rust port computes in f16 weights with f32 "
            "accumulation, so logits parity holds within a bf16-vs-f16 drift "
            "bound while greedy argmax tokens must match exactly. "
            "greedy_tokens is the forced 16-step argmax stream (EOS tokens "
            "kept in-stream and flagged) because the production eos-aware "
            "loop emits EOS at step 0 on this synthetic noise input."
        ),
        "files": {
            "features.f32": {
                "shape": [audio_pad_count, hidden],
                "dtype": "<f4",
                "sha256": sha256_file(features_path),
            },
            "logits.f32": {
                "shape": [len(dump_positions), vocab],
                "dtype": "<f4",
                "sha256": sha256_file(logits_path),
            },
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {features_path} ({features_path.stat().st_size} bytes)")
    print(f"wrote {logits_path} ({logits_path.stat().st_size} bytes)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
