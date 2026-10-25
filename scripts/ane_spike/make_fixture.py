#!/usr/bin/env python
"""ANE spike (WS-12b): deterministic 30 s mel fixture [1, 128, 3000].

Produces the single shared input used by every spike benchmark:

- ``mel.npy``    float32 [1, 128, 3000], frequency-major — the exact tensor fed
                 to the transformers/Core ML audio tower (Core ML conversion and
                 prediction) and to the MLX reference tower.
- ``mel_tm.f32`` float32 [3000, 128] row-major, time-major — the layout the
                 Rust CPU tower (``AudioTower::forward``) consumes.
- ``meta.json``  recipe, shapes, sha256s, feature-extractor provenance.

The audio is deterministic synthetic (multi-tone sines + seeded noise, same
family as scripts/export_asr_encoder_reference.py), so the fixture is
byte-stable across runs. The HF ``Qwen3ASRFeatureExtractor`` from the local
``Qwen/Qwen3-ASR-1.7B-hf`` cache produces the mel exactly like the production
pipeline (``Qwen3ASRModel._preprocess_audio``: return_attention_mask=True,
truncation=False, padding=True).

Usage::

    uv run python scripts/ane_spike/make_fixture.py [--out native/golden/ane_spike]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "native" / "golden" / "ane_spike"

# Must resolve from the local HF cache; spike runs fully offline.
FEATURE_EXTRACTOR_REPO = "Qwen/Qwen3-ASR-1.7B-hf"
DURATION_S = 30.0
SAMPLE_RATE = 16000
RNG_SEED = 20261025


def gen_wav() -> tuple[np.ndarray, dict]:
    """Deterministic 30 s mixture: five sines across the band + seeded noise.

    Same multi-tone recipe as the committed encoder goldens, widened to 30 s
    and dithered with fixed-seed noise so no mel bin is degenerate.
    """
    freqs = [220.0, 440.0, 1000.0, 3500.0, 7800.0]
    amps = [0.45, 0.30, 0.40, 0.25, 0.15]
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    wav = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps))
    # Slow amplitude modulation so consecutive 100-frame conv chunks differ.
    wav *= 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.37 * t)
    rng = np.random.default_rng(RNG_SEED)
    wav += 0.01 * rng.standard_normal(n)
    return np.clip(wav, -1.0, 1.0).astype(np.float32), {
        "type": "multi_tone_30s",
        "freqs_hz": freqs,
        "amplitudes": amps,
        "am_mod_hz": 0.37,
        "am_depth": 0.2,
        "noise_std": 0.01,
        "rng_seed": RNG_SEED,
        "duration_s": DURATION_S,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-extractor", default=FEATURE_EXTRACTOR_REPO)
    args = parser.parse_args()

    import transformers
    from transformers import AutoProcessor

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.feature_extractor)
    fe = processor.feature_extractor

    wav, recipe = gen_wav()
    audio_inputs = fe(
        wav,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    input_features = np.ascontiguousarray(
        audio_inputs["input_features"], dtype=np.float32
    )
    attention_mask = np.asarray(audio_inputs["attention_mask"])
    true_frames = int(attention_mask.sum(axis=-1)[0])

    if input_features.shape != (1, 128, 3000):
        print(
            f"FATAL: input_features shape {input_features.shape}, expected (1, 128, 3000)"
        )
        return 1
    if true_frames != 3000:
        print(
            f"FATAL: true frames {true_frames} != 3000 (padding tail would pollute the bucket)"
        )
        return 1

    # Time-major [frames, 128] for the Rust tower / any consumer of mel_tm.f32.
    mel_tm = np.ascontiguousarray(input_features[0].T, dtype="<f4")

    np.save(out_dir / "mel.npy", input_features)
    (out_dir / "mel_tm.f32").write_bytes(mel_tm.tobytes())

    meta = {
        "generator": "scripts/ane_spike/make_fixture.py",
        "feature_extractor": args.feature_extractor,
        "generated_with": {
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "audio_recipe": recipe,
        "samples": int(wav.size),
        "sample_rate": SAMPLE_RATE,
        "mel_npy_shape": list(input_features.shape),
        "mel_tm_f32_shape": list(mel_tm.shape),
        "mel_layout": "mel.npy: [1,128,3000] frequency-major f32; mel_tm.f32: [3000,128] row-major f32 (time-major)",
        "true_frames": true_frames,
        "sha256": {
            "mel.npy": sha256_file(out_dir / "mel.npy"),
            "mel_tm.f32": sha256_file(out_dir / "mel_tm.f32"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f"mel.npy    {input_features.shape} f32  {sha256_file(out_dir / 'mel.npy')[:16]}"
    )
    print(f"mel_tm.f32 {mel_tm.shape} f32  {sha256_file(out_dir / 'mel_tm.f32')[:16]}")
    print(
        f"stats: min {input_features.min():.4f} max {input_features.max():.4f} mean {input_features.mean():.4f}"
    )
    total = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"out: {out_dir} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
