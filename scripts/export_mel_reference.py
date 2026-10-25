#!/usr/bin/env python
"""Export golden mel-spectrogram fixtures for the native engine (WS-05).

Runs the Hugging Face ``Qwen3ASRFeatureExtractor`` (checkpoint
``Qwen/Qwen3-ASR-1.7B-hf``) over a set of deterministic synthetic signals and
writes, per case, into ``native/golden/mel/cases/<name>/``:

- ``input.f32``     raw waveform, little-endian float32, one sample per word
- ``output.f32``    reference log-mel, little-endian float32, row-major
                    ``[n_mels=128, frames]`` (frequency-major, torch layout)
- ``meta.json``     case metadata: shapes, the serialized ``MelConfig`` that
                    the Rust ``aimd-engine`` audio module consumes, and the
                    deterministic generator recipe

Plus shared artifacts:

- ``mel_filters.f64``  the 201x128 slaney filterbank, little-endian float64,
                       for a direct filterbank parity test
- ``manifest.json``    versions, config, tolerances, per-case index, sha256s

The exact reference chain (transformers ``Qwen3ASRFeatureExtractor``):

1. float32 waveform, 16 kHz
2. ``torch.stft(..., n_fft=400, hop_length=160, window=torch.hann_window(400),
   center=True (reflect pad, n_fft//2 = 200 each side), normalized=False,
   onesided=True)`` -> 201 bins, ``1 + floor(N / hop)`` frames
3. drop the last frame (``stft[..., :-1]``) -> ``floor(N / hop)`` frames
4. magnitudes**2 (power, no STFT normalization)
5. ``mel_filters.T @ magnitudes`` with the slaney-scale, slaney-norm
   filterbank (201 -> 128, 0..8000 Hz), numpy float64 construction cast to
   float32 by the extractor
6. ``clamp(min=1e-10).log10()``
7. floor at the global max minus 8.0
8. ``(x + 4.0) / 4.0``

The ``min_length`` (8000-sample zero pad), ``max_length`` (480000-sample
pad/truncate) and mel-axis pad to a multiple of ``2 * n_window`` (100) steps
of the extractor wrapper are *padding concerns of the caller* (the model
graph) and are intentionally not part of these fixtures; the fixtures lock
the core sample -> log-mel chain only.

A numpy float64 mirror of the same chain is computed for every case and
compared against the torch output as a self-check; the observed max abs diff
is recorded in the manifest and predicts the tolerance the Rust port must
meet (the Rust implementation follows the same f64-internal data flow).

Usage::

    uv run python scripts/export_mel_reference.py [--out native/golden/mel]

Deterministic: fixed seeds, no wall-clock, no threading effects on values
(torch CPU STFT is deterministic for a fixed build).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "native" / "golden" / "mel"

MEL_CONFIG = {
    "sampling_rate": 16000,
    "n_mels": 128,
    "n_fft": 400,
    "hop_length": 160,
    "f_min": 0.0,
    "f_max": 8000.0,
    "mel_scale": "slaney",
    "norm": "slaney",
    "power": 2.0,
    "center": True,
    "pad_mode": "reflect",
    "window": "hann",
    "clamp_min": 1e-10,
    "floor_offset": 8.0,
    "norm_offset": 4.0,
    "norm_scale": 4.0,
    "drop_last_frame": True,
    "dither": 0.0,
}

# Self-check bound: numpy f64 mirror vs torch f32 reference. Observed max is
# ~5e-5, dominated by torch's own f32 FFT round-off at quiet bins ~200 dB
# below the spectral peak (the f64 mirror is the accurate value there). The
# parity gate itself is 1e-2 abs / 1e-3 rel; this is just an early-warning
# tripwire for exporter/regressor drift.
MIRROR_TOLERANCE = 1e-4


# ---------------------------------------------------------------------------
# Signal generators (all deterministic)
# ---------------------------------------------------------------------------


def gen_sine_sweep(sr: int = 16000, duration: float = 2.0) -> tuple[np.ndarray, dict]:
    """Exponential chirp 20 Hz -> 8 kHz, amplitude 0.8."""
    f0, f1, amp = 20.0, 8000.0, 0.8
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    k = f1 / f0
    phase = (
        2.0 * np.pi * f0 * (duration / np.log(k)) * (np.power(k, t / duration) - 1.0)
    )
    wav = amp * np.sin(phase)
    return wav, {
        "type": "exponential_chirp",
        "f0_hz": f0,
        "f1_hz": f1,
        "amplitude": amp,
    }


def gen_multi_tone(sr: int = 16000, duration: float = 1.5) -> tuple[np.ndarray, dict]:
    """Sum of five sines spread across the 0-8 kHz band."""
    freqs = [220.0, 440.0, 1000.0, 3500.0, 7800.0]
    amps = [0.5, 0.3, 0.4, 0.25, 0.15]
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    wav = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps))
    wav = np.clip(wav, -1.0, 1.0)
    return wav, {"type": "multi_tone", "freqs_hz": freqs, "amplitudes": amps}


def gen_seeded_noise(sr: int = 16000, duration: float = 1.5) -> tuple[np.ndarray, dict]:
    """Gaussian noise from a fixed seed (PCG64, stable stream)."""
    amp, seed = 0.3, 1234
    n = int(sr * duration)
    rng = np.random.default_rng(seed)
    wav = amp * rng.standard_normal(n)
    return wav, {"type": "gaussian_noise", "seed": seed, "amplitude": amp}


def gen_near_silence(sr: int = 16000, duration: float = 1.0) -> tuple[np.ndarray, dict]:
    """Tiny tone + noise straddling the 1e-10 mel-power clamp boundary."""
    tone_amp, noise_amp, freq, seed = 1e-6, 3e-7, 440.0, 7
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    wav = tone_amp * np.sin(2.0 * np.pi * freq * t) + noise_amp * rng.standard_normal(n)
    return wav, {
        "type": "near_silence",
        "tone_hz": freq,
        "tone_amplitude": tone_amp,
        "noise_seed": seed,
        "noise_amplitude": noise_amp,
    }


def gen_silence(sr: int = 16000, duration: float = 0.5) -> tuple[np.ndarray, dict]:
    """Digital silence: locks the clamp/floor/normalize constants exactly."""
    n = int(sr * duration)
    return np.zeros(n), {"type": "silence"}


def gen_speech_am_burst(
    sr: int = 16000, duration: float = 1.8
) -> tuple[np.ndarray, dict]:
    """Formant-ish tone stack, 3 Hz AM, burst gating, faint breath noise."""
    freqs = [300.0, 700.0, 1220.0, 2600.0]
    amps = [0.45, 0.25, 0.2, 0.15]
    am_hz, gate_hz, noise_amp, seed = 3.0, 1.2, 0.02, 99
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    carrier = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps))
    am = (0.5 * (1.0 + np.sin(2.0 * np.pi * am_hz * t))) ** 2
    gate = np.clip(np.sin(2.0 * np.pi * gate_hz * t) * 4.0 + 0.5, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    wav = 0.6 * am * gate * carrier + noise_amp * rng.standard_normal(n)
    wav = np.clip(wav, -1.0, 1.0)
    return wav, {
        "type": "speech_am_burst",
        "freqs_hz": freqs,
        "am_amplitude": amps,
        "am_hz": am_hz,
        "gate_hz": gate_hz,
        "noise_seed": seed,
        "noise_amplitude": noise_amp,
    }


def gen_short_clip_nonmultiple(
    sr: int = 16000, duration: float = 0.61
) -> tuple[np.ndarray, dict]:
    """Sample count not a multiple of hop_length: locks floor(N/hop) frames."""
    freq, amp = 640.0, 0.5
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    wav = amp * np.sin(2.0 * np.pi * freq * t)
    return wav, {
        "type": "tone",
        "freq_hz": freq,
        "amplitude": amp,
        "nonmultiple_of_hop": True,
    }


CASES = [
    ("sine_sweep_20hz_8khz", gen_sine_sweep),
    ("multi_tone", gen_multi_tone),
    ("seeded_noise", gen_seeded_noise),
    ("near_silence_clamp", gen_near_silence),
    ("silence", gen_silence),
    ("speech_am_burst", gen_speech_am_burst),
    ("short_clip_nonmultiple", gen_short_clip_nonmultiple),
]


# ---------------------------------------------------------------------------
# Independent f64 mirror of the Rust implementation (self-check vs torch)
# ---------------------------------------------------------------------------


def build_filterbank_mirror(
    n_freq_bins: int, n_mels: int, sr: int, f_min: float, f_max: float
) -> np.ndarray:
    """Mirror of the HF slaney filterbank the Rust port implements."""

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        out = 3.0 * freq / 200.0
        log_region = freq >= 1000.0
        out[log_region] = 15.0 + np.log(freq[log_region] / 1000.0) * (
            27.0 / np.log(6.4)
        )
        return out

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        out = 200.0 * mels / 3.0
        log_region = mels >= 15.0
        out[log_region] = 1000.0 * np.exp(
            (np.log(6.4) / 27.0) * (mels[log_region] - 15.0)
        )
        return out

    mel_pts = np.linspace(
        hz_to_mel(np.array([f_min]))[0], hz_to_mel(np.array([f_max]))[0], n_mels + 2
    )
    filter_freqs = mel_to_hz(mel_pts)
    fft_freqs = np.linspace(0.0, float(sr // 2), n_freq_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
    fb = fb * enorm[None, :]
    return fb


def mirror_mel(wav_f32: np.ndarray, config: dict, filterbank: np.ndarray) -> np.ndarray:
    """float64 mirror of the planned Rust data flow (torch-layout output).

    Deliberately matches the Rust implementation step by step: f64 STFT,
    power cast to f32 (as torch produces f32 magnitudes), filterbank cast to
    f32 (as the extractor casts its float64 filterbank), f64 accumulation,
    f64 log chain, final f32 cast.
    """
    n_fft = config["n_fft"]
    hop = config["hop_length"]
    pad = n_fft // 2
    x = wav_f32.astype(np.float64)

    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(n_fft, dtype=np.float64) / n_fft
    )
    window = window.astype(np.float32).astype(np.float64)

    padded = np.pad(x, (pad, pad), mode="reflect")
    total_frames = 1 + len(x) // hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][
        :total_frames
    ]
    frames = (frames * window[None, :]).astype(np.float32).astype(np.float64)

    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    power = (spec.real**2 + spec.imag**2)[:-1]  # drop the last frame
    power = power.astype(np.float32).astype(np.float64)

    fb = filterbank.astype(np.float32).astype(np.float64)
    mel = power @ fb

    mel = np.maximum(mel, config["clamp_min"])
    log_spec = np.log10(mel)
    log_spec = np.maximum(log_spec, log_spec.max() - config["floor_offset"])
    out = (log_spec + config["norm_offset"]) / config["norm_scale"]
    return out.T.astype(np.float32)  # -> [n_mels, frames], torch layout


# ---------------------------------------------------------------------------
# Reference extraction via the actual HF feature extractor
# ---------------------------------------------------------------------------


def torch_reference(wav_f32: np.ndarray, fe) -> np.ndarray:
    """Run the real extractor core: [n_mels, frames] float32."""
    out = fe._torch_extract_fbank_features(wav_f32)
    if out.ndim != 2 or out.shape[0] != fe.feature_size:
        raise RuntimeError(f"unexpected extractor output shape {out.shape}")
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def full_call_crosscheck(wav_f32: np.ndarray, fe) -> float:
    """Cross-check the core output against the full padded __call__ path.

    The wrapper pads to 480000 samples (30 s) and computes over the padded
    signal; interior frames must agree with the unpadded core output up to
    the final frame (which touches the raw-signal reflect boundary).
    """
    batch = fe(wav_f32, padding="max_length", return_tensors="np")
    padded_mel = batch["input_features"][0]  # [n_mels, 3000]
    raw = fe._torch_extract_fbank_features(wav_f32)
    comparable = min(raw.shape[1] - 1, padded_mel.shape[1])  # skip boundary frame
    diff = np.abs(raw[:, :comparable] - padded_mel[:, :comparable]).max()
    return float(diff)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_f32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def write_f64(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype="<f8").tobytes())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-crosscheck", action="store_true")
    args = parser.parse_args()

    import torch  # imported late so --help works without the ML stack
    import transformers

    from transformers import AutoProcessor

    out_dir = args.out
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    fe = processor.feature_extractor

    # 1) Lock the extracted parameters: the fixture config must match the
    #    live extractor or the fixture set is measuring the wrong thing.
    live = {
        "sampling_rate": fe.sampling_rate,
        "n_mels": fe.feature_size,
        "n_fft": fe.n_fft,
        "hop_length": fe.hop_length,
    }
    for key, value in live.items():
        if MEL_CONFIG[key] != value:
            print(
                f"FATAL: extractor param {key}={value} != fixture config {MEL_CONFIG[key]}"
            )
            return 1

    # 2) Filterbank parity: independent slaney construction vs HF numpy.
    fb_mirror = build_filterbank_mirror(
        1 + fe.n_fft // 2, fe.feature_size, fe.sampling_rate, 0.0, 8000.0
    )
    fb_diff = float(np.abs(fb_mirror - fe.mel_filters).max())
    if fb_diff > 1e-9:
        print(f"FATAL: filterbank mirror mismatch {fb_diff}")
        return 1
    print(f"filterbank mirror vs HF: max abs diff {fb_diff:.3e}")

    # 3) Per-case fixtures.
    manifest_cases = []
    mirror_diffs = {}
    total_bytes = 0
    for name, generator in CASES:
        wav64, recipe = generator()
        wav32 = wav64.astype(np.float32)
        reference = torch_reference(wav32, fe)
        mirror = mirror_mel(wav32, MEL_CONFIG, fe.mel_filters)
        if mirror.shape != reference.shape:
            print(
                f"FATAL {name}: mirror shape {mirror.shape} != reference {reference.shape}"
            )
            return 1
        diff = float(
            np.abs(mirror.astype(np.float64) - reference.astype(np.float64)).max()
        )
        mirror_diffs[name] = diff
        if diff > MIRROR_TOLERANCE:
            print(
                f"FATAL {name}: mirror vs torch diff {diff:.3e} > {MIRROR_TOLERANCE:.0e}"
            )
            return 1

        case_dir = cases_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = case_dir / "input.f32"
        output_path = case_dir / "output.f32"
        meta_path = case_dir / "meta.json"
        write_f32(input_path, wav32)
        write_f32(output_path, reference)

        frames = reference.shape[1]
        expected_frames = wav32.size // fe.hop_length
        if frames != expected_frames:
            print(f"FATAL {name}: frames {frames} != floor(N/hop) {expected_frames}")
            return 1

        clamped = float((reference <= -1.499999).mean())
        meta = {
            "name": name,
            "samples": int(wav32.size),
            "frames": int(frames),
            "n_mels": int(reference.shape[0]),
            "layout": "row_major_[n_mels,frames]_f32le",
            "config": MEL_CONFIG,
            "generator": recipe,
            "fraction_below_clamp_floor": clamped,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        manifest_cases.append(
            {
                "name": name,
                "input": str(input_path.relative_to(out_dir)),
                "output": str(output_path.relative_to(out_dir)),
                "meta": str(meta_path.relative_to(out_dir)),
                "samples": int(wav32.size),
                "frames": int(frames),
                "sha256": {
                    "input.f32": sha256_file(input_path),
                    "output.f32": sha256_file(output_path),
                },
            }
        )
        total_bytes += input_path.stat().st_size + output_path.stat().st_size
        print(
            f"{name:28s} samples={wav32.size:6d} frames={frames:5d} "
            f"mirror_diff={diff:.3e} clamped={clamped:.3f}"
        )

    filters_path = out_dir / "mel_filters.f64"
    write_f64(filters_path, fe.mel_filters)
    total_bytes += filters_path.stat().st_size

    manifest = {
        "version": 1,
        "generator": "scripts/export_mel_reference.py",
        "checkpoint": "Qwen/Qwen3-ASR-1.7B-hf",
        "feature_extractor_type": fe.__class__.__name__,
        "generated_with": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "config": MEL_CONFIG,
        "reference_chain": [
            "f32 waveform @ 16 kHz",
            "torch.stft(n_fft=400, hop=160, hann periodic, center=reflect(200), normalized=False)",
            "1 + floor(N/hop) frames -> drop last -> floor(N/hop)",
            "power = |stft|^2 (no normalization)",
            "mel_filters.T @ power (slaney scale+norm, 201 -> 128, 0..8000 Hz)",
            "clamp(min=1e-10).log10()",
            "floor at global max - 8.0",
            "(x + 4.0) / 4.0",
        ],
        "wrapper_padding_note": (
            "min_length=8000 zero pad, max_length=480000 pad/truncate, and mel-axis "
            "pad to a multiple of 2*n_window=100 are caller-side padding concerns "
            "(model graph), not part of these fixtures."
        ),
        "tolerance": {
            "abs": 1e-2,
            "rel": 1e-3,
            "note": "element-wise pass if abs <= 1e-2 or rel <= 1e-3; observed mirror bound recorded below",
        },
        "mirror_vs_torch_max_abs_diff": mirror_diffs,
        "filterbank_mirror_max_abs_diff": fb_diff,
        "cases": manifest_cases,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.skip_crosscheck:
        wav64, _ = gen_multi_tone()
        crosscheck = full_call_crosscheck(wav64.astype(np.float32), fe)
        print(
            f"full __call__ cross-check (interior frames): max abs diff {crosscheck:.3e}"
        )
        if crosscheck > 1e-5:
            print("FATAL: core chain disagrees with the full padded extractor path")
            return 1

    print(f"\nwrote {len(manifest_cases)} cases + filterbank to {out_dir}")
    print(
        f"fixture bytes (inputs+outputs+filters): {total_bytes} ({total_bytes / 1e6:.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
