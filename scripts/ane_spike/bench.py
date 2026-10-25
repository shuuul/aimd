#!/usr/bin/env python
"""ANE spike (WS-12b): encoder-segment latency + parity benchmark.

Times, on the same deterministic 30 s mel fixture (steps 4 of the approved
spike design):

(a) the compiled Core ML audio tower (warm predictions, compute_units=
    CPU_AND_NE), plus Core ML load timings (compiled-model load vs fresh
    mlpackage load through the system compile cache),
(b) the MLX audio tower (mlx_audio internals, GPU/Metal reference) — the
    production-path backend,
and prints a table against the Rust CPU number produced by
``native/crates/aimd-engine/examples/bench_tower.rs`` (pass it via --rust-ms).

Parity: element-wise Core ML fp16 output vs MLX float32 output
(max-abs / mean-abs / p99.9-abs, abs-covered relative errors), plus the
sum|out| checksum cross-check against the Rust bench output.

Usage::

    uv run python scripts/ane_spike/bench.py \
        [--model /tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlmodelc] \
        [--mlpackage /tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlpackage] \
        [--fixture native/golden/ane_spike/mel.npy] \
        [--rounds 15] [--rust-ms 35535.10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mlx_reference import load_mlx_tower, mlx_tower_forward, mlx_versions  # noqa: E402

DEFAULT_MODEL = Path("/tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlmodelc")
DEFAULT_MLPACKAGE = Path("/tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlpackage")
DEFAULT_FIXTURE = REPO_ROOT / "native" / "golden" / "ane_spike" / "mel.npy"
AUDIO_SECONDS = 30.0
CORE_ML_CACHE_DIR = Path.home() / "Library" / "Caches" / "CaptureAndCoreML"


def parity_stats(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = actual.astype(np.float64)
    r = reference.astype(np.float64)
    diff = np.abs(a - r)
    denom = np.maximum(np.abs(r), 1e-3)  # rel is abs-covered below |ref| < 1e-3
    rel = diff / denom
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p999_abs": float(np.quantile(diff, 0.999)),
        "max_rel_abs_covered": float(rel.max()),
        "p999_rel_abs_covered": float(np.quantile(rel, 0.999)),
    }


def bench_predict(predict_fn, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        predict_fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        predict_fn()
        times.append(time.perf_counter() - t0)
    return times


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:,.1f} ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mlpackage", type=Path, default=DEFAULT_MLPACKAGE)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--rust-ms",
        type=float,
        default=None,
        help="median encoder ms from examples/bench_tower.rs",
    )
    args = parser.parse_args()

    import coremltools as ct

    mel = np.load(args.fixture).astype(np.float16)  # [1, 128, 3000]
    mel_f32 = np.load(args.fixture).astype(np.float32)
    rows: dict[str, dict] = {}
    out: dict[str, dict] = {}

    # ---- (a) Core ML / ANE -------------------------------------------------
    t0 = time.perf_counter()
    compiled = ct.models.CompiledMLModel(
        str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    load_compiled_1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled2 = ct.models.CompiledMLModel(
        str(args.model), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    load_compiled_2 = time.perf_counter() - t0
    del compiled2

    t0 = time.perf_counter()
    pkg_model = ct.models.MLModel(
        str(args.mlpackage), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    load_pkg_1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    pkg_model2 = ct.models.MLModel(
        str(args.mlpackage), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    load_pkg_2 = time.perf_counter() - t0
    del pkg_model, pkg_model2

    print(
        f"Core ML load: compiled-model {fmt_ms(load_compiled_1)} / reload {fmt_ms(load_compiled_2)}"
    )
    print(f"Core ML load: mlpackage {fmt_ms(load_pkg_1)} / reload {fmt_ms(load_pkg_2)}")

    predict = lambda: compiled.predict({"input_features": mel})  # noqa: E731
    times = bench_predict(predict, args.warmup, args.rounds)
    out["coreml"] = np.asarray(
        compiled.predict({"input_features": mel})["tower_output"]
    )
    rows["CoreML tower_output"] = {
        "device": "CPU_AND_NE (ANE-eligible)",
        "times": times,
    }
    print(
        f"CoreML predict: median {fmt_ms(statistics.median(times))} over {args.rounds} rounds"
    )

    if CORE_ML_CACHE_DIR.exists():
        cache_bytes = sum(
            p.stat().st_size for p in CORE_ML_CACHE_DIR.rglob("*") if p.is_file()
        )
        print(
            f"system Core ML compile cache: {CORE_ML_CACHE_DIR} ({cache_bytes / 1e6:.1f} MB)"
        )

    # ---- (b) MLX / GPU reference -------------------------------------------
    mlx_tower = load_mlx_tower()
    import mlx.core as mx

    def mlx_forward():
        out = mlx_tower(
            mx.array(mel_f32), mx.array(np.ones((1, mel_f32.shape[-1]), dtype=np.int32))
        )
        mx.eval(out)
        mx.synchronize()

    for _ in range(2):
        mlx_forward()
    times = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        mlx_forward()
        times.append(time.perf_counter() - t0)
    out["mlx"] = mlx_tower_forward(mlx_tower, mel_f32)
    rows["MLX AudioEncoder"] = {"device": "GPU (Metal)", "times": times}
    print(
        f"MLX predict: median {fmt_ms(statistics.median(times))} over {args.rounds} rounds"
    )

    # ---- parity Core ML vs MLX ---------------------------------------------
    stats = parity_stats(out["coreml"], out["mlx"])
    print(f"parity CoreML(fp16) vs MLX(f32): {json.dumps(stats)}")

    checksum = {
        "coreml_sum_abs": float(np.abs(out["coreml"].astype(np.float32)).sum()),
        "mlx_sum_abs": float(np.abs(out["mlx"]).sum()),
    }
    print(f"checksums: {json.dumps(checksum)}")

    # ---- table --------------------------------------------------------------
    def summarize(times: list[float]) -> dict:
        med = statistics.median(times)
        return {
            "median_ms": med * 1e3,
            "mean_ms": statistics.fmean(times) * 1e3,
            "min_ms": min(times) * 1e3,
            "rtf": med / AUDIO_SECONDS,
        }

    table = {name: {**meta, **summarize(meta["times"])} for name, meta in rows.items()}
    print()
    print(
        f"{'backend':<22}{'device':<28}{'median':>12}{'mean':>12}{'min':>12}{'x-realtime':>12}"
    )
    for name, r in table.items():
        print(
            f"{name:<22}{r['device']:<28}{r['median_ms']:>10,.1f}ms{r['mean_ms']:>10,.1f}ms"
            f"{r['min_ms']:>10,.1f}ms{1.0 / r['rtf']:>10.1f}x"
        )
    if args.rust_ms is not None:
        rust = {"median_ms": args.rust_ms, "rtf": args.rust_ms / 1e3 / AUDIO_SECONDS}
        print(
            f"{'Rust AudioTower':<22}{'CPU (native)':<28}{args.rust_ms:>10,.1f}ms"
            f"{'':>12}{'':>12}{1.0 / rust['rtf']:>10.1f}x"
        )

    coreml_med = table["CoreML tower_output"]["median_ms"]
    mlx_med = table["MLX AudioEncoder"]["median_ms"]
    print()
    print(
        f"speedup vs Rust/CPU: {args.rust_ms / coreml_med:.2f}x"
        if args.rust_ms
        else "speedup vs Rust: (pass --rust-ms)"
    )
    print(f"speedup vs MLX/GPU:  {mlx_med / coreml_med:.2f}x")

    result = {
        "fixture": str(args.fixture),
        "rounds": args.rounds,
        "loads": {
            "compiled_model_load_s": load_compiled_1,
            "compiled_model_reload_s": load_compiled_2,
            "mlpackage_load_s": load_pkg_1,
            "mlpackage_reload_s": load_pkg_2,
        },
        "table": table,
        "parity_coreml_vs_mlx": stats,
        "checksums": checksum,
        "versions": {
            "coremltools": ct.__version__,
            **mlx_versions(),
            "numpy": np.__version__,
        },
    }
    result_path = args.model.parent / "bench_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nresults -> {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
