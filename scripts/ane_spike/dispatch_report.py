#!/usr/bin/env python
"""ANE spike (WS-12b): per-op compute-device dispatch report.

Loads the compiled Core ML model (``.mlmodelc`` from convert_tower.py) through
``MLComputePlan.load_from_path`` (step 3 of the approved spike design) and
records, per operation, the preferred compute device under
``compute_units=CPU_AND_NE`` — the go gate is >=90% of ops dispatched to ANE.

Outputs (next to the model artifacts):
- ``dispatch_report.txt``  full per-op table (operator, device, cost weight)
- ``dispatch_summary.json`` machine-readable counts + percentages

Usage::

    uv run python scripts/ane_spike/dispatch_report.py \
        [--model /tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlmodelc]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/tmp/ane_spike/qwen3_asr_audio_tower_fp16.mlmodelc")

import coremltools as ct  # noqa: E402
from coremltools.models.compute_device import (  # noqa: E402
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import MLComputePlan  # noqa: E402


def device_name(device) -> str:
    if device is None:
        return "unassigned"
    if isinstance(device, MLNeuralEngineComputeDevice):
        return "ANE"
    if isinstance(device, MLGPUComputeDevice):
        return "GPU"
    if isinstance(device, MLCPUComputeDevice):
        return "CPU"
    return type(device).__name__


def iter_operations(function):
    """Yield (depth, operator) over the main function and nested blocks."""
    stack = [(0, op) for op in function.block.operations]
    while stack:
        depth, op = stack.pop()
        yield depth, op
        for block in op.blocks:
            for nested in block.operations:
                stack.append((depth + 1, nested))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: the model's parent directory",
    )
    args = parser.parse_args()

    mlmodelc = args.model
    if not mlmodelc.is_dir():
        print(f"FATAL: compiled model not found: {mlmodelc}")
        return 1
    out_dir = args.out_dir or mlmodelc.parent

    plan = MLComputePlan.load_from_path(
        str(mlmodelc), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    structure = plan.model_structure
    if structure.program is None:
        print("FATAL: model has no ML Program structure (neuralnetwork?)")
        return 1
    main_fn = structure.program.functions["main"]

    rows: list[dict] = []
    for _depth, op in iter_operations(main_fn):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        preferred = (
            getattr(usage, "preferred_compute_device", None)
            if usage is not None
            else None
        )
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
        weight = float(cost.weight) if cost is not None else None
        rows.append(
            {
                "operator": op.operator_name,
                "device": device_name(preferred),
                "weight": weight,
                "outputs": [f"{v.name}:{v.type}" for v in (op.outputs or [])[:1]],
            }
        )

    total = len(rows)
    per_device = Counter(r["device"] for r in rows)
    per_op = Counter((r["operator"], r["device"]) for r in rows)
    const_ops = per_op.get(("const", "unassigned"), 0)
    executable = total - const_ops
    ane_ops = per_device.get("ANE", 0)

    weights = [(r["device"], r["weight"]) for r in rows if r["weight"] is not None]
    weight_total = sum(w for _, w in weights)
    ane_weight = sum(w for d, w in weights if d == "ANE")

    # `const` nodes materialize weights/symbols — they are not runtime work.
    # The WhisperKit-style ANE coverage % counts executable ops only.
    ane_pct_ops = 100.0 * ane_ops / executable if executable else 0.0
    ane_pct_all = 100.0 * ane_ops / total if total else 0.0
    ane_pct_weight = 100.0 * ane_weight / weight_total if weight_total else None

    lines = []
    lines.append(f"model: {mlmodelc}")
    lines.append(f"compute_units: CPU_AND_NE   coremltools {ct.__version__}")
    lines.append(
        f"total ops in program: {total} ({const_ops} const materializations, {executable} executable)"
    )
    lines.append(
        f"per-device op counts (executable): {dict(Counter(r['device'] for r in rows if r['operator'] != 'const').most_common())}"
    )
    lines.append(f"ANE op % (executable): {ane_pct_ops:.2f}%")
    lines.append(
        f"ANE op % (incl. const): {ane_pct_all:.2f}%   ANE cost-weight %: "
        f"{f'{ane_pct_weight:.2f}%' if ane_pct_weight is not None else 'n/a'}"
    )
    lines.append("")
    lines.append(f"{'count':>6}  {'operator':<28} device")
    for (operator, device), count in sorted(
        per_op.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        lines.append(f"{count:>6}  {operator:<28} {device}")
    lines.append("")
    lines.append("== full per-op table (trace order, weight = est. cost share) ==")
    for i, r in enumerate(rows):
        w = f"{r['weight']:.6f}" if r["weight"] is not None else "   -   "
        lines.append(f"{i:5d}  {r['operator']:<30} {r['device']:<5} {w}")

    report = "\n".join(lines) + "\n"
    (out_dir / "dispatch_report.txt").write_text(report)
    summary = {
        "model": str(mlmodelc),
        "compute_units": "CPU_AND_NE",
        "coremltools": ct.__version__,
        "total_ops": total,
        "const_ops": const_ops,
        "executable_ops": executable,
        "per_device_op_counts": dict(per_device),
        "per_op_device_counts": {f"{o}|{d}": c for (o, d), c in per_op.items()},
        "ane_op_pct_executable": ane_pct_ops,
        "ane_op_pct_incl_const": ane_pct_all,
        "ane_cost_weighted_pct": ane_pct_weight,
    }
    (out_dir / "dispatch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n".join(lines[:9]))
    print(f"\nfull table -> {out_dir / 'dispatch_report.txt'}")
    print(f"summary    -> {out_dir / 'dispatch_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
