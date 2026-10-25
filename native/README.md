# aimd native engine

Rust workspace for the hand-written inference engine (spec
`specs/002-native-engine.md`, design in
`docs/native-reimplementation-design.md`). Exactly two models:
Qwen3-ASR-1.7B and Unlimited-OCR, consuming `mlx-community` 4-bit (and ASR
8-bit retry) safetensors checkpoints.

## Layout

```text
crates/
├── aimd-weights/   safetensors mmap + MLX affine 4/8-bit unpack; `dequantize_affine` is the
│                   bit-exact reference gated by golden/quant (tests only, not on the runtime path)
├── aimd-mlx/       C-ABI shim over MLX C++ (Metal; MLX CPU device when no GPU or AIMD_MLX_DEVICE=cpu)
└── aimd-engine/    tokenizer, audio/mel front-end, MLX model graphs (Qwen3-ASR, Unlimited-OCR), generation loops
```

MLX is the only compute backend (specs/002-native-engine.md, "MLX-ONLY
engine"). Requires brew `mlx` (or `MLX_PREFIX`).

## Commands

```bash
cd native
cargo test --workspace                       # Metal when available
AIMD_MLX_DEVICE=cpu cargo test --workspace   # MLX CPU device (CI / no-GPU hosts)
cargo clippy --workspace --all-targets
cargo fmt --all
```

Use a private target dir to avoid lock contention between builds:
`CARGO_TARGET_DIR=native/target-<name> cargo test -p <crate>`.

## Golden fixtures

`scripts/export_*_reference.py` write small reference tensors under
`native/golden/` (each file <= 1 MB; larger dumps are regenerated, not
committed). The
fixtures are committed so CI can run parity tests without multi-GB
checkpoints; checkpoint-dependent tests skip when the HF cache is absent.
