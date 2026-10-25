# aimd native engine

Rust workspace for the hand-written inference engine (spec
`specs/002-native-engine.md`, design in
`docs/native-reimplementation-design.md`). Exactly two models:
Qwen3-ASR-1.7B and Unlimited-OCR, consuming `mlx-community` 4-bit (and ASR
8-bit retry) safetensors checkpoints.

## Layout

```text
crates/
├── aimd-tensor/    dtype/shape/CPU tensor + CPU ops (rmsnorm, softmax, rope, swiglu, attention)
├── aimd-weights/   safetensors mmap + MLX affine 4/8-bit unpack/dequant (golden-parity gated)
└── aimd-engine/    tokenizer, audio/mel front-end, model graphs, generation loops
```

`aimd-native` (PyO3 cdylib exposing `aimd._native`) is added at binding time
(WS-10); GPU kernel crates (`aimd-metal`, `aimd-cuda`) land with their
phases.

## Commands

```bash
cd native
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --all
```

Parallel agents build with a private target dir to avoid lock contention:
`CARGO_TARGET_DIR=native/target-<agent> cargo test -p <crate>`.

## Golden fixtures

`scripts/export_quant_reference.py` and `scripts/export_mel_reference.py`
(WS-02 / WS-05) write small reference tensors under `native/golden/`. The
fixtures are committed so CI can run parity tests without multi-GB
checkpoints; checkpoint-dependent tests skip when the HF cache is absent.
