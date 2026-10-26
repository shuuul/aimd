---
id: "003"
title: "Linux on the mlx CUDA backend"
status: Draft
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 003 — Linux on the mlx CUDA backend

## Context

Spec 002 shipped the native engine on macOS only: `aimd-mlx` links brew
`mlx` (Metal), the Homebrew tap bottles `aimd` for macOS arm64, and CI has
no Linux job (`.github/workflows/ci.yml` TODO). Its WS-09 (Linux via the mlx
CUDA backend) never started and moved here when spec 002 closed
(`specs/archive/002-native-engine.md`, 2026-10-26 decision). Repository
evidence:

- `crates/aimd-mlx/build.rs` links `libmlx.dylib` from the brew prefix
  (`MLX_PREFIX` override) and assumes macOS.
- Homebrew's `mlx` formula is bottled for macOS only and Homebrew has no
  CUDA toolchain, so Linux needs a tap formula for a C++-only mlx built with
  the CUDA backend.
- Test hardware: metacube (Arch Linux, RTX 5090) and dspark1/dspark2
  (DGX Spark, aarch64 Ubuntu).
- The golden-corpus gate (`scripts/build_corpus.py`,
  `scripts/corpus_baseline.py`, `corpus_gate`) is macOS-only (`say`,
  Chrome), but its built corpus directory is portable.

## Goal and success criteria

`brew install shuuul/tap/aimd` works on Linux x86_64 with an NVIDIA GPU, with
behavior and accuracy unchanged vs macOS.

- [ ] A tap formula builds C++-only mlx with the CUDA backend and bottles it
      for Linux x86_64.
- [ ] `aimd-mlx` builds and links on Linux against that formula; the
      hermetic workspace tests pass there.
- [x] ASR and OCR pass the golden-corpus gate on metacube against the
      mlx-audio / mlx-vlm baseline (built on macOS, copied over).
      Evidence: 2026-10-26 entry below; all five checkpoints identical.
- [x] ASR >= 10x realtime and healthy OCR pages <= 15 s on metacube;
      numbers recorded in `docs/performance.md`. Evidence: 112x realtime
      on a 60 s clip, 0.72 s per dense page.
- [ ] Linux `aimd` bottle in the tap; a Linux CI job (self-hosted runner)
      runs the Rust gates.

## Scope and non-goals

In scope:

- Tap mlx-cuda formula, OS-conditional `aimd-mlx` build, Linux device
  preflight (`BackendUnavailable` without a CUDA GPU).
- Linux bottles for `aimd` and the self-hosted CI runner.

Not in scope:

- Hand-written CUDA or CUTLASS kernels (cancelled in spec 002); the WS-08
  Metal kernels stay macOS-only, with mlx ops as the Linux path.
- CPU compute paths, other GPU vendors, and aarch64 Linux bottles until
  x86_64 works.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | Spec opened as Draft from spec 002 WS-09; the mlx-cuda formula strategy and the CI runner host are open decisions | Spec 002 closed on macOS; Linux work needs its own contract | WS-01 |
| 2026-10-26 | The WS-08 custom kernels gate on `aimd_mlx::metal_kernels_available()` (GPU device and `metal::is_available()`), not `!device_is_cpu()`; CUDA runs plain mlx ops | `device_is_cpu` is false on CUDA, so the old gate dispatched Metal kernels there | WS-01 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Linux on metacube/dspark via the `mlx` CUDA backend: tap formula for a C++-only mlx built with CUDA, `aimd-mlx` build.rs made OS-conditional (Linux `.so`, `MLX_PREFIX`), ASR + OCR corpus gate against the Python reference, Linux bottle build + self-hosted CI runner (from spec 002 WS-09) | shuuul | In progress | None | Corpus gate PASS on metacube; `brew install` of the tap formulas + Linux bottle works; perf rows in docs/performance.md |

## Verification

- `cargo xtask check-specs` after any spec change.
- On metacube: `cargo test --workspace`, then
  `cargo run --release -p aimd-engine --example corpus_gate -- <corpus>`
  with a corpus directory and `baseline.json` built on macOS.
- `brew install shuuul/tap/aimd` on a clean Linux host pours the bottle and
  the three binaries run.

## Documentation sync

- Durable product/developer docs: `README.md` install section,
  `docs/performance.md` Linux rows.
- Nearest local `AGENTS.md`: distribution, CI, and MLX device sections.
- Parent/package guidance: tap formulas in `shuuul/homebrew-tap`.
- Root guidance and roadmap: this spec until closeout.

### Experiment refs

- (to record before archiving)

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-26 — shuuul/spec opened — WS-01

- Changed: spec created from spec 002 WS-09 at its closeout.
- Evidence: none yet.
- Remaining: everything in WS-01.
- Blockers: none.
- Next action: decide the mlx-cuda formula strategy, then make
  `aimd-mlx` build on metacube.

### 2026-10-26 — shuuul/metacube bring-up — WS-01

- Changed: `aimd-mlx` build.rs links `libmlx.so` + libstdc++ on Linux;
  new shim `mlx_metal_kernels_available` gates the fused MoE, decode SDPA
  and router kernels (macOS outputs unchanged: corpus 72/72, 5/5).
- Evidence (metacube, RTX 5090, CUDA 13.4, pacman `cudnn` 9.27 / `nccl`
  2.32, `mlx-cuda-13` 0.32.4 wheel as `MLX_PREFIX`):
  - Hermetic `AIMD_MLX_DEVICE=cpu cargo test --workspace`: 0 failures.
  - Checkpoint tier on CUDA (`-p aimd-mlx -p aimd-engine`): 74/76 pass.
  - Corpus gate: ASR 1.7B-4bit / 1.7B-8bit / 0.6B-6bit 72/72 identical
    (en WER 0.87/0.44/1.31%, zh CER 3.83/3.55/3.40%); OCR 4bit / 8bit
    similarity 1.0, stop 5/5, 5/5 pages identical.
  - Perf: ASR 112x realtime (60 s clip, 0.54 s), OCR 0.72 s per dense
    page; rows in `docs/performance.md`.
- Open findings:
  - `vision_tower_stages_golden` and `vision_final_features_golden` fail
    on CUDA: drift grows through the SAM/CLIP blocks (block0 0.3%, final
    4.6-5.3% rel rms vs the 3% bound). Gradual, consistent with cuBLAS
    bf16 accumulation order; end-to-end markdown is identical. Needs a
    decision: per-backend bounds vs CUDA goldens.
  - Dependents of `aimd-mlx` get no rpath (cargo applies `rustc-link-arg`
    only to the crate's own targets); macOS works through the dylib's
    absolute install name. Dev runs need `LD_LIBRARY_PATH`; the tap
    formula should place `libmlx.so` in the brew lib dir.
  - mlx's CUDA JIT needs `CUDA_HOME` at runtime (formula env / wrapper).
- Remaining: vision-bound decision, tap mlx-cuda formula, Linux bottle,
  self-hosted CI runner, dspark (aarch64).
- Blockers: none.
- Next action: decide the vision parity bounds, then the formula.

### 2026-10-26 — shuuul/vision parity investigation — WS-01

- Changed: nothing in the tree (experiments reverted).
- Evidence:
  - Op probe at vision shapes (bf16 vs a float64 reference, same seeded
    inputs on both backends): matmul, conv2d, layer_norm, GELU and
    precise softmax sit at the bf16 rounding floor on CUDA and Metal
    alike; CUDA cuDNN SDPA is slightly above it (2.1-2.3e-3 vs Metal
    1.66e-3).
  - Replacing SDPA with explicit f32 attention does not fix the CUDA
    failures, and the same patch on Metal fails both tests by the same
    margin (final features 4.0-4.2% vs the 3% bound): the vision tower
    amplifies any rounding change, so these two tests detect "not the
    Metal mlx rounding", not inaccuracy.
  - mlx v0.32.4 CUDA source: bf16 GEMM (`CUBLAS_COMPUTE_32F`), conv,
    SDPA, norms and erf accumulate in fp32; TF32 applies only to fp32.
    No mlx issue reports a CUDA bf16 precision bug of this kind; mlx's own
    suite loosened CUDA tolerances (ml-explore/mlx#2815).
- Remaining: choose how the two vision stage tests treat non-Metal
  backends (backend-specific bounds, or rely on the corpus gate there).
- Blockers: none.
- Next action: record that decision, then the formula.

## Completion summary

(to complete before archiving)
