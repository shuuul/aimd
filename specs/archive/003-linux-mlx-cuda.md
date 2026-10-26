---
id: "003"
title: "Linux on the mlx CUDA backend"
status: Completed
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 003 — Linux on the mlx CUDA backend

## Context

Spec 002 shipped the native engine on macOS only: `aimd-mlx` links brew
`mlx` (Metal), the Homebrew tap bottles `aimd` for macOS arm64, and CI has
no Linux job. Its WS-09 (Linux via the mlx
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

- [x] A tap formula provides C++ mlx with the CUDA backend for Linux
      (x86_64 and aarch64). Changed by decision: `mlx-cuda` repackages
      MLX's official `mlx-cuda-13` wheel instead of compiling mlx, and is
      not bottled. Evidence: installs, `brew test` and
      `brew audit --strict` pass on dspark1.
- [x] `aimd-mlx` builds and links on Linux against that formula; the
      hermetic workspace tests pass there. Evidence: `brew install --HEAD
      shuuul/tap/aimd` builds in 2 min on dspark1 and `brew test` passes;
      hermetic tests 0 failures and the checkpoint tier 76/76 against brew
      `mlx-cuda`.
- [x] ASR and OCR pass the golden-corpus gate on metacube against the
      mlx-audio / mlx-vlm baseline (built on macOS, copied over).
      Evidence: 2026-10-26 entry below; all five checkpoints identical.
- [x] ASR >= 10x realtime and healthy OCR pages <= 15 s on metacube;
      numbers recorded in `docs/performance.md`. Evidence: 112x realtime
      on a 60 s clip, 0.72 s per dense page.
- [x] Linux `aimd` install from the tap. Changed by decision: Linux builds
      from source, no bottle (and no Linux CI job: CI only packages).
      Evidence: the dspark1 `--HEAD` install above; the stable formula
      gains Linux with the first release after this spec (the v1.0.0
      tarball predates the Linux `build.rs`).

## Scope and non-goals

In scope:

- Tap mlx-cuda formula, OS-conditional `aimd-mlx` build, Linux device
  preflight (`BackendUnavailable` without a CUDA GPU).
- Linux install of `aimd` from the tap.

Not in scope:

- Hand-written CUDA or CUTLASS kernels (cancelled in spec 002); the WS-08
  Metal kernels stay macOS-only, with mlx ops as the Linux path.
- CPU compute paths and other GPU vendors.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | Spec opened as Draft from spec 002 WS-09; the mlx-cuda formula strategy and the CI runner host are open decisions | Spec 002 closed on macOS; Linux work needs its own contract | WS-01 |
| 2026-10-26 | `mlx-cuda` repackages MLX's `mlx-cuda-13` wheel (libmlx.so, headers, vendored BLAS/LAPACK; rpath `$ORIGIN`) and relies on the system NVIDIA driver, CUDA 13, cuDNN 9 and NCCL 2; it is not compiled in brew | Homebrew has no CUDA toolchain; the wheel is MLX's own release build, so brew skips an hour-long CUDA compile and matches pip mlx bit for bit | WS-01 |
| 2026-10-26 | No Linux bottles: `aimd` builds from source on Linux (about 2 minutes); the tap's test-bot stays macOS-only | Hosted runners have no NVIDIA driver, so `libmlx.so` cannot load there for `brew test`/linkage, and a self-hosted runner was rejected | WS-01 |
| 2026-10-26 | The shim sets `CUDA_HOME=/opt/cuda` when CUDA_HOME/CUDA_PATH are unset and `/usr/local/cuda` is absent | mlx's CUDA JIT needs the toolkit headers and only looks in `/usr/local/cuda` by default; Arch installs to `/opt/cuda` | WS-01 |
| 2026-10-26 | The vision stage/final-feature parity tests keep their Metal bounds and enforce them only when `metal_kernels_available()`; other GPU backends print the drift without failing, and the corpus gate covers their accuracy | The tower amplifies any rounding change past the bounds (f32 attention on Metal fails the same way), so the bounds test Metal rounding, not accuracy; CUDA corpus output is identical | WS-01 |
| 2026-10-26 | CI only packages (`release.yml`); `ci.yml` and its test jobs are removed, and no self-hosted runner is added. Tests (hermetic, checkpoint, corpus gate) run locally on the Mac and metacube | A self-hosted GPU runner is too much upkeep for this project; the GPU tiers need local weights anyway | WS-01 |
| 2026-10-26 | The WS-08 custom kernels gate on `aimd_mlx::metal_kernels_available()` (GPU device and `metal::is_available()`), not `!device_is_cpu()`; CUDA runs plain mlx ops | `device_is_cpu` is false on CUDA, so the old gate dispatched Metal kernels there | WS-01 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Linux on metacube/dspark via the `mlx` CUDA backend: tap mlx-cuda formula, `aimd-mlx` build.rs made OS-conditional (Linux `.so`, `MLX_PREFIX`), ASR + OCR corpus gate against the Python reference, Linux install from the tap (from spec 002 WS-09) | shuuul | Done | None | Corpus gate PASS on metacube; `brew install` of the tap formulas + Linux bottle works; perf rows in docs/performance.md |

## Verification

- `cargo xtask check-specs` after any spec change.
- On metacube: `cargo test --workspace`, then
  `cargo run --release -p aimd-engine --example corpus_gate -- <corpus>`
  with a corpus directory and `baseline.json` built on macOS.
- `brew install shuuul/tap/aimd` on a Linux host with the NVIDIA stack
  builds from source and `brew test shuuul/tap/aimd` passes.

## Documentation sync

- Durable product/developer docs: `README.md` install section,
  `docs/performance.md` Linux rows.
- Nearest local `AGENTS.md`: distribution, CI, and MLX device sections.
- Parent/package guidance: tap formulas in `shuuul/homebrew-tap`.
- Root guidance and roadmap: this spec until closeout.

### Experiment refs

- None: the work landed as commits on `main`; the op probe and the f32
  attention experiment were reverted and are recorded in the progress log.

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
- Remaining: tap mlx-cuda formula, Linux bottle, dspark (aarch64).
- Blockers: none.
- Next action: the tap mlx-cuda formula.

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


### 2026-10-26 — shuuul/vision bounds — WS-01

- Changed: `mlx_vision.rs` tests enforce the vision parity bounds only on
  Metal (`assert_vision_parity`); non-Metal backends print the drift.
- Evidence: `cargo test -p aimd-engine --lib vision_` passes on the Mac
  (Metal, enforced) and on metacube (CUDA, drift reported, 5 passed).
- Remaining: tap mlx-cuda formula, Linux bottle, dspark (aarch64).
- Blockers: none.
- Next action: the tap mlx-cuda formula.


### 2026-10-26 — shuuul/tap formula and closeout — WS-01

- Changed: tap `Formula/mlx-cuda.rb` (wheel repackage) and `aimd.rb`
  (`on_linux` depends on it); shim `CUDA_HOME` fallback; docs; closeout.
- Evidence:
  - dspark1 (aarch64 GB10, Ubuntu 24.04, Linuxbrew 7.0.10, CUDA 13.4):
    `brew install shuuul/tap/mlx-cuda` and `brew install --HEAD
    shuuul/tap/aimd` (2 min build); `brew test` and `brew audit --strict`
    pass for both; `aimd` resolves `libmlx.so` through brew's rpath and the
    CUDA libraries from ld.so.conf.
  - dspark1 corpus gate against brew `mlx-cuda`: identical metrics to the
    baseline for all five checkpoints (1.7B-4bit 71/72 clips identical,
    the rest 72/72; OCR 5/5 pages). Hermetic 0 failures, checkpoint tier
    76/76. No benchmark (the GPU was shared).
  - metacube: vision tests pass with CUDA_HOME unset (the shim fallback).
  - metacube has no Linuxbrew, so the x86_64 formula path is checked only
    through the same wheel used directly (`MLX_PREFIX`).
- Remaining: none in this spec; the next release makes the stable formula
  installable on Linux.
- Blockers: none.
- Next action: archive.

## Completion summary

aimd runs on Linux with an NVIDIA GPU through the mlx CUDA backend.
`aimd-mlx` links `libmlx.so` on Linux, the custom Metal kernels gate on
`metal_kernels_available()`, and the shim finds Arch's `/opt/cuda`. The
tap's `mlx-cuda` formula repackages MLX's `mlx-cuda-13` wheel, and `aimd`
builds from source on Linux against it. Scope deviations, all recorded in
Decisions: no mlx compile in brew, no Linux bottles, no Linux CI (CI only
packages; tests run locally), and the vision parity bounds are enforced on
Metal only. Verification: corpus gate identical to the mlx-audio / mlx-vlm
baseline on metacube (x86_64, RTX 5090) and dspark1 (aarch64, GB10) for all
five checkpoints; hermetic and checkpoint tiers pass on both; metacube perf
112x realtime ASR and 0.72 s per OCR page. Durable docs: `AGENTS.md`
(distribution, MLX-only, Linux build, backend-specific bounds, CI),
`README.md` install, `docs/performance.md` Linux section, and the tap's
README and formulas.
