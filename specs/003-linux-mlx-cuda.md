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
- [ ] ASR and OCR pass the golden-corpus gate on metacube against the
      mlx-audio / mlx-vlm baseline (built on macOS, copied over).
- [ ] ASR >= 10x realtime and healthy OCR pages <= 15 s on metacube;
      numbers recorded in `docs/performance.md`.
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

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Linux on metacube/dspark via the `mlx` CUDA backend: tap formula for a C++-only mlx built with CUDA, `aimd-mlx` build.rs made OS-conditional (Linux `.so`, `MLX_PREFIX`), ASR + OCR corpus gate against the Python reference, Linux bottle build + self-hosted CI runner (from spec 002 WS-09) | Unassigned | Pending | None | Corpus gate PASS on metacube; `brew install` of the tap formulas + Linux bottle works; perf rows in docs/performance.md |

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

## Completion summary

(to complete before archiving)
