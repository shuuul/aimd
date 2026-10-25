---
id: "002"
title: "Native Rust inference engine reimplementation"
status: Active
created: 2026-10-25
updated: 2026-10-25
coordinator: "shuuul"
---

# 002 — Native Rust inference engine reimplementation

## Context

aimd currently runs local inference through `mlx-audio`/`mlx-vlm` (Apple
Silicon) and `transformers`+`torch` (CUDA/MPS), wrapped behind MarkItDown
plugins. This dependency stack is multi-GB, carries upstream version gates
(`mlx-vlm>=0.6.4`, `transformers>=5.14.1`), and forces behavior-critical logic
(gundam cropping, sliding-window no-repeat n-gram, 4-bit repetition retry) to
live in adapter layers that break on upstream changes. Repository evidence:

- `docs/performance.md` records MLX baselines: Qwen3-ASR-1.7B-4bit ~0.39s for
  7.24s Chinese audio (~18x realtime, ~2.5 GB peak); Unlimited-OCR healthy
  pages ~6-11s on mlx-vlm, looping to ~80s/page before the n-gram fix.
- Local checkpoints pin the exact architectures: Qwen3-ASR-1.7B (128-mel
  front-end, conv downsample, 24-layer windowed encoder, Qwen3 dense decoder
  28L/GQA 16-8) and Unlimited-OCR (SAM ViT-B vision tower, linear projector,
  DeepSeek-V2-lite MoE decoder: 64 routed experts top-6 + 2 shared,
  sliding_window 128). Both ship MLX 4-bit affine group=64 quantization.
- `src/aimd/plugins/*/models/remote.py` already provides an
  OpenAI-compatible HTTP boundary used as a comparison baseline.

Design discussion (2026-10-25, `docs/native-reimplementation-design.md`)
settled all open decisions; this spec tracks execution.

## Goal and success criteria

Replace the local ML inference stack with a hand-written Rust engine exposed
to the retained Python interface layer as a `aimd._native` extension module,
supporting exactly two models on CPU, Metal (M2 Pro), and CUDA (metacube).

- [ ] MLX 4-bit affine (g=64) safetensors loader reproduces MLX-exported
      reference tensors bit-exactly in CI.
- [ ] ASR: on a labeled golden corpus, CER/WER degrades <= 1% absolute vs the
      current MLX backend; 4-bit repetition-loop 8-bit retry still works.
- [ ] OCR: text similarity >= MLX baseline - 2% on the golden corpus; healthy
      pages end with `finish=stop` at a rate no lower than today.
- [ ] Perf gates: ASR >= 5x realtime (CPU), >= MLX baseline on M2 Pro (Metal),
      >= 10x realtime on metacube (CUDA); OCR healthy pages <= 30s (CPU),
      <= 15s (CUDA), MLX-comparable (Metal). Results recorded in
      `docs/performance.md`.
- [ ] `aimd-tool` wheels publish to PyPI with the native engine embedded
      (macOS arm64 CPU+Metal, manylinux CPU, CUDA cubin/PTX embedded);
      interfaces (CLI/API/MCP) unchanged behaviorally.
- [ ] `mlx-audio`, `mlx-vlm`, `transformers`, `torch` removed from default
      installs after a one-minor-version deprecation window.

## Scope and non-goals

In scope:

- Rust engine under `native/`: Cargo workspace with crates `aimd-tensor`,
  `aimd-weights`, `aimd-engine` (later `aimd-native` PyO3 cdylib).
- Hand-written CPU ops (NEON/AVX), Metal MSL kernels (tuned for M2 Pro),
  CUDA .cu kernels (+ CUTLASS shim) with cubin/PTX embedding.
- `qwen3_asr` and `unlimited_ocr` model graphs, generation loops, and
  behavior ports (chat templates, gundam preprocessing, n-gram guard, markdown
  normalization, 4-bit->8-bit retry).
- PyO3 binding `aimd._native` with a minimal byte/string-level API.
- Weight loading via safetensors mmap + hf-hub download of the existing
  `mlx-community` 4-bit checkpoints (8-bit ASR checkpoint for retry).
- Python-side `native` backend in `capabilities.py`, made default; old
  backends demoted then removed.
- Golden-reference exporters under `scripts/` (Python) plus small committed
  fixtures under `native/golden/` for CI parity tests.

Not in scope:

- Any model beyond Qwen3-ASR-1.7B and Unlimited-OCR (GLM-OCR removed).
- Training, fine-tuning, or re-quantization infrastructure.
- Rewriting the Python interface layer (CLI/API/MCP) or the URL/doc pipelines.
- A generic engine for arbitrary architectures.
- Per-model tuning for Apple GPUs other than M2 Pro (conservative fallback
  parameters are acceptable).

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-25 | Language: Rust | PyO3/maturin for PyPI wheels; cudarc + metal-rs for GPU work; official Rust `tokenizers` removes tokenizer-parity risk | All |
| 2026-10-25 | Python interface layer retained; engine ships as `aimd._native` extension | Interfaces are thin and tested; the ML dependency stack is the actual pain; avoids the largest rewrite risk | WS-10 |
| 2026-10-25 | Model matrix narrowed to Qwen3-ASR-1.7B + Unlimited-OCR; GLM-OCR cut | Narrowing is a goal of the reimplementation | All |
| 2026-10-25 | ASR before OCR | ASR decoder is the shared operator base; retires the biggest risks on the simpler model | WS-06, WS-07 |
| 2026-10-25 | CUDA: hand-written .cu + CUTLASS for dense/grouped GEMM; no torch | Own kernels on quantized hot paths; CUTLASS templates for big GEMMs; runtime depends only on the NVIDIA driver | WS-09 |
| 2026-10-25 | Metal: hand-written MSL tuned for the local M2 Pro (19-core GPU, 32 GB, Metal 4); no wgpu | Local dev loop and daily-driver platform; function constants give other Apple Silicon conservative fallbacks | WS-08 |
| 2026-10-25 | CUDA validation on metacube machines | Available test hardware for interactive tuning and perf recording | WS-09 |
| 2026-10-25 | Consume mlx-community 4-bit checkpoints directly (plus 8-bit ASR retry checkpoint) | Zero re-quantization infra; direct numeric comparison vs MLX; reuses user caches | WS-02 |
| 2026-10-25 | Phase order: CPU -> Metal -> CUDA | CPU is the correctness baseline; Metal iterates locally; CUDA follows on metacube | All |
| 2026-10-25 | Workspace layout `native/crates/{aimd-tensor,aimd-weights,aimd-engine}`; `aimd-native` (PyO3) added at binding time | Crate boundaries double as parallel-worker ownership boundaries; shared deps declared once in the workspace root | WS-01 |
| 2026-10-25 | Wave-1 parallel dispatch: WS-02/03/04/05 run concurrently as subagents on disjoint paths; workers do not edit specs, workspace Cargo.toml, or other crates; integration and spec updates are coordinator-only; per-agent `CARGO_TARGET_DIR` avoids build-lock contention | Enables safe parallelism without worktrees for wave 1; later waves that touch shared Python files will use worktrees | WS-02, WS-03, WS-04, WS-05 |
| 2026-10-25 | Parity fixtures exported by Python scripts under `scripts/` into `native/golden/` (small slices, committed); model-checkpoint tests skip gracefully when caches are absent | CI must run without multi-GB checkpoints; golden slices keep parity checks hermetic | WS-02, WS-05 |
| 2026-10-25 | Dual-wheel packaging: default `aimd-tool` wheel ships only the hand-written CPU engine (~10-20 MB); `aimd-tool[mlx]` extra installs an `aimd-tool-mlx` variant built `--features mlx` linking MLX C++ (runtime measured: libmlx.dylib 21 MB + mlx.metallib 174 MB, ~+60-90 MB compressed) | Measured sizes: hand-written kernels keep the default CLI small; MLX reuse removes the highest-risk MSL tuning from the critical path; Mac perf parity with the MLX baseline holds by construction | WS-08, WS-10, WS-11 |
| 2026-10-25 | Mac performance path: near-term MLX via a C-ABI C++ shim crate (pattern identical to the CUTLASS shim); hand-written MSL kernels demoted to an optional late phase that replaces/retires the MLX variant | MLX being a C++ library does not force a C++ engine; Rust constraints (PyO3/maturin, tokenizers, cudarc, memory safety) are unchanged; MSL tuning (4-8 weeks, highest risk) leaves the critical path | WS-08, WS-11 |
| 2026-10-25 | CUDA: hand-written .cu + CUTLASS remains the primary path on metacube; `mlx[cuda]` (headers + libjaccl present in the installed MLX build) serves as emergency fallback and perf comparison | Explicit project requirement to own the CUDA kernels; upstream MLX CUDA backend is too young to depend on but useful as a cross-check | WS-09 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Native workspace scaffold: `native/` Cargo workspace, three crates with stub modules and documented API contracts, workspace deps pre-wired, `cargo test --workspace` green | shuuul | Done | None | `cd native && cargo test --workspace` |
| WS-02 | `crates/aimd-weights`: safetensors mmap, MLX 4-bit affine (g=64) unpack + dequant, 8-bit path, `scripts/export_quant_reference.py` + committed golden fixtures | agent-weights | Done | WS-01 | `cargo test -p aimd-weights` golden parity (bit-exact f16 vs MLX exports) |
| WS-03 | `crates/aimd-tensor`: dtype/shape/CPU tensor, rmsnorm, softmax, rope (theta, window semantics), swiglu, GQA attention (f32/f16) with naive-reference unit tests | agent-tensor | Done | WS-01 | `cargo test -p aimd-tensor` |
| WS-04 | `crates/aimd-engine/src/tokenizer`: `tokenizers` wrapper loading both checkpoints' `tokenizer.json`, encode/decode roundtrips (CJK-heavy), ASR chat-template builders (language prefix, context system message), OCR `document parsing.` prompt builder | agent-tokenizer + shuuul | Done | WS-01 | `cargo test -p aimd-engine --lib tokenizer` incl. checkpoint roundtrips (skip w/o cache) |
| WS-05 | `crates/aimd-engine/src/audio`: wav load (f32 mono 16k), 128-bin mel matching the Qwen3-ASR feature extractor, `scripts/export_mel_reference.py` + committed golden fixtures | agent-audio | Done | WS-01 | `cargo test -p aimd-engine --lib audio` golden parity (tolerance vs exported reference) |
| WS-06 | ASR model graph + generation loop on CPU (windowed encoder n_window=50/infer=800, Qwen3 decoder, greedy, retry-on-loop) | Unassigned | Pending | WS-02, WS-03, WS-04, WS-05 | Golden-corpus CER/WER gate + CPU perf gate |
| WS-07 | OCR model graph + generation loop on CPU (gundam preprocessing, SAM ViT-B, MoE decoder, n-gram guard, markdown normalization) | Unassigned | Pending | WS-06 | Golden-corpus similarity/finish=stop gate + CPU perf gate |
| WS-08 | Hand-written Metal MSL kernels (optional late phase: replace/retire the MLX variant; M2 Pro tuning targets unchanged) | Unassigned | Pending | WS-11 | Small default wheel with Metal perf on this machine |
| WS-09 | CUDA .cu kernels + CUTLASS shim (ASR then OCR) on metacube; `mlx[cuda]` as fallback/comparison | Unassigned | Pending | WS-06, WS-07 | CUDA perf gate + numeric parity harness |
| WS-10 | `aimd-native` PyO3 binding, Python `native` backend default, dual-wheel matrix (default CPU + `[mlx]`/CUDA variants), old-backend deprecation + removal | Unassigned | Pending | WS-06, WS-07 | Wheel smoke installs (default small, variants functional); behavior tests on retained interfaces |
| WS-11 | `aimd-mlx` C-ABI shim over `mlx::core` (array lifetime, `quantized_matmul`, GEMM/attention composition); model graphs on the MLX backend; `--features mlx` variant build | Unassigned | Pending | WS-06, WS-07 | Same-machine perf >= MLX baseline by construction; variant wheel builds |

## Verification

- Spec lifecycle: `uv run python scripts/check_specs.py` and
  `uv run pytest -q tests/test_check_specs.py` after any spec change.
- Workspace: `cd native && cargo test --workspace` and
  `cargo clippy --workspace --all-targets` must stay green after every
  integration pass.
- Operator parity: CI-visible tests comparing each op against MLX/NumPy
  reference tensors (golden set exported by `scripts/export_*_reference.py`
  into `native/golden/`; atol per dtype; checkpoint-dependent tests skip when
  HF caches are absent).
- Golden corpus: labeled ASR set (CER/WER) and OCR page set (text similarity,
  `finish=stop` rate) run against both the MLX backend and `aimd._native`;
  deltas must satisfy the success criteria.
- Perf recording: warm/cold runs per backend recorded in
  `docs/performance.md` with hardware/runtime rows, including M2 Pro (Metal)
  and metacube (CUDA) results.
- Packaging: install built wheels into clean venvs on macOS arm64 and
  manylinux; exercise `aimd audio.mp3`, `aimd scan.pdf`, `aimd-api`,
  `aimd-mcp` smoke paths.

## Documentation sync

- Durable product/developer docs: `docs/native-reimplementation-design.md`
  (design), `docs/performance.md` (new baseline rows), `CHANGELOG.md`.
- Nearest local `AGENTS.md`: replace MLX/Transformers backend sections with
  the native engine contract (models, precisions, devices, weight sources,
  kernel boundaries) at retirement time; add `native/` layout and cargo
  commands when WS-01 lands.
- Parent/package guidance: `pyproject.toml` entry points and extras updated
  when `aimd._native` lands; remove `mlx-audio`/`mlx-vlm`/`transformers`
  dependency groups in the retirement change.
- Root guidance and roadmap: keep the model matrix and phase status here
  until closeout, then collapse into `AGENTS.md`.

### Experiment refs

- (to record before archiving)

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-25 — shuuul/spec kickoff — WS-01

- Changed: design discussion completed and frozen in
  `docs/native-reimplementation-design.md` (v0.3); spec 002 created and
  indexed as Active.
- Evidence: model structures verified against local checkpoint configs;
  baselines cited from `docs/performance.md`.
- Remaining: WS-01 kickoff (workspace + loader parity harness).
- Blockers: none.
- Next action: scaffold the Rust workspace and export MLX reference tensors
  for the 4-bit affine layout.

### 2026-10-25 — shuuul/dispatch — WS-02, WS-03, WS-04, WS-05

- Changed: workstreams re-grained to dispatch size; wave-1 concurrency rules
  recorded under Decisions; WS-02..WS-05 claimed by subagents on disjoint
  paths (`crates/aimd-weights`, `crates/aimd-tensor`,
  `crates/aimd-engine/src/tokenizer`, `crates/aimd-engine/src/audio`,
  plus per-agent reference-exporter scripts).
- Evidence: coordinator performs all spec writes and integration; workers
  self-verify with `cargo test -p <crate>` under per-agent target dirs.
- Remaining: wave-1 integration pass (workspace-wide tests, fmt/clippy,
  spec status updates) once agents report.
- Blockers: none.
- Next action: wave-1 integration; then WS-06 dispatch.

### 2026-10-25 — shuuul/design revision v0.4 — WS-08, WS-10, WS-11

- Changed: mid-dispatch design discussion resolved three more decisions
  (dual-wheel packaging, MLX C++ shim for the Mac performance path, CUDA
  with `mlx[cuda]` fallback); design doc bumped to v0.4; WS-11 added,
  WS-08 re-scoped to the optional MSL replacement phase, WS-10 updated for
  the dual-wheel matrix.
- Evidence: on-machine measurements (mlx runtime = libmlx.dylib 21 MB +
  mlx.metallib 174 MB; current ML stack ~835 MB installed; `mlx/ops.h`
  exposes `quantized_matmul`; `mlx/backend/cuda/` + `libjaccl.dylib`
  present in the installed build).
- Remaining: wave-1 unaffected (CPU foundation + parity tooling needed by
  every backend); WS-11/WS-10 planning after WS-06.
- Blockers: none.
- Next action: complete wave-1 integration.

### 2026-10-25 — shuuul/wave-1 integration — WS-01, WS-02, WS-03, WS-04, WS-05

- Changed: all five wave-1 workstreams landed. WS-02 (agent-weights, glm-5.3-flash
  subagent): safetensors mmap store + MLX affine 4/8-bit dequant, 25 golden
  fixtures (528 KB), 15/15 tests. WS-03 (agent-tensor): CPU op layer (rmsnorm,
  softmax+mask, rope NeoX convention, swiglu, blocked matmul, GQA decode +
  causal prefill attention, KV cache with window views), 51/51 tests; fixed
  stub dtype size bug (F16/BF16 4->2 bytes). WS-04 (agent-tokenizer fixtures +
  exporter; tokenizer agent went inactive twice, Rust side completed by
  coordinator): exact ASR/OCR chat-template builders golden-locked (9 cases),
  6 tests. WS-05 (agent-audio): RIFF parser + mel front-end (n_fft=400
  four-step FFT, slaney filterbank, log chain), 7 fixtures, 25 tests,
  observed tolerance 4.7e-5 abs.
- Evidence: `cargo test --workspace` = 97 passed / 0 failed (includes
  bit-exact checkpoint integration tests against the local HF cache);
  `cargo fmt --all -- --check` clean; `cargo clippy --workspace
  --all-targets` zero findings.
- Critical knowledge landed in code docs: MLX affine packing (U32 payload
  LSB-first nibbles, f16-or-bf16 scales/biases per 64-group, sign-agnostic
  scale convention with max-as-bias); Qwen3-ASR mel parameter table
  (16000/128/400/160, periodic Hann, center+reflect, slaney/slaney,
  clamp-log10-floor-max-8, (x+4)/4); ASR/OCR template strings golden-locked.
- Remaining: WS-06 dispatch (ASR model graph + generation on CPU); CI wiring
  for cargo tests (success criterion requires parity checks in CI).
- Blockers: none. Note: two subagent dispatches went inactive/timeout and
  one hit the concurrency limit; retry + coordinator completion worked, but
  wave-2 dispatches should be batched <= 2 at a time.
- Next action: WS-06 (ASR graph: windowed encoder + Qwen3 decoder + greedy
  + retry), consuming the WS-02/03/04/05 surfaces.

### 2026-10-25 — shuuul/WS-06 dispatch + gate incident — WS-06

- Changed: qwen3_asr shared skeleton landed (mod.rs contracts + config.rs
  typed checkpoint config, verified against the cached 4-bit config);
  WS-06a (encoder port + per-stage activation parity) and WS-06b (Qwen3
  decoder over 4-bit weights + greedy/retry/parse generation) dispatched as
  parallel background subagents on disjoint files; `native.yml` CI workflow
  added (fmt/clippy/test on ubuntu + macos-14, hermetic golden fixtures).
- Evidence: checkpoint survey — audio tower is UNQUANTIZED bf16 under
  `audio_tower.*` (Whisper-style biased LayerNorm/GELU/windowed attn),
  decoder 4-bit under `model.*`; mlx-audio
  `stt/models/qwen3_asr/qwen3_asr.py` is the porting reference.
- Remaining: encoder/decoder integration pipeline (mel -> encoder ->
  decoder -> generate, coordinator-owned); commit of the wave-1 + WS-06
  work is staged but blocked by the mimosa commit gate (22 high "SSRF"
  findings verified as mock-object false positives in untouched
  tests/test_url.py; awaiting a triage decision — plugin allowlist, test
  edits, or deferral).
- Blockers: commit gate only; engineering unblocked.
- Next action: integrate WS-06a/06b on completion; run workspace gate.

## Completion summary

(to complete before archiving)
