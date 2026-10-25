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
- [ ] Perf gates (v0.5): ASR >= MLX baseline on M2 Pro via the MLX variant
      and >= 10x realtime on metacube (CUDA); OCR healthy pages <= 15s
      (CUDA), MLX-comparable (MLX variant). The CPU path has no perf gate —
      it is the correctness baseline and small-wheel fallback. Results
      recorded in `docs/performance.md`.
- [ ] `aimd` ships as a Homebrew formula in our own tap with bottles for
      macOS arm64 (Metal) and Linux (mlx CUDA backend); `brew install`
      gives working CLI/API/MCP binaries with behavior unchanged vs the
      Python interfaces. The PyPI package gets one final release pointing
      users to brew.
- [ ] `mlx-audio`, `mlx-vlm`, `transformers`, `torch` removed from default
      installs after a one-minor-version deprecation window.

## Scope and non-goals

In scope:

- Rust engine under `crates/`: Cargo workspace with crates `aimd-tensor`,
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
  fixtures under each crate's `tests/golden/` for CI parity tests.

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
| 2026-10-25 | CPU path demoted to correctness baseline + small-wheel fallback: no perf gate, no NEON sprint | First benchmark (docs/performance.md 2026-10-25): CPU load at parity with MLX (10.85s vs 10.01s) and token-identical outputs, but warm inference ~0.1x realtime vs MLX 19.9x on Metal — optimization effort is better spent on the MLX/Metal route | WS-02, WS-03 |
| 2026-10-25 | MLX/Metal route is the primary Mac performance path: WS-11 promoted to the next construction wave (ASR first); brew `mlx` 0.32.4 (bottled, same 0.32 line as the wheel) is the shim link target | User direction after the CPU benchmark; brew bottle removes the source-build risk | WS-11 |
| 2026-10-25 | ANE exploration: feasibility spike only (WS-12). ANE has no public API (Core ML is the only path, fp16 unquantized); candidate sweet spot is encoder-on-ANE + decoder-on-Metal; go/no-go follows the report | User direction; unknowns (stateful LLM conversion for Qwen3-ASR, weight size at fp16, Rust/Core ML integration) must be resolved before committing | WS-12 |
| 2026-10-25 | CPU "obvious wins only" optimization pass, quality bar = unit tests + parity tolerances (no full gate): vocab-axis rayon parallelism for small-m matmul (decode steps were serial under row-panel parallelism), transposed-B blocked logits matmul (no 600 MB embedding transpose, F32 out for argmax faithfulness), SIMD inner kernels (stable NEON f32x4 FMA on aarch64 — fp16 load intrinsics are nightly-only; fully vectorized AVX2+F16C+FMA on x86_64; scalar fallback; runtime detection) | User direction: strengthen MLX/Metal as primary, but land the obvious CPU wins in passing; kernel reordering stays within existing golden tolerances (verified: decoder golden logits parity + 16-token argmax stream green) | WS-03 |
| 2026-10-25 | Inference-perf roadmap for the MLX path (user-approved): (1) preallocated/ring KV cache replacing concatenate-grown caches (fixes hidden O(n^2)); (2) `mlx::core::compile` exposed through the shim and applied to the decoder layer/step functions (MLX's CUDA-Graphs analog: automatic kernel fusion + replay; C++ API verified in brew mlx 0.32.4 `mlx/compile.h`); (3) n-gram/prompt-lookup speculative decoding at the generation-loop level (model-free draft+verify, largest expected win on OCR's repetitive markdown); (4) hand-written MSL kernels remain the optional late phase (WS-08) | Post-WS-11b profiling logic: ~350 small kernel dispatches per decode token are launch-overhead bound; ordered by ROI; #1 is bug-class, #2 est 1.3-2x decode, #3 est 1.5-2.5x OCR decode, #4 highest ceiling but duplicates MLX value | WS-13, WS-14, WS-08 |
| 2026-10-25 | Full-Rust interface layer (WS-15): flips the v0.4 "retain Python interfaces" decision — CLI/doc/URL/API/MCP all port to Rust; the PyPI package reduces to a compatibility shim (PyO3 `process()` + console scripts onto native binaries). Sequencing: CLI -> doc pipeline -> URL pipeline -> HTTP API -> MCP -> test/release cleanup (~2-3 months) | Engine side is now fully proven in Rust (ASR done on both backends); marginal cost of the port dropped; CLI cold start becomes ms-scale; default wheel loses the last Python deps | WS-15, WS-10 |
| 2026-10-25 | yt-dlp becomes a managed CLI dependency with an auto-update interface: `yt-dlp --version` floor check, `aimd deps update` (runs `yt-dlp -U`, or `brew upgrade` when brew-installed), stale-detection (version age + known update-required stderr signatures) triggering one auto-update+retry, version surfaced in `aimd doctor` | yt-dlp requires frequent updates to keep working (site changes); the lib->CLI switch makes update management ours | WS-15 |
| 2026-10-25 | SUPERSEDED same day by the managed-binary decision below (no install-method probing). | | WS-15 |
| 2026-10-25 | SUPERSEDED by the Homebrew-distribution decision below for the tool-management half (brew `depends_on` replaces the manifest/downloader/`aimd deps`; the `dom_smoothie` readability choice still stands). Original: all subprocess tools become aimd-MANAGED BINARIES (user direction "just install the binaries"): a pinned tool manifest (url template + version + sha256) covers yt-dlp (official standalone), ffmpeg (static builds), pandoc (official single-file); on-demand download into ~/.cache/aimd/bin keeps the default wheel small; resolution order = per-tool env override > managed copy > system PATH; `aimd deps install/update/status` (update = GitHub latest-release check, confirmed pin bump, atomic replace); downloader hand-written with security gates (https-only, host allowlist github.com/api.github.com/objects.githubusercontent.com, resolve-then-reject loopback/private/reserved IPs, sha256 verify before atomic rename, no shell exec). yt-dlp self-heal = stderr signature -> `aimd deps update yt-dlp` -> one retry. Readability extraction uses the third-party crate `dom_smoothie` (readability.py lineage, actively maintained) — hand-writing was rejected as genuinely complex (user correction: too many details); `npx defuddle` stays as an env-var escape hatch during migration, validated against a fixture corpus; `defuddle-rs` (0.1.0, same algorithm as the current tool) is watched as the future exact-parity swap | Official standalone binaries are first-class distribution for these tools; no user-environment probing, controlled versions; bundling in-wheel would violate the small-wheel goal; readability is a genuinely complex domain (per the dependency policy -> third-party), unlike the hand-written items | WS-15 |
| 2026-10-25 | Dependency policy for WS-15: third-party crates only for genuinely complex domains — keep axum+tokio (async HTTP/SSE), rmcp (approved), serde/serde_json, pdfium-render, image, rustls (remote-backend TLS), miniz_oxide (deflate). Hand-write instead: CLI arg parsing (~200 lines, frozen flag surface), zip directory reading (store + miniz inflate), OPF/spine XML walking, SSE framing, Netscape cookie parsing, sidecar path checks | User direction: simple dependencies are worse than writing it ourselves; each hand-written item is small, dependency-light, and fully testable | WS-15 |
| 2026-10-25 | Nested Qwen cleanup LLM for ASR post-processing: DEFERRED (user decision "先不做"). Research conclusion recorded for reopening: tokenizer + Qwen3 graph/MLX-backend infrastructure ARE reusable (any Qwen3 dense checkpoint plugs in), but embedding WEIGHT tables must not be shared across models (fine-tuned distribution drift); context biasing already covers external-knowledge corrections (never feed first-pass output back as context — it amplifies errors); a cleanup LLM fills the no-external-clue gap and needs conservative guardrails (copy-mode prompting + edit-distance-ratio rejection) | Revisit after WS-15 if corpus evidence shows a residual error gap | — |
| 2026-10-25 | MLX-ONLY engine (user direction "只需要 MLX 版本，跟原来的 Python 版本对标"): supersedes the CPU-baseline, dual-wheel, and hand-written CUDA decisions; hand-written MSL (WS-08) stays as an active macOS optimization track, re-scoped to custom Metal kernels inside the MLX graph rather than a separate backend (user correction). The single compute backend is the `aimd-mlx` shim — Metal on macOS, `mlx` CUDA backend on Linux (metacube). All CPU compute paths (aimd-tensor ops, ASR CPU encoder/decoder, OCR CPU vision/decoder, CPU benches) are deleted; backend-agnostic pieces stay (weights loading, tokenizer, mel, gundam preprocessing, n-gram guard, markdown normalization, retry/parse logic). Correctness reference = the Python mlx-audio / mlx_vlm outputs (golden fixtures + E2E markdown/transcript identity), not a Rust CPU graph | Simplify the codebase; the CPU path was ~0.01x of MLX speed (OCR page 734 s vs 4.8 s) and only served as a parity baseline that the Python reference already provides | WS-03, WS-06, WS-07, WS-08, WS-09, WS-10, WS-11 |
| 2026-10-25 | Pure Homebrew distribution (user decision): the repo stops being a Python package once WS-15 lands. `aimd` ships as a Homebrew formula in our own tap (`shuuul/homebrew-tap`, not homebrew-core yet) with CI-built bottles for macOS arm64 and Linux; runtime deps via `depends_on` (`yt-dlp`, `ffmpeg`, `pandoc`; mlx on macOS). Linux GPU: brew mlx is macOS-only-bottled and Homebrew has no CUDA toolchain, so the tap carries its own C++-only mlx formula built with the CUDA backend, bottled on metacube/dspark (to be proven in WS-09). WS-10 (PyO3/wheels/PyPI) is cancelled; until WS-15 completes the Python package stays as-is (no new features, no Python-version changes), then gets one final PyPI release pointing to brew. Model weights still download at runtime from HF into the cache; API/MCP servers can run under `brew services` | Removes PyO3, the wheel matrix, release-please's Python flow, and the hand-written tool downloader; brew already distributes every subprocess tool | WS-09, WS-10, WS-15 |
| 2026-10-25 | Rust-style repo layout (user direction, replaces the queued `native/` -> `crates/` rename): the Cargo workspace root moves to the repo root (`Cargo.toml`/`Cargo.lock`), crates live in root `crates/` (`aimd-weights`, `aimd-mlx`, `aimd-engine`; WS-15 binaries join as `crates/aimd-cli` etc.), golden fixtures move from `native/golden/` into each owning crate's `tests/golden/` (located via `CARGO_MANIFEST_DIR`), `scripts/` exporters write there, `target/` at root is gitignored; the Python package (`src/aimd`, `pyproject.toml`, `tests/`) coexists until WS-15 removes it. Done with `git mv` in its own commit after the MLX-only cleanup lands | Standard Rust workspace shape for the brew-distributed binary; per-crate fixtures follow crate ownership | WS-13, WS-15 |
| 2026-10-25 | WS-15 runs before WS-13/WS-14 (user direction: move the interfaces to Rust first, then delete Python and rewrite the README). WS-15 becomes the umbrella/closeout row; the port splits into WS-16 (core + `aimd` CLI + local/remote ASR/OCR), WS-17 (doc), WS-18 (URL), WS-19 (HTTP API), WS-20 (MCP), WS-21 (Homebrew tap + release CI). Crates: `aimd-core` (models, errors, hooks, text shaping; no heavy deps), `aimd-doc` and `aimd-url` (depend only on `aimd-core`; URL audio fallback goes through a transcriber trait implemented by `aimd-process`), `aimd-process` (routing/orchestration over doc/url/engine), and one binary crate per interface (`aimd-cli` -> `aimd`, then `aimd-api`, `aimd-mcp`) | Removes the Python dependency before perf work; the core/doc/url split avoids crate cycles and lets WS-16/17/18 run in parallel | WS-13, WS-14, WS-15..WS-21 |
| 2026-10-25 | Rust document/PDF tool mapping: Pandoc (>= 3.8, brew) reads docx/pptx/xlsx/odt/epub/html/rtf/etc., replacing MarkItDown built-ins; legacy binary `.doc`/`.ppt`/`.xls` are dropped (no Pandoc reader); `.txt`/`.md` pass through. Text-layer PDFs and the scanned/encoding-issue OCR probe use the `pdf-inspector` crate (the same library behind the Python binding); scanned-page rendering for OCR shells out to brew `mupdf` (`mutool draw`, replacing PyMuPDF). pdfium is dropped (no brew formula) | Keeps the formula on brew-distributed tools; pdf-inspector keeps routing identical to Python | WS-16, WS-17 |
| 2026-10-26 | GLM-OCR option removed from the Rust crates entirely (user direction "直接移除 GLM-OCR 的选项"): `glm-ocr`, its aliases and HF ids follow the unknown-model error path; the frozen Python package keeps it until WS-15 deletes Python | Narrow model matrix; no dead option in help text | WS-16 |
| 2026-10-26 | WS-15 closeout (user direction: "按你的默认方案做 WS-15 收尾" + "release-please 都可以去掉"): the Python package, pytest suite, `pyproject.toml`/`uv.lock` and `scripts/check_specs.py` are deleted; the spec validator is `cargo xtask check-specs` (byte-identical output); reference exporters stay as standalone PEP 723 `uv run` scripts with pinned deps (lint via root `ruff.toml`); release-please is removed: version lives in `[workspace.package]` starting at 1.0.0, a pushed `vX.Y.Z` tag drives `release.yml` (version check, build + smoke, `gh release create --generate-notes`, tap PR); `CHANGELOG.md` is frozen history | Single toolchain; no Python packaging or release bot to keep in sync | WS-15, WS-21 |
| 2026-10-26 | WS-13 `mlx::core::compile` scope narrowed to fused elementwise ops (compiled SwiGLU in the shim, used by the ASR MLP and the OCR dense/shared/`SwitchGLU` MLPs); no whole-layer/step compile. The OCR `KvCache` (step-256 + ring) moves to `aimd_engine::models::kv_cache` and replaces the ASR concatenate cache; ASR decode adopts the OCR/mlx-lm `async_eval` pipelining. Bench baseline is `main` 8021879 (b96c031 predates the crates move), A/B interleaved | The KV cache grows every step, so a shape-keyed compile retraces per token and a shapeless one cannot express slice bounds (mlx-lm does not compile the step either); profiling shows ~0.4 ms host graph build vs ~8 ms GPU+encode per ASR token, so remaining headroom is WS-14/WS-08 | WS-13, WS-14, WS-08 |
| 2026-10-25 | MLX-only cleanup: `aimd-tensor` crate deleted; ANE spike artifacts (`golden/ane_spike`, `scripts/ane_spike`) deleted (report `docs/ane-spike-results.md` kept as history); `aimd_weights::dequantize_affine` kept as the test-only bit-exact oracle for `golden/quant`; `golden/asr_decoder/logits.f32` dropped (unread). MLX device = GPU unless `AIMD_MLX_DEVICE=cpu` or no GPU; hermetic CI runs on the CPU device and the checkpoint tier skips there; MLX's CPU `quantized_matmul` with bf16 inputs loses precision as K grows (vs float64: 1-2% of max|out| at K=128, 6.5-17% at K=4096; f16/dense/f32 <= 0.2%; Metal bf16 < 0.8%; reproduced in Python mlx 0.32.2), so CPU-device parity tests promote bf16 cases to f32 inputs and keep the Metal tolerance; the CPU device must never run real bf16-scale checkpoints | Hosted macOS runners may lack Metal; keep CI hermetic and inside the TEST RULES budget (CPU 4 s, Metal 12 s) | WS-02, WS-03, WS-12 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Native workspace scaffold: root Cargo workspace (`crates/`), three crates with stub modules and documented API contracts, workspace deps pre-wired, `cargo test --workspace` green | shuuul | Done | None | `cargo test --workspace` |
| WS-02 | `crates/aimd-weights`: safetensors mmap, MLX 4-bit affine (g=64) unpack + dequant, 8-bit path, `scripts/export_quant_reference.py` + committed golden fixtures | agent-weights | Done | WS-01 | `cargo test -p aimd-weights` golden parity (bit-exact f16 vs MLX exports) |
| WS-03 | `crates/aimd-tensor`: dtype/shape/CPU tensor, rmsnorm, softmax, rope (theta, window semantics), swiglu, GQA attention (f32/f16) with naive-reference unit tests | agent-tensor | Done | WS-01 | `cargo test -p aimd-tensor` |
| WS-04 | `crates/aimd-engine/src/tokenizer`: `tokenizers` wrapper loading both checkpoints' `tokenizer.json`, encode/decode roundtrips (CJK-heavy), ASR chat-template builders (language prefix, context system message), OCR `document parsing.` prompt builder | agent-tokenizer + shuuul | Done | WS-01 | `cargo test -p aimd-engine --lib tokenizer` incl. checkpoint roundtrips (skip w/o cache) |
| WS-05 | `crates/aimd-engine/src/audio`: wav load (f32 mono 16k), 128-bin mel matching the Qwen3-ASR feature extractor, `scripts/export_mel_reference.py` + committed golden fixtures | agent-audio | Done | WS-01 | `cargo test -p aimd-engine --lib audio` golden parity (tolerance vs exported reference) |
| WS-06 | ASR model graph + generation loop on CPU (windowed encoder n_window=50/infer=800, Qwen3 decoder, greedy, retry-on-loop) | agent-encoder + agent-decoder + shuuul | Done | WS-02, WS-03, WS-04, WS-05 | Commits 2cc53a9/8de3841: token-identical outputs vs MLX, golden argmax stream exact, E2E green. Open tail: labeled golden-corpus CER harness (two-clip identity is evidence, not the gate) |
| WS-07 | OCR model graph + generation loop on the MLX backend (gundam preprocessing, SAM ViT-B + CLIP + projector, MoE decoder with ring sliding-window KV, n-gram guard, markdown normalization); CPU prototype retired per the MLX-only decision | agent-ocr-vision + agent-ocr-decoder + agent-ocr-integration + agent-ocr-mlx | Done | WS-11 | Vision + prefill bit-exact vs mlx_vlm goldens; E2E markdown identical (similarity 1.0, finish=stop, equal token counts) on both `crates/aimd-engine/tests/golden/ocr_e2e` fixtures; latency at parity with mlx_vlm (0.83 s / 4.7 s vs 0.88 s / 4.6 s). Open tail: labeled multi-page OCR corpus gate |
| WS-08 | Hand-written Metal MSL kernels for macOS hot spots, plugged into the MLX graph (`mlx::core::fast::metal_kernel` custom kernels through the shim, e.g. fused MoE dispatch, fused attention/ring-KV update, quantized GEMV variants tuned for M2 Pro); chosen by profiling after WS-13/WS-14, kept only where they beat MLX's built-ins | Unassigned | Pending | WS-13 | Per-kernel A/B vs the MLX built-in with outputs unchanged; end-to-end decode gain recorded in docs/performance.md |
| WS-09 | Linux on metacube/dspark via the `mlx` CUDA backend: tap formula for a C++-only mlx built with CUDA, `aimd-mlx` build.rs made OS-conditional (Linux `.so`, `MLX_PREFIX`), ASR + OCR E2E against the Python reference, Linux bottle build + self-hosted CI runner; hand-written .cu/CUTLASS cancelled | Unassigned | Pending | WS-07 | ASR/OCR E2E parity on metacube; `brew install` of the tap mlx formula + Linux bottle works; perf rows in docs/performance.md |
| WS-10 | PyO3 binding / wheels / PyPI — CANCELLED by the pure Homebrew distribution decision (no work done) | shuuul | Done | None | Cancellation recorded in Decisions (2026-10-25 pure Homebrew) |
| WS-11 | `aimd-mlx` C-ABI shim over `mlx::core` (array lifetime, `quantized_matmul`, GEMM/attention composition); model graphs on the MLX backend; `--features mlx` variant build. Promoted to the active wave (ASR first); brew mlx 0.32.4 link target | agent-mlx + agent-mlxgraph + shuuul | Done | WS-06 | ASR done (commit b96c031: 0.92-0.93x mlx-audio latency, 24x faster load, token-identical outputs); OCR graph rides WS-07 completion |
| WS-13 | MLX perf pack: preallocated/ring KV cache (replace concatenate growth) + `mlx::core::compile` through the shim applied to decoder layer/step functions + A/B benchmarks recorded in docs/performance.md | shuuul | Done | WS-07 | Decode-latency improvement vs b96c031 baseline with outputs still text-identical; bench rows recorded |
| WS-14 | Speculative decoding (n-gram/prompt-lookup draft + single-forward verify k tokens) in the greedy generation loops; enabled per-model after accuracy passes the golden-corpus gates | Unassigned | Pending | WS-13 | OCR decode >= 1.3x vs WS-13 with corpus similarity unchanged; ASR measured opportunistically |
| WS-15 | Umbrella + closeout for the full-Rust port and Homebrew release: port the pytest behavior scenarios onto the Rust crates, final PyPI release pointing to brew, delete `src/aimd`/`tests/`/`pyproject.toml`/uv tooling, rewrite README/AGENTS/docs for Homebrew. Per the dependency policy: hand-write the simple parts, third-party for complex domains | shuuul | In progress | WS-16, WS-17, WS-18, WS-19, WS-20, WS-21 | Behavior parity vs the Python interfaces on the ported pytest scenarios; Python tree removed; README documents only the brew install |
| WS-16 | `aimd-core` (text shaping, precision, remote config, extension tables) + `aimd-process` (routing incl. pdf-inspector OCR probe, local Qwen3-ASR over aimd-engine with ffmpeg decode/segmenting/start-end/context/8-bit retry, Unlimited-OCR for images and `mutool`-rendered PDFs, OpenAI-compatible remote ASR/OCR, model alias/precision -> HF checkpoint resolution + download) + `aimd` CLI (hand-written arg parse, frozen flag surface from the Typer CLI) | agent-core | Done | WS-07, WS-11 | Hermetic core 31 / process 31 / cli 11 green on GPU and CPU device; Rust `aimd` output byte-identical to `uv run aimd` on a wav, a png, a scanned PDF (mutool) and a YouTube subtitle URL. Open: URL audio fallback only mocked; 6bit/0.6B/HF download not exercised (no cache) |
| WS-17 | `aimd-doc`: Pandoc conversion, EPUB spine/cleanup/flat image extraction, text-layer PDF via pdf-inspector, txt/md passthrough | agent-doc | Done | None | `cargo test -p aimd-doc` 16 green; AIMD_E2E parity vs the Python converter identical (rich + minimal EPUB Markdown/chapters/images, rst, 15-case cleaner corpus). Open: xlsx not exercised on real Pandoc; Zip64/encrypted EPUB unsupported |
| WS-18 | `aimd-url`: URL detection/platform, yt-dlp subtitles (raw/plain), metadata + ASR metadata context, Netscape/browser cookies with automatic probing, audio download handed to a transcriber trait, `dom_smoothie` readable HTML, `brew upgrade yt-dlp` hint on stale-extractor stderr | agent-url | Done | None | 35 unit + 35 pipeline + 2 golden-parity tests green; real YouTube Markdown identical to Python; dom_smoothie vs defuddle F1 0.88-0.999 on articles (0.32 on an index page) |
| WS-19 | `aimd-api` (axum): `/healthz`, `/readyz`, `/v1/process`, `/v1/jobs` lifecycle + SSE, blobs, sidecar auth/loopback/`AIMD_ALLOWED_ROOTS` | agent-api | Done | WS-16 | `cargo test -p aimd-api` 49 green (all four test_api*.py suites ported); release `/v1/process` on page_text.png byte-identical to `uv run aimd-api`; jobs/SSE/shutdown smoke green |
| WS-20 | `aimd-mcp` (rmcp stdio): `healthz`, `process_input` | agent-mcp | Done | WS-16 | `cargo test -p aimd-mcp` 11 green; tools/list schemas, healthz and the OCR result identical to `uv run aimd-mcp`; error text keeps the domain message (intentional deviation) |
| WS-21 | Homebrew tap formula (`shuuul/homebrew-tap`, `depends_on` mlx/pandoc/yt-dlp/ffmpeg/mupdf) + bottle CI (macOS arm64; Linux after WS-09) + Rust release automation replacing release-please Python flow | agent-tap | In progress | WS-16, WS-17, WS-18, WS-19, WS-20 | `brew install shuuul/tap/aimd` from a bottle works on macOS arm64 |
| WS-12 | ANE feasibility spike: Core ML routes (stateful LLM on ANE, encoder-on-ANE), fp16/weight-size constraints, mlx-or-torch conversion source, Rust integration path, expected gains vs Metal on M2 Pro; go/no-go recommendation in `docs/ane-feasibility.md` | agent-ane + agent-ane-spike + shuuul | Done | None | Report reviewed; NO-GO recorded (commit 8fc1650): gates passed technically, ~4-5% end-to-end gain does not justify the artifact/maintenance cost |

## Verification

- Spec lifecycle: `cargo xtask check-specs` and `cargo test -p xtask`
  after any spec change (ported from the deleted Python validator).
- Workspace: `cargo test --workspace` and
  `cargo clippy --workspace --all-targets` must stay green after every
  integration pass.
- Operator parity: CI-visible tests comparing each op against MLX/NumPy
  reference tensors (golden set exported by `scripts/export_*_reference.py`
  into each owning crate's `tests/golden/`; atol per dtype; checkpoint-dependent tests skip when
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
  kernel boundaries) at retirement time; add `crates/` layout and cargo
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

### 2026-10-25 — agent-decoder/WS-06b complete — WS-06

- Changed: `qwen3_asr/{decoder.rs, generate.rs}` landed: Qwen3 decoder over
  4-bit affine weights (all weights via `dequantize_affine` only),
  audio-pad embedding replacement (`audio_token_id` positions replaced
  in order, `min(#pads, #features)`), prefill/step over the shared KV
  cache, greedy loop with EOS-before-append semantics, repetition-loop
  detector port (regex-free) reporting `loop_detected` for the 4-bit->8-bit
  retry, `parse_qwen_output` port; `scripts/export_asr_decoder_reference.py`
  + golden fixtures under `native/golden/asr_decoder/` (byte-identical rerun).
- Evidence: forced 16-token greedy argmax stream + 5 prefill positions match
  MLX exactly; logits max-abs 1.035 on ~±25 scale documented at tolerance
  1.5 (bf16-reference vs f16-compute store rounding; weights bit-exact,
  both accumulate f32); chunked prefill == full prefill bit-identically;
  8-bit checkpoint load tested. 15 tests (`decoder_`x7, `generate_`x8).
- Checkpoint facts recorded: layernorms and `q_norm`/`k_norm` are stored
  UNQUANTIZED bf16 (MLX quantizes only Linear/Embedding); per-head
  q/k RMSNorm applies before NeoX RoPE.
- Remaining: WS-06a (encoder) still in progress (sibling observed an
  E0277 in encoder.rs under active development); integration pass after it
  lands: workspace gate, mod.rs fmt ordering, pipeline wiring, fold in the
  stale `qwen3_asr.rs` deletion missed by the wave-1 commit (HEAD currently
  has both module files; do not push HEAD).
- Blockers: none (decoder-side).
- Next action: await WS-06a completion; then integrate.

### 2026-10-25 — shuuul/WS-06 integration — WS-06

- Changed: coordinator pipeline landed (`qwen3_asr/pipeline.rs`): wav ->
  mel (transpose to time-major) -> audio tower -> prompt/pads ->
  `greedy_generate` -> `parse_qwen_output`, plus `transcribe_with_retry`
  (4-bit -> 8-bit on loop detection, falls back to primary on retry
  failure). AIMD_E2E=1-gated full-stack tests added. WS-06a encoder
  delivered earlier this session (18 tests; staged parity 7.8e-5 interior /
  8.5e-7 final projection; frames->pads formula locked vs all fixtures).
- Key fix during integration: the mlx-community 4-bit repo ships
  `vocab.json`+`merges.txt` (no `tokenizer.json`) — `Tokenizer::from_checkpoint`
  now reconstructs the byte-level BPE (ByteLevel pretokenizer/decoder +
  `added_tokens_decoder` specials), mirroring mlx-audio's
  `AutoTokenizer` path; equivalence vs the `-hf` tokenizer.json locked by
  test (`tokenizer_vocab_merges_equivalence`, identical ids incl. specials
  and CJK).
- Evidence: release-mode E2E 3/3 green — sine-noise input returns
  `EmptyTranscription` as expected; macOS TTS clip transcribes correctly:
  "Today's weather is great, and context engineering just got simpler."
- Gate + perf recorded: workspace 134/134 tests green (68 engine incl.
  checkpoint integration, 979s debug), fmt clean, clippy zero findings
  after integration fixes; first benchmark numbers in
  `docs/performance.md` (native CPU: load 10.85s at parity with MLX's
  10.01s; warm inference ~0.1x realtime vs MLX 19.9x on Metal;
  transcription text token-identical to MLX on both clips, including the
  same mishearing). `examples/bench_asr.rs` is the reproducible harness.
- Remaining for WS-06 Done: golden-corpus CER/WER comparison harness vs
  the MLX backend (the two-clip identity result is promising evidence,
  not the gate).
- Blockers: none.
- Next action: commit WS-06; corpus harness; then WS-07 (OCR) dispatch
  decision.

### 2026-10-25 — shuuul/v0.5 re-prioritization + wave dispatch — WS-11, WS-12

- Changed: after the first CPU benchmark (load parity 10.85s vs 10.01s,
  token-identical outputs, but ~0.1x realtime warm vs MLX 19.9x), the CPU
  perf route is dropped (no gate, no NEON sprint); MLX/Metal is the
  primary Mac performance path — WS-11 promoted into the active wave
  (ASR first, brew mlx 0.32.4 link target). WS-12 ANE feasibility spike
  dispatched in parallel (Core ML is the only ANE path; fp16 unquantized;
  candidate sweet spot: encoder-on-ANE + decoder-on-Metal).
- Evidence: benchmark rows in `docs/performance.md` (2026-10-25);
  `brew info mlx` shows a bottled 0.32.4 formula on the installed wheel's
  0.32 line.
- Remaining: WS-11a shim bring-up + parity vs golden quant fixtures;
  WS-12 report; WS-06 corpus harness still open.
- Blockers: none.
- Next action: integrate WS-11a/WS-12 on completion; decide WS-11b
  (ASR graph on the MLX backend) and the ANE go/no-go.

### 2026-10-25 — agent-ane/WS-12 report delivered — WS-12

- Changed: `docs/ane-feasibility.md` landed (335 lines, sources cited,
  verified facts vs inference separated). Recommendation: **skip ANE
  (option a)**; encoder-on-ANE (option b) kept as a time-boxed 1-2
  person-week spike ONLY if energy/long-recording prefill becomes a
  product goal.
- Evidence: ANE decode is bandwidth-starved (44-70% of GPU effective
  bandwidth per MacPaw M2-Pro-inclusive measurements; Apple's own Llama
  guide targets GPU for decode) and our workloads are decode-dominated;
  fp16-only weights conflict with MLX 4-bit and the small default wheel;
  MLX->CoreML has no official export path; Apple announced Core AI
  (macOS 27) as Core ML's successor — build-then-freeze risk.
- Remaining: user go/no-go decision on the report.
- Blockers: none.
- Next action: record the ANE decision; WS-11a integration when the shim
  agent reports.

### 2026-10-25 — shuuul/commit 2cc53a9 — WS-06

- Changed: WS-06 + CPU SIMD pass + ANE report + v0.5 decisions committed
  as `2cc53a9` (user-side terminal; mimosa gate still blocks agent
  commits on the pre-existing test_url.py false positives).
- Evidence: committed tree builds (mlx wiring excluded, lands with
  WS-11a); benchmarks in `docs/performance.md` (SIMD pass 1.4-1.6x,
  load parity, token-identical outputs).
- Remaining: WS-11a (in flight), WS-06 corpus harness (open), ANE
  go/no-go user decision.
- Blockers: none.
- Next action: WS-11a integration on completion.

### 2026-10-25 — agent-mlx/WS-11a complete + feature wiring — WS-11

- Changed: `aimd-mlx` shim landed: brew mlx 0.32.4 discovery + cc build
  (C++20) + rpath linking; ~20-symbol C ABI (lifecycle with bit-exact
  byte import/export, matmul, quantized_matmul — bf16 scales accepted
  natively via mlx dtype promotion, no fp16 workaround — transpose,
  reshape, elementwise, softmax, silu composed, native fused rms_norm);
  `MlxArray` RAII wrapper. aimd-engine gained the optional `mlx` feature
  (`--features mlx`, default CPU wheel stays mlx-free).
- Evidence: 17/17 tests; quantized parity over all 25 golden fixtures
  (worst 3.6e-3 relative = bf16 ulp); f16 matmul parity 2.2e-4 relative;
  clean-env from-scratch run without DYLD env. Key finding recorded:
  mlx transpose is a lazy strided view — readbacks must materialize via
  contiguous()+eval() (a SIGSEGV taught us).
- Remaining: WS-11b (ASR graph on the MLX backend) dispatched next;
  mlx wiring still uncommitted (lands with WS-11b integration).
- Blockers: none.
- Next action: WS-11b integration + bench vs mlx-audio.

### 2026-10-25 — shuuul/decisions + WS-12b dispatch — WS-12, WS-03

- Changed: user decisions recorded — (1) ANE: run the time-boxed partial
  spike (encoder-on-ANE per docs/ane-feasibility.md §最小 spike 设计;
  WS-12b dispatched as a background agent: coremltools conversion of the
  audio tower, fixed [1,128,3000] mel bucket, ANE dispatch >=90% gate,
  >=2x encoder speedup go condition; coremltools installed into the
  project venv spike-only); (2) CPU casual-optimization pack approved
  (f16->f32 vectorized bit-trick + f32 cache option).
- Evidence: the bit-trick NEON conversion landed first — stable integer
  intrinsics only, exhaustively verified: all 65536 f16 bit patterns
  bit-exact vs half's scalar conversion (denormals/±0/inf/NaN payloads
  included); tensor suite 54/54. Bench + engine parity re-run in flight.
- Remaining: WS-11b (ASR on MLX) and WS-12b (ANE spike) both in flight;
  bench numbers for the bit-trick to be recorded in
  docs/performance.md.
- Blockers: none.
- Next action: integrate both agents' results; record ANE go/no-go once
  the spike reports.

### 2026-10-25 — shuuul/commit 8de3841 — WS-03

- Changed: CPU casual-optimization pack closed and committed as `8de3841`
  (user-side terminal; bit-trick NEON conversion + exhaustive 65536
  pattern test + benchmark row). Cumulative CPU result: 1.75-1.9x over
  naive (zh 62->33.4 s, en 43->24.6 s), outputs token-identical to MLX.
- Evidence: git log 8de3841; docs/performance.md rows.
- Remaining: WS-11b (MLX backend) and WS-12b (ANE spike) in flight
  (working tree shows their in-progress files: aimd-mlx/, ane_spike/).
- Blockers: none.
- Next action: integrate both on completion.

### 2026-10-25 — shuuul/WS-12 closed: ANE NO-GO — WS-12

- Changed: WS-12b spike delivered (docs/ane-spike-results.md, 219 lines,
  full reproduction commands); ANE decision recorded as NO-GO and WS-12
  closed. scripts/ane_spike/** + native/golden/ane_spike/** +
  examples/bench_tower.rs staged for commit.
- Evidence vs the pre-agreed gates: dispatch 100% (1027/1027 executable
  ops on ANE) — PASS; encoder latency 63.9-65.3 ms vs MLX Metal
  127.6-139.4 ms on the 30 s mel fixture = 2.00-2.14x — straddles the 2x
  gate; parity SNR 21.4 dB vs MLX (ANE fp16-accumulation profile,
  conversion itself lossless at 52.6 dB on GPU) — below the §5.7-level
  gate; compile cache stable only via pre-compiled .mlmodelc. Deciding
  product math: the encoder is ~8% of the MLX 18x end-to-end baseline, so
  even a perfect ANE encoder saves ~4-5% end-to-end (measured), against a
  0.62 GB fp16 artifact + conversion-shim maintenance + Apple's Core AI
  toolchain churn. NO-GO per the combined criteria; skip-ANE
  recommendation from docs/ane-feasibility.md stands.
- By-product data point: the Rust CPU AudioTower needs 35.5 s for the
  30 s fixture (0.8x realtime) vs MLX 128 ms — on the CPU fallback path,
  LONG audio is tower-dominated; acceptable for a fallback, worth
  remembering if CPU ever needs a tower pass.
- Also fixed: root .gitignore's bare `examples/` rule was ignoring
  native/crates/*/examples/** — bench_asr.rs (referenced by commits
  2cc53a9/8de3841) had never actually been committed; rule scoped to
  `/examples/` and the examples staged.
- Cleanup note: coremltools 9.0 was installed into .venv spike-only
  (`uv pip install`); remove with `uv sync --dev --upgrade`.
- Remaining: WS-11b (MLX backend) still in flight.
- Blockers: none.
- Next action: WS-11b integration on completion.

### 2026-10-25 — agent-mlxgraph/WS-11b complete — WS-11

- Changed: `MlxAsrPipeline` landed behind `--features mlx` (mlx_backend.rs,
  1587 lines, 11 tests): full-MLX architecture (conv2d + fast::sdpa +
  fast::rope/layer_norm all present in mlx 0.32.4 C++ headers — no hybrid
  fallback), 4-bit/8-bit triples fed DIRECTLY into quantized_matmul
  (embeddings, every projection, and transposed logits — no eager
  dequant), KV cache as MlxArray state, greedy loop with in-mlx argmax +
  4-byte scalar readback, retry composed as mlx_transcribe_with_retry.
  Shim extended by 10 symbols (+12 op tests; aimd-mlx 29/29).
- Evidence (release, idle machine, same clips/checkpoint as baseline):
  load 0.42 s warm / 3.81 s cold vs mlx-audio 10.01 s; zh 8.56 s audio
  0.40 s (21.6x realtime) vs mlx-audio 0.43 s; en 3.75 s 0.23 s (16.6x)
  vs 0.25 s — at 0.92-0.93x of mlx-audio latency, i.e. slightly FASTER,
  with token-identical outputs (incl. the same mishearing) and the golden
  16-token argmax stream token-exact; tower parity 2.0e-7 abs vs the
  committed reference. Two bring-up bugs found via ground-truth isolation:
  missing conv biases, and mlx stream thread-affinity (fixed with a
  process-global serialized stream; MlxArray Send+Sync, documented).
- Also: coordinator landed `[profile.dev.package.aimd-*] opt-level = 3`
  after the agent's accidental full-CPU debug suites burned 45 min (killed
  and re-scoped mid-run; checkpoint tests now ~10x faster).
- Remaining: workspace gate (in flight), commit of all mlx wiring.
- Blockers: none.
- Next action: gate green -> commit; WS-07 (OCR engine) is the next
  construction wave.

### 2026-10-25 — shuuul/commit b96c031 + WS-07 dispatch — WS-11, WS-07

- Changed: WS-11 (MLX backend, faster than mlx-audio) committed as
  `b96c031` — 18 files, +4884 lines. WS-07 (OCR engine) dispatched as two
  parallel workstreams: WS-07a vision (gundam preprocessing + SAM ViT-B +
  projector; reference mlx_vlm unlimited_ocr/deepseekocr) and WS-07b
  decoder (DeepSeek-V2-lite MoE + ring sliding-window KV + n-gram guard +
  markdown normalization ports). Shared OCR config skeleton landed and
  verified against the real checkpoint (dual-variant vision width map
  handled); `image` crate wired; stale module stub removed.
- Evidence: gate before commit — 167 tests (default) + 80 (`--features
  mlx`) green, fmt clean, clippy 0/0 both modes; dev opt-level took the
  checkpoint suite from 979 s to 44 s.
- Remaining: WS-07a/07b in flight; then pipeline integration + E2E;
  WS-10 (PyO3 + dual-wheel) after that.
- Blockers: none.
- Next action: WS-07 integration on completion.

### 2026-10-25 — shuuul/perf roadmap scheduled — WS-13, WS-14

- Changed: user approved the inference-perf roadmap; WS-13 (MLX perf
  pack: ring KV cache + `mlx::core::compile` via shim + A/B bench) and
  WS-14 (n-gram/prompt-lookup speculative decoding in the greedy loops)
  added as workstreams pending WS-07; hand-written MSL stays WS-08
  (optional late phase). Also queued: rename `native/` -> `crates/` at
  WS-07 integration, before WS-10 packaging bakes paths in.
- Evidence: `mlx/compile.h` C++ API verified present in brew mlx 0.32.4
  (CompileMode enum + compile overloads); WS-11b's concatenate-grown KV
  cache is the hidden O(n^2) motivating WS-13 item 1.
- Remaining: WS-07a/07b in flight.
- Blockers: none.
- Next action: WS-07 integration -> crates/ rename -> WS-13 dispatch.

### 2026-10-25 — shuuul/WS-15 scheduled + dependency policy — WS-15

- Changed: user confirmed the full-Rust interface port (flips v0.4's
  retain-Python decision) with three riders: yt-dlp gains a managed
  auto-update interface (version floor, `aimd deps update`, stale-detect
  auto-retry, `aimd doctor` visibility); rmcp approved for MCP; dependency
  policy = third-party only for complex domains (axum/tokio/rmcp/serde/
  pdfium/image/rustls/miniz_oxide), hand-write the simple parts (arg
  parsing, zip reading, OPF walking, SSE framing, cookie parsing, path
  checks). WS-15 row + three decision rows recorded.
- Remaining: WS-07a/07b in flight; then crates/ rename -> WS-13 -> WS-14
  -> WS-15 -> WS-10 (re-scoped: wheel = compatibility shim + native
  binaries).
- Blockers: none.
- Next action: WS-07 integration on completion.

### 2026-10-25 — agent-ocr-decoder/WS-07b complete — WS-07

- Changed: decoder.rs (1727 lines) + generate.rs (1205 lines) +
  exporter + golden/ocr_decoder fixtures landed after a network-
  interruption resume. 25/25 tests.
- Mission-brief corrections (all runtime-verified against the reference):
  attention is LlamaAttention (standard MHA 10x128, NeoX theta 1e4) —
  `qk_nope_head_dim: 0` flips attn_type, not MLA; logits come from the
  checkpoint's quantized UNTIED lm_head, not tied embeddings; MoE
  scoring is softmax (config default), not sigmoid.
- Ring sliding-window KV semantics reverse-engineered empirically and
  ported exactly: full prompt retained; ring overwrites
  `prefill_length + (pos mod window)`; decode attention unmasked;
  single-token-prompt yields prefill_length=0 (reference quirk ported).
  Cache-row parity 0.035 (bf16-vs-f16 rounding).
- Parity: prefill logits max-abs 0.4824/129280-vocab; window-128 forced
  16-token argmax stream EXACT; window-4 ring prefix exact.
- Known inherent limitation recorded: 6th-vs-7th expert softmax margins
  of 1e-4..9e-4 flip the selected expert set between bf16-reference and
  f16-compute past ring step 4 on the fixture — checkpoint-inherent, not
  a port bug; the text-similarity gate is unaffected. Optional follow-up:
  bf16-grid rounding of intermediates on CPU; the MLX backend (bf16
  compute) should track the reference natively.
- n-gram guard + markdown normalizer: line-by-line ports with every test
  expectation generated by running the actual Python code (the
  normalizer's regex backtracking semantics were caught and fixed).
- Remaining: WS-07a vision still in flight (its vision.rs mid-edit was
  transiently breaking the crate build during this agent's final runs).
- Blockers: none for WS-07b.
- Next action: WS-07a completion -> integration.

### 2026-10-25 — shuuul/MLX-only + pure Homebrew redirection — WS-06, WS-07, WS-08, WS-09, WS-10, WS-15

- Changed: user direction "只需要 MLX 版本" — CPU compute paths deleted
  (ASR CPU encoder/decoder/pipeline/benches removed, ~4.2k lines; the MLX
  backend is now `qwen3_asr::pipeline`, `aimd-mlx` non-optional with an
  empty `mlx` feature alias); OCR being ported to MLX (agent-ocr-mlx),
  after the CPU integration pass proved the token layout and produced
  mlx_vlm-identical markdown on `native/golden/ocr_e2e` fixtures (CPU
  734 s/page vs mlx_vlm 4.8 s motivated the cut). WS-08 kept and
  re-scoped to custom Metal kernels inside the MLX graph (user
  correction). Distribution switched to a Homebrew tap; WS-10 cancelled;
  WS-09/WS-15 carry the tap, mlx-cuda formula and bottle builds. CI
  reworked (single macOS native job with brew-mlx + rust-cache reuse;
  Python lint/spec once; path-split triggers) and TEST RULES added to
  AGENTS.md (one checkpoint load per test binary, three tiers, budgets).
- Evidence: ASR bench transcripts byte-identical before/after the CPU
  deletion; `cargo test --lib qwen3_asr` 17/17 in 1.9 s (was 6.4 s);
  pytest 7.9 s (was 15.1 s); actionlint clean.
- Remaining: OCR MLX port + OCR CPU deletion; aimd-tensor cleanup; MLX
  tests on hosted macOS runners may lack Metal (plan: MLX CPU device for
  the hermetic tier in CI).
- Blockers: none.
- Next action: OCR MLX port completion -> cleanup -> gate -> commit.

### 2026-10-25 — agent-ocr-mlx/WS-07 complete on MLX — WS-07

- Changed: OCR ported to MLX (`mlx_weights.rs`, `mlx_vision.rs`,
  `mlx_decoder.rs`, `preprocess.rs`, `test_support.rs`; pipeline.rs on
  MLX with the CPU-verified token layout). Generation loop mirrors
  mlx_vlm `generate_step` (2048-token chunked prefill, pipelined
  async_eval decode, on-device n-gram guard). CPU vision.rs/decoder.rs
  (5.6k lines) and CPU-era goldens deleted; goldens shrunk to <= 1 MB.
  19 shim ops added (gather_qmm, argpartition, put_along_axis,
  slice_update, async_eval, ...).
- Evidence: ~22 vision stages + final features 0/0 rel-rms vs goldens;
  prefill logits 0/0; forced 16-token stream identical; E2E similarity
  1.0000 with 69/69 and 197/197 tokens, finish=stop; bench in
  docs/performance.md. OCR tests 27 in 3.9 s (was ~239 s on CPU).
- Known difference: single-token decode ~1 bf16 ulp vs Python (brew mlx
  0.32.4 vs Python mlx 0.32.2 small-M qmv) — bbox coords shift by 2/1000,
  markdown unchanged; exactness would need pinning mlx 0.32.2.
- Remaining: cleanup pass (aimd-tensor deletion, CI CPU-device fallback,
  golden size audit) then commit.
- Blockers: none.
- Next action: gate -> commit; then WS-13.

### 2026-10-25 — agent-core/WS-16 ready for review — WS-16

- Changed: `crates/aimd-core` gained `text` (exact port of title
  extraction + header/paragraph chunking, `MAX_CHUNK_SIZE`), `precision`,
  `remote` (`AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`, `/v1` suffix), `ext`
  (extension tables), `output` (persistence + help texts), `url`, `log`
  (stderr logger + `log_*!` macros + thread-local `capture`), `pyfmt`.
  Contract change: `ProcessInput.start/end` are now `Option<i64>` (negative
  values reach the Python validation messages); `TaskType::parse` added.
  `crates/aimd-process`: `process_input(&ProcessInput, &dyn ProcessHooks)`
  over a `Runtime` seam (`NativeRuntime` = real system); routing identical
  to `get_input_route`/`ensure_supported_input` with the pdf-inspector OCR
  probe; doc dispatch calls `aimd_doc::convert_document`; URL dispatch is a
  `Runtime::url_transcript` stub returning `BackendUnavailable` until WS-18
  lands (`asr::transcribe_with_options` + `AsrOptions` is the transcriber
  entry point for the URL audio fallback). Local Qwen3-ASR (ffmpeg decode to
  16 kHz mono unless the wav already loads natively, >600 s segmenting,
  context, language, lazy 4-bit -> 8-bit repetition retry), local
  Unlimited-OCR for images and `mutool draw -r 144 -c rgb` PDFs, remote
  OpenAI-compatible ASR/OCR (ureq + rustls), alias/precision -> mlx-community
  id resolution, HF cache lookup + HTTPS snapshot download, one pipeline load
  per model per process. `crates/aimd-cli`: binary `aimd`, hand-written
  parser over the Typer flag surface, Python output-path/messages/exit codes.
- Evidence: `cargo test -p aimd-core -p aimd-process -p aimd-cli` = 31 + 29 +
  11 pass on the default device and with `AIMD_MLX_DEVICE=cpu`; fmt and
  clippy `-D warnings` clean on the three crates. `AIMD_E2E=1 cargo test -p
  aimd-process --test e2e` passes (OCR png + ASR wav). Parity:
  `aimd clip.wav -o clip_rust.md --language en` vs `uv run aimd clip.wav -o
  clip_py.md --language en` (`say` 16 kHz clip) and `aimd page_text.png` vs
  `uv run aimd page_text.png` (`ocr_e2e/page_text.png`): `diff` empty for
  both.
- Deviations: GLM-OCR, Transformers model ids, and bf16 checkpoints return
  `BackendUnavailable`; local inference on the MLX CPU device returns
  `BackendUnavailable`; missing ffmpeg is `BackendUnavailable`; model
  resolution fails fast before PDF rendering; OCR progress is reported per
  page and cancellation is also checked between PDF pages; remote ASR never
  retries with 8-bit; `.env` is read from the cwd only; plain-text help and
  usage errors (no rich boxes, no completion flags).
- Blockers: none. Open: wire WS-18 into `Runtime::url_transcript`; `mutool`
  is not installed on this machine, so scanned-PDF E2E is unverified (mocked
  hermetically); 6-bit and 0.6B checkpoints untested (none cached).

### 2026-10-25 — agent-url/WS-18 complete — WS-18

- Changed: `crates/aimd-url` ported from `aimd.plugins.url` (yt-dlp CLI, no
  Python). API: `convert_url(&ProcessInput, &dyn AudioTranscriber, &dyn
  ProcessHooks) -> Result<UrlOutput{markdown, title, platform}>` (tools from
  `AIMD_YT_DLP`/`AIMD_READABLE_BACKEND`/`AIMD_NPX`/`AIMD_CURL`) and
  `convert_url_with(.., &UrlConfig, ..)`; `trait AudioTranscriber {
  transcribe(&Path, &TranscribeRequest, hooks) }` where `TranscribeRequest`
  mirrors `aimd_process::asr::AsrOptions` plus the merged `context`;
  `is_url`, `detect_platform`, `merge_asr_context`, `convert_readable_html`
  + `readable_html_requested` (same opt-in trigger) over `dom_smoothie`
  (`AIMD_READABLE_BACKEND=defuddle` escape hatch). Metadata via
  `--dump-single-json` with the same cookie probe order; subtitles via
  `--load-info-json` + `--write-(auto-)subs --sub-langs --sub-format` per
  candidate language/format (Python selection logic, json3/srv1/srv3 -> SRT,
  bilibili inline `data`); audio via the same format strategies; hand-written
  Netscape cookie parser, TTML/srv XML parser, subtitle cleanup and Python
  `str` semantics. Output gets MarkItDown's post-normalization (line
  `rstrip`, `\n{3,}` -> `\n\n`). Golden exporter:
  `scripts/export_url_subtitle_reference.py`.
- Evidence: `cargo test -p aimd-url` = 35 unit + 2 golden parity (25
  subtitle inputs incl. a real YouTube track in 6 formats; language
  preference/detection, metadata context, `format_content`, cookie specs,
  URL detection, linear transcript) + 35 hermetic pipeline tests (fake
  `sh` yt-dlp, mock transcriber; ports of the `test_url.py` and
  `test_asr_context.py` URL scenarios), ~1 s; also green with
  `AIMD_MLX_DEVICE=cpu`. fmt + clippy `-D warnings` clean. `AIMD_E2E=1`:
  `jNQXAC9IVRw` and `dQw4w9WgXcQ` Rust Markdown == `uv run aimd <url>`
  (view count masked); one earlier `dQw4w9WgXcQ` run failed transiently
  and passed on 3 reruns. Readable F1 (word multiset) dom_smoothie vs defuddle:
  fixtures blog 0.978, docs 0.989, news 0.991; saved real pages:
  paulgraham greatwork 0.999, docs.python.org re 0.950, Wikipedia Markdown
  0.876, github.blog/engineering (index page) 0.321.
- Deviations: explicit cookie options yt-dlp rejects fail fast as
  `UnsupportedInput`; transient subtitle retries are yt-dlp's (`--retries 2
  --retry-sleep http:linear=0.5:1.5:1`, socket timeout 40 s) instead of the
  Python urlopen loop; the audio auth hint is raised only when every cookie
  source failed; readable pages are fetched with `curl`.
- Blockers: none. Next (WS-16): implement `AudioTranscriber` over
  `asr::transcribe_with_options` and wire `Runtime::url_transcript` to
  `aimd_url::convert_url`; route `readable_html` (no `TaskType` yet) to
  `convert_readable_html`.

### 2026-10-25 — agent-core/WS-16 URL wiring ready for review — WS-16

- Changed: `aimd-process` depends on `aimd-url`. `UrlAudioTranscriber`
  implements `aimd_url::AudioTranscriber` over `asr::transcribe_with_options`,
  and `convert_url(rt, request, hooks)` wraps `aimd_url::convert_url`.
  `NativeRuntime::url_transcript` calls it (the `Runtime` seam stays
  mockable). `UrlOutput` is now the `aimd_url` re-export, and
  `ProcessResult.platform` comes from `UrlOutput.platform`. Local
  doc/OCR/ASR markdown now goes through `markitdown_normalize`, as
  MarkItDown does for every converter; files a converter writes are left
  as written. Readable HTML is not routed: Python core/CLI/API/MCP never
  pass `task_type="readable_html"` or `defuddle=True`, so the converter
  is unreachable there too. User decision: GLM-OCR removed. `glm-ocr`,
  `GLM OCR`, `mlx-community/GLM-OCR-*` and `zai-org/GLM-OCR` now fail
  through the unknown-model path (`Unsupported MLX VLM OCR model ...`,
  ProcessingFailed); the error text no longer lists GLM.
- Evidence: `cargo test -p aimd-core -p aimd-process -p aimd-cli` = 31 +
  31 + 11 pass (default device and `AIMD_MLX_DEVICE=cpu`); fmt and clippy
  `-D warnings` clean. Ported the URL precision forwarding test
  (`test_use_case_url_passes_precision_to_url_processor`) plus adapter and
  normalization tests. Release `aimd https://www.youtube.com/watch?v=jNQXAC9IVRw
  -o yt_rust.md` vs `uv run aimd <same> -o yt_py.md`: byte-identical (712
  bytes; the view-count line matched too).
- Blockers: none.

### 2026-10-26 — agent-mcp/WS-20 `aimd-mcp` ready for review — WS-20

- Changed: `crates/aimd-mcp` (lib + `aimd-mcp` bin) on `rmcp` 3.7 (features
  `server`, `transport-io`; no macros/schemars). Hand-written `ServerHandler`
  with the Python tool schemas/description reproduced verbatim (tools/list is
  byte-identical to FastMCP), manual pydantic-style argument validation,
  `Runtime` injected (`AimdMcp::new(Arc<dyn Runtime>)`), processing on
  `spawn_blocking`. Ported `test_mcp_server` scenarios (11 tests, mock runtime,
  duplex JSON-RPC round trip). No changes outside `crates/aimd-mcp/`.
- Verification: `cargo fmt/clippy -D warnings` clean, `cargo test -p aimd-mcp`
  green; stdio smoke vs `uv run aimd-mcp` on `page_text.png`: tools/list and
  OCR result identical. Known diffs: initialize capabilities (`tools: {}` vs
  Python's listChanged/prompts/resources stubs); error text for domain errors
  includes the message (Python's masks it); pydantic URL line omitted.
- Blockers: none.

### 2026-10-26 — agent-api/WS-19 `aimd-api` ready for review — WS-19

- Changed: `crates/aimd-api` (lib + `aimd-api` bin) on axum 0.8 (`http1`,
  `tokio`, `multipart`) + tokio + `http-body`. Routes `/healthz`, `/readyz`,
  `POST /v1/process`, `POST /v1/blobs` (streamed to disk), `POST /v1/jobs`,
  `GET`/`DELETE /v1/jobs/{id}`, `GET /v1/jobs/{id}/events` (hand-written SSE
  framing, `Last-Event-ID` resume), `/openapi.json` (the FastAPI schema frozen
  as `src/openapi.json`, version/help texts/`bearerAuth` patched at startup),
  `/docs`, `/redoc`; FastAPI-shaped 404/405/401 bodies and pydantic-compatible
  422 bodies (hand-written lax validation: `"3"`->3, `"yes"`->true, ...).
  Hand-written UUIDs, pydantic UTC timestamps, constant-time bearer check,
  `realpath`-style allow-root checks, blob store, loopback bind validation.
  `Processor` seam (`RuntimeProcessor` over `Arc<dyn Runtime>`); processing
  on `spawn_blocking` with a `ProcessHooks` impl backed by the job record
  (progress stages mapped onto `JobStage`, `recognizing`->`ocr`). SIGINT/
  SIGTERM: cooperative cancel, 5 s grace, then abort remaining runners
  (`Job task cancelled`) and exit. Same env surface (`AIMD_HTTP_HOST`,
  `AIMD_HTTP_PORT`, `AIMD_API_TOKEN`, `AIMD_ALLOWED_ROOTS`, `AIMD_BLOB_DIR`,
  `AIMD_TEMP_DIR`); no CLI flags (as Python). No changes outside the crate.
- Evidence: `cargo test -p aimd-api` = 49 hermetic tests (ports of
  `test_api`, `test_api_jobs`, `test_api_blobs`, `test_api_sidecar`, plus a
  bound-socket SSE + shutdown test and captured FastAPI error bodies), 5/5
  stable reruns, <0.1 s; fmt + clippy `-D warnings` clean. Release smoke:
  `/healthz` ok; `POST /v1/process` on `ocr_e2e/page_text.png` is
  byte-identical to `uv run aimd-api`; job + SSE stream (queued, running/
  extracting, ocr 1/1, completed) and graceful SIGTERM exit verified.
- Deviations: OpenAPI keys are emitted sorted (same content);
  `/docs/oauth2-redirect` not served; I/O error texts use Rust wording;
  JSON-decode `ctx.error` messages are mapped approximately from serde_json;
  ints beyond i64 are rejected (`int_parsing_size`); `Allow` on 405 also
  lists `HEAD` for GET routes.
- Blockers: none.

### 2026-10-25 — agent-tap/WS-21 tap + release automation ready for review — WS-21

- Changed: new tap working copy `/Users/shuuul/repo/homebrew-tap` (git
  init, one local commit, no remote) in `brew tap-new` layout:
  `Formula/aimd.rb` (source tarball of a release tag; `rust` build dep;
  `depends_on :macos`, `arch: :arm64`, `mlx`, `ffmpeg`, `mupdf`, `pandoc`,
  `yt-dlp`; `cargo install` per binary crate with `std_cargo_args`, a shared
  `CARGO_TARGET_DIR`, `MLX_PREFIX=formula_opt_prefix("mlx")`, release
  debuginfo off; hermetic `test do`: `--help`, `.md` passthrough, `.html`
  via pandoc, api/mcp binaries present), `.github/workflows/tests.yml`
  (brew test-bot, macos-26 arm64; Linux matrix entry commented out pending
  WS-09), `publish.yml` (standard `brew pr-pull`), `dependabot.yml`,
  README. The template's `autobump.yml` is dropped (aimd opens the bump
  PRs). aimd side: `.github/workflows/rust-release.yml` (`push` on
  `aimd-rs-v*` tags, `workflow_dispatch`, `workflow_call`): verifies the
  tag against the workspace version via `cargo metadata --locked`, builds
  the three binaries on macos-14 with brew mlx, runs the same hermetic
  smoke as the formula test, hashes the GitHub source tarball, and opens
  an `aimd <version>` PR on `shuuul/homebrew-tap` (secret
  `HOMEBREW_TAP_TOKEN`). `release.yml`, `ci.yml` and the release-please
  config are untouched, so Python keeps releasing.
- Decisions taken (to confirm): Rust tags are `aimd-rs-vX.Y.Z` while
  Python releases `vX.Y.Z`, because workspace 0.1.0 would collide with
  existing Python tags (v0.5.0..v0.23.3). Release-please is not switched yet:
  a root Python package and a root Cargo workspace cannot share the `.`
  package path, and a version bump would leave `Cargo.lock` stale for
  `--locked`. The proposal for WS-15 closeout is in the WS-21 report:
  release-type `simple` with toml `extra-files` for
  `$.workspace.package.version`, `release.yml` calling `rust-release.yml`
  via `workflow_call`, and the AGENTS.md "centralized release.yml" rule
  restored.
- Evidence: `actionlint` clean on all aimd and tap workflows; `brew style`
  and `brew audit --strict --formula` clean. Local proof against the
  working tree (rsync snapshot without `target*`/`.git`/`.venv` -> tarball,
  temp tap `ws21tmp/proof` with a `file://` url): the first attempt failed
  while `aimd-api` was mid-edit (`can't find lib aimd_api`); the retry
  passed. `brew install --build-from-source` took 2 m 16 s (59 MB, links
  only `/opt/homebrew/opt/mlx/lib/libmlx.dylib` plus system libs), and
  `brew test` passed. `--build-bottle` + `brew bottle` gave a relocatable
  bottle (`cellar: :any`, arm64_golden_gate). Pouring that bottle through a
  `file://` `root_url` and running `brew test` again passed. Afterwards aimd
  was uninstalled and the temp taps untapped.
- Blockers: none for the code. Acceptance (`brew install shuuul/tap/aimd`
  from a published bottle) needs the coordinator to create and push
  `shuuul/homebrew-tap`, add `HOMEBREW_TAP_TOKEN`, and cut the first
  `aimd-rs-v*` tag. Linux bottles wait on WS-09 (tap mlx-cuda formula,
  Linux `MLX_PREFIX` in build.rs, self-hosted CUDA runner).

### 2026-10-25 — shuuul/WS-15 split and kickoff — WS-15, WS-16, WS-17, WS-18

- Changed: WS-15 moved ahead of WS-13/WS-14 and split into WS-16..WS-21
  (decision rows above). Skeleton crates `aimd-core` (errors, hooks,
  models), `aimd-doc`, `aimd-url`, `aimd-process`, `aimd-cli` added to the
  workspace. README gained a migration-status note.
- Next: WS-16/WS-17/WS-18 run in parallel (each owns its crate directory and
  its own `CARGO_TARGET_DIR`); WS-19/WS-20 start once the WS-16 process API
  is stable.

### 2026-10-25 — agent-doc/WS-17 complete — WS-17

- Changed: `crates/aimd-doc` ported from `aimd.plugins.doc`. API:
  `convert_document(path, output_dir, temp_dir, &dyn ProcessHooks) ->
  aimd_core::Result<DocOutput{markdown, title: Option<String>}>`,
  `is_document_extension(ext)`, `DOCUMENT_EXTENSIONS`. Pandoc CLI for every
  Python format plus `.pptx`/`.xlsx`; `.txt`/`.md` pass through; legacy
  `.doc/.ppt/.xls` unsupported; `.pdf` goes through the `pdf-inspector` crate
  (default options, same as the Python binding's `process_pdf`).
  EPUB: hand-written zip (central directory, stored + `miniz_oxide`
  inflate, CRC check), container -> OPF -> spine regex walk, per-chapter
  Pandoc, all cleanup regexes (`regex` crate; the `_dedup_headings`
  backreference is hand-written), flat image extraction, progress and
  cancellation per chapter.
- Evidence: `cargo test -p aimd-doc` 16 unit + 1 parity (parity skipped
  unless `AIMD_E2E=1`); `AIMD_E2E=1` parity vs the Python converter passes
  (identical Markdown/title/chapters/image bytes on a rich generated EPUB
  and `tests/test_files/minimal.epub`, identical `.rst` conversion, 15-case
  cleaner differential corpus). fmt + clippy `-D warnings` clean.
- Blockers: none. Needs from WS-16: call `convert_document` only after the
  scanned/encoding-issue OCR probe; `.pdf` is in `DOCUMENT_EXTENSIONS`.

### 2026-10-26 — shuuul/WS-15 closeout — WS-15, WS-21

- Changed: `src/aimd`, `tests/`, `pyproject.toml`, `uv.lock`,
  `.python-version`, `scripts/check_specs.py`,
  `scripts/export_url_subtitle_reference.py` deleted; `minimal.epub` moved
  to `crates/aimd-doc/tests/fixtures/`; Python-parity tests removed
  (`aimd-doc/tests/parity.rs`) or made Rust-only (`aimd-url` YouTube smoke);
  `crates/xtask` (`cargo xtask check-specs`, 18 tests); release-please
  config/manifest and the Python CI/release workflows deleted, `rust.yml` ->
  `ci.yml`, `rust-release.yml` -> `release.yml` (tag `v*`); version 1.0.0;
  README/AGENTS/docs rewritten for the Rust + Homebrew product.
- Next: push the branch, final PyPI release from `main` pointing to brew,
  `HOMEBREW_TAP_TOKEN` secret, tag `v1.0.0` -> first bottle (closes WS-21,
  then WS-15).

### 2026-10-26 — shuuul/WS-13 MLX perf pack — WS-13

- Changed: shared `models/kv_cache.rs` (moved from the OCR decoder with
  its ring/step-256 unit test); ASR `MlxKvCache` (concatenate growth)
  deleted; ASR decode pipelined (device-side argmax -> `embed_indices`,
  `async_eval`, `clear_cache` every 256 tokens; a step queued behind EOS
  is discarded); shim `mlx_swiglu` (`mlx::core::compile(shapeless=true)`)
  + `MlxArray::swiglu`, bit-identical to `silu * up` in bf16 on GPU and
  CPU (`mlx_parity` test).
- Evidence: ASR 1.7B-4bit 60 s clip median 4.18 s -> 3.13 s (-25%), short
  clips ~5% faster, transcripts byte-identical; OCR `page_text.png`
  unchanged within noise. Rows in docs/performance.md. Engine goldens
  (argmax/forced streams) pass.
- Next action: WS-14 (speculative decoding) is unblocked.

## Completion summary

(to complete before archiving)
