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
| 2026-10-25 | MLX-only cleanup: `aimd-tensor` crate deleted; ANE spike artifacts (`golden/ane_spike`, `scripts/ane_spike`) deleted (report `docs/ane-spike-results.md` kept as history); `aimd_weights::dequantize_affine` kept as the test-only bit-exact oracle for `golden/quant`; `golden/asr_decoder/logits.f32` dropped (unread). MLX device = GPU unless `AIMD_MLX_DEVICE=cpu` or no GPU; hermetic CI runs on the CPU device and the checkpoint tier skips there; quantized-matmul parity tolerance loosened on the CPU device only | Hosted macOS runners may lack Metal; keep CI hermetic and inside the TEST RULES budget (CPU 4 s, Metal 12 s) | WS-02, WS-03, WS-12 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Native workspace scaffold: `native/` Cargo workspace, three crates with stub modules and documented API contracts, workspace deps pre-wired, `cargo test --workspace` green | shuuul | Done | None | `cd native && cargo test --workspace` |
| WS-02 | `crates/aimd-weights`: safetensors mmap, MLX 4-bit affine (g=64) unpack + dequant, 8-bit path, `scripts/export_quant_reference.py` + committed golden fixtures | agent-weights | Done | WS-01 | `cargo test -p aimd-weights` golden parity (bit-exact f16 vs MLX exports) |
| WS-03 | `crates/aimd-tensor`: dtype/shape/CPU tensor, rmsnorm, softmax, rope (theta, window semantics), swiglu, GQA attention (f32/f16) with naive-reference unit tests | agent-tensor | Done | WS-01 | `cargo test -p aimd-tensor` |
| WS-04 | `crates/aimd-engine/src/tokenizer`: `tokenizers` wrapper loading both checkpoints' `tokenizer.json`, encode/decode roundtrips (CJK-heavy), ASR chat-template builders (language prefix, context system message), OCR `document parsing.` prompt builder | agent-tokenizer + shuuul | Done | WS-01 | `cargo test -p aimd-engine --lib tokenizer` incl. checkpoint roundtrips (skip w/o cache) |
| WS-05 | `crates/aimd-engine/src/audio`: wav load (f32 mono 16k), 128-bin mel matching the Qwen3-ASR feature extractor, `scripts/export_mel_reference.py` + committed golden fixtures | agent-audio | Done | WS-01 | `cargo test -p aimd-engine --lib audio` golden parity (tolerance vs exported reference) |
| WS-06 | ASR model graph + generation loop on CPU (windowed encoder n_window=50/infer=800, Qwen3 decoder, greedy, retry-on-loop) | agent-encoder + agent-decoder + shuuul | Done | WS-02, WS-03, WS-04, WS-05 | Commits 2cc53a9/8de3841: token-identical outputs vs MLX, golden argmax stream exact, E2E green. Open tail: labeled golden-corpus CER harness (two-clip identity is evidence, not the gate) |
| WS-07 | OCR model graph + generation loop on the MLX backend (gundam preprocessing, SAM ViT-B + CLIP + projector, MoE decoder with ring sliding-window KV, n-gram guard, markdown normalization); CPU prototype retired per the MLX-only decision | agent-ocr-vision + agent-ocr-decoder + agent-ocr-integration + agent-ocr-mlx | Done | WS-11 | Vision + prefill bit-exact vs mlx_vlm goldens; E2E markdown identical (similarity 1.0, finish=stop, equal token counts) on both `native/golden/ocr_e2e` fixtures; latency at parity with mlx_vlm (0.83 s / 4.7 s vs 0.88 s / 4.6 s). Open tail: labeled multi-page OCR corpus gate |
| WS-08 | Hand-written Metal MSL kernels for macOS hot spots, plugged into the MLX graph (`mlx::core::fast::metal_kernel` custom kernels through the shim, e.g. fused MoE dispatch, fused attention/ring-KV update, quantized GEMV variants tuned for M2 Pro); chosen by profiling after WS-13/WS-14, kept only where they beat MLX's built-ins | Unassigned | Pending | WS-13 | Per-kernel A/B vs the MLX built-in with outputs unchanged; end-to-end decode gain recorded in docs/performance.md |
| WS-09 | Linux on metacube/dspark via the `mlx` CUDA backend: tap formula for a C++-only mlx built with CUDA, `aimd-mlx` build.rs made OS-conditional (Linux `.so`, `MLX_PREFIX`), ASR + OCR E2E against the Python reference, Linux bottle build + self-hosted CI runner; hand-written .cu/CUTLASS cancelled | Unassigned | Pending | WS-07 | ASR/OCR E2E parity on metacube; `brew install` of the tap mlx formula + Linux bottle works; perf rows in docs/performance.md |
| WS-10 | PyO3 binding / wheels / PyPI — CANCELLED by the pure Homebrew distribution decision (no work done) | shuuul | Done | None | Cancellation recorded in Decisions (2026-10-25 pure Homebrew) |
| WS-11 | `aimd-mlx` C-ABI shim over `mlx::core` (array lifetime, `quantized_matmul`, GEMM/attention composition); model graphs on the MLX backend; `--features mlx` variant build. Promoted to the active wave (ASR first); brew mlx 0.32.4 link target | agent-mlx + agent-mlxgraph + shuuul | Done | WS-06 | ASR done (commit b96c031: 0.92-0.93x mlx-audio latency, 24x faster load, token-identical outputs); OCR graph rides WS-07 completion |
| WS-13 | MLX perf pack: preallocated/ring KV cache (replace concatenate growth) + `mlx::core::compile` through the shim applied to decoder layer/step functions + A/B benchmarks recorded in docs/performance.md | Unassigned | Pending | WS-07 | Decode-latency improvement vs b96c031 baseline with outputs still text-identical; bench rows recorded |
| WS-14 | Speculative decoding (n-gram/prompt-lookup draft + single-forward verify k tokens) in the greedy generation loops; enabled per-model after accuracy passes the golden-corpus gates | Unassigned | Pending | WS-13 | OCR decode >= 1.3x vs WS-13 with corpus similarity unchanged; ASR measured opportunistically |
| WS-15 | Full-Rust interface layer + Homebrew release: CLI (hand-written arg parse), doc pipeline (brew pandoc + hand-written zip/OPF + pdfium), URL pipeline (brew yt-dlp with stderr-signature hint to `brew upgrade yt-dlp` + `dom_smoothie` readability with defuddle fixture validation), HTTP API (axum, SSE, jobs), MCP (rmcp); tap formula + bottle CI (macOS arm64, Linux from WS-09) + Rust release automation replacing release-please's Python flow; final PyPI release pointing to brew, then Python package removal. Per the dependency policy: hand-write the simple parts, third-party for complex domains | Unassigned | Pending | WS-14 | Behavior parity vs the Python interfaces on the ported pytest scenarios; readability output similarity vs defuddle on a fixture corpus; `brew install shuuul/tap/aimd` from bottles works on macOS arm64 (and Linux once WS-09 lands) |
| WS-12 | ANE feasibility spike: Core ML routes (stateful LLM on ANE, encoder-on-ANE), fp16/weight-size constraints, mlx-or-torch conversion source, Rust integration path, expected gains vs Metal on M2 Pro; go/no-go recommendation in `docs/ane-feasibility.md` | agent-ane + agent-ane-spike + shuuul | Done | None | Report reviewed; NO-GO recorded (commit 8fc1650): gates passed technically, ~4-5% end-to-end gain does not justify the artifact/maintenance cost |

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

## Completion summary

(to complete before archiving)
