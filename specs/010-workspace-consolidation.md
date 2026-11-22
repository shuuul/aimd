---
id: "010"
title: "Workspace consolidation: one aimd library crate over the engine"
status: Active
created: 2026-11-23
updated: 2026-11-23
coordinator: "pi (Claude Opus 5.5) session with shuuul"
---

# 010 — Workspace consolidation: one aimd library crate over the engine

## Context

The workspace has 9 product crates plus `xtask` (about 38k lines of Rust in
`src/`). Line counts from `src/` only, at `42ba1a6` (v1.5.0):

| Crate | src lines | Role |
| --- | --- | --- |
| `aimd-engine` | 18.4k | Model graphs, tokenizer, mel, generation |
| `aimd-process` | 5.5k | Routing, orchestration, `Runtime`, ASR/OCR, remote, HF |
| `aimd-url` | 4.2k | yt-dlp, subtitles, cookies, metadata, readable HTML |
| `aimd-core` | 3.0k | Models, errors, hooks, text, anchors, output, misc helpers |
| `aimd-mlx` | 2.2k + 3.3k C++ | MLX C-ABI shim, `build.rs` linking |
| `aimd-doc` | 1.9k | Pandoc, EPUB, cleanup, pdf-inspector |
| `aimd-cli` | 1.3k | `aimd` binary |
| `aimd-weights` | 1.1k | safetensors store + test-only dequant oracle |
| `aimd-mcp` | 0.8k | `aimd-mcp` binary |

The crate count is not the problem in itself. The problem is that several
boundaries cost more than they buy:

- **`aimd-weights` is too thin.** `aimd-engine` uses only `WeightStore`;
  `dequantize_affine` / `DequantizedTensor` are a test oracle used only by
  `aimd-mlx`'s tests (dev-dependency) and `aimd-weights/tests`.
- **Utilities are duplicated across the doc/url/process split.** Three
  temp-dir implementations (`aimd-url/src/tempdir.rs`,
  `aimd-process/src/tempdir.rs`, `aimd-doc/src/util.rs::TempDir`) and two
  Python-string shims (`aimd-url/src/pytext.rs`, `aimd-core/src/pyfmt.rs`,
  both with `is_space` / `strip`).
- **The `Runtime` seam covers half the system.** `aimd-process` routes env,
  `which` and subprocesses through `Runtime`, but `aimd-doc` has its own
  `pandoc::Runner` and `aimd-url` calls `yt-dlp`, `firecrawl-local` and
  `npx defuddle` with `Command::new` directly. Because doc/url cannot depend
  on `aimd-process`, `Runtime` exposes coarse `convert_document` /
  `url_transcript` methods and tests fake those instead of the tools.
- **`aimd-core` is a grab bag.** `remote`, `anchors`, `ext`, `precision` and
  `url` are used only by `aimd-process`. The genuinely shared contract is
  `model`, `error`, `hooks`, `output`, `log`.
- **`aimd_url::AudioTranscriber` exists to break a crate cycle.** It is a
  fine interface, but it is public API only because of the crate split.

What the current split does buy, and must be kept:

- `aimd-engine` (heaviest) and `aimd-mlx` (C++ build, MLX link) stay
  separate compilation units, so orchestration edits do not recompile them.
- `aimd-cli` does not pull in `tokio` / `rmcp`.
- Dependency-direction rules are enforced by the compiler.

## Goal and success criteria

Collapse the workspace to five crates: `aimd-mlx`, `aimd-engine`, `aimd`
(library), `aimd-cli`, `aimd-mcp` (plus `xtask`), with byte-identical
product behavior, one system seam, and module-level boundary rules enforced
by `xtask`.

- [ ] Workspace members are exactly `aimd-mlx`, `aimd-engine`, `aimd`,
  `aimd-cli`, `aimd-mcp`, `xtask` (`cargo metadata --no-deps`).
- [ ] No duplicated temp-dir or Python-string helpers: one `util::tempdir`
  and one `util::pytext` in `aimd` (`rg 'struct TempDir'` -> 1 hit).
- [ ] Every subprocess and env/`PATH` lookup in `aimd` goes through the
  `sys::System` seam (`rg 'Command::new|std::env::var' crates/aimd/src`
  -> only `sys.rs` and `#[cfg(test)]` code).
- [ ] `cargo xtask check-modules` enforces the module dependency rules (see
  Decisions) and has regression tests in `cargo test -p xtask`.
- [ ] `cargo check -p aimd --no-default-features` builds without linking
  MLX; local ASR/OCR then fail with `BackendUnavailable`, remote backends,
  documents and URLs (subtitles path) still work.
- [ ] Output parity: the hermetic tier passes with the same test count or a
  recorded reason for each deleted test; frozen goldens
  (`tests/golden/subtitles`) are moved byte-for-byte, never edited; MCP
  tool schemas and CLI `--help` output are byte-identical to `42ba1a6`.
- [ ] Build cost recorded before/after (clean `cargo build --release -p
  aimd-cli -p aimd-mcp`, and incremental rebuild after touching one file in
  doc, url and process); any regression > 25% has a recorded decision.
- [ ] `release.yml` and the Homebrew formula build unchanged
  (`-p aimd-cli -p aimd-mcp`, `crates/aimd-cli` path).

## Scope and non-goals

In scope:

- Merge `aimd-weights` into `aimd-engine` (store) and `aimd-mlx` (test
  oracle + `golden/quant`).
- Merge `aimd-core`, `aimd-doc`, `aimd-url`, `aimd-process` into one
  `crates/aimd` library with internal modules.
- Introduce `sys::System` (env, `which`, subprocess) and route doc/url tool
  calls through it; `Runtime` extends it with model loading and preflight.
- Make `aimd-engine`'s `mlx` feature real and add `aimd`'s default
  `native` feature.
- `xtask check-modules` validator, AGENTS.md / `docs/architecture.md`
  rewrite.

Not in scope:

- Any behavior, output, flag, MCP schema or error-kind change.
- Splitting `aimd-engine` per model, or merging `aimd-cli` and `aimd-mcp`.
- Publishing `aimd` to crates.io or promising a stable library API.
- Changing the `aimd-mlx` shim or MLX linking.
- New third-party crates.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-11-23 | Target layout: `aimd-mlx`, `aimd-engine`, `aimd` (lib), `aimd-cli`, `aimd-mcp`, `xtask`. | Keeps the two expensive compilation units (engine, mlx) and the two binaries separate; removes boundaries that only produce duplication and cycle-breaking traits. | All |
| 2026-11-23 | `aimd-weights` splits: `store` -> `aimd_engine::weights`; `mlx_quant` oracle -> `aimd-mlx` test support (`tests/common/`), `tests/golden/quant` -> `crates/aimd-mlx/tests/golden/quant`. | `aimd-mlx` tests cannot dev-depend on `aimd-engine` without a dev-dependency cycle that duplicates `aimd-mlx` types. The oracle is only used by tests. | WS-02 |
| 2026-11-23 | `aimd` module tree: `model`, `error`, `hooks`, `output`, `log`, `text`, `anchors` (contract); `sys`, `util::{tempdir,pytext}` (infrastructure); `doc`, `url` (source adapters); `route`, `runtime`, `native`, `asr`, `ocr`, `audio`, `remote`, `hf`, `http`, `models`, `precision`, `ext` (orchestration). `aimd_core::url` becomes `url::parse`. | Flat, discoverable; mirrors the current crates so `git mv` keeps history. | WS-03 |
| 2026-11-23 | Module rules (enforced by `xtask check-modules` on `use crate::` paths): contract modules import only contract modules; `sys`/`util` import only contract; `doc` and `url` import only contract, `sys`, `util`, `ext`, and each other never; orchestration may import anything; `aimd-cli` / `aimd-mcp` use only `aimd`'s public API. | Replaces the compiler-enforced crate graph with a cheap, tested lint so the rules in AGENTS.md stay true. | WS-01, WS-03 |
| 2026-11-23 | `sys::System` trait holds `env`, `which`, `run` (current `Runtime` defaults); `Runtime: System` adds `preflight_local`, `load_asr`, `load_ocr`, `pdf_*`. `doc` and `url` take `&dyn System`. `pandoc::Runner` is deleted. `convert_document` / `url_transcript` stay on `Runtime` until WS-04 shows tests no longer need them, then are removed. | One seam for every subprocess; fakes move from coarse methods to tool level. | WS-04 |
| 2026-11-23 | `AudioTranscriber` becomes `pub(crate)` inside `aimd::url`. | Its only reason to be public was the crate cycle. | WS-03 |
| 2026-11-23 | `aimd-engine` feature `mlx` (default on) makes `aimd-mlx` optional and gates graphs/pipelines; mlx-free helpers used by remote paths (`parse_qwen_output`, `detect_repetition_loop`, `normalize_unlimited_ocr_markdown`, `render_unlimited_ocr2`, tokenizer language tables, wav loading) compile without it. `normalize_unlimited_ocr_markdown` moves out of `unlimited_ocr/generate.rs` into an mlx-free module. `aimd` feature `native` (default) enables `aimd-engine/mlx` and `NativeRuntime` model loading. | Enables a remote-only build without MLX; the pure helpers already have no `aimd_mlx` imports except `generate.rs`. | WS-05 |
| 2026-11-23 | WS-05 (`native` feature) stays in this spec rather than a follow-up (shuuul). | The engine feature split touches the same imports WS-03 rewrites; doing it once avoids a second pass over the moved modules. | WS-05 |
| 2026-11-23 | Package names `aimd-cli` / `aimd-mcp` and their paths stay; binary names `aimd` / `aimd-mcp` stay. | Release workflow and Homebrew formula build by package/path. | WS-03, WS-06 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Baseline + guardrails: record build timings, hermetic test counts per crate, MCP schema and CLI `--help` snapshots at `42ba1a6`; add `cargo xtask check-modules` (rules table configurable, initially a no-op until `crates/aimd` exists) with regression tests. | Unassigned | Pending | None | `cargo test -p xtask`; baseline numbers in Progress |
| WS-02 | Fold `aimd-weights`: `store.rs` -> `aimd-engine/src/weights.rs`; `mlx_quant.rs` + `golden/quant` -> `aimd-mlx` test support; update `mlx_parity.rs` fixture path; delete crate. | Unassigned | Pending | None | `AIMD_MLX_DEVICE=cpu cargo test -p aimd-engine -p aimd-mlx`; checkpoint tier on Metal |
| WS-03 | Create `crates/aimd` by `git mv` of core/doc/url/process sources into the module tree; merge the three temp-dirs and two pytext shims into `util`; make `AudioTranscriber` `pub(crate)`; move integration tests and fixtures (`aimd-url/tests/{golden,fixtures}`, `aimd-doc/tests/fixtures`, `aimd-process/tests/e2e.rs`) under `crates/aimd/tests`; switch `aimd-cli` / `aimd-mcp` imports; delete the four crates; enable `check-modules` rules. | Unassigned | Pending | WS-01 | Hermetic tier, `cargo xtask check-modules`, `--help`/schema diff vs WS-01 snapshot, `git diff --stat -M` shows renames |
| WS-04 | `sys::System` seam: route pandoc, yt-dlp, firecrawl-local, npx defuddle, mutool, ffmpeg through it; delete `pandoc::Runner`; convert doc/url tests to tool-level fakes; remove `Runtime::convert_document` / `url_transcript` if unused. | Unassigned | Pending | WS-03 | `rg 'Command::new\|std::env::var' crates/aimd/src` clean; hermetic tier; `AIMD_E2E=1 cargo test -p aimd --test e2e` |
| WS-05 | `native` feature: real `aimd-engine/mlx` feature, mlx-free helper modules, `aimd` default `native`; `NativeRuntime` without `native` returns `BackendUnavailable` for local ASR/OCR. | Unassigned | Pending | WS-02, WS-03 | `cargo check -p aimd --no-default-features`; `cargo tree -p aimd --no-default-features` has no `aimd-mlx`; hermetic tier with default features |
| WS-06 | Docs sync and acceptance: AGENTS.md (crate map, diagram, where-to-look, conventions, anti-patterns, golden fixture paths, commands), `docs/architecture.md`, `.agents/skills/aimd` if it names crates; after-numbers for build timings; checkpoint tier + corpus gate on Metal; release dry-run build. | Unassigned | Pending | WS-04, WS-05 | All Verification commands; `cargo xtask check-specs` |

WS-01 and WS-02 can run in parallel. WS-03 is a single large mechanical
move and should be one worker in one commit series, so other branches touching
`crates/aimd-{core,doc,url,process}` must land first or rebase after it.

## Verification

```bash
# Every workstream
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo xtask check-modules
cargo xtask check-specs && cargo test -p xtask

# Feature split (WS-05)
cargo check -p aimd --no-default-features
cargo tree -p aimd --no-default-features -e normal | rg aimd-mlx  # no output

# Contract parity (WS-03 onward, against WS-01 snapshots)
./target/release/aimd --help | diff - target/baseline/aimd-help.txt
# MCP tools/list JSON diff vs target/baseline/mcp-tools.json

# Before closeout (Metal Mac; metacube for CUDA)
cargo test --workspace
AIMD_E2E=1 cargo test -p aimd --test e2e
cargo run --release -p aimd-engine --example corpus_gate
cargo build --release --locked -p aimd-cli -p aimd-mcp
```

The corpus gate must not move: this spec changes no model code.

## Documentation sync

- Durable product/developer docs: `docs/architecture.md` (crate graph,
  module rules, `System`/`Runtime` seam, `native` feature).
- Nearest local `AGENTS.md`: root `AGENTS.md` only (no nested AGENTS.md
  today); rewrite CRATE MAP, ARCHITECTURE DIAGRAM, WHERE TO LOOK, the
  core-is-interface-independent / Runtime-seam conventions, ANTI-PATTERNS
  (`aimd-url` -> `aimd-process` edge rule becomes the `url` module rule),
  TEST RULES golden fixture owners (`quant` moves to `aimd-mlx`; `subtitles`
  to `aimd`), COMMANDS (`-p aimd` test invocations).
- Parent/package guidance: `.agents/skills/aimd/SKILL.md` if it names
  crates; Homebrew formula in `shuuul/homebrew-tap` only if a path changes
  (expected: none).
- Root guidance and roadmap: `README.md` architecture section, if any.

### Experiment refs

None expected.

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-11-23 — pi / coordinator — spec draft

- Changed: drafted this spec from the crate audit at `42ba1a6`.
- Evidence: line counts and cross-crate usage in Context; duplicated
  helpers found with `rg 'struct TempDir'` and `pytext`/`pyfmt`;
  `Command::new` in `aimd-doc/src/pandoc.rs`, `aimd-url/src/{ytdlp,readable}.rs`.
- Remaining: review Decisions with shuuul, then set `status: Active`.
- Blockers: none.
- Next action: decide whether WS-05 (`native` feature) stays in this spec or
  becomes a follow-up; claim WS-01 and WS-02.

### 2026-11-23 — pi / coordinator — activation

- Changed: recorded the WS-05 scope decision; status Draft -> Active.
- Evidence: shuuul confirmed WS-05 stays in scope.
- Remaining: all workstreams Pending.
- Blockers: none.
- Next action: claim WS-01 and WS-02 (parallel).

## Completion summary

Not started.
