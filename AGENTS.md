# aimd repository guidance

`aimd` converts URLs, media, and documents to Markdown and chunked text context.
The Rust CLI (`aimd-cli`) and rmcp stdio server (`aimd-mcp`) call
`aimd::process_input`. `aimd` owns routing, adapters, and orchestration.
`aimd-engine` owns model graphs and preprocessing. `aimd-mlx` wraps MLX C++.
The default `native` feature enables local GPU inference. The library also
supports document and URL processing and remote inference without MLX.

- [Architecture and code map](docs/architecture.md)
- [Model and runtime implementation](docs/models.md)
- [Development, testing, and releases](docs/development.md)
- [Performance and accuracy measurements](docs/performance.md)
- [User-facing usage and supported models](README.md)

## Processing contracts

- Keep routing and orchestration in `aimd`. `doc` and `url` are independent
  adapters. Use `System` for environment lookup, executable lookup, and
  subprocesses. `Runtime` extends it for model loading, preflight, and PDF
  probes. Run `cargo xtask check-modules` after changing module boundaries.
- Use `InputRoute { source_kind, task_type }`. URLs default to `transcript`.
  `readable` is opt-in for URLs and `.html`/`.htm` files. For URL transcripts,
  try subtitles in language and format preference order, then audio and ASR.
  Preserve automatic browser-cookie probing when no cookie option was supplied.
  Reject invalid explicit cookie options and report failure if neither path
  produces text. Test changes to this fallback order.
- Return a typed `AimdError`. Use `BackendUnavailable` for missing tools,
  checkpoints, or GPUs, `InputNotFound` for missing paths, `UnsupportedInput`
  for unsupported inputs or options, `ProcessingFailed` for conversion failures,
  and `Cancelled` for cooperative cancellation.
- Resolve models and check required tools and the MLX GPU before rendering,
  downloading, or transcoding. A configured remote backend bypasses MLX preflight.
- Thread `ProcessHooks` through processing. Check cancellation at safe boundaries
  between segments, pages or page windows, and chapters. Preserve work that
  completes during an uninterruptible model call or subprocess.
- CLI and MCP own output paths and persist through `aimd::output`. Write
  `ProcessResult.markdown` byte-for-byte and preserve `TextContext.chunk_list`
  for context consumers. Asset-producing conversions own their `output_dir`
  and ignore `output_file`. Directory asset URIs end in `/`.
- Keep MCP JSON field names, error bodies, and tool schemas wire-compatible.
  Changes require an explicit decision. With `anchors` and `include_segments`
  both disabled, preserve byte-for-byte Markdown and skip forced alignment.

## Models and dependencies

- Keep canonical model aliases kebab-case. The defaults are `qwen3-asr-1.7b`
  and `unlimited-ocr-2`. Preserve `unlimited-ocr` for v1 and accept legacy aliases
  and supported explicit checkpoint IDs. The README lists supported families.
- Keep precision separate from the model alias and forward it through CLI, MCP,
  URL audio fallback, ASR, and OCR. Qwen3-ASR and Unlimited-OCR 1/2 run at 4-bit
  (default), 6-bit, and 8-bit. Native `bf16` returns `BackendUnavailable`.
  PP-OCRv6 and remote inference ignore local precision with one warning.
  Reject precision that conflicts with an explicit checkpoint ID.
- Forward ASR context and the metadata toggle through CLI, MCP, and URL audio
  fallback. Preserve language auto-detection when omitted. Retry only a local
  4-bit segment flagged for repetition, once, with the matching 8-bit
  checkpoint. Keep first-pass ASR text out of the context.
- Use MLX for local compute. Gate custom Metal kernels on
  `aimd_mlx::metal_kernels_available()` and use MLX ops elsewhere. Keep DASH
  block-sparse on all backends. Preserve early submission of the next decode
  step and automatic buffer bounds. Use shapeless compilation for fused
  elementwise ops and keep results identical to the unfused graph.
- Hand-write simple parsers and helpers. Justify new dependencies for complex
  domains and get coordinator sign-off before adding shared workspace versions.
  Install external tools through Homebrew dependencies. Firecrawl Local stays
  a subprocess because it is AGPL-3.0.

## Verification and specs

Before each commit, run the hooks and Rust gates. Review any hook fixes and
include them in the commit. Use Conventional Commits.

```bash
prek run --all-files
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
AIMD_MLX_DEVICE=cpu cargo test --workspace
```

- Keep hermetic and checkpoint tests offline. Load each checkpoint once per
  test binary through a shared lazy fixture. Gate real tools and end-to-end
  tests behind `AIMD_E2E=1`. Use short polling intervals for test synchronization.
- Extend existing golden tests and regenerate fixtures with their exporters.
  Keep committed fixtures at most 1 MB each. The subtitle fixtures are frozen
  and require a reviewed behavior change to update.
- Run the corpus gate for changes that can affect model outputs. Check
  checkpoint parity and the corpus on a Linux GPU for changes that can affect
  CUDA. Keep the hermetic suite around two minutes and use a separate
  `CARGO_TARGET_DIR=target-<name>` for each concurrent worker.
- Track multi-workstream, multi-agent, and handoff-heavy work in `specs/`.
  Reserve and index a spec before parallel work starts. Follow the lifecycle in
  [specs/README.md](specs/README.md) and run `cargo xtask check-specs` after changes.
- The workspace version comes from root `Cargo.toml`. Follow the
  [release process](docs/development.md#release-process) for tag-driven releases.
