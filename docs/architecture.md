# Architecture

## Overview

`aimd` is a Rust workspace at the repo root. It ships three binaries through Homebrew
(`aimd`, `aimd-api`, `aimd-mcp`). The interface crates are thin. All routing and
orchestration lives in `aimd-process`, which sits on interface-independent pipeline
crates:

- `aimd-core`: canonical request/response models (`ProcessInput`, `ProcessResult`,
  `TextContext`, `TaskType`, `SourceKind`, `InputRoute`), `AimdError`/`ErrorKind`,
  `ProcessHooks`, Markdown title/chunk shaping, precision parsing, remote-backend
  configuration, extension tables, and interface-owned output persistence (`output`).
- `aimd-doc`: document conversion. EPUB uses hand-written zip reading and an OPF spine
  walk, runs Pandoc per chapter, applies cleanup, and extracts images into a flat
  directory. Other formats go through the Pandoc CLI directly. Text-layer PDFs use the
  `pdf-inspector` crate. `.txt`/`.md` pass through.
- `aimd-url`: URL extraction over the yt-dlp CLI. It handles metadata, the subtitle-first
  transcript, cookie handling with automatic browser probing, metadata ASR context, and
  the audio download. The audio is handed to an `AudioTranscriber` trait.
- `aimd-process`: `process_input`, routing (`route.rs`), the `Runtime` system seam,
  local Qwen3-ASR (ffmpeg decode/segmenting, 8-bit retry), local Unlimited-OCR (images and
  `mutool`-rendered PDF pages), OpenAI-compatible remote ASR/OCR, model alias/precision
  resolution, and HF cache lookup/download.
- `aimd-engine`: tokenizer, mel front-end, and the MLX model graphs and generation loops
  for Qwen3-ASR and Unlimited-OCR.
- `aimd-mlx`: C-ABI shim over MLX C++ (Metal; device selection via `AIMD_MLX_DEVICE`).
- `aimd-weights`: safetensors mmap and the bit-exact MLX affine unpack used as a test
  oracle.
- `aimd-cli`, `aimd-api`, `aimd-mcp`: the CLI (hand-written argument parser), the axum
  HTTP API (jobs, SSE, blobs, sidecar auth), and the rmcp stdio MCP server.

## Dependency Rules

- Interface crates depend on `aimd-core` and `aimd-process`, not on pipeline internals.
- `aimd-doc` and `aimd-url` depend only on `aimd-core`. The URL audio fallback reaches
  ASR through `aimd_url::AudioTranscriber`, which `aimd-process` implements
  (`UrlAudioTranscriber`). This keeps the crate graph acyclic.
- No pipeline crate depends on an interface crate.
- System access in `aimd-process` (env, `PATH`, subprocesses, device preflight, model
  loading, PDF probing) goes through the `Runtime` trait. `NativeRuntime` is the real
  system, and tests substitute fakes.
- Output destinations are interface concerns. `ProcessInput` does not carry
  `output_file`; CLI/API/MCP persist requested outputs through `aimd_core::output`.
- API/MCP request validation and response shaping live in the API/MCP crates.

## Primary Flow

1. CLI/API/MCP receives the request and builds a `ProcessInput`.
2. The interface calls `aimd_process::process_input(&input, &hooks)` (the API and MCP run
   it on a blocking worker).
3. `ensure_supported_input` classifies the source into an `InputRoute { source_kind,
   task_type }`. For PDFs, the pdf-inspector probe sends scanned PDFs and PDFs with
   broken font encodings to OCR.
4. Dispatch:
   - URL -> `aimd-url` (subtitles, else audio download + ASR through the transcriber).
   - `transcript` -> local or remote ASR.
   - `ocr` -> local or remote OCR (images, or PDF pages rendered by `mutool`).
   - `convert` -> `aimd-doc`.
5. Local results are normalized (trailing whitespace stripped per line, runs of three or
   more newlines collapsed). `process_input` then returns the lossless Markdown, its
   asset base, and the shaped `TextContext`.
6. The interface maps `ProcessResult` to its response or output and persists
   `output_file` if requested.

```diagram
╭──────────────╮     ╭──────────────────────╮     ╭───────────────────────╮
│ aimd-cli     │     │ aimd-process         │     │ aimd-url  (yt-dlp)    │
│ aimd-api     │────▶│ process_input()      │────▶│ aimd-doc  (pandoc,    │
│ aimd-mcp     │     │ route + dispatch     │     │   pdf-inspector)      │
╰──────┬───────╯     ╰──────────┬───────────╯     │ asr / ocr / remote    │
       │                        │                 ╰──────────┬────────────╯
       │                        ▼                            ▼
       │             ╭──────────────────────╮     ╭───────────────────────╮
       │             │ aimd-engine          │◀────│ local ASR/OCR         │
       │             │ aimd-mlx (Metal)     │     │ (ffmpeg, mutool)      │
       │             ╰──────────────────────╯     ╰───────────────────────╯
       ▼
╭─────────────────────────╮       ╭─────────────────────────╮
│ aimd_core::output +     │◀──────│ ProcessResult           │
│ interface mapping       │       │ Markdown + asset base   │
│                         │       │ + TextContext           │
╰─────────────────────────╯       ╰─────────────────────────╯
```

`ProcessResult.markdown` is the exact converter output before context splitting.
`TextContext.chunk_list` is the context-window representation and is not a lossless
viewer artifact. `asset_base_uri` takes one of these values:

- the original URL, for URL processing;
- the generated asset directory, for document conversions that emit one;
- the input's parent directory, for other local convert/OCR work;
- `None`, for local transcriptions.

Local directory URIs always end in `/` so standards-based URL joining keeps directory
semantics.

Asset-producing document conversions (`.docx`, `.epub`, `.odt`) create `output_dir`
beside the local source. The directory is durable caller-owned output, not
request-scoped temporary storage. aimd does not clean it up when an interface call
completes. Interfaces therefore ignore a separately requested `output_file` for these
results, so relative links stay valid. Other requested files are written byte-for-byte
from `ProcessResult.markdown`; interfaces must not rebuild them from
`TextContext.chunk_list`.

## Backend And Model Boundaries

| Task | Boundary | Model source |
|------|----------|--------------|
| Transcript | `aimd-url` for URL subtitles/audio; `aimd-process::asr` for transcription | Qwen3-ASR 1.7B/0.6B (`mlx-community/Qwen3-ASR-*-{4,6,8}bit`) on MLX, or remote `/v1/audio/transcriptions` when `AIMD_ASR_BASE_URL` is set. |
| Convert | `aimd-doc` | Pandoc CLI, pdf-inspector, passthrough for `.txt`/`.md`. |
| OCR | `aimd-process::ocr` | Unlimited-OCR (`mlx-community/Unlimited-OCR-{4,6,8}bit`) on MLX, or remote `/v1/chat/completions` when `AIMD_OCR_BASE_URL` is set. |

The README is the user-facing source of truth for `--model` values. Resolution lives in
`aimd-process/src/models.rs`, and the CLI help texts live in `aimd-core/src/output.rs`.
Each checkpoint is loaded once per process and reused. Before loading, local inference
checks that the MLX GPU device is active; on the CPU device it fails with
`BackendUnavailable`.

Remote settings flow through `ProcessInput`, so CLI, API, MCP, local-file work, and the
URL audio fallback share one selection path. Explicit request values override
`AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`. A configured remote URL bypasses the local MLX
preflight. Without one, local inference is used.

For performance expectations and benchmarks, see [Performance](performance.md).
[Local desktop sidecar contract](sidecar.md) describes the authenticated loopback,
path, job, and output-lifetime boundary that desktop clients use.

## Cancellation And Progress

Every pipeline receives `&dyn ProcessHooks`. Pipelines report progress stages
(`extracting`, `downloading`, `transcribing`, `converting`, `recognizing`) with unit
counts when known. They call `check_cancelled()` only at safe boundaries: between ASR
segments, OCR pages, and EPUB chapters. Model calls and subprocesses are never
interrupted. The API maps the hooks onto job records and stages. A cancellation that
arrives during uninterruptible work ends as `completed_after_request` with the artifact
kept.

## Error Handling

Errors are `aimd_core::AimdError { kind, message }` end-to-end. Each pipeline maps
failures at its boundary:

- Missing tools, models, or device (`pandoc`, `yt-dlp`, `ffmpeg`, `mutool`, MLX GPU,
  unsupported bf16 checkpoint) -> `BackendUnavailable`
- Missing input paths -> `InputNotFound`
- Unsupported inputs, options, or rejected cookie options -> `UnsupportedInput`
- Conversion/transcription failures, unknown models -> `ProcessingFailed`
- Cooperative cancellation -> `Cancelled`

HTTP API status codes come from `ErrorKind::status_code`:

- `UnsupportedInput` -> 400
- `InputNotFound` -> 404
- `Cancelled` -> 409
- `BackendUnavailable` -> 422
- `ProcessingFailed` -> 500

The CLI logs the message to stderr and exits non-zero. MCP returns the message as a tool
error.
