# Architecture

## Overview

The workspace ships `aimd` and `aimd-mcp` through Homebrew. Both interfaces build
a `ProcessInput` and call the `aimd` library. The CLI owns argument parsing and
the MCP crate owns the rmcp stdio server, request validation, and response shaping.

- `aimd` holds request and result types, Markdown shaping, routing, format adapters,
  local and remote orchestration, checkpoint resolution, and output helpers.
- `aimd-engine` owns the safetensors store, tokenizer, audio and image preprocessing,
  and model graphs and generation for Qwen3-ASR, Qwen3-ForcedAligner,
  Unlimited-OCR 1/2, and PP-OCRv6.
- `aimd-mlx` wraps the MLX C++ API. It runs on Metal on macOS and the MLX CUDA
  backend on Linux. The bit-exact quantization oracle lives in its tests.
- `xtask` validates specs and module imports.

## Dependency Rules

`cargo xtask check-modules` checks `use crate::` imports in `aimd`. The interface
crates call the public `aimd` API. The `model`, `error`, `hooks`, `output`, `log`,
`text`, and `anchors` modules share the request and result contract. They import
other contract modules, and `text` shares Python-compatible string helpers with
`util::pytext`. `sys` and `util` depend on the contract. `doc` and `url` use the
contract, `sys`, `util`, and `ext`, with independent adapter implementations. The
orchestration modules can use any of these modules.

`sys::System` provides environment lookup, executable lookup, and subprocess
capture. The document and URL adapters accept this interface for Pandoc, yt-dlp,
Firecrawl Local, and Defuddle. `Runtime` extends `System` with MLX preflight,
model loading, and PDF probing. `NativeRuntime` provides the real system for the
CLI and MCP. The `url` module's internal `AudioTranscriber` lets the orchestrator
supply ASR for the audio fallback. Adapter tests substitute `System` to verify
tool arguments and control responses.

CLI and MCP own output destinations and persist requested files through
`aimd::output`. `ProcessInput` carries source processing options.

## Primary Flow

1. CLI/MCP receives the request and builds a `ProcessInput`.
2. The interface calls `aimd::process_input(&input, &hooks)` (MCP runs
   it on a blocking worker).
3. `ensure_supported_input` classifies the source into an `InputRoute { source_kind,
   task_type }`. For PDFs, the pdf-inspector probe sends scanned PDFs and PDFs with
   broken font encodings to OCR.
4. Dispatch:
   - URL transcript requests use `aimd::url` for subtitles, then audio download and ASR.
   - `readable` uses Firecrawl Local for URLs and `dom_smoothie` for files. Setting
     `AIMD_READABLE_BACKEND=defuddle` selects Defuddle for both.
   - `transcript` -> local or remote ASR.
   - `ocr` -> local or remote OCR (images, or PDF pages rendered by `mutool`).
   - `convert` uses `aimd::doc` for Pandoc, EPUB, PDF, and text passthrough.
5. Local results are normalized (trailing whitespace stripped per line, runs of three or
   more newlines collapsed). `process_input` then returns the lossless Markdown, its
   asset base, and the shaped `TextContext`.
6. The interface maps `ProcessResult` to its response or output and persists
   `output_file` if requested.

```mermaid
flowchart TD
    CLI[aimd-cli] --> Route
    MCP[aimd-mcp] --> Route
    subgraph Aimd[aimd library]
        Route[process_input and route] --> Doc[doc: Pandoc, EPUB, pdf-inspector]
        Route --> Url[url: readable HTML, subtitles and audio]
        Route --> Inference[ASR and OCR]
        Url -->|AudioTranscriber| Inference
        Doc --> Sys[System: env, PATH and subprocesses]
        Url --> Sys
        Inference --> Runtime[Runtime: System, model loading and PDF probing]
        Runtime --> Sys
        Inference --> Remote[OpenAI-compatible HTTP]
    end
    Runtime -->|native| Engine[aimd-engine]
    Engine -->|mlx| MLX[aimd-mlx]
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
| Transcript | `aimd::url` for URL subtitles/audio; `aimd::asr` for transcription | Qwen3-ASR 1.7B/0.6B (`mlx-community/Qwen3-ASR-*-{4,6,8}bit`) on MLX, or remote `/v1/audio/transcriptions` when `AIMD_ASR_BASE_URL` is set. |
| Convert | `aimd::doc` | Pandoc CLI, pdf-inspector, passthrough for `.txt`/`.md`. |
| OCR | `aimd::ocr` | Unlimited-OCR 1/2 or PP-OCRv6 on MLX, or remote `/v1/chat/completions` when `AIMD_OCR_BASE_URL` is set. |

The README is the user-facing source of truth for `--model` values. Resolution lives in
`crates/aimd/src/models.rs`, and the CLI help texts live in `crates/aimd/src/output.rs`.
Each checkpoint is loaded once per process and reused. Before loading, local inference
checks that the MLX GPU device is active; on the CPU device it fails with
`BackendUnavailable`.

Remote settings flow through `ProcessInput`, so CLI, MCP, local-file work, and the
URL audio fallback share one selection path. Explicit request values override
`AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`. A configured remote URL bypasses the local MLX
preflight. Without one, local inference is used.

The `aimd` library enables `native` by default, which enables the engine's `mlx`
feature and local model loading. With `default-features = false`, the library
builds and runs without MLX and reports `BackendUnavailable` for local ASR and
OCR. Document conversion, readable HTML, URL subtitles, and remote ASR and OCR
are available in this build. The shipped binaries use the default features.

HTTP connections (including proxy negotiation and TLS handshakes) have a 15-second
budget, capped by the caller's total timeout. Remote inference keeps its 600-second
total budget with no separate response-header deadline: non-streaming servers may
not send headers until generation finishes. This requires ureq 3.5 or later.

ASR segmentation owns a unique temporary subdirectory for each request, including
when `--temp-dir` is supplied. Only that directory is removed on completion or
failure; same-named inputs cannot overwrite another request's segments.

HF cache hits require configuration, tokenizer files (or the PP-OCRv6 recognition
character table), and weights. A safetensors index requires every referenced shard.
Partial snapshots are repaired by downloading missing files when online; offline
requests fail explicitly rather than returning an incomplete checkpoint.

For performance expectations and benchmarks, see [Performance](performance.md).

## Cancellation And Progress

Every pipeline receives `&dyn ProcessHooks`. Pipelines report progress stages
(`extracting`, `downloading`, `transcribing`, `converting`, `recognizing`) with unit
counts when known. They call `check_cancelled()` only at safe boundaries: between ASR
segments, OCR pages or page windows, and EPUB chapters. Model calls and subprocesses are never
interrupted; a cancellation that arrives during uninterruptible work keeps the
completed artifact instead of discarding it.

## Error Handling

Errors are `aimd::AimdError { kind, message }` end-to-end. Each pipeline maps
failures at its boundary:

- Missing tools, models, or device (`pandoc`, `yt-dlp`, `ffmpeg`, `mutool`, MLX GPU,
  unsupported bf16 checkpoint) -> `BackendUnavailable`
- Missing input paths -> `InputNotFound`
- Unsupported inputs, options, or rejected cookie options -> `UnsupportedInput`
- Conversion/transcription failures, unknown models -> `ProcessingFailed`
- Cooperative cancellation -> `Cancelled`

The CLI logs the message to stderr and exits non-zero. MCP returns the message as a tool
error.

## Adapter and output details

- **Source anchors** (spec 005): `ProcessInput.anchors` (default `false`; CLI
  `--anchors`, MCP `anchors`; `--no-anchors` is still accepted as a no-op) adds OCR
  `<!-- aimd:loc page=P bbox=x0,y0,x1,y1 kind=K -->` comments (1-based page, bbox
  normalized 0..1, 3 decimals) and `[hh:mm:ss] ` transcript paragraphs to `markdown`.
  `ProcessInput::wants_positions()` (`anchors || include_segments`) gates forced
  alignment and `ProcessResult.segments`; CLI `--format json/srt/vtt` set
  `include_segments`. With both off the output must match the pre-anchor output byte
  for byte and skip forced alignment. Positions come through the
  `OcrModel::recognize_located` / `AsrModel::transcribe_aligned` seams (defaults: none);
  local ASR aligns 295 s segments (the aligner caps at 300 s; ffmpeg overshoots a 300 s
  cut by milliseconds) with Qwen3-ForcedAligner-0.6B-8bit, loaded lazily once per process.
  It covers zh, yue, en, de, es, fr, it, pt, ru and ko (not Japanese). Anything unaligned
  (remote ASR, unsupported language, failed alignment) spreads its sentences over the
  segment span instead of failing. JSON segment coordinates round to 3 decimals, times
  to ms. `include_segments` (MCP; CLI `--format json`)
  returns `segments`. MCP response fields are additive (`segments` last).

- **URL cookie behavior**: keep automatic browser-cookie probing when no cookie option is
  supplied; it is intentional user convenience for restricted media. Explicit cookie
  options that yt-dlp rejects fail fast (`UnsupportedInput`). Subtitle and audio
  fallback failures must not be hidden behind metadata-only success results.

- **URL fallback ordering**: subtitles first (language preference order, then formats),
  then audio download + ASR. If both produce no text, the request is a processing
  failure. Stale-extractor yt-dlp stderr gets the `brew upgrade yt-dlp` hint.

- **Document tools**: Pandoc (brew) reads every supported document format; `.txt`/`.md`
  pass through; legacy `.doc`/`.ppt`/`.xls` and MOBI/AZW3 are unsupported. Text-layer
  PDFs use the `pdf-inspector` crate; scanned pages render with `mutool draw`.
  Pandoc 3.12 rejects xlsx workbooks whose rels use absolute targets (openpyxl
  output); Excel-style relative targets convert.

- **Readable HTML**: `TaskType::Readable` (CLI `--task readable`, MCP
  `task_type: "readable"`) goes through `aimd::url` via `System`. URLs run `firecrawl-local scrape <url> --format
  markdown` (`AIMD_FIRECRAWL` override) and parse its JSON envelope; `success: false` or
  empty markdown (blocked pages carry a `warning`) is `ProcessingFailed`, a missing binary
  `BackendUnavailable`. Firecrawl Local does not read files, so local HTML uses
  `dom_smoothie`. `AIMD_READABLE_BACKEND=defuddle` routes URLs and files to `npx defuddle
  parse` (`AIMD_NPX`); Defuddle is optional and not a brew dependency.

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Input routing | `crates/aimd/src/route.rs` | `get_input_route` / `ensure_supported_input` map source kind + task into an `InputRoute`; `parse_task`. |
| Orchestration | `crates/aimd/src/runtime.rs` | `process_input` / `process_input_with(request, hooks, &dyn Runtime)`; local-file dispatch, MarkItDown-style normalization, `TextContext` wrapping. |
| System seam | `crates/aimd/src/sys.rs`, `runtime.rs`, `native.rs` | `System` handles env, `PATH` and subprocesses. `Runtime` extends it with preflight, model loading and PDF probing. |
| PDF/OCR routing | `crates/aimd/src/native.rs` | pdf-inspector probe: scanned or `has_encoding_issues` PDFs route to OCR. |
| Model + precision resolution | `crates/aimd/src/models.rs` | Alias/precision to model keys; `ocr_family` decodes them for `NativeRuntime::load_ocr`; native gates; 8-bit fallback; ASR language codes. |
| Precision parsing | `crates/aimd/src/precision.rs` | Normalizes `4bit`/`6bit`/`8bit`/`bf16` and dash/space/case variants. |
| HF checkpoints | `crates/aimd/src/hf.rs` | Cache lookup (`HF_HUB_CACHE` > `HF_HOME/hub` > `~/.cache/huggingface/hub`), HTTPS snapshot download, `HF_TOKEN`, `HF_ENDPOINT`, `HF_HUB_OFFLINE`. |
| Local ASR | `crates/aimd/src/asr.rs`, `audio.rs` | ffmpeg decode to 16 kHz mono, >600 s segmenting, context, 4-bit -> 8-bit repetition retry. |
| Local OCR | `crates/aimd/src/ocr.rs` | Images and `mutool draw` PDF pages rendered in the background (`PdfRenderJob`); progress and cancellation between pages or page windows. |
| Remote backends | `crates/aimd/src/remote_config.rs`, `crates/aimd/src/remote.rs` | `AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`, `/v1` suffix; ureq + rustls. |
| Request/result models | `crates/aimd/src/model.rs` | `ProcessInput`, `ProcessResult`, `TextContext`, `TaskType`, `SourceKind`; serde names are the wire format. |
| Errors | `crates/aimd/src/error.rs` | `AimdError { kind, message }`, `ErrorKind` with a stable snake_case name. |
| Cancellation/progress | `crates/aimd/src/hooks.rs` | `ProcessHooks` (`is_cancelled`, `progress`, `check_cancelled`), `NoHooks`. |
| Text shaping | `crates/aimd/src/text.rs` | Title extraction and header/paragraph chunking (`MAX_CHUNK_SIZE`). |
| Source anchors | `crates/aimd/src/anchors.rs` | OCR `<!-- aimd:loc -->` comments, sentence split, unit-to-sentence timing, `[hh:mm:ss]` paragraphs, subtitle cues, SRT/VTT write/parse. |
| Output persistence + help texts | `crates/aimd/src/output.rs` | `persist_result_output_if_requested`; `OutputFormat`, `result_payload`/`payload_text` (MCP response, CLI `--format json`), `subtitle_text`; `MODEL_HELP_TEXT`, `PRECISION_HELP_TEXT`, `CONTEXT_HELP_TEXT`. |
| Extension tables | `crates/aimd/src/ext.rs` | Audio/video/image/OCR/asset extensions and the Pandoc reader map. |
| CLI | `crates/aimd-cli/src/args.rs`, `app.rs` | Hand-written parser over the frozen flag surface; output naming; `.env` loading. |
| MCP server | `crates/aimd-mcp/src/lib.rs` | Tool schemas, argument validation, `spawn_blocking` processing. |
| Document conversion | `crates/aimd/src/doc/` | `pandoc.rs`, `epub.rs` + `zip.rs` (hand-written), `clean.rs`, `pdf.rs` (pdf-inspector). |
| URL conversion | `crates/aimd/src/url/` | `ytdlp.rs`, `subtitles.rs`, `cookies.rs`, `metadata.rs` (`build_metadata_context`, 2000 chars), `audio.rs`, `readable.rs`. |
| Model graphs | `crates/aimd-engine/src/models/{qwen3_asr,qwen3_aligner,unlimited_ocr,pp_ocrv6}/` | `unlimited_ocr` also holds v2 (`dash.rs`, `markdown2.rs`). MLX graphs, generation loops, n-gram guard, `test_support.rs` shared fixtures. PP-OCRv6 host ops (cv2-exact resize, contours, min-area rect, unclip, perspective crop, CTC) are hand-written in `pp_ocrv6/{image_ops,db,ctc}.rs`. |
| MLX shim | `crates/aimd-mlx/{build.rs,shim/,src/lib.rs}` | Links brew mlx (`MLX_PREFIX` override); `AIMD_MLX_DEVICE`. `src/dash.rs`: DASH router/prefill/decode ops. |
| Golden corpus | `scripts/corpus/`, `scripts/{build_corpus,corpus_baseline,corpus_baseline_paddle}.py`, `crates/aimd-engine/examples/corpus_gate.rs` | Labeled ASR/OCR accuracy gate vs the mlx-audio / mlx-vlm / mlx4ocr reference. `--ocr-model` picks the OCR family. `scripts/omnidoc_{build,score}.py` compare OCR models with ground truth on an OmniDocBench subset. [Performance](performance.md) |
| Tracked specs | [`specs/`](../specs/README.md) | Lifecycle, numbering, and index. |
