# OCR implementation plan

This document tracks the planned OCR expansion for `aimd`. The goal is to add
OCR for scanned PDFs and images without disrupting the existing URL transcript,
local document conversion, API, or MCP flows.

## Goal

Add OCR routing that turns image files and scanned PDF pages into the same stable
`TextContext(title, chunk_list, split_header_level)` output used by the rest of
`aimd`.

Platform-specific OCR backends:

- **macOS / Apple Silicon**: use `mlx4ocr` as the OCR dependency and runtime.
- **Linux**: use a `transformers` backend for OCR-capable vision-language or OCR
  models.

## Non-goals for the first pass

- Do not rewrite the existing transcript or document conversion pipeline.
- Do not add a custom plugin registry; keep routing centralized in `aimd`.
- Do not hide OCR inside the generic convert path when the input requires OCR
  options, page rendering, or platform-specific engine selection.
- Do not change the `TextContext` contract.
- Do not require users to install both macOS and Linux OCR stacks.

## Dependency shape

Keep OCR dependencies platform-scoped so `aimd-tool[all]` remains installable on
the target platform without pulling unusable OCR runtimes.

Planned dependency split:

```toml
dependencies = [
    "mlx4ocr>=0.1.2; sys_platform == 'darwin'",
    "transformers>=4.57.6; sys_platform == 'linux'",
    "torch>=2.9.1; sys_platform == 'linux'",
]
```

Notes:

- `mlx4ocr` already provides the macOS CLI/runtime surface for MLX OCR.
- `transformers` is already part of the current package for ASR-related Linux
  model support, but OCR should still have its own engine boundary.
- If the Linux OCR model needs additional runtime packages, add them only after
  selecting and smoke-testing the model.
- Keep `mlx4ocr` out of Linux resolution and keep Linux OCR-only packages out of
  Darwin resolution.

## Public interface

Expose OCR through the unified `aimd <input>` CLI. Users should not have to pick
a task; routing decides how to produce Markdown from the input.

### CLI

Proposed examples:

```bash
aimd scan.pdf
aimd page.png
aimd scan.pdf --engine auto
aimd scan.pdf --engine mlx4ocr      # macOS only
aimd scan.pdf --engine transformers # Linux only
aimd scan.pdf --model paddleocr_v6  # default macOS OCR model
aimd scan.pdf --model glm_ocr       # optional mlx4ocr VLM backend
aimd scan.pdf --model paddleocr_vl  # optional mlx4ocr VLM backend
```

Initial engine values:

- `auto`: choose the platform default.
- `mlx4ocr`: require macOS and the `mlx4ocr` dependency.
- `transformers`: require Linux and the selected Transformers OCR model stack.

Initial macOS OCR model values:

- `paddleocr_v6`: default; maps to mlx4ocr `ppocrv6` with the `medium` variant.
- `glm_ocr`: maps to mlx4ocr `glm-ocr` and requires optional VLM dependencies.
- `paddleocr_vl`: maps to mlx4ocr `paddleocr-vl` and requires optional VLM
  dependencies.

### API / MCP

Expose the same OCR task through existing request models rather than adding a
separate endpoint/tool unless the OCR input contract becomes substantially
different.

Request fields to consider:

- `task_type="ocr"`
- `engine="auto" | "mlx4ocr" | "transformers"`
- `model`: optional backend-specific model id.
- `start` / `end`: optional 0-based inclusive PDF page range.
- `language`: optional hint if the selected backend supports it.

## Routing changes

1. Extend `TaskType` with `ocr`.
2. Extend input classification for OCR-capable sources:
   - image files: `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`
   - scanned PDFs: `.pdf` without extractable text, detected before generic PDF
     conversion when local PDF text-layer inspection is available
3. Route OCR inputs to `src/aimd/ocr/processor.py`.
4. Keep normal document conversion as the default for PDFs that have an
   extractable text layer.

## Engine boundary

Create a small OCR engine boundary inside `src/aimd/ocr/`:

```text
src/aimd/ocr/
├── processor.py       # task orchestration and TextContext wrapping
├── engines.py         # engine resolution / availability checks
├── mlx4ocr_engine.py  # macOS adapter
└── transformers_engine.py # Linux adapter
```

The boundary should return a simple internal result shape, for example:

```python
@dataclass(frozen=True)
class OCRPage:
    page_index: int | None
    text: str

@dataclass(frozen=True)
class OCRResult:
    title: str
    pages: tuple[OCRPage, ...]
```

Then `processor.py` converts pages into `TextContext` chunks.

## macOS backend: `mlx4ocr`

Implementation outline:

1. Add a platform-conditional dependency on `mlx4ocr`.
2. Import `mlx4ocr` lazily inside the macOS engine so non-macOS imports do not
   fail.
3. For images, call the Python API directly for `paddleocr_v6` and use VLMOCR
   for `glm_ocr` / `paddleocr_vl` when optional dependencies are installed.
4. For PDFs, call the `mlx4ocr` CLI so PDF rendering and page-range handling stay
   inside the OCR backend.
5. Normalize output into page text and then `TextContext`.
6. Add a fail-fast error if `mlx4ocr` is requested on non-Darwin platforms.

## Linux backend: `transformers`

Implementation outline:

1. Select one default OCR-capable Transformers model for Linux.
2. Keep model loading lazy and fail fast with a clear error when dependencies or
   model support are missing.
3. For images, run the selected processor/model pair and return plain text.
4. For PDFs, render pages to images before OCR.
5. Normalize generated OCR text into the same internal `OCRResult` shape used by
   the `mlx4ocr` engine.
6. Add an explicit `--model` escape hatch after the default model is proven.

Open Linux model decision:

- Choose the default after a smoke test on CPU and, if available, CUDA.
- Prefer a model with reasonable CPU fallback and standard `transformers`
  loading APIs.
- Document model size and first-run download behavior.

## Output formatting

For single images:

```markdown
recognized text...
```

For PDFs or multi-page inputs:

```markdown
## Page 1

recognized text...

## Page 2

recognized text...
```

Chunking should reuse the existing markdown/TextContext conventions rather than
introducing an OCR-specific result contract at adapter boundaries.

## Tests

Start with fast tests before model-heavy tests:

1. Routing tests for image files and scanned PDF OCR detection.
2. Engine resolution tests for Darwin/Linux/unsupported platform behavior.
3. Processor tests with fake OCR engines returning deterministic page text.
4. CLI/API/MCP payload tests for OCR options and engine/model mapping.
5. Dependency/import tests to ensure platform-specific engines import lazily.
6. `mlx4ocr` command-building tests that do not download model weights.
7. Optional smoke tests gated by markers for real `mlx4ocr` and Transformers OCR
   inference.

Do not make the default test suite download OCR model weights.

## Implementation order

1. Add request/routing model support for `TaskType.OCR`.
2. Add OCR processor with fake/injected engine tests.
3. Add macOS `mlx4ocr` engine and local image/PDF smoke tests.
4. Add PDF page handling and page-range support.
5. Add Linux Transformers engine after selecting a default model.
6. Expose OCR through CLI/API/MCP consistently.
7. Update README install/usage docs and release notes.
8. Add optional real-backend smoke tests that are not part of the default CI
   path unless model caches are controlled.
