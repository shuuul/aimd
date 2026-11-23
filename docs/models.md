# Model implementation

See the [README](../README.md#models) for model selection and precision.

## Local model behavior

The user-facing model selection contract separates model family/size from
precision:

```bash
aimd audio.wav --model qwen3-asr-1.7b --precision 4bit
aimd audio.wav --model qwen3-asr-0.6b --precision 8bit
aimd scan.pdf --model unlimited-ocr --precision 8bit
```

Supported model families are intentionally narrow:

- **ASR**: `qwen3-asr-1.7b` (default) and `qwen3-asr-0.6b`. Omitted precision defaults
  to `4bit` and resolves to `mlx-community/Qwen3-ASR-{1.7B,0.6B}-{4bit,6bit,8bit}`.
  Omitted `language` means auto-detect (no forced language in the prompt) for local
  ASR, every segment, the 8-bit retry, and the URL audio fallback; URL metadata and
  title heuristics steer subtitle choice only. Explicit codes come from
  `aimd_engine::tokenizer::ASR_LANGUAGES` (30 languages).
- **Unlimited-OCR vision + R-SWA** (v1 and v2 share the towers): the numerical source of
  truth is Baidu's `deepencoder.py` / `modeling_deepseekv2.py`, not mlx-vlm. CLIP uses
  `quick_gelu` and LayerNorm eps 1e-5; crop pos embeds resample the CLIP grid spatially
  (bicubic antialias) and SAM rel pos with the half-pixel linear index. v1's ring cache
  keeps the whole prompt (`KvCache::retain_prompt`), then a 128-token decode ring.
- **OCR v1**: `unlimited-ocr`, resolving to `mlx-community/Unlimited-OCR-{4bit,6bit,8bit}`
  (default 4bit). OCR uses single-image gundam mode (`document parsing.`), the
  sliding-window no-repeat n-gram guard (35/128), and `max_tokens=8192` per page.
  PDFs go in 5-page `recognize_pages` windows (upstream `infer_multi`: one 1024 base
  view per page, `Multi page parsing.`, no `<PAGE>` in the prompt). v1 writes `<PAGE>`
  as 3 text tokens; the page-count guard matches that sequence and substitutes it for
  an EOS before the last page. A window of one page keeps the gundam request. v1
  matched page-by-page output through 6 pages; 7+ loop or repeat page 1 ([Spec 008](../specs/archive/008-unlimited-ocr-2.md)).
- **Unlimited-OCR-2**: `unlimited-ocr-2`, resolving to
  `mlx-community/Unlimited-OCR-2-{4bit,6bit,8bit}` (default 4bit). The default OCR model (omitted `model`);
  a separate model from v1, which stays `unlimited-ocr`. The decoder uses DASH block-sparse
  attention (block 32, local window 128, top-k 12 routed blocks, entmax15 prior, page2d
  RoPE) through the `aimd_mlx::dash` shim ops: custom Metal kernels, the same block
  selection from mlx gathers elsewhere. Never fall back to dense attention; the dense
  `DashMode` is a test oracle only. Prompt: `[bos] + ([<PAGE>] + <image>*n)` per page +
  `Multi page parsing.`. Every page is gundam-tiled (max 16 crops), single images and
  multi-page requests alike (mlx-vlm's default; upstream `infer` uses base mode for
  several pages, which misreads small text and loops on long documents). PDFs go through `OcrModel::recognize_pages` in 10-page windows
  (`pages_per_request`), with progress and cancellation between windows. Generation
  carries the upstream page-count guard (EOS banned until every `<PAGE>`, forced past
  it, 20000 tokens per page) and widens the n-gram window to 1024 for multi-page
  requests. `markdown2::render_unlimited_ocr2` turns the 0..999 page-relative boxes
  into per-page located items.
- **PP-OCRv6**: `pp-ocrv6` (`pp-ocrv6-medium`), medium tier only. Model key
  `PaddlePaddle/PP-OCRv6_medium` names the `_det_safetensors` / `_rec_safetensors`
  repo pair. The HF rec checkpoint's `head.encoder.conv_block.{0,1}` are Paddle's
  `skip_conv` / `conv_reduce` swapped; the loader swaps them back (never download the
  Paddle `.pdparams` pickles). Output: lines scoring >= 0.5 in reading order. Parity
  with mlx4ocr is bit-exact from decoded pixels on; JPEG decoding (zune-jpeg vs
  libjpeg-turbo) differs by a few levels.
- **bf16** resolves but is rejected by the native gate for Qwen3-ASR and Unlimited-OCR 1/2 (`ensure_native_asr` /
  `ensure_native_ocr`).
- Explicit `mlx-community/*` ids for these families remain compatibility inputs; unknown
  models fail with the "Unsupported MLX ... model" `ProcessingFailed` message.

Checkpoints load once per model per process (MCP reuses loaded pipelines).

## ASR context and retries

- **ASR context biasing**: `ProcessInput.context` (explicit text) and
  `ProcessInput.metadata_context` (default `true`) are forwarded by CLI
  (`--context`/`--no-context`) and MCP. `aimd::url` builds context from page
  metadata (title/author/description/tags/chapters, capped at 2000 chars) and merges it
  into the audio fallback. Qwen3-ASR consumes it as the system message. Never feed
  first-pass ASR output back as context.

- **4-bit retry**: a 4-bit Qwen3-ASR segment that trips the repetition-loop detector is
  retried once with the matching 8-bit checkpoint. Other precisions and remote ASR do not
  auto-switch.

## Remote inference

- **Remote inference**: `AIMD_ASR_BASE_URL` / `AIMD_OCR_BASE_URL` (or the explicit
  CLI/MCP fields) opt into OpenAI-compatible HTTP backends and bypass local MLX
  preflight. Optional `AIMD_{ASR,OCR}_{MODEL,API_KEY}` have matching fields. Bare server
  URLs gain `/v1`. Explicit values override the environment. Remote precision is
  server-owned, so a local `precision` is ignored with one warning. Remote OCR picks
  the instruction from the served model name: `Unlimited-OCR-2*` gets
  `<PAGE><image>\nMulti page parsing.` and `render_unlimited_ocr2`; anything else
  gets v1's `<image>\ndocument parsing.`.

## MLX runtime

- **Fused elementwise ops**: shapeless compiled shim ops (`swiglu`, `gelu`,
  `bias_act`) run on every backend and must match
  the unfused graph value for value (tested in `aimd-mlx/tests`). Do not
  compile whole models shape-keyed: every new page/clip shape would retrace
  (spec 006).

- **Host/GPU overlap** (spec 007): decode loops submit step k+1
  (`async_eval`) before reading token k back, like mlx-lm `generate_step`.
  The shim defaults `MLX_MAX_MB_PER_BUFFER` to 30 on every backend (a user
  value wins). Larger Metal command buffers keep the previous step's KV arrays
  alive, so `slice_update` copies the cache instead of donating it (MoE
  Unlimited-OCR lost 15%). Host-side image work threads through
  `aimd_engine::parallel` (`for_row_bands`, ordered `map`), computing each
  element exactly as one thread would. PDF OCR renders pages on a
  background thread (`PdfRenderJob`, PPM for local engines, PNG for remote)
  at most two pages ahead.

- **Buffer memory**: mlx keeps freed buffers for reuse and reuses one only
  for a similar size, so shape-varying calls (PP-OCRv6 rec batch widths,
  image crops) used to grow the cache to 25 GB. Three automatic layers bound
  memory; models must not depend on hand-placed `clear_cache` calls: the
  30 MB `MLX_MAX_MB_PER_BUFFER` above also bounds the temporaries a pending
  buffer holds (mlx's CUDA default held a whole vision tower: 11 GB per
  Unlimited-OCR page, 4 GB at 30 MB, same speed), the shim
  caps the cache at 1 GB on every device (`set_cache_limit` at device
  selection), and `aimd::native` wraps every native model in `Released`,
  which clears the cache after each call so a long-lived MCP process keeps
  only the weights between requests (Metal hands the pages back a few
  seconds later). `aimd_mlx::memory_stats()` reads the counters,
  `aimd-mlx/tests/mlx_memory.rs` guards the cap, and `corpus_gate` prints
  each model's MLX peak; check it when adding a model.

- **Linux build**: `aimd-mlx` links `libmlx.so` from `MLX_PREFIX` (brew `mlx-cuda`, or
  an unpacked `mlx-cuda-13` wheel for dev). Cargo applies the shim's rpath only to
  `aimd-mlx`'s own targets, so dev runs of dependents need
  `LD_LIBRARY_PATH=$MLX_PREFIX/lib` (brew's linker shim adds the rpath for installs).
  The shim points `CUDA_HOME` at `/opt/cuda` when unset and `/usr/local/cuda` is
  absent, because mlx's CUDA JIT needs the toolkit headers. It also caps mlx's memory
  limit at 90% of the VRAM free at device selection. mlx sizes that limit from total
  VRAM, so without the cap its buffer cache runs a shared GPU out of memory.
