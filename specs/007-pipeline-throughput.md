---
id: "007"
title: "End-to-end throughput: host/GPU overlap, pipelining, CPU preprocessing"
status: Active
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 007 — End-to-end throughput: host/GPU overlap, pipelining, CPU preprocessing

## Context

Spec 006 fused kernels inside the new models. A read-only survey of the rest
of the pipeline (2026-10-26) found that most remaining time is outside the
kernels:

- The Qwen3-ASR (`qwen3_asr/pipeline.rs`), PaddleOCR-VL
  (`paddleocr_vl/generate.rs`) and Unlimited-OCR (`unlimited_ocr/generate.rs`)
  decode loops build the next step's graph but submit it (`async_eval`) only
  after reading the current token back. MLX encodes at eval time, so the GPU
  idles while the host encodes each step (~1.8 ms of a ~3.1 ms PaddleOCR-VL
  4-bit step, measured in spec 006). mlx-lm submits the next step before
  `y.item()`.
- `aimd-process/src/ocr.rs` renders every PDF page with `mutool draw` to PNG
  before OCR starts; PNG compression and decode sit on the critical path.
- PP-OCRv6 recognition evaluates and reads back each 6-crop batch before
  preparing the next (`pp_ocrv6/pipeline.rs`); crops, perspective warps,
  resizes and normalization are single-threaded.
- The Pillow-exact bicubic resize shared by PaddleOCR-VL and Unlimited-OCR
  (`unlimited_ocr/preprocess.rs`) is single-threaded.
- Qwen3-ASR and the aligner read audio features back to the host, merge the
  prompt embeddings on the host and re-upload them.
- Long audio is fully segmented by ffmpeg before any transcription, and each
  segment WAV is parsed up to three times.
- PaddleOCR-VL decodes the page image twice when spotting; the aligner runs
  `lm_head` on every position; PaddleOCR-VL quantizes at load with one eval
  per weight.
- All decoding runs at batch size 1.

The user asked for all of these, plus any further optimization found along
the way.

## Goal and success criteria

- [ ] Decode tok/s improves for Qwen3-ASR, Unlimited-OCR and PaddleOCR-VL
      (bf16 and 4-bit) with identical outputs.
- [ ] Multi-page scanned PDF wall time improves for every OCR model
      (render/OCR overlap), with identical Markdown.
- [ ] PP-OCRv6 per-page latency improves (rec pipelining, threaded host
      ops) with identical text.
- [ ] Long-audio (>= 10 min) transcription wall time improves (streamed
      segments, device-side features, single WAV load) with identical text.
- [ ] Every tried change, kept or not, is recorded in `docs/performance.md`
      with its measured effect; new findings get their own row here.
- [ ] Batched decoding is measured; it ships only under the decision below.
- [ ] Bit-identical changes: model goldens pass on Metal, `corpus_gate`
      identical for all OCR models and ASR, `aligner_parity` gated MAE
      unchanged. fmt, clippy `-D warnings`, prek, hermetic tier pass.

## Scope and non-goals

In scope:

- Host/GPU overlap in decode and prefill paths; async submission.
- Pipelining of PDF rendering, audio segmentation and model work
  (threads/channels inside `aimd-process` and `aimd-engine`).
- Multi-threaded host image/audio preprocessing and postprocessing.
- Removing host round trips and duplicate decodes/parses.
- Load-time improvements (batched quantize, avoided re-work).
- Batched decoding across pages/segments (evaluation workstream).

Not in scope:

- Changing models, precisions, defaults or output formats.
- New third-party crates (threads use `std::thread`; channels use
  `std::sync::mpsc`).
- CUDA-specific kernels.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | Every change except batched decoding must be bit-identical (goldens, corpus gate identical, aligner MAE unchanged) | Same rule as specs 002 and 006; references define correctness | WS-01..WS-07 |
| 2026-10-26 | Batched decoding may change outputs; it is kept only if the corpus gate passes its accuracy thresholds for every affected model and it speeds up the multi-page/multi-segment case by >= 1.3x; otherwise it is recorded as not kept | Batched matmuls pick different kernels, so bit-identity is impossible; the corpus gate is the accuracy contract | WS-08 |
| 2026-10-26 | Concurrency uses `std::thread` / `std::sync::mpsc` only; cancellation stays at page/segment boundaries and a cancelled pipeline joins its workers | Dependency policy; AGENTS.md cooperative cancellation | WS-02, WS-05 |
| 2026-10-26 | One coordinator, workstreams in order, A/B runs interleaved on one GPU | Shared GPU skews parallel benchmarks | all |
| 2026-10-26 | Builds on branch `fix/anchors-default-off` | It touches the same ASR/OCR orchestration code | all |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Decode loops submit step k+1 before reading token k (Qwen3-ASR, PaddleOCR-VL, Unlimited-OCR with its n-gram guard) | coordinator | Done | None | bench tok/s; goldens; corpus gate identical |
| WS-02 | PDF pages rendered in the background and fed to OCR as they finish; PNM instead of PNG | coordinator | Done | None | multi-page PDF wall time; corpus gate identical; cancellation tests |
| WS-03 | PP-OCRv6: async rec batches, threaded crop/warp/resize/normalize | coordinator | Pending | None | `ppocrv6` example; checkpoint test; corpus gate identical |
| WS-04 | Pillow bicubic resize threaded; Unlimited-OCR's two resizes concurrent; PaddleOCR-VL spotting decodes the image once | coordinator | Pending | None | preprocess goldens; OCR page latency |
| WS-05 | ASR: device-side feature merge, single WAV load per segment, streamed ffmpeg segmentation with next-segment prep overlapping decode | coordinator | Pending | WS-01 | long-audio wall time; corpus gate ASR identical; aligner parity |
| WS-06 | Aligner `lm_head` at slots only (if argmax identical); PaddleOCR-VL batched quantize at load | coordinator | Pending | None | aligner parity; load time |
| WS-07 | New findings discovered during WS-01..06 (one row each when found) | coordinator | In progress | None | per finding |
| WS-07a | Metal command buffers capped at 30 MB (shim default) so early-submitted MoE steps keep KV-cache donation | coordinator | Done | WS-01 | Unlimited-OCR tok/s; all-model A/B; goldens |
| WS-08 | Batched decoding across pages/segments: prototype, measure, decide | coordinator | Pending | WS-01..05 | corpus gate thresholds; speedup |
| WS-09 | `docs/performance.md`, AGENTS.md, final gates, closeout | coordinator | Pending | WS-01..08 | `cargo xtask check-specs`; commit gates |

## Verification

```bash
cargo run --release -p aimd-engine --example bench_paddleocr_vl -- --rounds 3 --quiet page.png
cargo run --release -p aimd-engine --example bench_paddleocr_vl -- --bits 4 --rounds 3 --quiet page.png
cargo run --release -p aimd-engine --example ppocrv6 -- <det_dir> <rec_dir> <images>
cargo run --release -p aimd-engine --example aligner_parity
cargo test --release -p aimd-engine --lib            # Metal checkpoint tier
cargo run --release -p aimd-engine --example corpus_gate -- --only ocr --ocr-model <model>
cargo run --release -p aimd-engine --example corpus_gate -- --only asr
./target/release/aimd <multi-page scanned PDF> -m <model>      # wall time
./target/release/aimd <>= 10 min audio>                        # wall time
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo xtask check-specs
```

## Documentation sync

- Durable product/developer docs: `docs/performance.md`.
- Nearest local `AGENTS.md`: root `AGENTS.md` (pipelining and concurrency
  conventions).
- Parent/package guidance: none.
- Root guidance and roadmap: root `AGENTS.md`.

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-26 — coordinator — all

- Changed: opened the spec from the throughput survey.
- Evidence: survey findings in Context (file references).
- Remaining: WS-01..09.
- Blockers: none.
- Next action: WS-01 baselines and the decode-loop change.

### 2026-10-26 — coordinator — WS-01

- Changed: Qwen3-ASR, PaddleOCR-VL and Unlimited-OCR decode loops
  `async_eval` step k+1 before reading token k. Unlimited-OCR submits only
  the forward early, because its n-gram guard needs token k on the host
  before sampling. The shim sets `MLX_MAX_MB_PER_BUFFER=30` on Apple unless
  the user set it.
- Finding (WS-07 row): early submit alone made Unlimited-OCR slower
  (241 -> 223 tok/s). In that mode `async_eval` blocked about 3.1 ms per
  step in mlx's in-flight throttle (`wait_for_one`), and `slice_update` took
  the full `copy_gpu` path (host samples: 17 vs 1).
  - mlx commits a Metal command buffer once its inputs pass the MB cap. On
    M2 Pro that cap is 50 MB, and the MoE `gather_qmm` expert weights exceed
    it, so each step is a few large buffers.
  - A buffer keeps its inputs alive until it completes, so the previous
    step's KV arrays cannot be donated and every layer copies its cache.
  - Early-submit tok/s by MB cap: 30: 272-276, 40: 254, 50: 223,
    unlimited: 214. Old loop: about 241 at every cap. The op-count cap
    (`MLX_MAX_OPS_PER_BUFFER`) had no effect.
  - The dense models fill buffers by op count, so they never hit this. They
    gain at every cap and are neutral to 30 MB.
- Evidence (M2 Pro, `article-p1.png`, interleaved A/B vs b2cf0b1, 3 rounds):
  - Unlimited-OCR decode: 227-239 -> 263-276 tok/s. Vision and prefill
    unchanged.
  - PaddleOCR-VL 4-bit: 264-310 -> 324-334 tok/s.
  - PaddleOCR-VL bf16: 143-154 -> 157-160 tok/s.
  - Qwen3-ASR `en_60s.wav`: 2.35-2.49 -> 2.29-2.37 s.
  - PP-OCRv6 (cap only): latency neutral, text byte-identical.
  - Bit-identity: `cargo test --release -p aimd-engine --lib` (119) and
    `aimd-mlx` pass on Metal. `corpus_gate` identical to baseline for ASR
    (72/72), Unlimited-OCR, PaddleOCR-VL bf16 and 4-bit, and PP-OCRv6 (5/5
    each). `aligner_parity` PASS, gated MAE 0.17 ms.
- Remaining: WS-02..09.
- Blockers: none.
- Next action: WS-02 PDF render pipelining.

### 2026-10-26 — coordinator — WS-02

- Changed:
  - `PdfRenderJob` (`aimd-process/src/ocr.rs`) renders one page per
    `mutool draw` run on a scoped background thread.
  - Pages go to OCR over a `sync_channel`, at most two pages ahead, and each
    page image is deleted once consumed.
  - Local models get binary PPM (`-F pnm`); remote servers still get PNG.
  - A render failure stops the renderer at that page. A consumer error or a
    cancellation drops the channel, and the scope joins the renderer.
- Evidence (10-page PDF built from the corpus pages, 144 DPI):
  - Rendering every page up front took 0.42 s before the first OCR (PNG,
    42 ms/page). A per-page run costs about 16 ms of process start, which
    the overlap hides.
  - PPM renders in 23 ms/page and decodes in 0.4 ms vs PNG's 1.9 ms.
    Decoded pixels are identical to PNG.
  - Markdown is byte-identical to bf6c24b for pp-ocrv6, unlimited-ocr and
    paddleocr-vl.
  - CLI wall time on this machine varied 7-57 s for identical PP-OCRv6 runs,
    all of it GPU waits; that noise is far larger than the saving (about
    40-60 ms per page, about 7% of a PP-OCRv6 page), so wall time is not
    quoted.
  - Hermetic tests cover per-page PPM args, a mid-range render failure with
    cleanup, and PNG for remote OCR.
- Remaining: WS-03..09.
- Next action: WS-03 PP-OCRv6 rec pipelining.

## Completion summary

Pending.
