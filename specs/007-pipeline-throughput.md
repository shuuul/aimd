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
| WS-01 | Decode loops submit step k+1 before reading token k (Qwen3-ASR, PaddleOCR-VL, Unlimited-OCR with its n-gram guard) | coordinator | Claimed | None | bench tok/s; goldens; corpus gate identical |
| WS-02 | PDF pages rendered in the background and fed to OCR as they finish; PNM instead of PNG | coordinator | Pending | None | multi-page PDF wall time; corpus gate identical; cancellation tests |
| WS-03 | PP-OCRv6: async rec batches, threaded crop/warp/resize/normalize | coordinator | Pending | None | `ppocrv6` example; checkpoint test; corpus gate identical |
| WS-04 | Pillow bicubic resize threaded; Unlimited-OCR's two resizes concurrent; PaddleOCR-VL spotting decodes the image once | coordinator | Pending | None | preprocess goldens; OCR page latency |
| WS-05 | ASR: device-side feature merge, single WAV load per segment, streamed ffmpeg segmentation with next-segment prep overlapping decode | coordinator | Pending | WS-01 | long-audio wall time; corpus gate ASR identical; aligner parity |
| WS-06 | Aligner `lm_head` at slots only (if argmax identical); PaddleOCR-VL batched quantize at load | coordinator | Pending | None | aligner parity; load time |
| WS-07 | New findings discovered during WS-01..06 (one row each when found) | coordinator | Pending | None | per finding |
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

## Completion summary

Pending.
