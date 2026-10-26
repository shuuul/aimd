---
id: "006"
title: "Fast paths for PaddleOCR-VL, PP-OCRv6, and Qwen3-ForcedAligner"
status: Active
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 006 — Fast paths for PaddleOCR-VL, PP-OCRv6, and Qwen3-ForcedAligner

## Context

Specs 004 and 005 added three MLX models: PaddleOCR-VL-1.6, PP-OCRv6
medium, and Qwen3-ForcedAligner-0.6B. They were ported op for op against
their Python references and have none of the fast paths that Qwen3-ASR and
Unlimited-OCR picked up in spec 002 (`docs/performance.md`, "Decode-step
kernel profile"):

- shapeless `mlx::core::compile` fused elementwise ops (`swiglu`, `gelu`);
- custom Metal kernels gated on `metal_kernels_available()`
  (`mlx_sdpa_decode`, fused MoE and router kernels);
- dispatch trimming in the decode step.

The user asked for these optimizations on every newly added model.
Baselines (Apple M2 Pro, release, warm):

| Model | Input | Native now | Reference |
| --- | --- | --- | --- |
| PaddleOCR-VL bf16 | `ocr_e2e/page_text.png`, 1253-token prompt, 105 tokens | prefill 1.95 s (642 tok/s), decode 167 tok/s | mlx-vlm 630 / 167 tok/s |
| PaddleOCR-VL 4-bit | same | prefill 2.21 s (567 tok/s), decode 327 tok/s | mlx-vlm 540 / 321-332 tok/s |
| PP-OCRv6 medium | general_ocr_002 / img_10 / sample_doc | first 0.54 / 0.57 / 0.52 s, warm 0.50 / 0.31 / 0.24 s | mlx4ocr non-compiled warm 0.54 / 0.33 / 0.24 s; compiled warm 0.49 / 0.30 / 0.22 s, first 6.73 / 1.55 / 0.95 s |
| ForcedAligner 8-bit | corpus `en_300s` / `zh_300s` | 5.13 / 5.92 s | mlx-audio 4.70 / 5.57 s |

The 4-bit PaddleOCR-VL decode step reads ~200 MB, a ~1.2 ms roofline, but
takes ~3.1 ms. Per step it issues ~300 small rotary ops (8 per q/k per layer)
plus a host-built mRoPE table, so dispatch, not bandwidth, bounds it.

## Goal and success criteria

- [ ] PaddleOCR-VL 4-bit decode improves >= 10% and bf16 decode does not
      regress; the prefill/vision gain is recorded either way.
- [ ] PP-OCRv6 warm latency reaches mlx4ocr's compiled warm numbers without
      its per-shape first-call cost (first call stays within 10% of today).
- [ ] ForcedAligner closes the gap to mlx-audio on the 300 s probes, or the
      remaining gap is explained.
- [ ] Outputs stay identical: model goldens pass on Metal, `corpus_gate`
      reports identical pages/clips for all OCR models and ASR, and
      `aligner_parity` passes with unchanged gated MAE.
- [ ] Every tried change is recorded in `docs/performance.md` with its
      measured effect (kept or not).
- [ ] fmt, clippy `-D warnings`, prek, hermetic tier pass.

## Scope and non-goals

In scope:

- Fused elementwise sub-graphs (shapeless `compile` in the shim) and custom
  Metal kernels with the stock-op fallback.
- Host-side dispatch trimming: precomputed rotary tables, reuse of
  `mlx_sdpa_decode`.
- Profiling tools and examples needed to measure the above.

Not in scope:

- Whole-model shape-keyed `compile` (see Decisions).
- Changing model numerics, precisions, or defaults; speculative decoding.
- CUDA-specific kernels (CUDA keeps the stock ops).

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | No whole-model shape-keyed `compile` (mlx4ocr `compile_models=True` style) | Measured: 8-10% warm gain, but each new image size retraces (first call 1.13 -> 6.73 s on general_ocr_002); aimd sees a new shape per page and runs it once | WS-02, WS-03 |
| 2026-10-26 | Fast paths must be bit-identical to the current graph on Metal: same goldens, identical corpus outputs | Same rule as the spec 002 kernels; the references define correctness | all |
| 2026-10-26 | Custom Metal kernels are gated on `metal_kernels_available()` with the stock ops as fallback; shapeless compiled ops run everywhere | AGENTS.md "MLX only" convention | all |
| 2026-10-26 | One coordinator works through the workstreams in order | Benchmarks share one GPU; parallel runs would skew A/B timings | all |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | PaddleOCR-VL decoder: cached decode rotary tables, fused rotary apply, `sdpa_decode` | coordinator | Claimed | None | `bench_paddleocr_vl` bf16 + 4-bit; checkpoint goldens on Metal; corpus gate `paddleocr-vl` identical |
| WS-02 | PaddleOCR-VL vision prefill: profile, fuse rotary and other elementwise chains | coordinator | Pending | WS-01 (shared rotary op) | vision stage goldens on Metal; bench prefill |
| WS-03 | PP-OCRv6: profile det/rec, fuse activation chains (shapeless) | coordinator | Pending | None | `ppocrv6` example first/warm; checkpoint test; corpus gate `pp-ocrv6` identical |
| WS-04 | ForcedAligner: profile against mlx-audio, close the gap | coordinator | Pending | None | `aligner_parity` latency and MAE |
| WS-05 | `docs/performance.md`, AGENTS.md, final gates, closeout | coordinator | Pending | WS-01..04 | `cargo xtask check-specs`; commit gates |

## Verification

```bash
cargo run --release -p aimd-engine --example bench_paddleocr_vl -- --rounds 3 --quiet page.png
cargo run --release -p aimd-engine --example bench_paddleocr_vl -- --bits 4 --rounds 3 --quiet page.png
cargo run --release -p aimd-engine --example ppocrv6 -- <det_dir> <rec_dir> <images>
cargo run --release -p aimd-engine --example aligner_parity
cargo test -p aimd-engine --lib paddleocr_vl pp_ocrv6 qwen3_aligner   # Metal checkpoint tier
cargo run --release -p aimd-engine --example corpus_gate -- --only ocr --ocr-model <model>
cargo run --release -p aimd-engine --example corpus_gate -- --only asr
cargo xtask check-specs
```

## Documentation sync

- Durable product/developer docs: `docs/performance.md`.
- Nearest local `AGENTS.md`: root `AGENTS.md` (new shim ops, if any).
- Parent/package guidance: none.
- Root guidance and roadmap: root `AGENTS.md`.

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-26 — coordinator — all

- Changed: opened the spec with the baselines above.
- Evidence: the baseline table (bench runs on this machine).
- Remaining: WS-01..05.
- Blockers: none.
- Next action: WS-01.

## Completion summary

Pending.
