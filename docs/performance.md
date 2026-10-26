# Performance

This document records practical performance expectations for `aimd` processing paths. It is not a leaderboard: real throughput depends on input length, page resolution, model size, first-run model download time, GPU memory, and whether the path can reuse model weights already loaded in the current process.

## High-level expectations

| Path | Typical bottleneck | Warm-cache behavior | Notes |
|------|--------------------|---------------------|-------|
| URL with subtitles | Network and yt-dlp metadata/subtitle download | Usually fast after metadata is available | `aimd` prefers subtitles over ASR whenever usable subtitles are found. |
| URL without subtitles | Media download, audio extraction, then ASR | Model reuse helps only inside the same long-running process | Each CLI invocation loads the checkpoint again; MCP reuses loaded models. |
| Local audio/video transcription | ffmpeg decode to 16 kHz mono and ASR inference | Better in MCP because loaded pipelines are cached in-process | Inputs over 10 minutes are transcribed in segments. |
| Document conversion | Pandoc / pdf-inspector / document structure | Mostly CPU and file I/O bound | EPUB conversion also writes chapter/image output trees. |
| Image OCR | OCR model inference | MCP reuses the loaded pipeline | VLM OCR models are heavier than classic detector/recognizer OCR. |
| PDF OCR | PDF rasterization plus one OCR pass per page | Model reuse helps; page rendering still scales with page count and DPI | PDF pages render with `mutool draw` (144 dpi, RGB) before OCR inference. |

## Model/runtime tradeoffs

| Task | Backend | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | native MLX | Qwen3-ASR 1.7B/0.6B (`4bit`, `6bit`, `8bit`) | Default `qwen3-asr-1.7b` 4-bit. The 0.6B model and lower bit widths trade quality for lower memory and faster load. 4-bit segments that loop retry once on 8-bit. |
| OCR | native MLX | Unlimited-OCR (`4bit`, `6bit`, `8bit`) | Default 4-bit. Single-image gundam mode per page, no-repeat n-gram guard (`ngram=35`, `window=128`), `max_tokens=8192`. Without the guard the model can loop (e.g. `R. R. R. ...`) until max_tokens and take minutes per page. |
| ASR / OCR | remote OpenAI-compatible | server-defined | Latency is network plus server-side; no local load. |

## Native Rust engine (MLX backend)

The engine runs only on MLX through the `aimd-mlx` C-ABI shim (brew `mlx`
0.32.4). Numbers are from Apple M2 Pro (19-core GPU, 32 GB), release builds
of `crates/aimd-engine/examples/{bench_asr_mlx,bench_ocr}.rs`, rounds 2-3
steady state. "Reference" rows are mlx-audio / mlx_vlm on the same machine
(`scripts/ocr_bench_mlx_vlm.py`, the corpus baseline tooling).

### Qwen3-ASR-1.7B-4bit

| Date | Backend | Input | Load | Warm latency | Realtime | Notes |
|------|---------|-------|------|--------------|----------|-------|
| 2026-10-25 | mlx-audio (Metal) | zh 8.56 s / en 3.75 s TTS clips | 10.01 s | zh 0.43 s; en 0.25 s | 19.9x / 15x | Reference. |
| 2026-10-26 | mlx-audio 0.5.3 (Metal) | 59.96 s en TTS clip (797-token prompt, 191 tokens) | — | 2.88-2.95 s | ~20.5x | Reference for the long clip. |
| 2026-10-26 | native MLX | zh 4.83 s / en 4.09 s / 59.96 s en | 0.42-0.45 s warm / 3.81 s cold | zh 0.30-0.31 s; en 0.29-0.30 s; long 2.73-2.75 s | ~16x / ~14x / ~22x | Transcripts token-identical to the reference. Long clip: encoder 0.44 s, prefill 0.67 s (~1.2k tok/s), decode 1.65 s (~117 tok/s). A 5-minute clip takes ~22 s. 4-bit triples go straight into `quantized_matmul`, hence the fast load. |

### Unlimited-OCR-4bit

| Date | Backend | Input | Load | Vision | Prefill | Decode | Total | Notes |
|------|---------|-------|------|--------|---------|--------|-------|-------|
| 2026-10-25 | mlx_vlm 0.7.0 | `golden/ocr_e2e/small_text.png` (no crops) | 3.26 s | 0.35-0.37 s | in prompt | 199-206 tok/s | 0.87-0.88 s | Reference. |
| 2026-10-25 | mlx_vlm 0.7.0 | `golden/ocr_e2e/page_text.png` (4x5 crops, 2327-token prompt) | — | 2.50-2.55 s | ~0.77 s | 170-177 tok/s | 4.51-4.65 s | Reference. |
| 2026-10-26 | native MLX | `page_text.png` / `small_text.png` | 0.58 s | 2.20 s / 0.31 s | 0.69 s (3390 tok/s) / 0.10 s | 254 / 305 tok/s | 3.78 s / 0.63 s | Fused MoE expert + router, decode sdpa, compiled GELU. Markdown identical to the reference, finish=stop. Vision + prefill are bit-exact vs the mlx_vlm goldens; single-token decode can drift ~1 bf16 ulp (brew mlx 0.32.4 vs Python mlx 0.32.2 small-M qmv), shifting some bbox coordinates by 2/1000. |

### PaddleOCR-VL-1.6

Apple M2 Pro, release build, `mutool draw -r 144` page (1224x1584, 1253-token
prompt, 995 generated tokens). Markdown byte-identical to mlx-vlm 0.7.10 in
both precisions.

| Date | Backend | Precision | Load | Prefill | Decode | Total |
|------|---------|-----------|------|---------|--------|-------|
| 2026-10-26 | mlx-vlm 0.7.10 | bf16 | — | 630 tok/s | 167 tok/s | 8.0 s |
| 2026-10-26 | mlx-vlm 0.7.10 | 4bit (mlx-community repo) | — | 540 tok/s | 321-332 tok/s | 5.3-5.5 s |
| 2026-10-26 | native MLX | bf16 | 1.6-3.3 s | 620 tok/s | 161 tok/s | 8.3 s |
| 2026-10-26 | native MLX | 4bit (quantized at load) | 0.5-2.3 s | 565 tok/s | 326 tok/s | 5.3 s |

### PP-OCRv6 medium

Apple M2 Pro, release build, mlx4ocr 0.1.2 example images; load 0.21 s.
Det + rec outputs are bit-exact with mlx4ocr from decoded pixels on.

| Image | Native first / warm | mlx4ocr |
|-------|---------------------|---------|
| sample_doc | 0.24 / 0.21 s | 0.26 s |
| img_10 (1280x720) | 0.31 / 0.28 s | 0.34 s |
| general_ocr_002 | 0.56 / 0.50 s | 0.60 s |
| word_1 (no text) | 1.58 / 0.64 s | 0.67 s |

Spec 006's fused conv epilogues later cut warm to 0.445 / 0.283 / 0.203 s
(general_ocr_002 / img_10 / sample_doc); see "New-model fast paths".

ASR decode loop: in the pipelined ASR loop the host builds a step's graph in
~0.4 ms, and the remaining ~8 ms per token is GPU work plus Metal encode for
~25 kernels per layer x 28 layers. Elementwise fusion (`mlx::core::compile`)
buys little there. A whole-step `compile` does not fit because the KV cache
grows every step, so a shape-keyed compile would retrace each token and a
shapeless one cannot express the slice bounds (mlx-lm doesn't compile the
step either).

Prompt-lookup speculative decoding (not shipped): an
m-token Unlimited-OCR verify forward costs 4.3 / 6.5 / 11.0 / 19.4 / 24.6 ms
for m = 1 / 2 / 4 / 8 / 16, nearly linear because every routed token reads
its own 6 of 64 experts. On six real OCR pages, n-gram drafts were accepted
for only 1.01-1.15 tokens per verify, so simulated decode came out
0.63-1.01x.

### Source anchors (spec 005)

Apple M2 Pro, release CLI, wall clock including model load.

| Input | Default (anchored Markdown) | Other |
|-------|-----------------------------|-------|
| 394 s English clip, Qwen3-ASR-1.7B-4bit | `-f srt` 31.9 s (two 295 s segments, both aligned) | `--no-anchors` 23.0 s (unsegmented) |
| 2-page scanned PDF, Unlimited-OCR | 9.1 s | `-f json` 5.5 s (warm cache) |
| same, PP-OCRv6 | 1.6 s | `-f json` 1.3 s |
| same, PaddleOCR-VL bf16 | 6.6 s | `-f json` 11.0 s (adds the `Spotting:` pass, ~2.2 s/page) |

Forced alignment costs ~9 s on the 394 s clip: aligner load plus the first
alignment take ~6 s, and segmenting adds a second ASR prefill. Unlimited-OCR
blocks and PP-OCRv6 quads come from the existing forward pass, so OCR anchors
are free; only PaddleOCR-VL line segments need the extra pass, and only when
segments are requested.

### New-model fast paths (spec 006)

Apple M2 Pro, release, warm, A/B runs interleaved. Every kept change is
bit-identical: model goldens pass on Metal, `corpus_gate` pages/clips are
identical for all OCR models and ASR, and `aligner_parity` keeps its gated
MAE (0.17 ms).

| Model | Change | Before | After | Kept |
|-------|--------|--------|-------|------|
| PaddleOCR-VL decoder | one mRoPE table per generation (sliced per step), rotary as one compiled kernel (`rope_half_split`: halves-reversed strided view times `[-s, s]`, no slice/negate/concat), `sdpa_decode` for single-token steps | 4-bit 305 tok/s, bf16 162-171 tok/s (360-token page) | 4-bit 342-345 tok/s, bf16 unchanged | yes |
| PaddleOCR-VL vision | same fused rotary in the tower | prefill 1.95 s | 1.86 s | yes |
| PaddleOCR-VL vision | pad head dim 72 -> 80 to use fused sdpa | 1.86 s | 1.94 s | no |
| PP-OCRv6 | conv epilogues `act(conv + b) + residual` as one compiled kernel (`bias_act`: relu, hard sigmoid, gelu) | warm 0.470 / 0.297 / 0.212 s | 0.445 / 0.283 / 0.203 s | yes |
| ForcedAligner (and ASR) | host log-mel frames split over all cores, zero filterbank weights skipped | mel 700 ms per 300 s; aligner en/zh_300s 5.10 / 5.92 s | mel 95 ms; 4.52 / 5.33 s (mlx-audio 4.70 / 5.57 s) | yes |

PP-OCRv6 images are general_ocr_002 / img_10 / sample_doc. The fused build
is faster warm than mlx4ocr with `compile_models=True` (0.49 / 0.30 / 0.22
s) and keeps the non-compiled first call (0.49 / 0.31 / 0.22 s). Whole-model
shape-keyed `compile` was rejected: it retraces for every new image size
(first call 1.13 -> 6.73 s on general_ocr_002), and aimd sees a new shape
per page. Detection is ~95% of a page except on dense pages (39 lines: rec
0.21 s).

The PaddleOCR-VL decode step is host-bound, not bandwidth-bound: ~0.4 ms
graph build, ~1.8 ms Metal encode in `async_eval`, ~0.8 ms GPU wait, against
a ~1.2 ms read roofline at 4-bit, so fewer dispatches pay off. The vision
tower is 1.65 s of the 1.86 s prefill and compute-bound (~63% of peak);
biases already go through `addmm`.

After the mel fix the aligner's 300 s cost is tower ~1.5 s and the
cache-free decoder pass 2.8-3.6 s, both on the shared Qwen3-ASR fast paths.

### Pipeline throughput (spec 007)

Apple M2 Pro, release, warm, A/B runs interleaved against the previous
commit. Everything kept is bit-identical: Metal checkpoint tests pass,
`corpus_gate` is identical for ASR (72/72) and every OCR model (5/5), and
`aligner_parity` keeps its gated MAE (0.17 ms).

| Area | Change | Before | After | Kept |
|------|--------|--------|-------|------|
| Decode loops (ASR, PaddleOCR-VL, Unlimited-OCR) | submit step k+1 (`async_eval`) before reading token k, like mlx-lm `generate_step`; Unlimited-OCR submits the forward early and samples after its n-gram guard | Unlimited-OCR 227-239, PaddleOCR-VL 4-bit 264-310, bf16 143-154 tok/s; ASR en_60s 2.35-2.49 s | 263-276, 324-334, 157-160 tok/s; 2.29-2.37 s | yes |
| Metal command buffers | shim sets `MLX_MAX_MB_PER_BUFFER=30` unless the user set it | Unlimited-OCR early submit at mlx's 50 MB: 223 tok/s | 272-276 tok/s; dense models neutral | yes |
| PDF OCR | render pages one `mutool draw` run at a time on a background thread, at most two ahead; PPM for local engines (PNG for remote) | all pages rendered first, PNG 42 ms + decode 1.9 ms per page | overlapped; PPM 23 ms + 0.4 ms | yes |
| PP-OCRv6 host | resize, det normalize, crops, rec preprocess threaded (`aimd_engine::parallel`); all rec batches submitted before any readback | warm 0.93-0.97 / 0.58-0.64 / 0.60-0.67 s | 0.82-0.92 / 0.55-0.60 / 0.58-0.64 s | yes |
| Unlimited-OCR preprocess | Pillow bicubic passes on HWC with fixed-point taps, threaded by row bands | 0.104-0.111 s | 0.013-0.016 s | yes |
| Unlimited-OCR preprocess | global-view and crop resizes concurrently | - | - | no: each pass already uses every core |
| PaddleOCR-VL spotting | reuse the OCR pass's decoded page | two decodes | one | yes |
| ForcedAligner | gather the timestamp slots before the 5000-class `lm_head` | `[seq, 5000]` logits | `[slots, 5000]`; latency within noise | yes |
| PaddleOCR-VL load | quantize-at-load via `async_eval` instead of one blocking eval per tensor | 4-bit 0.52-0.54 s | 0.42-0.44 s | yes |
| Qwen3-ASR host | device-side feature merge, single WAV parse, streamed segmentation | merge 0.02-0.04 s, WAV 4 ms, mel 0.2 s per 600 s; ffmpeg segmentation 1.3 s per hour | - | no: under 1% of a segment |
| Batched decoding | B sequences per step (microbenchmark, Qwen3-1.7B-shaped 4-bit, KV 2000) | 8.1 ms/token at B=1 | B=2 5.4, B=4 4.9, B=8 5.4 ms per sequence | no: at most ~1.5x on decode, under the 1.3x end-to-end bar |

Notes:

- mlx commits a Metal command buffer when its op count or input size passes
  a cap (50 ops / 50 MB on M2 Pro). A buffer holds its inputs until it
  completes. When the next step is encoded while the previous one is in
  flight, the previous KV arrays are still referenced, so `slice_update`
  cannot donate them and copies each layer's cache. Unlimited-OCR's MoE
  `gather_qmm` expert weights fill buffers by size, so it hit this; the
  dense models fill buffers by op count and did not. At 30 MB the buffers
  are small enough to complete before the next step needs the cache.
- Long-audio ASR decode is bandwidth-bound, not copy-bound. A 300 s decode
  is 8.8-9.4 s at buffer caps from 10 to 1000 MB. For a 600 s segment the
  phases are: tower 4.7 s, prefill of 7.8k tokens 9.0 s, decode of 2427
  tokens 35 s.
- The PaddleOCR-VL vision tower is 1.83 s of a page with 4960 patches. That
  is about 7 TFLOP (27 layers with full attention through fused sdpa),
  close to M2 Pro peak. The LM prefill of 1253 tokens is 0.26 s.

### Decode-step kernel profile (Metal)

`cargo run --release -p aimd-mlx --example bench_ops` times every
decode-step op at the real shapes as a dependent chain (GPU time plus Metal
encode per kernel). Weights rotate through ~96 MB of distinct random 4-bit
copies, so the system-level cache cannot hide reads. The practical read
roofline is ~166 GB/s (the lm_head qmv; spec is 200 GB/s). Summing the ops
reproduces the measured steps: OCR ~4.75 ms estimated vs ~5.1 ms measured,
ASR ~8.9 ms vs ~8.5 ms. Dispatch floor is 2.3 µs per kernel.

| Op (per layer unless noted) | µs | Bytes | % of roofline |
|---|---|---|---|
| ASR qmv 2048->2048 (q, o) | 20.1 | 2.36 MB | 71% |
| ASR qmv 2048->1024 (k, v) | 11.9 | 1.18 MB | 60% |
| ASR qmv up\|gate + down (2048<->6144) | 101.9 per pair | 14.2 MB | 84% |
| ASR sdpa 16q/8kv, kv 256 / 1024 | 15.6 / 43.1 | 1.0 / 4.2 MB | 40% / 59% |
| ASR lm_head 2048->151936 (per token) | 1063 | 175 MB | ~100% |
| ASR rms_norm, rope, kv row write, swiglu | 2.2-2.8 each | — | dispatch-bound |
| OCR qmv 1280->1280 (q, k, v, o) | 11.4 | 0.92 MB | 49% |
| OCR shared expert up\|gate + down (1280<->1792) | 31.0 per pair | 2.58 MB | 50% |
| OCR gather_qmm up + down, 6 of 64 experts, + sum | 82.4 | 7.74 MB | 57% |
| OCR routed experts unfused (gate\|up + swiglu + down + score sum) | 105 | 11.6 MB | 70% |
| OCR routed experts, fused kernel pair (shipped) | 75 | 11.6 MB | ~95% |
| OCR router unfused (gemv + softmax + top-6 + take) | 30 | 0.16 MB | dispatch-bound |
| OCR router, gemv + fused softmax/top-6 (shipped) | 15 | 0.16 MB | dispatch-bound |
| OCR router: gate matmul (W_g.T view) + softmax + top-6 | ~8.5 + 2.6 + ~12 | — | dispatch-bound |
| OCR sdpa 10 heads, kv 2455 (prefill + ring) | 102.2 | 12.6 MB | 74% |
| OCR `sdpa_decode`, kv 2455 (shipped) | 80 | 12.6 MB | ~95% |
| OCR lm_head 1280->129280 (per token) | 561 | 93 MB | ~100% |

Where an OCR token's ~5.1 ms goes: attention over the kept 2327-token prefill
KV ~1.2 ms, routed experts ~1.0 ms, q/k/v/o ~0.55 ms, shared experts ~0.5 ms,
lm_head 0.56 ms, router ~0.25 ms, the rest small ops. ASR is dominated by
qmv (~6 ms of ~8.5 ms) already at 60-84% of roofline, plus lm_head at
roofline.

Changes tried, by expected saving per token (shipped ones are the current code):

1. ~~Fused q|k|v projection~~ (tried 2026-10-26, not kept): outputs were
   byte-identical (3 ASR clips, 7 OCR images) but decode moved <1% (OCR
   196-204 tok/s both ways). In the real graph the three projections share
   an input, so MLX dispatches them concurrently: 41.8 µs per ASR layer vs
   35.6 µs fused, not the 44 µs the serial chain suggests.
2. Fused MoE expert kernel (shipped 2026-10-26): `mlx_moe_swiglu_4bit`,
   two custom Metal kernels (gate|up + SwiGLU, down + weighted sum) for
   single-token steps. Experts per layer 105 -> 75 µs in the chain (~95%
   of roofline); OCR decode 229-232 -> 238-247 tok/s (+7%), markdown
   byte-identical on 7 images. The kernels mirror mlx `qmv_impl`
   accumulation order and bf16 rounding, so the output is bit-identical
   to the `gather_qmm` graph. Layouts that kept more activations per lane
   (registers or threadgroup memory) lost occupancy and ran 100-173 µs.
   The shared-expert and layer-0 dense MLPs run through the same kernel as
   one expert with score 1 (bit-identical to the `quantized_matmul` MLP:
   M=1 takes `qmv_impl` for K=1280/1792/6848): OCR decode ~259 -> ~262
   tok/s (+1.5%), markdown byte-identical on 7 images. The down kernel
   also takes up to two `[D]` addends (shared-expert output, then the
   residual), each rounded like mlx `add`, removing two dispatches per
   layer: OCR decode ~267 -> ~274 tok/s (+2.7%), byte-identical.
3. Decode attention (shipped 2026-10-26): mlx already splits long KV
   (`sdpa_vector_2pass` for kv >= 1024 on M2 Pro, 64 blocks for 1 query
   head per KV head); `MLX_SDPA_BLOCKS` 32-512 did not help.
   `mlx_sdpa_decode` is the same two kernels without pass 1's 4-key
   register prefetch: one key per step keeps more simdgroups resident.
   Same blocks, token order and float ops, so bit-identical; it falls back
   to mlx sdpa wherever mlx would not take the plain 2-pass kernel.
   89 -> 80 µs (OCR kv 2455), 39 -> 35 µs and 89 -> 78 µs (ASR GQA kv
   1024 / 3000). OCR decode ~263 -> ~271 tok/s (+3%), markdown
   byte-identical on 7 images; a 5-minute ASR clip 22.85 -> 22.17 s, text
   identical (short clips stay under 1024 KV and are unchanged). Packing
   2-16 blocks per threadgroup and 2/8-key prefetch were no better.
4. Fused router (shipped 2026-10-26): `mlx_moe_router` keeps mlx's gate
   gemv and replaces softmax + `argpartition` + `take_along_axis` with one
   simdgroup kernel (mirrors `softmax_single_row`; `argpartition` is a stable
   sort on Metal, so top-6 come out in ascending (score, index) order).
   Router chain 30 -> 15 µs per layer; OCR decode ~245 -> ~259 tok/s
   (+5.5%), markdown byte-identical on 7 images. A mirrored one-row-per-
   threadgroup gemv saved only 0.8 µs more (dispatch-floor bound), not kept.
   A single-threadgroup gemv + softmax + top-k ran 70 µs (one core).
5. Small-op fusion (measured 2026-10-26, not kept): a bit-identical
   residual add + RMSNorm kernel (mirrors `rms_single_row`) saves only
   ~0.5 µs per fusion in a chain (6.4 -> 5.9 µs at d 2048). Wired into
   both decoders (OCR -1, ASR -2 dispatches per layer) it was within noise
   end to end: OCR decode ~276 tok/s either way, the 5-minute ASR clip
   21.94 -> 21.82 s but the 1-minute clip 2.58 -> 2.60 s. Fusing rope for
   q and k saves one such dispatch; a custom kernel cannot write the KV
   row in place (`metal_kernel` outputs are fresh buffers), so the
   remaining small ops are below the measurable floor.

Caveat: the chain serializes every op, so it overstates ops that run
concurrently in the decoder (q/k/v, gate/up, shared vs routed experts).
Treat the per-op times as upper bounds and the serially dependent paths
(router -> experts, attention) as the reliable targets.

OCR vision tower (2026-10-26, sym-6, 12 crops + global view, ~1.5 s):
SAM over the crops is ~1.0 s (window blocks ~75 ms, global blocks ~95 ms
at B = 12), the global view's SAM ~0.27 s, CLIP ~0.23 s. The bf16 GEMMs
(qkv, lin1, lin2) run at 4-5 TFLOPS. `gelu` was five unfused elementwise
kernels (7.5 ms per crop block); it is now compiled like `mlx.nn.gelu`
(`@mx.compile(shapeless=True)`, intermediates keep their dtype) into one
kernel (3.1 ms): vision 1.505 -> 1.432 s per page, OCR markdown and ASR
text identical (the first run on a machine pays a one-time Metal compile).
The rel-pos bias broadcast add already writes ~104 GB/s in isolation; a
custom kernel was no faster (the 25 ms seen under per-op evals is
first-touch allocation).

## Golden-corpus accuracy (2026-10-26)

Labeled corpus from `scripts/corpus/`, built by `uv run scripts/build_corpus.py`:
72 ASR clips (12 en + 12 zh sentences, 3 macOS `say` voices each, 16 kHz)
and 4 OCR documents / 5 A4 pages (Pandoc + headless Chrome, rasterized with
`mutool draw -r 144` like production). Scored by
`cargo run --release -p aimd-engine --example corpus_gate` against the
mlx-audio 0.5.3 / mlx-vlm 0.7.0 outputs from `uv run scripts/corpus_baseline.py`.
WER is English words, CER Chinese characters, both after dropping
punctuation; OCR similarity is 1 - edit distance / longer length over
alphanumeric and CJK characters with markup removed.

| Checkpoint | en WER | zh CER | OCR similarity | finish=stop | Identical to reference |
|------------|--------|--------|----------------|-------------|------------------------|
| Qwen3-ASR-1.7B-4bit | 0.87% | 3.83% | — | — | 72/72 |
| Qwen3-ASR-1.7B-8bit | 0.44% | 3.55% | — | — | 72/72 |
| Qwen3-ASR-0.6B-6bit | 1.31% | 3.40% | — | — | 72/72 |
| Unlimited-OCR-4bit | — | — | 1.0000 | 5/5 | 5/5 pages |
| Unlimited-OCR-8bit | — | — | 1.0000 | 5/5 | 5/5 pages |
| PaddleOCR-VL-1.6 bf16 | — | — | 1.0000 | 5/5 | 5/5 pages |
| PaddleOCR-VL-1.6 4bit | — | — | 1.0000 | 5/5 | 5/5 pages |
| PP-OCRv6 medium | — | — | 1.0000 | 5/5 | 5/5 pages |

PaddleOCR-VL and PP-OCRv6 baselines come from
`uv run scripts/corpus_baseline_paddle.py` (mlx-vlm 0.7.10, mlx4ocr 0.1.2);
score with `corpus_gate -- --only ocr --ocr-model paddleocr-vl [--ocr-bits 4]`
or `--ocr-model pp-ocrv6`.

The speech is synthetic and the pages are clean renders, so absolute
error rates are optimistic. Zh errors are real homophone/near-sound
mistakes on the weaker voices (构建 -> 断线, 她 -> 他); en errors include
"twenty" -> "20". Identity with the mlx-audio / mlx-vlm reference is the strong signal.

On Linux with the mlx CUDA backend (metacube, see below) the same corpus
and the macOS-built baseline give identical results for all five
checkpoints: 72/72 clips and 5/5 pages identical to the reference. The
same holds for PaddleOCR-VL-1.6 (bf16 and 4bit) and PP-OCRv6 medium: 5/5
pages identical.

## Linux, mlx CUDA backend

metacube: RTX 5090 32 GB, Arch Linux, CUDA 13.4, cuDNN 9.27, NCCL 2.32,
the `mlx-cuda-13` 0.32.4 wheel's `libmlx.so` (`MLX_PREFIX`). Release builds
of the same examples, rounds 2-3 steady state. The custom Metal kernels are
off on CUDA (`aimd_mlx::metal_kernels_available`), so these are plain mlx
ops.

| Date | Model | Input | Load | Warm | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-26 | Qwen3-ASR-1.7B-4bit | zh 4.83 s / en 4.09 s / 59.96 s en | 0.68 s | 0.05-0.06 s / 0.05 s / 0.54 s | ~85x / ~77x / 112x realtime; transcripts identical to macOS. Round 1 adds ~0.7 s of CUDA JIT. |
| 2026-10-26 | Unlimited-OCR-4bit | `page_text.png` (4x5 crops, 2327-token prompt) | 0.74 s | 0.72 s | Vision 0.12 s, prefill 0.24 s (~9.5k tok/s), decode 750-770 tok/s, finish=stop. Round 1 2.19 s (JIT). |
| 2026-10-26 | Unlimited-OCR-4bit | `small_text.png` | — | 0.14 s | Decode ~850 tok/s. |

mlx JIT-compiles kernels against the CUDA headers (`CUDA_HOME`, else
`/usr/local/cuda`; the shim falls back to Arch's `/opt/cuda`). Vision-tower
stage parity against the Metal goldens drifts past the bf16 bounds on CUDA
(final features 4.6-5.3% rel rms vs 3%) while end-to-end markdown stays
identical, so those bounds are enforced on Metal only
(`specs/archive/003-linux-mlx-cuda.md`).

dspark1 (DGX Spark GB10, aarch64, Ubuntu 24.04, CUDA 13.4) with the tap's
brew `mlx-cuda`: the corpus gate matches the baseline for all five
checkpoints. No benchmark was recorded there (shared GPU).

### PaddleOCR-VL-1.6 and PP-OCRv6 on CUDA

metacube, 2026-10-26, brew `shuuul/tap/mlx-cuda` 0.32.4 (CUDA 13.4). The
GPU was shared with another process holding ~15 GB, so treat the timings as
indicative.

| Check | PaddleOCR-VL-1.6 bf16 | PaddleOCR-VL-1.6 4bit | PP-OCRv6 medium |
| --- | --- | --- | --- |
| Corpus gate (5 pages) | 5/5 identical, finish=stop 5/5 | 5/5 identical, finish=stop 5/5 | 5/5 identical |
| E2E `page_text.png` | 0.76 s (prefill 2.2k tok/s, decode 557 tok/s) | 0.26 s (prefill 12k tok/s, decode 667 tok/s) | — |
| E2E golden images (warm) | — | — | 40 / 129 / 51 ms (sample_doc / general_ocr_002 / img_10) |

Stage drift against the Metal goldens, with end-to-end output unchanged:

- **PaddleOCR-VL**: the bf16 patch-embed conv rounds ~1 ulp differently
  (rel rms 5e-5), and the 27-layer vision tower amplifies that to 3-4%
  (bf16) and 3-8% (4bit) rel rms at the projector. The vision-stage test
  reports this on CUDA and enforces it on Metal only, as for Unlimited-OCR.
  Decoder logprobs stay within the noise bound (rel rms <= 6e-3).
- **PP-OCRv6**: mlx's CUDA backend runs f32 convs and matmuls in TF32 by
  default (`MLX_ENABLE_TF32=1`). Against the goldens the detector
  probability map drifts up to 2.3e-2, rec step maxima up to 3e-2 (one
  step argmax flips on img_10 and collapses to the same text), and one box
  corner moves 1 px. Recognized text matches exactly. With
  `MLX_ENABLE_TF32=0`, the drift falls to 1e-5 (det) / 1e-4 (rec), at a
  cost of ~12-15% in end-to-end time on the corpus pages. aimd keeps mlx's
  default; the checkpoint test holds CUDA to noise bounds plus exact text.

On CUDA, mlx's allocator sizes its memory limit from total VRAM and
reclaims its buffer cache only past that limit. When another process
already holds VRAM, the cache keeps growing until `cudaMallocAsync` runs
out of memory. PP-OCRv6 on the corpus needs 7-12 GB of transient buffers
per page, so it hit this on the third page. On Linux the shim now caps
mlx's memory limit at 90% of the VRAM that is free when the device is
first selected.

## Measurement guidance

When comparing models, measure warm and cold runs separately:

1. **Cold run**: clear model cache/process state and include model download/load time.
2. **Warm run**: run multiple inputs in one MCP process, or use the engine bench examples, so loaded pipelines are reused.
3. **Separate stages** when possible: download/extraction, PDF rasterization, audio conversion, and model inference.
4. **Record input shape**: audio duration, page count, image dimensions, PDF DPI, language, and model ID.
5. **Record environment**: OS, chip/GPU, aimd version, brew `mlx` version, and whether weights were already cached.

A minimal local timing harness can wrap the public CLI:

```bash
/usr/bin/time -p aimd input.mp3 --model qwen3-asr-0.6b --output /tmp/out.md
/usr/bin/time -p aimd scan.pdf --model unlimited-ocr --output /tmp/ocr.md
```

Engine-level timings (load, vision, prefill, decode) come from the release bench
examples:

```bash
cargo run --release -p aimd-engine --example bench_asr_mlx
cargo run --release -p aimd-engine --example bench_ocr
```

For MCP model-reuse measurements, keep one MCP server session and send repeated requests against the same process.

## Dependency-upgrade smoke tests

Before bumping brew `mlx`, run the
checkpoint and E2E tiers on a Metal machine with the HF cache present. Also re-record the
bench rows above if the numbers move:

```bash
cargo test --workspace
AIMD_E2E=1 cargo test -p aimd-engine --lib
AIMD_E2E=1 cargo test -p aimd-process --test e2e
```
