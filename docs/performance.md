# Performance

This document records practical performance expectations for `aimd` processing paths. It is not a leaderboard: real throughput depends on input length, page resolution, model size, first-run model download time, GPU memory, and whether the path can reuse model weights already loaded in the current process.

Sections marked **Python era** record measurements of the Python package (`aimd-tool`
<= 0.23.x: mlx-audio, mlx-vlm, Transformers). They are kept as historical baselines; the
Python package and those backends are gone. Current numbers are in
[Native Rust engine (MLX backend)](#native-rust-engine-mlx-backend).

## High-level expectations

| Path | Typical bottleneck | Warm-cache behavior | Notes |
|------|--------------------|---------------------|-------|
| URL with subtitles | Network and yt-dlp metadata/subtitle download | Usually fast after metadata is available | `aimd` prefers subtitles over ASR whenever usable subtitles are found. |
| URL without subtitles | Media download, audio extraction, then ASR | Model reuse helps only inside the same long-running process | Each CLI invocation loads the checkpoint again; API/MCP reuse loaded models. |
| Local audio/video transcription | ffmpeg decode to 16 kHz mono and ASR inference | Better in API/MCP because loaded pipelines are cached in-process | Inputs over 10 minutes are transcribed in segments. |
| Document conversion | Pandoc / pdf-inspector / document structure | Mostly CPU and file I/O bound | EPUB conversion also writes chapter/image output trees. |
| Image OCR | OCR model inference | API/MCP reuse the loaded pipeline | VLM OCR models are heavier than classic detector/recognizer OCR. |
| PDF OCR | PDF rasterization plus one OCR pass per page | Model reuse helps; page rendering still scales with page count and DPI | PDF pages render with `mutool draw` (144 dpi, RGB) before OCR inference. |

## Model/runtime tradeoffs

| Task | Backend | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | native MLX | Qwen3-ASR 1.7B/0.6B (`4bit`, `6bit`, `8bit`) | Default `qwen3-asr-1.7b` 4-bit. The 0.6B model and lower bit widths trade quality for lower memory and faster load. 4-bit segments that loop retry once on 8-bit. |
| OCR | native MLX | Unlimited-OCR (`4bit`, `6bit`, `8bit`) | Default 4-bit. Single-image gundam mode per page, no-repeat n-gram guard (`ngram=35`, `window=128`), `max_tokens=8192`. Without the guard the model can loop (e.g. `R. R. R. ...`) until max_tokens and take minutes per page. |
| ASR / OCR | remote OpenAI-compatible | server-defined | Latency is network plus server-side; no local load. |

### Python-era backends (historical)

| Task | Backend | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | `mlx` | Qwen3-ASR 1.7B/0.6B (`4bit`, `6bit`, `8bit`, `bf16`) | Best for Apple Silicon local transcription; lower-bit and 0.6B models trade quality/capability for lower memory and faster startup. |
| Transcription | `transformers` | Qwen3-ASR | Default on CUDA-capable non-Darwin platforms and explicit opt-in on macOS/MPS when a `Qwen/Qwen3-ASR-*-hf` (or legacy `Qwen/Qwen3-ASR-*`) model ID is provided. Uses native Transformers Qwen3-ASR (`transformers>=5.14.1`); no vendored model code, `qwen-asr`, vLLM, or SGLang runtime. The 0.6B-hf model is the lower-memory option; 1.7B-hf is the default quality-oriented CUDA option. |
| OCR | `mlx-vlm` | Unlimited-OCR (`4bit`, `6bit`, `8bit`, `bf16`) | Default macOS OCR path via `mlx-community/Unlimited-OCR-4bit` and `mlx-vlm>=0.6.4`. Each page uses single-image gundam mode with Baidu's sliding-window no-repeat n-gram guard (`ngram=35`, `window=128`) and `max_tokens=8192`. PDFs are OCR'd page-by-page. Without the n-gram guard the model can loop (e.g. `R. R. R. ...`) until max_tokens and take minutes per page. |
| OCR | `mlx-vlm` | GLM-OCR (`4bit`, `6bit`, `8bit`, `bf16`) | Optional macOS VLM OCR path using the four `mlx-community/GLM-OCR-*` checkpoints. |
| OCR | `transformers` | Unlimited-OCR | Default Linux/CUDA OCR path using Baidu's custom `infer`/`infer_multi` API and trusted remote code. Results are read from the model's saved output files. |
| OCR | `transformers` | GLM-OCR | Linux/CUDA OCR path using the dedicated `zai-org/GLM-OCR` adapter. |

## Apple Silicon ASR comparison (Python era)

MPS can run Qwen3-ASR through native Transformers (`transformers>=5.14.1`), but macOS still defaults to MLX. In our smoke/latency checks, the MPS path was viable and useful as an explicit `Qwen/Qwen3-ASR-*-hf` opt-in, but the quantized MLX models remained the better default for Apple Silicon because they are simpler to maintain, smaller, and at least as fast in warm-cache runs.

Input shapes differed slightly between runs, so treat these as practical observations rather than a strict leaderboard.

| Date | Hardware/runtime | Backend/model | Input | Warm latency | Memory observation | Result / decision |
|------|------------------|---------------|-------|--------------|--------------------|-------------------|
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-0.6B-4bit` | 7.24s generated Chinese WAV | ~0.18s | MLX peak ~1.46 GB | Fastest/lower-memory observed Qwen3-ASR path; suitable Apple Silicon opt-in for lower-memory MLX use. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-4bit` | 7.24s generated Chinese WAV | ~0.39s | MLX peak ~2.5 GB | Balanced default-quality Apple Silicon path; macOS default remains MLX. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-8bit` | 7.24s generated Chinese WAV | ~0.56s | MLX peak ~3.32 GB | Higher memory and slower than 4-bit in this smoke. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `Qwen/Qwen3-ASR-0.6B` (legacy ID; now maps to `...-0.6B-hf`) | generated Chinese WAV | ~0.25s after local SDPA + KV-cache adaptation on a short smoke; earlier 7.24s run was ~0.45s | MPS allocated ~1.5 GB in earlier run | Historical smoke on vendored backend. Current path uses native Transformers `Qwen/Qwen3-ASR-*-hf` and still does not replace MLX as the macOS default. |

Policy at the time: MLX stayed the macOS default and `Qwen/Qwen3-ASR-*-hf` ids opted into Transformers. The Rust release removed the Transformers path; only the MLX checkpoints remain.

## Native Rust engine (MLX backend)

The native engine (spec 002) runs only on MLX through the `aimd-mlx` C-ABI
shim (brew `mlx` 0.32.4). Numbers are from Apple M2 Pro (19-core GPU,
32 GB), release builds of
`crates/aimd-engine/examples/{bench_asr_mlx,bench_ocr}.rs`, rounds
2-3 steady state. The "Reference" rows are the Python-era baselines (mlx-audio,
mlx_vlm) on the same machine, measured before the Python package was removed.

### Qwen3-ASR-1.7B-4bit

| Date | Backend | Input | Load | Warm latency | Realtime | Notes |
|------|---------|-------|------|--------------|----------|-------|
| 2026-10-25 | mlx-audio (Metal) | zh 8.56 s / en 3.75 s TTS clips | 10.01 s | zh 0.43 s; en 0.25 s | 19.9x / 15x | Reference. |
| 2026-10-25 | native MLX | same | 0.42-0.45 s warm / 3.81 s cold | zh 0.40-0.44 s; en 0.23-0.27 s | ~20x / ~15x | Token-identical text (incl. the same 上下文忠诚 mishearing); 4-bit triples go straight into `quantized_matmul` (no eager dequant), hence ~24x faster load. Unchanged after the CPU-path deletion (transcripts byte-identical). |
| 2026-10-26 | mlx-audio 0.5.3 (Metal) | 59.96 s en TTS clip (797-token prompt, 191 tokens) | — | 2.88-2.95 s | ~20.5x | Reference for the long clip. |
| 2026-10-26 | native MLX, `main` (concatenate KV, sync per step) | zh 4.83 s / en 4.09 s / 59.96 s en | — | zh 0.32 s; en 0.31 s; long 3.47-6.05 s (median 4.18 s) | ~15x / ~13x / ~14x | WS-13 A/B baseline, interleaved with the next row on a loaded machine. |
| 2026-10-26 | native MLX, WS-13 | same | — | zh 0.30-0.31 s; en 0.29-0.30 s; long 3.07-3.20 s (median 3.13 s) | ~16x / ~14x / ~19x | Step-256 preallocated KV (`slice_update`), `async_eval`-pipelined decode, compiled SwiGLU. Long clip -25% and far steadier; transcripts byte-identical. On a quiet machine: long 2.73-2.75 s; encoder 0.44 s, prefill 0.67 s (~1.2k tok/s), decode 1.65 s (~117 tok/s). |

### Unlimited-OCR-4bit

| Date | Backend | Input | Load | Vision | Prefill | Decode | Total | Notes |
|------|---------|-------|------|--------|---------|--------|-------|-------|
| 2026-10-25 | mlx_vlm 0.7.0 | `golden/ocr_e2e/small_text.png` (no crops) | 3.26 s | 0.35-0.37 s | in prompt | 199-206 tok/s | 0.87-0.88 s | Reference. |
| 2026-10-25 | native MLX | same | 0.58 s | 0.36-0.37 s | 0.12 s (2300 tok/s) | 200-205 tok/s | 0.82-0.84 s | Markdown identical (similarity 1.0), finish=stop, 69/69 tokens. |
| 2026-10-25 | mlx_vlm 0.7.0 | `golden/ocr_e2e/page_text.png` (4x5 crops, 2327-token prompt) | — | 2.50-2.55 s | ~0.77 s | 170-177 tok/s | 4.51-4.65 s | Reference. |
| 2026-10-25 | native MLX | same | — | 2.61-2.63 s | 0.79-0.82 s | 165-175 tok/s | 4.67-4.75 s | Markdown identical, finish=stop, 197/197 tokens. Vision + prefill are bit-exact vs mlx_vlm goldens; single-token decode drifts ~1 bf16 ulp (brew mlx 0.32.4 vs Python mlx 0.32.2 small-M qmv), shifting some bbox coordinates by 2/1000. |
| 2026-10-26 | native MLX, `main` vs WS-13 | same, interleaved on a loaded machine | — | 2.59-2.80 s both | 0.77-0.92 s both | 166-177 tok/s both | 4.64-4.87 s both | Compiled SwiGLU (dense MLP, shared experts, `SwitchGLU`) is neutral within noise; OCR already had the step-256/ring KV and `async_eval` pipelining. |

Retired: the hand-written CPU prototype reached 0.1-0.3x realtime on ASR
and ~734 s per OCR page (vs 4.8 s on MLX); it was deleted under the
MLX-only decision.

WS-13 findings: in the pipelined ASR loop the host builds a step's graph in
~0.4 ms, and the remaining ~8 ms per token is GPU work plus Metal encode for
~25 kernels per layer x 28 layers. Elementwise fusion (`mlx::core::compile`)
buys little there. A whole-step `compile` does not fit because the KV cache
grows every step, so a shape-keyed compile would retrace each token and a
shapeless one cannot express the slice bounds (mlx-lm doesn't compile the
step either).

WS-14 findings (prompt-lookup speculative decoding, not shipped): an
m-token Unlimited-OCR verify forward costs 4.3 / 6.5 / 11.0 / 19.4 / 24.6 ms
for m = 1 / 2 / 4 / 8 / 16, nearly linear because every routed token reads
its own 6 of 64 experts. On six real OCR pages, n-gram drafts were accepted
for only 1.01-1.15 tokens per verify, so simulated decode came out
0.63-1.01x. Remaining decode headroom is WS-08 (custom Metal kernels inside
the MLX graph).

### WS-08 decode-step kernel profile

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
| OCR router: gate matmul (W_g.T view) + softmax + top-6 | ~8.5 + 2.6 + ~12 | — | dispatch-bound |
| OCR sdpa 10 heads, kv 2455 (prefill + ring) | 102.2 | 12.6 MB | 74% |
| OCR lm_head 1280->129280 (per token) | 561 | 93 MB | ~100% |

Where an OCR token's ~5.1 ms goes: attention over the kept 2327-token prefill
KV ~1.2 ms, routed experts ~1.0 ms, q/k/v/o ~0.55 ms, shared experts ~0.5 ms,
lm_head 0.56 ms, router ~0.25 ms, the rest small ops. ASR is dominated by
qmv (~6 ms of ~8.5 ms) already at 60-84% of roofline, plus lm_head at
roofline.

Candidates ranked by expected saving per token:

1. ~~Fused q|k|v projection~~ (tried 2026-10-26, not kept): outputs were
   byte-identical (3 ASR clips, 7 OCR images) but decode moved <1% (OCR
   196-204 tok/s both ways). In the real graph the three projections share
   an input, so MLX dispatches them concurrently: 41.8 µs per ASR layer vs
   35.6 µs fused, not the 44 µs the serial chain suggests.
2. Fused MoE expert kernel (gather up|gate + SwiGLU + down + weighted sum):
   82 µs at 57% -> ~50 µs per layer, OCR ~0.35 ms.
3. Split-K decode attention for long KV (flash-decoding style): OCR
   102 -> ~80 µs per layer (~0.25 ms), ASR ~0.3 ms on long clips.
4. Fused router (gemv + softmax + top-6 in one kernel): ~23 -> ~4 µs per
   layer, OCR ~0.2 ms.
5. Small-op fusion (rms_norm + rope + KV row write): ~10 µs per layer, both
   models.

Caveat: the chain serializes every op, so it overstates ops that run
concurrently in the decoder (q/k/v, gate/up, shared vs routed experts).
Treat the per-op times as upper bounds and the serially dependent paths
(router -> experts, attention) as the reliable targets. Ceiling if all land:
OCR decode ~1.2x, ASR ~1.1x. Prefill and the OCR
vision tower (2.6 s of a 4.8 s page) are separate targets.

## Smoke-test record (Python era)

Functional smoke tests of the Python package, not benchmarks. They verified that the
heavyweight model paths could load and run in development environments.

| Date | Hardware | Command/path | Input | Observed result |
|------|----------|--------------|-------|-----------------|
| 2026-06-22 | NVIDIA GeForce RTX 5090, Linux/WSL2 | `process_ocr(..., model="unlimited_ocr")` | Generated 512×160 PNG containing `Hello OCR 123` | Returned `HelloOCR 123` after first loading `baidu/Unlimited-OCR`. |
| 2026-07-31 | Apple Silicon, macOS, mlx-vlm 0.6.8 | `aimd biorxiv.pdf --task ocr --start 0 --end 0` (default `unlimited_ocr`) | bioRxiv 10.64898/2026.06.01.729423v1 full PDF, page 0 only | Cold run ~22s including model fetch; output contained title `Direct Detection and Atomic Modeling of Ligands in Cryo-EM Maps Using Deep Learning` plus author/affiliation text with grounding `<\|det\|>` markers. |
| 2026-07-31 | Apple Silicon, macOS, mlx-vlm 0.6.8 | Unlimited-OCR page timing after n-gram fix | bioRxiv pages 0/4/10 at aimd 2× render | Healthy pages ~6–11s (`finish=stop`, ~350–826 gen tokens). Pre-fix page 10 looped to `max_tokens` (~80s at 8192). Post-fix page 10 ~9s with stop. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `transcribe_file(..., model="Qwen/Qwen3-ASR-0.6B", language="zh")` | Generated short Chinese WAV | Historical smoke: non-empty Chinese text via vendored Transformers backend. Prefer `Qwen/Qwen3-ASR-0.6B-hf` with `transformers>=5.14.1` now. |

## Measurement guidance

When comparing models, measure warm and cold runs separately:

1. **Cold run**: clear model cache/process state and include model download/load time.
2. **Warm run**: run multiple inputs in one API/MCP process, or use the engine bench examples, so loaded pipelines are reused.
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

For API/MCP model-reuse measurements, start the server once and send repeated requests against the same process.

## Dependency-upgrade smoke tests

Before bumping brew `mlx` (and `MLX_BREW_VERSION` in `.github/workflows/ci.yml`), run the
checkpoint and E2E tiers on a Metal machine with the HF cache present. Also re-record the
bench rows above if the numbers move:

```bash
cargo test --workspace
AIMD_E2E=1 cargo test -p aimd-engine --lib
AIMD_E2E=1 cargo test -p aimd-process --test e2e
```
