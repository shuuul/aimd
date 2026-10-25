# Performance

This document records practical performance expectations for `aimd` processing paths. It is not a leaderboard: real throughput depends on input length, page resolution, model size, first-run model download time, GPU memory, and whether the path can reuse cached model weights in the current Python process.

## High-level expectations

| Path | Typical bottleneck | Warm-cache behavior | Notes |
|------|--------------------|---------------------|-------|
| URL with subtitles | Network and yt-dlp metadata/subtitle download | Usually fast after metadata is available | `aimd` prefers subtitles over ASR whenever usable subtitles are found. |
| URL without subtitles | Media download, audio extraction, then ASR | Model reuse helps only inside the same long-running process | CLI invocations pay process startup costs each time; API/MCP can reuse loaded models. |
| Local audio/video transcription | Audio decode/transcode in `aimd.plugins.asr` and ASR model inference | Better in API/MCP because model objects are cached in-process | Non-WAV inputs may be converted to WAV before transcription. |
| MarkItDown document conversion | Parser/Pandoc/document structure | Mostly CPU and file I/O bound | EPUB conversion also writes chapter/image output trees. |
| Image OCR | OCR model inference | API/MCP can reuse loaded model objects | VLM OCR models are heavier than classic detector/recognizer OCR. |
| PDF OCR | PDF rasterization plus one or more OCR passes | Model reuse helps; page rendering still scales with page count and DPI | PDF pages render with PyMuPDF before OCR model inference. |

## Model/runtime tradeoffs

| Task | Backend | Model family | Expected performance profile |
|------|--------|--------------|------------------------------|
| Transcription | `mlx` | Qwen3-ASR 1.7B/0.6B (`4bit`, `6bit`, `8bit`, `bf16`) | Best for Apple Silicon local transcription; lower-bit and 0.6B models trade quality/capability for lower memory and faster startup. |
| Transcription | `transformers` | Qwen3-ASR | Default on CUDA-capable non-Darwin platforms and explicit opt-in on macOS/MPS when a `Qwen/Qwen3-ASR-*-hf` (or legacy `Qwen/Qwen3-ASR-*`) model ID is provided. Uses native Transformers Qwen3-ASR (`transformers>=5.14.1`); no vendored model code, `qwen-asr`, vLLM, or SGLang runtime. The 0.6B-hf model is the lower-memory option; 1.7B-hf is the default quality-oriented CUDA option. |
| OCR | `mlx-vlm` | Unlimited-OCR (`4bit`, `6bit`, `8bit`, `bf16`) | Default macOS OCR path via `mlx-community/Unlimited-OCR-4bit` and `mlx-vlm>=0.6.4`. Each page uses single-image gundam mode with Baidu's sliding-window no-repeat n-gram guard (`ngram=35`, `window=128`) and `max_tokens=8192`. PDFs are OCR'd page-by-page. Without the n-gram guard the model can loop (e.g. `R. R. R. ...`) until max_tokens and take minutes per page. |
| OCR | `mlx-vlm` | GLM-OCR (`4bit`, `6bit`, `8bit`, `bf16`) | Optional macOS VLM OCR path using the four `mlx-community/GLM-OCR-*` checkpoints. |
| OCR | `transformers` | Unlimited-OCR | Default Linux/CUDA OCR path using Baidu's custom `infer`/`infer_multi` API and trusted remote code. Results are read from the model's saved output files. |
| OCR | `transformers` | GLM-OCR | Linux/CUDA OCR path using the dedicated `zai-org/GLM-OCR` adapter. |

## Apple Silicon ASR comparison

MPS can run Qwen3-ASR through native Transformers (`transformers>=5.14.1`), but macOS still defaults to MLX. In our smoke/latency checks, the MPS path was viable and useful as an explicit `Qwen/Qwen3-ASR-*-hf` opt-in, but the quantized MLX models remained the better default for Apple Silicon because they are simpler to maintain, smaller, and at least as fast in warm-cache runs.

Input shapes differed slightly between runs, so treat these as practical observations rather than a strict leaderboard.

| Date | Hardware/runtime | Backend/model | Input | Warm latency | Memory observation | Result / decision |
|------|------------------|---------------|-------|--------------|--------------------|-------------------|
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-0.6B-4bit` | 7.24s generated Chinese WAV | ~0.18s | MLX peak ~1.46 GB | Fastest/lower-memory observed Qwen3-ASR path; suitable Apple Silicon opt-in for lower-memory MLX use. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-4bit` | 7.24s generated Chinese WAV | ~0.39s | MLX peak ~2.5 GB | Balanced default-quality Apple Silicon path; macOS default remains MLX. |
| 2026-06-23 | Apple Silicon, macOS, MLX | `mlx-community/Qwen3-ASR-1.7B-8bit` | 7.24s generated Chinese WAV | ~0.56s | MLX peak ~3.32 GB | Higher memory and slower than 4-bit in this smoke. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `Qwen/Qwen3-ASR-0.6B` (legacy ID; now maps to `...-0.6B-hf`) | generated Chinese WAV | ~0.25s after local SDPA + KV-cache adaptation on a short smoke; earlier 7.24s run was ~0.45s | MPS allocated ~1.5 GB in earlier run | Historical smoke on vendored backend. Current path uses native Transformers `Qwen/Qwen3-ASR-*-hf` and still does not replace MLX as the macOS default. |

Operational policy from these measurements:

- macOS Apple Silicon automatic transcription backend remains `mlx`.
- Passing `--model Qwen/Qwen3-ASR-0.6B-hf` or `--model Qwen/Qwen3-ASR-1.7B-hf` (legacy non-`-hf` IDs still work) explicitly opts into the native Transformers backend, including on MPS.
- Dependency upgrades that affect Transformers should run the opt-in Qwen3-ASR integration smoke below before release.

## Native Rust engine (WS-06, CPU baseline)

First measurements of the hand-written CPU engine (spec 002 WS-06) vs
mlx-audio on the same machine (Apple M2 Pro, 19-core GPU, 32 GB). Native
numbers are `cargo run --release` of `native/crates/aimd-engine/examples/bench_asr.rs`
(2026-10-25; un-optimized scalar/blocked CPU kernels, no NEON yet).

| Date | Backend | Input | Load (cold) | Warm latency | Realtime | Notes |
|------|---------|-------|-------------|--------------|----------|-------|
| 2026-10-25 | native CPU (release) | `mlx-community/Qwen3-ASR-1.7B-4bit` | 10.85 s (mmap + eager dequant to f16 caches; tokenizer 0.13 s) | zh 8.56 s TTS: ~61 s; en 3.75 s: ~43 s | ~0.1x | Text token-identical to MLX on both clips (incl. the same 上下文忠诚 mishearing). No first-run JIT penalty (flat across rounds). |
| 2026-10-25 | mlx-audio (Metal) | same checkpoints/clips | 10.01 s | zh 8.56 s: 0.43 s; en 3.75 s: 0.25 s | 19.9x / 15x | First round carries Metal warm-up (1.19 s -> 0.43 s). |

Interpretation:

- Startup is at parity with MLX (~11 s); the eager f16 dequant cache does
  not regress cold start on this machine.
- Warm CPU inference is ~0.1x realtime — far below the >= 5x CPU gate and
  ~150x behind MLX's Metal path. This is the expected un-optimized
  baseline (scalar f16 kernels, naive tied-embedding logits matmul over
  the 152k vocab per step). Performance remains the job of the MLX shim
  variant (WS-11) and the GPU phases per the v0.4 design; obvious CPU
  headroom exists (NEON, blocked+rayon logits matmul) if we want the
  default CPU wheel to be interactive.
- Behavioral parity: identical transcription text (including identical
  errors) vs MLX on both benchmark clips is the strongest end-to-end
  fidelity signal so far.

| 2026-10-25 | native CPU (release, SIMD pass) | same | 10.12 s | zh 8.56 s TTS: ~39 s (0.2x); en 3.75 s: ~30 s | 0.2x | "Obvious wins" pass: vocab-axis rayon for small-m matmul + transposed-B blocked logits + stable NEON f32x4 FMA kernels (fp16 vector loads are nightly-only) + AVX2/F16C/FMA on x86_64. 1.4-1.6x vs the naive pass; outputs still token-identical to MLX. Per v0.5 the CPU path stays the correctness baseline — further CPU tuning is not planned; perf rides the MLX/Metal variant. |

| 2026-10-25 | native CPU (release, bit-trick cvt) | same | 10.44 s | zh 8.56 s TTS: ~33 s (0.3x); en 3.75 s: ~25 s (0.2x) | 0.2-0.3x | f16->f32 conversion now fully vectorized on stable NEON via the integer bit algorithm (exhaustively bit-exact over all 65536 f16 patterns). 1.2x over the prior SIMD pass; cumulative 1.75-1.9x over naive. Conversion is no longer the dominant cost — further CPU gains need register-tiled microkernels (real kernel work, out of the casual pass; CPU stays the fallback path per v0.5). |

| 2026-10-25 | native MLX backend (release, `--features mlx`) | same checkpoints/clips | **0.42 s warm / 3.81 s cold** | zh 8.56 s: 0.40 s (21.6x); en 3.75s: 0.23 s (16.6x) | 21.6x / 16.6x | WS-11b: full-MLX graph (tower conv2d + SDPA + fast::rope/layer_norm on Metal), 4-bit triples directly into quantized_matmul (no eager dequant — hence the 24x faster load vs mlx-audio), argmax in-mlx with 4-byte readback. **Slightly faster than mlx-audio** (0.92-0.93x of baseline latency) with token-identical outputs incl. the same mishearing; golden 16-token argmax stream token-exact, tower parity 2.0e-7 abs. |

## Current smoke-test record

The following are functional smoke tests, not benchmarks. They verify that heavyweight model paths can load and execute in development environments.

| Date | Hardware | Command/path | Input | Observed result |
|------|----------|--------------|-------|-----------------|
| 2026-06-22 | NVIDIA GeForce RTX 5090, Linux/WSL2 | `process_ocr(..., model="unlimited_ocr")` | Generated 512×160 PNG containing `Hello OCR 123` | Returned `HelloOCR 123` after first loading `baidu/Unlimited-OCR`. |
| 2026-07-31 | Apple Silicon, macOS, mlx-vlm 0.6.8 | `aimd biorxiv.pdf --task ocr --start 0 --end 0` (default `unlimited_ocr`) | bioRxiv 10.64898/2026.06.01.729423v1 full PDF, page 0 only | Cold run ~22s including model fetch; output contained title `Direct Detection and Atomic Modeling of Ligands in Cryo-EM Maps Using Deep Learning` plus author/affiliation text with grounding `<\|det\|>` markers. |
| 2026-07-31 | Apple Silicon, macOS, mlx-vlm 0.6.8 | Unlimited-OCR page timing after n-gram fix | bioRxiv pages 0/4/10 at aimd 2× render | Healthy pages ~6–11s (`finish=stop`, ~350–826 gen tokens). Pre-fix page 10 looped to `max_tokens` (~80s at 8192). Post-fix page 10 ~9s with stop. |
| 2026-06-23 | Apple Silicon, macOS, Transformers 5.12.1 / MPS | `transcribe_file(..., model="Qwen/Qwen3-ASR-0.6B", language="zh")` | Generated short Chinese WAV | Historical smoke: non-empty Chinese text via vendored Transformers backend. Prefer `Qwen/Qwen3-ASR-0.6B-hf` with `transformers>=5.14.1` now. |

## Measurement guidance

When comparing models, measure warm and cold runs separately:

1. **Cold run**: clear model cache/process state and include model download/load time.
2. **Warm run**: run multiple inputs in one API/MCP process or one Python process so model objects can be reused.
3. **Separate stages** when possible: download/extraction, PDF rasterization, audio conversion, and model inference.
4. **Record input shape**: audio duration, page count, image dimensions, PDF DPI, language, and model ID.
5. **Record environment**: OS, Python version, CPU/GPU, CUDA/Metal runtime, dependency versions, and whether weights were already cached.

A minimal local timing harness can wrap the public CLI:

```bash
/usr/bin/time -p aimd input.mp3 --model Qwen/Qwen3-ASR-0.6B-hf --output /tmp/out.md
/usr/bin/time -p aimd scan.pdf --model unlimited_ocr --output /tmp/ocr.md
```

For API/MCP model-reuse measurements, start the server once and send repeated requests against the same process.

## Dependency-upgrade smoke tests

Qwen3-ASR uses native Transformers support (`transformers>=5.14.1`, `Qwen/Qwen3-ASR-*-hf`). Before upgrading Transformers, run the normal tests and the opt-in real inference smoke:

```bash
uv run pytest -q tests/test_transformers_asr.py tests/test_capabilities.py
AIMD_RUN_QWEN3_ASR_INTEGRATION=1 uv run pytest -q tests/test_transformers_asr.py -k real_inference_smoke
```

Set `AIMD_QWEN3_ASR_TEST_AUDIO=/path/to/audio.wav` to use a fixed local audio fixture; otherwise the macOS smoke creates a short synthetic speech file with `say` and `ffmpeg`.
