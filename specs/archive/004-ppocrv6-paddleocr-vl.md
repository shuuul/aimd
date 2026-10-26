---
id: "004"
title: "PP-OCRv6 and PaddleOCR-VL-1.6 OCR engines; ASR language auto-detection"
status: Completed
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 004 — PP-OCRv6 and PaddleOCR-VL-1.6 OCR engines; ASR language auto-detection

## Context

The native engine (spec 002) ships one OCR family, Unlimited-OCR, and
`resolve_mlx_ocr_model` rejects every other model. Two more local OCR
families are requested:

- **PP-OCRv6** (classic det + rec), largest tier only: `medium`.
  Checkpoints: `PaddlePaddle/PP-OCRv6_medium_det_safetensors` (88 MB, f32)
  and `PaddlePaddle/PP-OCRv6_medium_rec_safetensors` (77 MB, f32, 18710-class
  multilingual CTC dict in `preprocessor_config.json`). Reference MLX port:
  `shuuul/mlx4ocr` (Python, `src/mlx4ocr/{models,preprocess,postprocess,pipeline}`).
  It uses OpenCV (`resize`, `findContours`, `minAreaRect`, `warpPerspective`)
  and `pyclipper` for DB unclip.
- **PaddleOCR-VL-1.6** (0.9B VLM: NaViT/SigLIP vision 27 layers + 2x2 merge
  projector + ERNIE-4.5 0.3B decoder, 18 layers, GQA 16/2, 3-D mRoPE
  `[16, 24, 24]`). Official checkpoint `PaddlePaddle/PaddleOCR-VL-1.6`
  (bf16, 1.9 GB). mlx-community publishes only `-4bit` and `-5bit`. Reference
  MLX port: `mlx_vlm/models/paddleocr_vl` (mlx-vlm 0.7.10).

Also requested: `--language` stays, but omitting it must auto-detect.
Today `aimd_process::models::resolve_asr_language(None)` returns `Chinese`,
which forces the Qwen3-ASR prompt even though the engine already supports an
unforced prompt (`build_asr_prompt(context, None, ..)` and
`parse_qwen_output("language X<asr_text>...")`). OCR models here are all
multilingual and take no language hint.

Verified finding (2026-10-26): mlx4ocr downloads the 232 MB Paddle
`*_rec_pretrained.pdparams` pickle to "repair" the HF `small`/`medium` rec
heads. Comparing tensors shows the HF file is not corrupt: its
`head.encoder.conv_block.0` is exactly `skip_conv` and `conv_block.1` is
exactly `conv_reduce` (max abs diff 0.0, BN stats included). A key swap at
load replaces the pickle download.

## Goal and success criteria

- [x] `aimd scan.png --model pp-ocrv6` runs PP-OCRv6 medium det + rec natively
      on MLX; det probability map and rec logits match mlx4ocr goldens
      (hermetic slices + checkpoint tier), and DB boxes / final text match
      the mlx4ocr e2e golden on its `examples/images` (lossless inputs; see
      the JPEG decision).
- [x] `aimd scan.png --model paddleocr-vl` runs PaddleOCR-VL-1.6 natively
      (bf16 default; `4bit`/`6bit`/`8bit` quantize at load with the mlx-vlm
      convert rule: every module with `in_dim % 64 == 0`);
      vision features, decoder logits, and greedy text match mlx-vlm goldens.
- [x] Omitting `--language` auto-detects for local ASR (no forced language
      in the prompt), for every segment and for the URL audio fallback;
      explicit `--language` still forces it. Accepted codes widen to the
      engine's 30-language Qwen3-ASR table.
- [x] CLI help, MCP schema descriptions, README, and AGENTS.md list the new
      aliases and the auto-detect default; no GLM-OCR/Transformers mentions.
- [x] Corpus gate: both new models score within the AGENTS.md bounds of
      their Python reference (mlx4ocr / mlx-vlm) on `scripts/corpus` pages.
- [x] `prek`, fmt, clippy `-D warnings`, and the hermetic tier pass; hermetic
      runtime stays about 2 minutes.

## Scope and non-goals

In scope:

- `aimd-mlx` ops: grouped/dilated conv, transposed conv, unary/binary/reduce
  families, `where`, `clip`, tanh GELU, affine `quantize`.
- `aimd-engine::models::pp_ocrv6` and `aimd-engine::models::paddleocr_vl`.
- Host-side image ops for PP-OCRv6 (cv2-exact bilinear resize, contour
  tracing, min-area rect, rectangle unclip, perspective crop) hand-written.
- Model resolution, `NativeRuntime::load_ocr` dispatch, CLI/MCP help, docs.
- ASR language auto-detection.

Not in scope:

- PP-OCRv6 `tiny`/`small`, textline orientation or doc-unwarping models,
  PP-DocLayout layout analysis, PaddleOCR-VL table/formula/chart task
  prompts beyond full-page `OCR:`.
- Changing the default OCR model (stays `unlimited-ocr`).
- A language hint for OCR (all three families are multilingual; a local OCR
  run ignores `--language`).

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | PP-OCRv6 ships `medium` only; alias `pp-ocrv6` (also `pp-ocrv6-medium`); internal model key `PaddlePaddle/PP-OCRv6_medium` resolves both `_det_safetensors` and `_rec_safetensors` repos | User asked for the largest size only | WS-02, WS-05 |
| 2026-10-26 | Swap HF rec `head.encoder.conv_block.{0,1}` at load instead of downloading Paddle pretrained pickles | Tensor comparison shows an exact swap, not corruption; avoids a 232 MB non-HF download and a pickle parser | WS-02 |
| 2026-10-26 | PP-OCRv6 has no precision: `--precision` is ignored with one warning (same pattern as remote backends) | Checkpoints are dense f32 conv nets; quantization does not apply | WS-05 |
| 2026-10-26 | PaddleOCR-VL loads the official `PaddlePaddle/PaddleOCR-VL-1.6` bf16 checkpoint for every precision; omitted precision means bf16 (dense); `4bit`/`6bit`/`8bit` quantize at load (affine, group 64) with the mlx-vlm `convert` layer predicate | mlx-community has no 6/8-bit repos; one source keeps parity simple, and the 0.3B decoder gains little from 4-bit by default. 4-bit parity is checked against `mlx-community/PaddleOCR-VL-1.6-4bit` | WS-03, WS-05 |
| 2026-10-26 | PaddleOCR-VL runs full-page `OCR:` with the checkpoint's image processor bounds (min 112896, max 1003520 pixels) | Matches mlx4ocr's VLM usage; layout analysis is out of scope | WS-03 |
| 2026-10-26 | Omitted ASR language = auto-detect everywhere (local, retry, URL fallback); remote ASR already omits the field | User request; engine already supports the unforced prompt | WS-04 |
| 2026-10-26 | Local OCR ignores `--language` silently (help text says it is a transcription hint) | All OCR families are multilingual | WS-05 |
| 2026-10-26 | PaddleOCR-VL quantization follows the mlx-vlm convert rule (212 modules incl. vision qkv/out_proj/fc1, projector, position embedding; vision `fc2` stays dense), not decoder-only | Keeps 4-bit bit-exact with `mlx-community/PaddleOCR-VL-1.6-4bit` | WS-03 |
| 2026-10-26 | Quantized PaddleOCR-VL model keys are `PaddlePaddle/PaddleOCR-VL-1.6:<precision>` | A colon cannot appear in a Hub repo id, so the key never collides with a real repo; `aimd_process::models::ocr_family` decodes it | WS-05 |
| 2026-10-26 | Accept JPEG decode drift for PP-OCRv6 (zune-jpeg vs PIL libjpeg-turbo, up to 4 levels/pixel); parity is bit-exact from decoded pixels and on lossless inputs (PNG, mutool-rendered PDF pages) | A libjpeg-turbo-exact decoder would be a new crate or a hand-written codec; on mlx4ocr's JPEG examples text differs on one line of 39 | WS-02 |
| 2026-10-26 | New OCR baselines live in `scripts/corpus_baseline_paddle.py` (mlx 0.32.4, mlx-vlm 0.7.10, mlx4ocr 0.1.2); `corpus_gate --ocr-model` selects the family and its baseline key | The Unlimited-OCR baseline stays pinned to mlx-vlm 0.7.0 | WS-06 |
| 2026-10-26 | Engine API contracts (below) are frozen before WS-02/WS-03 start so WS-05 can wire against them | Parallel workers in separate worktrees | WS-02, WS-03, WS-05 |

Engine API contracts:

```rust
// aimd_engine::models::pp_ocrv6::pipeline
pub struct PpOcrV6Pipeline { .. }
impl PpOcrV6Pipeline {
    pub fn load(det_dir: &Path, rec_dir: &Path) -> Result<Self, PipelineError>;
    pub fn recognize_path(&self, image: &Path) -> Result<PpOcrOutput, PipelineError>;
}
pub struct PpOcrOutput { pub lines: Vec<TextLine>, pub markdown: String }
pub struct TextLine { pub text: String, pub score: f32, pub quad: [[f32; 2]; 4] }
pub enum PipelineError { EmptyContent, /* load / image / mlx variants */ .. }

// aimd_engine::models::paddleocr_vl::pipeline
pub struct PaddleOcrVlPipeline { .. }
impl PaddleOcrVlPipeline {
    /// `quant_bits`: None = dense bf16; Some(4 | 6 | 8) quantizes at load.
    pub fn load(dir: &Path, quant_bits: Option<u32>) -> Result<Self, PipelineError>;
    pub fn recognize_path(&self, image: &Path, request: VlRequest) -> Result<VlOutput, PipelineError>;
}
#[derive(Default)] pub struct VlRequest { pub max_tokens: Option<usize> }
pub struct VlOutput { pub markdown: String, pub tokens: usize }
pub enum PipelineError { EmptyContent, .. }
```

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | `aimd-mlx` shim ops for conv nets and the VL graph | coordinator | Done | None | `cargo test -p aimd-mlx --test mlx_ops_conv` (CPU and Metal) |
| WS-02 | `aimd-engine::models::pp_ocrv6`: det (PPLCNetV4 + RepLKPAN + IntraCL + DB head), rec (PPLCNetV4 + LightSVTR + CTC), cv2-exact preprocessing, DB postprocess, crop, CTC decode, reading-order markdown; goldens via `scripts/export_ppocrv6_reference.py` (mlx4ocr) | agent ws02 | Done | WS-01 | hermetic golden tests; checkpoint parity tier; e2e text vs mlx4ocr on its examples |
| WS-03 | `aimd-engine::models::paddleocr_vl`: image processor, vision tower, projector, ERNIE decoder with mRoPE + KV cache, greedy generation, load-time quantization; goldens via `scripts/export_paddleocr_vl_reference.py` (mlx-vlm 0.7.10) | agent ws03 | Done | WS-01 | hermetic golden tests; checkpoint parity tier (bf16 + 4-bit vs mlx-community 4bit); e2e text vs mlx-vlm |
| WS-04 | ASR language auto-detection (process, Runtime trait, URL fallback, tests, help) | coordinator | Done | None | hermetic tests in `aimd-process` |
| WS-05 | Model resolution, `NativeRuntime` dispatch, CLI/MCP help, README/AGENTS docs | coordinator | Done | WS-02/WS-03 API | hermetic tests; CLI smoke |
| WS-06 | Corpus gate + performance notes for both models | coordinator | Done | WS-02, WS-03, WS-05 | `corpus_gate` within bounds; `docs/performance.md` |

## Verification

```bash
cargo test -p aimd-mlx --test mlx_ops_conv
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo test -p aimd-engine pp_ocrv6 paddleocr_vl     # checkpoint tier when cached
AIMD_E2E=1 cargo test -p aimd-engine --lib pp_ocrv6 paddleocr_vl
uv run scripts/corpus_baseline.py && cargo run --release -p aimd-engine --example corpus_gate
cargo xtask check-specs
```

## Documentation sync

- Durable product/developer docs: README (models, precision, language),
  `docs/performance.md`.
- Nearest local `AGENTS.md`: root `AGENTS.md` (crate map, model naming,
  precision, golden sets, ASR language).
- Parent/package guidance: none.
- Root guidance and roadmap: root `AGENTS.md`.

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-26 — coordinator — WS-01

- Changed: `crates/aimd-mlx/shim/shim.cpp`, `crates/aimd-mlx/src/lib.rs`
  (`conv2d_ex`, `conv_transpose2d`, `unary`/`binary`/`reduce` + `UnaryOp`/
  `BinaryOp`/`ReduceOp`, `where_`, `clip`, `gelu_tanh`, `quantize`),
  `crates/aimd-mlx/tests/mlx_ops_conv.rs`.
- Evidence: 5/5 pass with `AIMD_MLX_DEVICE=cpu` and on Metal.
- Remaining: none.
- Blockers: none.
- Next action: WS-02 and WS-03 start in parallel worktrees.

### 2026-10-26 — agent ws03 — WS-03

- Changed: `crates/aimd-engine/src/models/paddleocr_vl/`, `examples/bench_paddleocr_vl.rs`,
  `tests/golden/paddleocr_vl/`, `scripts/export_paddleocr_vl_reference.py` (commit `b0aaacc`).
- Evidence: vision stages bit-exact vs mlx-vlm on Metal; decoder first 6 steps bit-exact;
  greedy text identical in bf16 and 4-bit on both fixtures and a 1224x1584 page; 4-bit
  weights bit-exact (212/212) with the mlx-community repo. Page: bf16 8.3 s, 4-bit 5.3 s,
  parity with mlx-vlm. Extra API: `recognize_path_with_stats`, `recognize_rgb8`.
- Remaining: CUDA checkpoint tier; corpus gate (WS-06).
- Blockers: none.
- Next action: coordinator wires it (WS-05).

### 2026-10-26 — coordinator — WS-04, WS-05

- Changed: ASR language auto-detect (`5e3c79d`); OCR model keys + help text (`1a8fbe8`);
  `NativeRuntime::load_ocr` family dispatch with PaddleOCR-VL (`229a768`).
- Evidence: hermetic tier 341 passed; `aimd small_text.png --model paddleocr-vl`
  (bf16 and `--precision 4bit`) prints the expected text.
- Remaining: PP-OCRv6 dispatch after WS-02 merges; README/AGENTS model docs.
- Blockers: WS-02.
- Next action: merge WS-02.

### 2026-10-26 — agent ws02 — WS-02

- Changed: `crates/aimd-engine/src/models/pp_ocrv6/`, `examples/ppocrv6.rs`,
  `tests/golden/ppocrv6/`, `scripts/export_ppocrv6_reference.py` (commit `c64f7c0`).
- Evidence: prob maps, rec argmax/max, lines, scores, and quads identical to mlx4ocr
  on Metal for sample_doc, general_ocr_002, img_10; warm 0.21-0.50 s per image vs
  mlx4ocr 0.26-0.60 s.
- Remaining: CUDA checkpoint tier.
- Blockers: none.
- Next action: coordinator wires it (WS-05).

### 2026-10-26 — coordinator — WS-05, WS-06

- Changed: PP-OCRv6 dispatch + README/AGENTS docs (`65aecd1`); `corpus_gate --ocr-model`,
  `scripts/corpus_baseline_paddle.py`, `docs/performance.md`.
- Evidence: `aimd general_ocr_002.png --model pp-ocrv6` 1.3 s end to end; corpus gate
  PASS with 5/5 pages identical to the reference for PaddleOCR-VL bf16, PaddleOCR-VL
  4bit, and PP-OCRv6; Unlimited-OCR unchanged (5/5); hermetic tier 351 passed.
- Remaining: Linux CUDA checkpoint tier and corpus gate for both models (metacube).
- Blockers: none.
- Next action: none in this spec.

## Completion summary

Shipped two native OCR families and ASR language auto-detection.

- `pp-ocrv6`: PP-OCRv6 medium det + rec on MLX with hand-written cv2-exact host ops;
  bit-exact with mlx4ocr from decoded pixels; precision ignored with a warning.
- `paddleocr-vl`: PaddleOCR-VL-1.6 from the official bf16 checkpoint; `4bit`/`6bit`/
  `8bit` quantize at load (4-bit bit-identical to the mlx-community repo); greedy text
  identical to mlx-vlm 0.7.10; speed at parity.
- Omitted `--language` auto-detects for every local ASR path; 30 explicit codes.
- README documents the new models and how to use YouTube/podcast descriptions as ASR
  context; AGENTS.md, help text, and the MCP description list the new aliases.
- Follow-up outside this spec: run the checkpoint tier and corpus gate for both new
  models on Linux CUDA (metacube).
