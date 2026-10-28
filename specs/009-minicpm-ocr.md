---
id: "009"
title: "MiniCPM-OCR 1.0 with the PP-DocLayoutV3 region pipeline"
status: Active
created: 2026-10-28
updated: 2026-10-28
coordinator: "shuuul"
---

# 009 — MiniCPM-OCR 1.0 with the PP-DocLayoutV3 region pipeline

## Context

Requested: native support for `openbmb/MiniCPM-OCR-1.0`
(https://huggingface.co/openbmb/MiniCPM-OCR-1.0,
https://github.com/OpenBMB/MiniCPM-OCR).

Verified facts (2026-10-28):

- **Checkpoint**: one bf16 `model.safetensors` (~2.6 GB, `transformers`
  5.21, `model_type: minicpm_ocr`). Third-party MLX conversions exist
  (`nativ-community/MiniCPM-OCR-1.0-{bf16,4bit,6bit,8bit}`, mlx-vlm layout,
  group 64 affine); the official repo has no quantized MLX weights.
- **Vision**: SigLIP-400M NaViT tower (27 layers, hidden 1152, 16 heads,
  patch 14, `gelu_pytorch_tanh`, LN eps 1e-6, 70x70 learned position table
  indexed by nearest-neighbour bucketing of the patch grid). Default
  `downsample_mode: "4x"` **skips** the `vit_merger` window-attention merger
  (`use_vit_merger = mode != "4x"`); features go through `post_layernorm`
  and one `MiniCPMOcrDownsampleMLP` (2x2 patch merge -> LN(4608, 1e-6) ->
  Linear 4608 -> exact GELU -> Linear 1024).
- **Processor**: MiniCPM-V slicing (`scale_resolution` 448, `max_slice_nums`
  9, `slice_mode`, `use_image_id`), mean/std 0.5. Placeholders are
  `<|image_pad|>` (id 248056) runs wrapped in the MiniCPM-V image/slice
  markers.
- **Language model**: Qwen3.5 text (`qwen3_5_text`), 24 layers, hidden 1024,
  vocab 248100, tied embeddings. Every 4th layer is full attention (GQA 8/2,
  head_dim 256, q/k RMSNorm, sigmoid output gate `attn_output_gate`,
  partial RoPE 0.25 = 64 dims, theta 1e7); the other 18 are Gated DeltaNet
  linear attention (16 k/v heads, 128 dims, causal depthwise conv k=4,
  `A_log`, `dt_bias`, gated RMSNorm, recurrent state kept in f32).
  RMSNorm is `(1 + w)`, evaluated in f32. MiniCPM-V positions are 1-D, so the
  mRoPE sections collapse to plain RoPE.
- **Chat format**: `<|im_start|>user\n<image>\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
  EOS ids 248044/248046.
- **The model recognises regions, not pages.** The official SDK
  (`sdk/minicpmvocr`) runs **PP-DocLayoutV3** first (RT-DETR: HGNetV2-L
  backbone, hybrid encoder, 6-layer deformable decoder, mask-enhanced query
  init, global-pointer reading-order head; 25 PaddleX labels; 800x800 input;
  `PaddlePaddle/PP-DocLayoutV3_safetensors`, f32, 133 MB), crops every
  region, sends each crop with a label-specific prompt (`Text Recognition:`,
  `Table Recognition:`, `Formula Recognition:`, `Chart to Table:`,
  `Seal Recognition:`) and token budget, then assembles Markdown in reading
  order (`postprocess/markdown.py`, OTSL -> HTML tables).
  Decode: greedy (`top_k 1`), `no_repeat_ngram_size 100`.
- **References**: mlx-vlm `models/minicpm_ocr` (subclasses `minicpmv4_6`,
  `qwen3_5`, `gated_delta.py`) and `models/pp_doclayout_v3`;
  transformers `models/minicpm_ocr`; the SDK's
  `layouts/ppdoclayout3_torch.py` (numpy post-process: threshold 0.3, NMS
  same 0.6 / diff 0.98, containment filter, `MERGE_BBOXES_MODE`,
  reading order from `order_logits`).

Current repo: `OcrFamily` covers Unlimited-OCR 1/2, PaddleOCR-VL and
PP-OCRv6 (`crates/aimd-process/src/models.rs`); the engine has no linear
attention, no recurrent state cache, no deformable attention and no
`grid_sample`.

## Goal and success criteria

- [ ] `aimd scan.png --model minicpm-ocr` runs PP-DocLayoutV3 + MiniCPM-OCR
      natively on MLX and returns SDK-equivalent Markdown (titles, HTML
      tables, `$$` formulas, ignored header/footer/number labels).
- [ ] Omitted precision = dense bf16; `--precision 4bit|6bit|8bit` quantize
      at load (affine, group 64), bit-exact with an mlx-vlm conversion of the
      same checkpoint. PP-DocLayoutV3 always runs f32.
- [ ] Goldens: SigLIP stages + merger features, Gated DeltaNet / full-attn
      layer outputs and greedy text vs mlx-vlm; layout logits, boxes, labels
      and reading order vs the SDK torch provider (or mlx-vlm port);
      Markdown assembly and OTSL conversion vs the SDK functions.
- [ ] `--anchors` emits one `<!-- aimd:loc ... kind=<label> -->` per region
      from the layout boxes; anchors off stays byte-identical to the
      no-anchor output.
- [ ] Gated DeltaNet runs a custom Metal kernel when
      `metal_kernels_available()` and a stock-mlx-ops recurrence elsewhere;
      both match a dense reference within bf16 tolerance (`aimd-mlx/tests`).
- [ ] Corpus gate: `--ocr-model minicpm-ocr` scores within the AGENTS.md
      bounds on `scripts/corpus` pages, Metal and CUDA (no Python baseline;
      see Decisions).
- [ ] DSpark speculative decoding (`openbmb/MiniCPM-OCR-1.0-DSpark`, vLLM
      `dspark` semantics, 15 proposals per round) is on whenever the draft
      loads; it emits the same greedy text as plain decoding on the
      fixtures and corpus and decodes faster.
- [ ] CLI help, MCP schema text, README, AGENTS.md list `minicpm-ocr`; the
      default OCR model stays `unlimited-ocr-2`.
- [ ] `prek`, fmt, clippy `-D warnings`, hermetic tier (~2 min budget) pass.

## Scope and non-goals

In scope:

- `aimd-mlx`: Gated DeltaNet recurrence op (Metal kernel + stock fallback),
  causal depthwise conv1d with carried state, bilinear `grid_sample`
  (zeros padding, `align_corners=False`) for deformable attention, plus any
  missing elementwise ops (softplus, etc.).
- `aimd-engine::models::minicpm_ocr` (processor, vision, merger, hybrid
  decoder, generation) and `aimd-engine::models::pp_doclayout_v3`
  (backbone, encoder, decoder, heads, post-process, reading order).
- Region pipeline: crop, per-label prompt/budget, skip/ignore labels,
  Markdown assembly, OTSL -> HTML (hand-written port of the SDK).
- Model resolution, `NativeRuntime::load_ocr`, HF download of the three
  repos (recogniser, DSpark draft, layout), anchors, help text, docs,
  exporters, corpus baseline.
- DSpark speculative decoding for the recogniser (greedy verification).

Not in scope:

- `16x` downsample mode / `vit_merger` (weights are skipped at load).
- SDK extras: upright/orientation correction, cross-page table merging
  (SDK default off), test-time scaling / reward model, KIE prompts.
- DSpark adaptive verification (confidence head; the official serving
  config disables it) and sampled (non-greedy) drafting.
- Remote-backend instruction routing for a served MiniCPM-OCR (follow-up).
- Changing the default OCR model.
- Multi-page batching (`pages_per_request` stays 1).

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-28 | Ship the full layout + region pipeline in one change; no whole-page-only interim. | User decision: the model is a region recogniser; page-level `Text Recognition:` loses tables/formulas. | All |
| 2026-10-28 | Weights: always `openbmb/MiniCPM-OCR-1.0` bf16; omitted precision = bf16; 4/6/8-bit quantize at load with mlx-vlm's convert predicate. Model key = repo id plus `:<precision>` when quantized (PaddleOCR-VL pattern). | User decision; no third-party repo dependency. | WS-03, WS-06 |
| 2026-10-28 | Layout weights: `PaddlePaddle/PP-DocLayoutV3_safetensors`, f32, never quantized; loaded with the recogniser as one OCR family. | Official HF safetensors, small (133 MB). | WS-04, WS-06 |
| 2026-10-28 | Only `4x` downsample mode; `vit_merger.*` tensors are dropped at load. | Checkpoint and SDK default; 4x never runs `vit_merger`. | WS-02 |
| 2026-10-28 | Labels follow the SDK's `PX_LABELS` order (25 PaddleX labels: `display_formula`/`inline_formula`, `header_image`, `vertical_text`), not the HF `id2label` (which collapses duplicates). | SDK prompt map and assembly key on PaddleX names. | WS-04, WS-05 |
| 2026-10-28 | Prompts, per-label token budgets, `ignore_labels`, title/formula/table formats and post-processing copy the SDK `DEFAULT_PROMPT_MAP` and `postprocess/markdown.py`; image regions emit nothing (no `![image]()` placeholder, no asset). | Parity with the official pipeline; aimd has no image-asset path for OCR crops. | WS-05 |
| 2026-10-28 | Decode greedy with the n-gram guard at size 100 over the whole generation; no sampling. | SDK decode config (`top_k 1`, `no_repeat_ngram_size 100`). | WS-03 |
| 2026-10-28 | Gated DeltaNet: custom Metal kernel gated on `metal_kernels_available()`, stock-op recurrence otherwise; no CUDA kernel. | AGENTS.md MLX-only / WS-08 kernel rule. | WS-01 |
| 2026-10-28 | Alias `minicpm-ocr` (also `minicpm-ocr-1.0`); `OcrFamily::MiniCpmOcr { quant_bits }`. | Kebab-case naming convention. | WS-06 |
| 2026-10-28 | Superseded below: the numerical reference for the recogniser is mlx-vlm (pinned commit in the exporter) with the merger GELU patched to exact erf. | transformers' merger uses `nn.GELU()` (erf); mlx-vlm's `GELU(approx="precise")` is the tanh form. Everything else in mlx-vlm matches transformers' graph and is the MLX-native op order to pin. | WS-02, WS-03 |
| 2026-10-28 | Superseded below: Qwen3.5 norms follow mlx-vlm exactly: sanitized `w + 1` weights stay f32 and `nn.RMSNorm` output is cast back to the activation dtype; DeltaNet q/k use `rms_norm(eps = 1e-6 / head_dim)` (L2 norm with the epsilon on the sum) times `1/128` and `128^-0.5`; the gated norm weight is not shifted. | Each one changes bf16 rounding; together they make every layer bit-exact on Metal. | WS-03 |
| 2026-10-28 | Superseded below: Partial RoPE goes through a shim port of mlx-vlm's `mrope_apply` Metal kernel (`MlxArray::rope_positions`: fast `metal::cos`/`sin` on f32 angles); other backends use stock ops. | mlx's `cos`/`sin` ops are the precise variants and move full-attention outputs by 1 ulp. | WS-01, WS-03 |
| 2026-10-28 | Superseded below: Image features splice as `e + bf16(f - e)` (mlx-vlm `_apply_vision_delta`); the prompt starts at `<|im_start|>` with no extra BOS. | Bit-exact `inputs_embeds`. | WS-02 |
| 2026-10-28 | Only full-attention q/k/v/o and the MLPs quantize (96 modules); DeltaNet projections, embeddings, vision and merger stay dense. | mlx-vlm's convert predicate for this checkpoint. | WS-03 |
| 2026-10-28 | The source of truth is the original implementation: transformers `MiniCPMOcrForConditionalGeneration` / `qwen3_5` and the official SDK. mlx-vlm stays the MLX op-order oracle only where it matches them; the exporter patches mlx-vlm toward the original wherever it diverges (the `patch_mlx_vlm_to_baidu` precedent), and torch-f32 goldens bound the bf16 port. | User decision ("align with the original"). | WS-02, WS-03, WS-07 |
| 2026-10-28 | DSpark is in scope and on by default when `openbmb/MiniCPM-OCR-1.0-DSpark` loads. Semantics follow vLLM `qwen3_dspark.py` + the DSpark speculator: target features = outputs of layers `[3, 7, 11, 15, 19]` (vLLM's `+1` aux-layer convention), `fc` -> `hidden_norm` -> per-layer context K/V cached across rounds; anchor + 14 masks, non-causal, all 15 positions predict (`sample_from_anchor`); sequential greedy Markov bias `markov_w2(markov_w1[prev])` in the reduced draft vocab, mapped with `id + d2t[id]`; `prev` is the previous mapped proposal. No confidence head (adaptive verification off in the official config). | User decision; official serving runs vLLM with `num_speculative_tokens: 15`. mlx-vlm's DSpark drafter ignores `d2t` and the draft `lm_head`, so it is not the reference. | WS-08 |
| 2026-10-28 | Verification is greedy: one target forward over `[anchor, 15 drafts]`, emitted tokens are the target's argmax per position with the n-gram guard applied to the emitted prefix (a banned argmax re-samples that row masked). Rollback trims KV caches and replays each DeltaNet recurrence over the kept prefix from the saved pre-block state (same step kernel, so the state equals the one the block passed through); conv state = the kept window. Batched verify rounds differently from one-token steps, so near-ties may resolve differently from plain decoding (as in vLLM); acceptance is identical fixture text. | No per-step state kernel needed; the replay costs one small recurrence per DeltaNet layer on partial rounds. | WS-08 |
| 2026-10-28 | The WS-05 crop chain ports the SDK `ppdoclayout3_torch` path (`_filter_overlap_boxes` -> `int()` crops -> `_merge_blocks` with `NON_MERGE_LABELS` -> `_build_layout_from_blocks`), without polygon masking. Folded blocks drop out and add their bbox to the leader's `regions`; `order` is the index in the crop list. | The default `ppdoclayout3_paddlex` provider documents the same crop chain; our layout detector returns boxes only, so there are no polygons to mask. | WS-05 |
| 2026-10-28 | `html.unescape` is ported exactly for numeric references and for the HTML4 named set (252 names, with HTML5 values) plus `apos` and the legacy no-semicolon prefixes. Other HTML5-only names stay literal. `str.isalpha` is approximated by `char::is_alphabetic`. | The SDK only meets these through OTSL/table text; the full 2231-entry HTML5 table is not worth carrying. Goldens cover the supported set. | WS-05 |
| 2026-10-28 | In `regions::recognize_page`, a recogniser error aborts the page (`Err` propagates), unlike the SDK, which records a failed block and keeps going. Cancellation and backend errors must not turn into silent empty blocks. | AGENTS.md typed-error and cancellation rules. | WS-05, WS-06 |
| 2026-10-28 | The recogniser goldens (`tests/golden/minicpm_ocr/`) come from transformers 5.21 in torch bf16 (CPU) via `scripts/export_minicpm_ocr_reference.py`. The 4-bit reference is the same transformers graph with mlx-vlm's quantized weights (`mx.quantize`, checked bit-exact against `mlx_vlm.convert`) dequantized to f32 and applied as an f32 matmul rounded once to bf16. | Original-implementation alignment; transformers has no MLX affine quantization, and dequantizing reproduces what the quantized matmul computes. | WS-02, WS-03, WS-07 |
| 2026-10-28 | The port follows torch's bf16 rounding (f32 opmath, one rounding per op) instead of mlx-vlm's: DeltaNet q/k l2norm + query scale, `a + dt_bias` / softplus and the recurrence in f32 with y cast to bf16; RoPE cos/sin from f32 angles (precise `cos`/`sin`) cast to bf16, rotation through `rope_half_split`; MLP and conv SiLU in f32 rounded once; vision `gelu_tanh`, merger erf GELU and vision LayerNorms in f32 with one cast; patch embedding as one `addmm`; the image splice is an exact replacement (`masked_scatter`); argmax over f32 logits. `rope_positions` (the mlx-vlm fast-`cos` kernel) is removed. | Audit of transformers `qwen3_5` / `minicpm_ocr` vs the port; each item moved bf16 rounding away from torch. | WS-01, WS-02, WS-03 |
| 2026-10-28 | Crops resize with a port of torchvision's uint8 antialiased bicubic (`a = -0.5`, int16 fixed-point weights, horizontal then vertical pass), the HF processor's `TorchvisionBackend` path; adaptive cv2 upsizing (`small_image_upsize` non-default) is not supported. | The HF processor resizes uint8 tensors with torchvision; PIL differs by 1-2 levels. Verified bit-exact against torchvision. | WS-02 |
| 2026-10-28 | The n-gram guard scans generated tokens only (transformers `no_repeat_ngram_size` also sees the prompt ids). | A 100-gram from the prompt never recurs in practice; scanning the prompt costs time on every step. | WS-03 |
| 2026-10-28 | The corpus gate needs no Python baseline for MiniCPM-OCR: the native pipeline scores similarity 1.0000 on every corpus document (stop 5/5) at bf16 and 4bit, so any reference baseline (<= 1.0) is met; `corpus_baseline_minicpm.py` is dropped. | A transformers CPU baseline of the full region pipeline takes hours for no extra signal. | WS-07 |
| 2026-10-28 | The region goldens (`tests/golden/minicpm_regions/`) come from the SDK at commit `458718f80d7684230b8ff5a090a27ca1d895ee9a` via `scripts/export_minicpm_regions_reference.py`. They compare exact strings and FNV-1a hashes of crop pixels, with `emit_image_placeholder = true` (SDK behaviour); aimd runs with `false`. | Pins the SDK revision the port matches. | WS-05 |
| 2026-10-28 | A page's VLM blocks run as region batches (`recognize_batch_rgb8`, `regions::recognize_page_batch`): blocks sorted by prompt length, up to 16 per batch while padding stays <= 1/3. Rows are left-padded with zero embeddings (exact: zeros stay zero through every layer and leave the DeltaNet state zero), masks hide padding, RoPE positions count each row's own columns. Without a drafter decode is one batched step per token; with one, DSpark drafts and verifies every row in one forward per round, rows accept different prefixes, caches keep the longest (`rollback_rows` masks each row's extra DeltaNet steps) and the shorter rows' rejected columns stay as masked holes. A batch of one runs the serial path. | Per-region decode is launch-bound (CUDA) or small-M gemm-bound (Metal, where B=2..8 steps cost 35-39 ms vs 14 ms at B=1); batched DSpark amortizes both. Batched kernels round differently, so near-ties may resolve differently from the serial path (as batched verify already does). | WS-09 |
| 2026-10-28 | Vision stays per crop (no batched SigLIP). | Crops differ in shape; on Metal the tower is compute-bound (~270 ms/crop) and on CUDA it is ~24 ms/crop, so batching gains little for the padding and bookkeeping it needs. | WS-09 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | `aimd-mlx` ops: gated-delta recurrence (Metal + stock), stateful causal depthwise conv1d, bilinear `grid_sample`, missing elementwise ops | shuuul/Claude | Done | None | `aimd-mlx/tests` vs unfused/dense references on CPU and GPU |
| WS-02 | MiniCPM-OCR processor (slicing, placeholders, image ids), SigLIP NaViT tower, 2x2 merger | shuuul/Claude | Done | WS-01 (none for vision) | `minicpm_ocr` golden: pixel slices, tower stages, merger features vs mlx-vlm |
| WS-03 | Qwen3.5 hybrid decoder (Gated DeltaNet + gated full attention, partial RoPE, f32 `(1+w)` RMSNorm), KV + recurrent cache, greedy generation, load-time quantization | shuuul/Claude | Done | WS-01 | decoder layer/logit golden + greedy text vs mlx-vlm; quantized weights bit-exact |
| WS-04 | PP-DocLayoutV3: HGNetV2-L, hybrid encoder, deformable decoder, mask/order heads, SDK post-process + reading order | Claude WS-04 agent | Done | WS-01 (`grid_sample` only, added here) | `pp_doclayout_v3` golden: logits/boxes/order + final regions vs SDK torch provider |
| WS-05 | Region pipeline: crops, prompt map, Markdown assembly, OTSL -> HTML, anchors | shuuul/Claude | Done | WS-02, WS-03, WS-04 | hermetic assembly/OTSL goldens from SDK functions; checkpoint e2e page |
| WS-06 | `aimd-process` integration: aliases, precision, `OcrFamily`, `load_ocr`, HF download of both repos, help/MCP text | shuuul/Claude | Done | WS-05 | `models.rs` unit tests; fake-runtime routing tests |
| WS-08 | DSpark speculative decoding: draft loader (`d2t`, reduced-vocab `lm_head`, Markov head), target feature capture, verify + walk + cache rollback, pipeline default | shuuul/Claude | Done | WS-03 | e2e fixtures: DSpark text == plain greedy text (bf16, 4-bit); acceptance logged |
| WS-07 | Exporters (`scripts/export_minicpm_ocr_reference.py`, `export_pp_doclayout_v3_reference.py`), corpus baseline, gate runs (Metal + metacube), docs/AGENTS sync | shuuul/Claude | In progress | WS-02..WS-06 | corpus gate within bounds; docs reviewed |
| WS-09 | Region batching: batched prefill/decode with per-row DeltaNet state, batched DSpark draft + verify, page-level batch in `MiniCpmDocument` | shuuul/Claude | Done | WS-05, WS-08 | e2e: batch text == serial text on the fixtures (plain + DSpark, bf16 + 4bit, Metal + CUDA); profile pages timed |

## Verification

```bash
uv run scripts/export_minicpm_ocr_reference.py
uv run scripts/export_pp_doclayout_v3_reference.py
prek run --all-files
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo test -p aimd-engine minicpm_ocr pp_doclayout_v3      # checkpoint tier
AIMD_E2E=1 cargo test -p aimd-engine --lib minicpm_ocr
cargo run --release -p aimd-engine --example corpus_gate -- --only ocr --ocr-model minicpm-ocr
cargo xtask check-specs
```

Manual: `aimd scan.pdf --model minicpm-ocr [--precision 4bit] [--anchors]` on a
page with a table and a display formula.

## Documentation sync

- Durable product/developer docs: README model table, `docs/` performance notes.
- Nearest local `AGENTS.md`: root `AGENTS.md` (model naming, MODEL AND PRECISION DESIGN, golden sets, WHERE TO LOOK).
- Parent/package guidance: none.
- Root guidance and roadmap: `AGENTS.md` overview line listing OCR families.

## Progress and handoff

Append entries rather than rewriting another worker's record.

### 2026-10-28 — shuuul/Claude — coordination

- Changed: reserved spec 009, recorded architecture facts and decisions.
- Evidence: HF configs, mlx-vlm `minicpm_ocr` / `qwen3_5` / `pp_doclayout_v3`, transformers `modeling_minicpm_ocr.py`, SDK `prompts.py`, `constants.py`, `layouts/ppdoclayout3_torch.py`.
- Remaining: all workstreams.
- Blockers: none.
- Next action: WS-01.

### 2026-10-28 — Claude WS-04 agent — WS-04 PP-DocLayoutV3

- Changed: `aimd_engine::models::pp_doclayout_v3` (`config`, `preprocess`, `backbone`, `encoder`, `decoder`, `model`, `postprocess`, `pipeline`, `test_support`, `tests`). `PpDocLayoutV3::load(dir)` checks `config.json` and loads the f32 snapshot (batch norms folded at load, reusing `pp_ocrv6::layers`); `detect(&RgbImage) -> Vec<LayoutRegion { index, label, class_id, score, bbox: [i32; 4] }>` in reading order; `raw_outputs` exposes logits/boxes/order. `aimd-mlx`: `MlxArray::grid_sample` (bilinear, zeros padding, `align_corners=False`, NHWC; stock gathers, every backend) as an appended shim block plus `src/grid_sample.rs`, tested in `tests/mlx_grid_sample.rs` against a host port of torch's CPU kernel (GPU and CPU device).
- Reference: `scripts/export_pp_doclayout_v3_reference.py` runs transformers 5.21 `PPDocLayoutV3ForObjectDetection` (torch 2.14, f32, CPU) behind the SDK preprocessing (RGB->BGR, `cv2.resize` 800x800 `INTER_CUBIC`, `* (1/255)`) and the SDK numpy post-process copied verbatim, i.e. exactly the SDK torch provider; mlx-vlm was not used (it resizes RGB to 1024 and post-processes differently). Golden set `crates/aimd-engine/tests/golden/pp_doclayout_v3/` (largest file 352 KB): `report` (corpus page) and `landscape` (PIL-drawn) pages, resized PNGs, stage slices, top-300 indices, initial/per-layer boxes, decoder hidden slices, logits/boxes/order, SDK regions, plus a crafted raw-output case exercising NMS, the image-area filter and containment.
- Evidence (M-series Metal): the cv2 cubic port is bit-exact with `cv2.resize` and the pixel slices are identical; the post-process on golden raw outputs reproduces the SDK regions exactly on both pages and the synthetic case (42 regions). Checkpoint tier: backbone/PAN/mask-feature slices max abs err <= 4e-4 (rel <= 1.1e-5); top-300 queries 300/300 and initial boxes 300/300 identical. Decoder from the reference queries: layer-0 hidden rel 1e-5, layer-5 rel <= 1.5e-3, boxes abs <= 3.3e-4, logits rel <= 8e-4, order logits rel <= 1.5e-3. `detect` reproduces the SDK regions on both pages with identical labels, order and boxes (0 px) and scores within 1e-4. Gates: fmt, clippy `-D warnings`, `AIMD_MLX_DEVICE=cpu cargo test --workspace`, `cargo test -p aimd-engine pp_doclayout_v3` (Metal, 1.3 s), prek.
- Remaining: CUDA (metacube) checkpoint run not done; JPEG pages decode through the `image` crate, not cv2/PIL (the goldens are PNG). No aimd-process wiring (WS-06) or crop pipeline (WS-05).
- Blockers: none.
- Next action: WS-05 can consume `PpDocLayoutV3::detect`.

### 2026-10-28 — shuuul/Claude — WS-01..WS-03 MiniCPM-OCR recogniser

- Changed: `aimd-mlx` `gated_delta_gates`, `gated_delta` (mlx-vlm `gated_delta_step` Metal kernel, stock-op loop elsewhere), grouped `conv1d`, `rope_positions` (tests in `tests/mlx_gated_delta.rs`, Metal and CPU). `aimd_engine::models::minicpm_ocr` (`config`, `preprocess`, `vision`, `decoder`, `generate`, `pipeline`, `test_support`, `checkpoint_tests`): MiniCPM-V slicing with PIL-exact bicubic resize, SigLIP NaViT + exact-GELU merger, Qwen3.5 hybrid decoder with KV/conv/recurrent caches, chunked-prefill greedy generation with `async_eval` pipelining and the n-gram-100 guard. `scripts/export_minicpm_ocr_reference.py` and golden set `crates/aimd-engine/tests/golden/minicpm_ocr/`.
- Evidence (Metal): preprocessing and prompt ids identical to the HF processor; vision stages and merger features exact; decoder `inputs_embeds`, layers 0/3/11/23, `final_norm` and step logprobs exact at bf16 and 4bit; quantized module set equals mlx-vlm's; E2E greedy text identical on `small_text` and `page_text` (bf16 and 4bit, ~82 / ~104 tok/s decode on this Mac).
- Remaining: WS-05 (region pipeline), WS-06, WS-07; CUDA checkpoint run.
- Blockers: none.
- Next action: WS-05.

### 2026-10-28 — shuuul/Claude — WS-08 DSpark

- Changed: `minicpm_ocr::dspark` (`DSparkDrafter::load` checks the draft config against the target, reads `d2t` as int64 deltas; `draft` = context K/V append, anchor + mask block, sequential Markov argmax). `decoder`: `Capture` (layer outputs + DeltaNet recurrence inputs), `forward_capture`, `rollback`; `KvCache::trim`. `generate::speculative_stream` (+ `SpecStats`). `pipeline`: `attach_drafter`, `OcrRequest::speculative` (default on), `OcrStats::speculative`. The shared test fixture attaches the draft when its snapshot is cached.
- Evidence (Metal, e2e tier): DSpark text equals plain greedy on `small_text` and `page_text` at bf16 and 4bit. `page_text` decode bf16 81 -> 220 tok/s (7 rounds, 90/105 drafts accepted), 4bit 105 -> 252 tok/s (90/120); `small_text` (18 tokens) is too short to gain.
- Remaining: aimd-process download/attach of the draft repo (WS-06); corpus-gate run with DSpark on; CUDA run.
- Blockers: none.
- Next action: WS-06 wiring.

### 2026-10-28 — Claude WS-05 agent — WS-05 region pipeline

- Changed: new modules `aimd_engine::models::minicpm_ocr::{regions, markdown, latex, otsl}` plus `regions_tests`. The `pub mod` block is in `minicpm_ocr/mod.rs`.
  - `regions` holds the crop planning and the label -> `ElementRecipe` map (prompt, `max_new_tokens`, `slice_nums` 9, output format). Its entry points are `plan_page`, `finish_page` and `recognize_page`, a driver that takes a recogniser closure.
  - `markdown` covers `classify_engine_text`, `postprocess_content`, `normalize_crop_text`, `split_text_block`, the repetition truncators, `normalize_and_dedupe_blocks` and `assemble_markdown(emit_image_placeholder)`.
  - `latex` is `repair_latex_formula`.
  - `otsl` covers `otsl_pad_to_sqr_v2`, `convert_otsl_to_html` and `html_unescape`.
  - The exporter is `scripts/export_minicpm_regions_reference.py`, which imports the SDK clone by path. Its golden set `crates/aimd-engine/tests/golden/minicpm_regions/` holds `text_cases.json` (445 KB), `crop_cases.json` and `e2e_cases.json`.
- Evidence:
  - `text_cases`: exact match on 1808 cases across 12 SDK functions plus 43 block lists, compared after normalize/dedupe and after Markdown assembly.
  - `crop_cases`: 7 pages (the two PP-DocLayoutV3 golden pages and 5 synthetic merge/filter pages: cross merge, up/down chain, tall group not merged, size/reference/overlap filters, aligned chains). Order, label, bbox, folded regions, crop size, crop pixel hash and recipe match for every block.
  - `e2e_cases`: each planned block runs through the SDK `_execute_block` with a stub engine fed canned raw outputs, then through `_normalize_and_dedupe_blocks` and `assemble_markdown`. The model inputs (pixel hash, prompt, budget, slices) and the final blocks and page Markdown are identical. With `emit_image_placeholder = false` the output differs only by dropping `![image]()`.
  - Gates: `cargo test -p aimd-engine --lib minicpm_ocr` (hermetic, 0.07 s for the region tests), clippy `-D warnings`, rustfmt.
- Remaining:
  - WS-06 must wire `PpDocLayoutV3::detect` -> `regions::recognize_page` -> `MiniCpmOcrPipeline::recognize_rgb8`, mapping `ElementRecipe` to `OcrRequest { task: prompt, max_tokens: Some(max_new_tokens), max_slices: Some(slice_nums), .. }`, with `emit_image_placeholder: false`.
  - Not reproduced: paddlex polygon masking, HTML5-only named entities, and the SDK's concurrent/priority scheduling. Order does not matter for the output, because blocks are reassembled by `order`.
- Blockers: none.
- Next action: WS-06.

### 2026-10-28 — shuuul/Claude — original alignment, WS-06 wiring, corpus gate

- Changed: recogniser numerics moved from mlx-vlm to torch rounding (see Decisions); `preprocess::resize_bicubic_aa` (torchvision uint8 AA bicubic); `rope_positions` shim op removed. `scripts/export_minicpm_ocr_reference.py` rewritten on transformers 5.21 (torch bf16 CPU, 4-bit via dequantized mlx-vlm weights). New `minicpm_ocr::document::MiniCpmDocument` (layout -> `regions::recognize_page` -> recogniser, `truncated_regions`). `aimd-process` `NativeOcr::MiniCpm`: resolves/downloads the recogniser, DSpark draft and layout repos, attaches the drafter, maps blocks to `LocatedOcr` segments (normalized bboxes, label as kind). `corpus_gate --ocr-model minicpm-ocr [--ocr-bits N]`.
- Evidence: official transformers bf16 greedy == the native output on both fixtures (19 / 98 tokens). Corpus gate (Metal, DSpark on): similarity 1.0000 on all 4 documents, stop 5/5, bf16 (33 s) and 4bit (26 s). `aimd article-p1.png --model minicpm-ocr --anchors`: per-region anchors with `kind=<label>`, 11 s including load. clippy `-D warnings` clean.
- Remaining: re-pin the recogniser goldens and tolerances from the transformers exporter; CUDA (metacube) checkpoint tier and corpus gate.
- Blockers: none.
- Next action: checkpoint tier against the new goldens.

### 2026-10-28 — shuuul/Claude — recogniser goldens re-pinned to transformers

- Changed: `tests/golden/minicpm_ocr/` regenerated by the transformers exporter (largest file 462 KB). `checkpoint_tests` bounds are measured torch bounds (`STAGES` rel rms 5e-2 / max 12.5%, `LOGPROBS` 1e-2 / 2%) instead of Metal-exact mlx-vlm parity.
- Evidence (Metal): preprocessing (torchvision AA bicubic crops, patch rows) and prompt ids exact; patch embed rel rms 1.6e-6; deepest vision stage 3.4e-2, decoder `final_norm` 3.2e-2 (mlx-vlm's goldens sit at 8.1e-2 / max 20% from the same torch reference); step logprobs <= 6.6e-3. 4-bit: 96 modules quantized, 0 mismatched vs `mlx_vlm.convert`. `AIMD_E2E=1`: plain and DSpark greedy text equal transformers on both fixtures at bf16 and 4bit (4bit reproduces torch's `2026\nnative` line break); `page_text` DSpark 244 / 249 tok/s vs 80 / 100 plain. Gates: fmt, clippy `-D warnings`, prek, `AIMD_MLX_DEVICE=cpu cargo test --workspace`, `cargo xtask check-specs`.
- Remaining: CUDA (metacube) checkpoint tier and corpus gate; spec closeout after that.
- Blockers: none.
- Next action: metacube run.

### 2026-10-28 — shuuul/Claude — WS-09 region batching

- Changed: `generate::{prefill_batch, greedy_batch, speculative_batch, batch_stream}`; `decoder`: batch-aware attention/DeltaNet/RoPE, `rope_rows`, array masks in `forward_capture`, `validity_mask`, `rollback_rows`, `select_rows`; `dspark::{draft_batch, DraftBatch}`; `KvCache::select_rows`; `pipeline::recognize_batch_rgb8` (clears the MLX buffer cache after each image's vision pass: without it, 20 differently shaped crops ran a 32 GB card out of memory); `regions::recognize_page_batch`; `MiniCpmDocument::recognize` uses them. The e2e test checks a 3-image batch against the serial texts.
- Evidence (5 profile pages, 58 regions, warm pages 1-4): CUDA (metacube) DSpark 38.2 -> 20.7 s, greedy 42.1 -> 24.4 s; Metal (M2 Pro) DSpark 54.1 -> 48.2 s, greedy 95.4 -> 88.4 s. 57/58 regions identical to serial; the rest diverge at near-ties (a newline vs a space; a formula that continues past the serial EOS). `AIMD_E2E=1` e2e passes on Metal and CUDA; hermetic tier, fmt, clippy `-D warnings` pass.
- Remaining: prefill on CUDA is still dominated by the stock-op `gated_delta` loop (WS-07 follow-up).
- Blockers: none.
- Next action: WS-07 closeout.

## Completion summary

Not complete.
