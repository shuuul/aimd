---
id: "008"
title: "Unlimited-OCR-2: DASH decoder and windowed multi-page PDF OCR"
status: Active
created: 2026-10-27
updated: 2026-10-28
coordinator: "Claude (Opus 5.5) session with shuuul"
---

# 008 — Unlimited-OCR-2: DASH decoder and windowed multi-page PDF OCR

## Context

`baidu/Unlimited-OCR-2` replaces Unlimited-OCR. The vision stack (SAM ViT-B +
CLIP-L + 2048 -> 1280 projector) and the DeepSeek-V2 MoE decoder weights keep
their shapes. What changes is the decoder's attention and the protocol:

- `sparse_attention_config.training_stage == "dash"`: every layer has a
  `self_attn.dash_router` (`q_proj`, `k_proj` 1280 -> 128, Gemma-style
  `q_norm`/`k_norm`, `head_cls`). Each query attends a 128-token local window
  (4 blocks of 32) plus the 12 remote blocks the router picks (head_cls block
  summaries, top-16 scores, 1.5-entmax prior, `log(max(prior, 1e-4))` centred
  over valid blocks as an additive bias). Prefill queries see only the image
  blocks of their own page. The warmup-stage `indexer.*` weights are unused.
- page2d RoPE (main branch and router): angles from a (token-in-page, page)
  coordinate, `tok_theta 1e4` resample frequencies over 63 pairs and one page
  pair (`page_theta 256`, page coordinate folded mod 2). Coordinates come
  from `<PAGE>` (128827) markers; the answer's page restarts at 0 and each
  generated `<PAGE>` advances it. The ring R-SWA cache of v1 is gone (full
  KV cache).
- Prompt: `"<PAGE><image>" * n + "Multi page parsing."`. One page uses gundam
  (<= 16 crops); several pages use one 1024 base view each (273 tokens).
- Output is not Markdown: `<PAGE><BLOCK>kind[<LEVELn>][x0,y0,x1,y1]<FORK>text`,
  with `cross_<t>` / `append_<t>` chains for blocks that continue across a
  page break. Upstream ships the converter `utils/process_unlimitedOCR2.py`.

The MLX reference is the mlx-vlm port in `shuuul/mlx-vlm` (`mlx_vlm/models/
unlimited_ocr/dash.py`), checked against a torch reference run on mayura (A6000).

## Goal and success criteria

aimd OCRs images and PDFs with Unlimited-OCR-2 natively, PDFs as native
multi-page requests of up to 10 pages.

- [x] `mlx-community/Unlimited-OCR-2-{4bit,6bit,8bit,bf16}` exist (converted
      with the mlx-vlm port; vision dense bf16, `dash_router` dense).
- [x] `aimd scan.pdf --model unlimited-ocr-2` (and `--precision 8bit`) runs
      locally; a PDF is OCR'd in windows of at most 10 pages, base mode, with
      progress and cancellation between windows.
- [x] The native decoder matches the mlx-vlm port: committed goldens for
      prefill logits rows, routing (`topi`/`bias`/`valid` on sampled layers)
      and a forced greedy stream, enforced on Metal like the v1 decoder goldens.
- [x] Raw output converts to Markdown with a port of
      `process_unlimitedOCR2.py`, unit-tested against upstream outputs.
- [x] Corpus gate: Unlimited-OCR-2 OCR similarity >= the mlx-vlm baseline
      - 2% and stop rate no lower; numbers recorded here.
- [x] Hermetic gates pass; help text, README, `docs/`, `AGENTS.md` updated.

## Scope and non-goals

In scope:

- `aimd-mlx` shim entry points for the DASH router/selection (fixed-shape
  compiled decode kernels, like the WS-08 fused ops).
- `aimd-engine` `unlimited_ocr`: config, DASH decoder variant, base-mode
  preprocessing, multi-image prompt/splice, `<PAGE>`-aware generation (page
  guard, n-gram window 1024 for multi-page), v2 Markdown converter, located
  blocks per page for anchors.
- `aimd-process`: model alias/precision resolution, PDF windowing (10 pages),
  progress per window, anchors per page.

Not in scope:

- v1 behavior changes (v1 stays byte-identical).
- KV reuse across windows (SGLang's "KV tail of the previous window"); windows
  are independent requests and the converter joins their chains.
- Batch size > 1, remote-backend protocol changes, gundam multi-page mode.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-27 | PDFs run native multi-page in windows of 10 pages, base mode; progress/cancel per window | User decision. Matches upstream SGLang's 10-page window; base mode keeps 273 tokens/page | WS-04 |
| 2026-10-27 | Upload 4bit, 6bit, 8bit, bf16 to mlx-community (no mxfp8) | User decision | WS-01 |
| 2026-10-27 | Alias `unlimited-ocr-2`; `unlimited-ocr` stays v1 until the corpus gate passes, then becomes v2 with `unlimited-ocr-1` for v1 | Upgrade without silently moving outputs before evidence | WS-04 |
| 2026-10-27 | Port the mlx-vlm decode design: gather 12x32+128 keys per step, summaries cached per completed block, page/RoPE/candidate state in-graph | Constant per-token cost; keeps the early-submit decode pipeline (spec 007) because no host sync on the token is needed | WS-02 |
| 2026-10-27 | Single images keep the gundam layout with the `<PAGE>` prefix | Upstream auto policy | WS-03 |
| 2026-10-27 | mlx-vlm: the v2 PR includes every Metal kernel and `mx.compile` optimization that speeds v2, shared MoE/decode kernels included; v1-only optimizations go in a separate later PR | User decision | WS-01 |
| 2026-10-27 | Supersedes the row above: the DASH/MoE kernels ship in an aimd release first; the mlx-vlm v2 PR carries only the port (compiled decode selection included). The kernels go to mlx-vlm afterwards, with comments citing aimd as the source | User decision | WS-01, WS-02 |
| 2026-10-27 | Supersedes the alias row: `unlimited-ocr` stays v1 for good; Unlimited-OCR-2 is a separate, newer model under `unlimited-ocr-2`, not a replacement | User decision | WS-04 |
| 2026-10-27 | `unlimited-ocr-2` is the default OCR model (omitted `model`); `unlimited-ocr` keeps naming v1 | User decision | WS-04 |
| 2026-10-27 | v1 stays page by page | Upstream v1 `infer_multi` (one `<image>`, base 1024) dropped page 2 of the 2-page article in mlx-vlm (one `<PAGE>`, EOS after page 1); v1's `<PAGE>` is 3 text tokens, so the v2 page guard does not apply. PaddleOCR-VL / PP-OCRv6 are single-image models | WS-04 |
| 2026-10-27 | Supersedes the row above: v1 reads PDFs in 5-page requests (`infer_multi` layout: one 1024 base view per page, no `<PAGE>` in the prompt, `Multi page parsing.`); a lone page keeps the gundam request | The earlier probe ran mlx-vlm at 640 px with no guard. Natively at 1024 with the page guard on v1's 3-token text `<PAGE>`, and the marker substituted for an EOS before the last page, v1 matched page-by-page output through 6 pages; the 7th looped and the 8th repeated page 1 | WS-04 |
| 2026-10-27 | No dense-attention fallback on any backend: block-sparse prefill (custom Metal kernel; sparse gather from mlx ops on CUDA/CPU) and gathered decode. A dense masked path exists only as a test oracle | User decision; prefill with a full [L, S] mask costs as much as dense attention | WS-02 |
| 2026-10-28 | Supersedes the 10-page base-mode row's layout: v2 tiles every page (gundam, <= 16 crops), multi-page requests included; 10-page windows stay. v1 multi-page keeps base | User decision, following mlx-vlm c3ae82e2 (base mode looped on page 3 of the 6-page technical report; tiles finished with EOS). On the corpus the text is identical and 10 pages take 41.4 s vs 23.2 s | WS-03, WS-04 |
| 2026-10-28 | The vision towers and v1 R-SWA follow Baidu's native code (`deepencoder.py`, `modeling_deepseekv2.py`), not mlx-vlm 0.7.0: CLIP `quick_gelu`, CLIP layer LayerNorm eps 1e-5, spatial bicubic-antialias CLIP pos resample, half-pixel SAM rel-pos resample, and the ring keeps the whole prompt. The `ocr_vision` golden comes from `deepencoder.py` in torch f32; mlx-vlm-based exporters and corpus baselines call `patch_mlx_vlm_to_baidu` | User decision; mlx-vlm PR #2603 found the same divergences, confirmed against the Baidu source | WS-02, WS-03 |

## Workstreams

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | mlx-vlm port, turtle parity, conversions + uploads, upstream PR | Claude | In progress | None | turtle torch reference vs mlx-vlm; HF repos exist |
| WS-02 | Shim DASH ops + native DASH decoder | Claude | Completed | WS-01 parity | decoder goldens (exporter from mlx-vlm), `AIMD_MLX_DEVICE=cpu cargo test` |
| WS-03 | Pipeline: base mode, multi-image prompt, page guard, v2 Markdown converter, located blocks | Claude | Completed | WS-02 | unit tests + E2E on corpus pages |
| WS-04 | aimd-process: alias, 10-page windows, progress/cancel, help/docs | Claude | In progress | WS-03 | hermetic route/model tests, corpus gate |

## Verification

```bash
prek run --all-files
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo test -p aimd-engine unlimited_ocr        # checkpoint tier (Metal)
AIMD_E2E=1 cargo test -p aimd-process --test e2e
cargo run --release -p aimd-engine --example corpus_gate -- --only ocr --ocr-model unlimited-ocr-2
cargo xtask check-specs
```

## Documentation sync

- Durable product/developer docs: README model table, `docs/` performance notes.
- Nearest local `AGENTS.md`: none below the root.
- Parent/package guidance: n/a.
- Root guidance and roadmap: `AGENTS.md` model naming, MODEL AND PRECISION
  DESIGN, golden fixture list.

## Progress and handoff

### 2026-10-27 — Claude — WS-01

- Changed: mlx-vlm `unlimited_ocr` gains `dash.py` (page2d RoPE, DASH router,
  gathered decode, compiled decode selection), v2 processor layout, tests,
  README. Committed on `shuuul/mlx-vlm` main (local, not pushed).
- Evidence: 1- and 2-page smoke tests produce correct text (cross-page
  `cross_text`/`append_text` chain); 4-bit decode 6.9 ms/token (v1 6.3);
  1725 mlx-vlm tests pass.
- Remaining: turtle torch parity, conversions/uploads, upstream PR.
- Blockers: turtle GPUs nearly full (shared machine).
- Next action: compare against the turtle reference dumps.

### 2026-10-27 — Claude — WS-01 (kernels)

- Changed: mlx-vlm local main gains `unlimited_ocr/dash_kernels.py` (fused
  decode routing + gathered attention; block-sparse simdgroup-matrix prefill),
  a tile-union gather prefill off Metal (the dense mask is now only a test
  oracle), and fused DeepSeek-V2 MoE decode kernels ported from the aimd shim
  (`kernels/moe_softmax_topk.py`, `kernels/moe_swiglu.py`, shared with v1).
  Commits a048293b, 78bb479f, 0a236a2f (not pushed).
- Evidence (M2 Pro, 4-bit): raw v2 decode 6.9 -> 5.3 ms/step, v1 4.1-4.4 ->
  3.6; prefill attention over 10 base pages 1222 -> 417 ms, one gundam page
  64 -> 57 ms. Greedy tokens unchanged on 1- and 2-page runs (the 2-page
  stream has an exact bf16 logprob tie at token 32 that the decode kernel can
  flip). Kernel vs reference: f32 <= 2e-6. 97 targeted mlx-vlm tests pass.
- Lessons for the aimd port (WS-02): per-lane scattered loads and register
  spills (Q + O fragments) dominated the first prefill kernel; staging K/V as
  f32 in threadgroup memory with `simdgroup_load`, Q staged in threadgroup
  memory, 8-key chunks and next-chunk register prefetch made it win. The
  mlx-op gather path is exact but only reaches dense speed at ~10 pages.
- Remaining: torch parity (mayura), conversions/uploads, upstream PR.
- Next action: compare against the mayura reference dumps.

### 2026-10-27 — Claude — WS-01 (parity)

- Changed: mlx-vlm processor `__call__` defaults `cropping=None` (the
  DeepSeek-OCR parent forced gundam tiles on multi-page requests: 3033 vs 553
  prompt tokens for two pages). Kernel commits moved to the local
  `uocr2-kernels` branch; `main` holds the port only (decision above).
- Evidence (mayura, MLX CUDA 0.32.4 bf16 vs torch 2.11 bf16 on an A6000,
  `page_text.png` gundam and a 2-page base article): prompt ids identical
  (1519, 553). Torch `prompt_embeds` fed into the MLX decoder: teacher-forced
  argmax agrees on 220/220 and 627/627 tokens, per-layer prefill attention
  output relative error 0.2-2.2%, last-position logits 0.6-0.7%. End to end
  (MLX vision) greedy text matches up to bf16 vision drift (bbox +-2 px; the
  2-page stream diverges at token 122/627). Prefill selection Jaccard vs torch
  is 0.90 (single) / 0.77 (multi) with identical inputs; the differing blocks
  carry negligible weight given the attention-output error.
- Next action: conversions + uploads (running on mayura), then the mlx-vlm PR.

### 2026-10-27 — Claude — WS-01 (uploads)

- Changed: converted on mayura with the port (affine, group 64; SAM/CLIP,
  `dash_router` bf16; 4bit 2.5 GB, 6bit 3.2 GB, bf16 6.4 GB) and uploaded
  `mlx-community/Unlimited-OCR-2-{4bit,6bit,8bit,bf16}` with a card pointing
  at `shuuul/mlx-vlm` until upstream merges. Each passed a greedy smoke OCR
  of `page_text.png` before upload. The port (no kernels) is pushed to
  `shuuul/mlx-vlm` main; it gained a processor fix (generate-only load kwargs
  `expert_cache_gb`/`max_kv_size` broke `mlx_vlm.generate`). Fork CI green
  except lint, fixed by a black commit.
- Remaining: upstream mlx-vlm PR (port only); WS-02..04.
- Next action: open the PR to Blaizzy/mlx-vlm from fork main.

### 2026-10-27 — Claude — WS-02

- Changed: `aimd-mlx` shim gains the DASH ops (`dash.rs`, shim `namespace
  dash`): page2d RoPE tables, router q/k (compiled for decode), head_cls
  summaries, block-sparse prefill (simdgroup-matrix Metal kernel; tile-union
  gather from mlx ops elsewhere; dense masked path only as the test oracle),
  gathered decode (fused Metal kernel; compiled select + take + sdpa
  elsewhere) and the in-graph decode context. Ported from the local mlx-vlm
  `uocr2-kernels` branch with one fix: the gather prefill reshaped K/V with
  the query head count (GQA bug, latent upstream since HQ == HK there).
  `aimd-engine` `unlimited_ocr/dash.rs` + decoder wiring: v2 configs get
  full KV caches plus `DashRun`; `splice_image_features` fills one `<image>`
  run per page.
- Evidence: `aimd-mlx/tests/mlx_dash.rs` (sparse paths vs the dense oracle,
  Metal and CPU). Decoder goldens `tests/golden/ocr2_decoder/` (exporter
  `scripts/export_ocr2_decoder_reference.py`, 2 base pages, 553 prompt
  tokens): prefill logits rel rms 9.2e-3, decode logits 1.1e-2; prefill
  routing on layers 0/5/11 selects identical blocks (layer-0 bias error
  5e-7); the 243-token forced stream (both `<PAGE>` markers, EOS) agrees
  except 4 exact bf16 ties (gap < 0.5). Hermetic `aimd-engine`/`aimd-mlx`
  pass on the CPU device.
- Next action: WS-03 pipeline.

### 2026-10-27 — Claude — WS-03

- Changed: `pipeline.rs` `recognize_pages` (1..=10 pages; one page = gundam
  with the 16-crop ratio table, several = base mode, 273 tokens each; prompt
  `[bos] + ([<PAGE>] + <image>*n)` per page + `Multi page parsing.`, checked
  against the mlx-vlm processor); `recognize` routes v2 single images through
  it. `generate.rs`: upstream page-count guard (`PageCountGuard`: EOS banned
  until every page marker, forced past it, 20000 tokens per page) and the
  1024-token n-gram window for multi-page requests. `markdown2.rs`: port of
  `process_unlimitedOCR2.py` plus `render_unlimited_ocr2`, which drops image
  placeholders and returns per-page located items (0..999 boxes normalized).
- Evidence: `ocr2_markdown` upstream cases pass byte for byte;
  `ocr2_pages_prompt_ids_match_processor`; `AIMD_E2E=1`
  `ocr2_two_page_request_e2e`: similarity 0.9911 vs mlx-vlm, 242 tokens, both
  `<PAGE>` markers, finish=stop, prefill 233 ms, decode 1.42 s.
- Next action: WS-04.

### 2026-10-27 — Claude — WS-04

- Changed: alias `unlimited-ocr-2` (`OcrFamily::UnlimitedOcr2`,
  `mlx-community/Unlimited-OCR-2-{4bit,6bit,8bit}`; bf16 rejected like v1);
  `OcrModel::{pages_per_request, recognize_pages}` (default 1 / unsupported);
  `NativeOcr` returns 10 for v2 and splits located items per page; the PDF
  OCR path groups rendered pages into windows, checks cancellation between
  windows (`OCR cancelled between page windows`) and reports
  `Processed OCR pages a-b of N`. Help text, MCP description, README,
  `docs/architecture.md`, `AGENTS.md`, the aimd skill. Corpus gate
  `--ocr-model unlimited-ocr-2` (one request per document) with
  `scripts/corpus_baseline_ocr2.py` (mlx-vlm port, same guards; raw output
  converted by the same Markdown port).
- Evidence: hermetic `pdf_ocr_multi_page_windows` (5 pages, window 2:
  requests [2, 2, 1], blank page renders empty, progress per window,
  cancellation after the first window). Corpus gate v2 4-bit: similarity
  1.0000 (baseline 1.0000), stop 4/4 requests, 4/4 identical; v1 gate
  unchanged (1.0000, 5/5). CLI on a 14-page scanned PDF (`--task ocr`):
  windows 10 + 4, 14 pages, 39.6 s wall including the checkpoint load.
  Gates: prek, fmt, clippy `-D warnings`, `AIMD_MLX_DEVICE=cpu cargo test
  --workspace` (56 s), check-specs.
- 8-bit: `--precision 8bit` OCRs `article.pdf` (2 pages) locally. Its first
  download dropped mid-file and the next run took the partial snapshot as
  cached (`config.json` downloaded before the weights); `hf::download_snapshot`
  now fetches `config.json` last. Metal checkpoint tier
  `cargo test -p aimd-engine unlimited_ocr`: 35 passed.
- Next action: commit, release; then the mlx-vlm PRs. `unlimited-ocr` stays
  v1 (user decision).

### 2026-10-27 — Claude — WS-01 / WS-04

- Changed: opened the mlx-vlm port PR (Blaizzy/mlx-vlm#2601, branch
  `shuuul:add-unlimited-ocr-2`, port only; kernels follow after it merges).
  `unlimited-ocr-2` is now the OCR default (help, MCP description, README,
  AGENTS.md, docs). v1 multi-page evaluated and rejected (decision above).
- Evidence: native vision is on par with mlx-vlm on the same checkpoint
  (M2 Pro, warm: base page 347 vs 352 ms, gundam 12 crops 1647 vs 1604 ms);
  folding SAM's decomposed rel-pos bias into Q/K (one-hot keys, fused SDPA
  at D=192/256) is 2x slower than the materialized bias (44.5 vs 22.6 ms per
  global block at 64x64), so the bias stays. 2-page request end to end:
  native 2.29 s vs mlx-vlm 2.99 s (decode 177 vs 133 tok/s). The 0.72 s
  `page_text.png` figure in `docs/performance.md` is the RTX 5090.
- Next action: gates, commit.

### 2026-10-27 — Claude — WS-04

- Changed: v1 multi-page (decision above). `PageCountGuard` takes a token
  marker (`[<PAGE>]` for v2, `UNLIMITED_OCR_V1_PAGE_MARKER` for v1) and
  forces a partial marker token by token; for v1 an EOS logit moves to the
  marker's first token while pages remain. `recognize_pages` builds the v1
  prompt and splits the output on `<PAGE>`; `max_pages()` is 10 for v2 and
  5 (`UNLIMITED_OCR_V1_MAX_PAGES`) for v1.
- Evidence (M2 Pro, 4-bit, the 14-page article/zh/table mix): word
  similarity to page-by-page v1 per page is 1.00 for 2-6 page requests; 7
  pages: last page 0.05 (loop); 8 pages: 0.01 (page 1 again); 10 pages:
  pages 7-10 <= 0.71. Without the EOS substitution page 2 of the 2-page
  article was empty (EOS ban -> hallucinated list). 10 pages page by page
  34.5 s vs two 5-page requests 17.7 s, all 10 pages 1.00; 11 pages
  (5+5+1) all 1.00.
- Next action: gates, commit.

### 2026-10-28 — Claude — WS-02/WS-03

- Changed: the four vision divergences and the R-SWA prompt boundary
  (decision above). `mlx_vision.rs`: CLIP `quick_gelu`, layer LN eps 1e-5.
  `preprocess.rs`: `clip_pos_embed_interp` resamples the `[16, 16, dim]`
  grid spatially with the torch bicubic antialias filter;
  `get_rel_pos_table` uses the half-pixel linear index; host tables stay
  f32 and round on upload. `KvCache::retain_prompt` pins `prefill_length`
  to the prompt length (`OcrDecoder::new_cache`, v1 only); before, chunked
  prefill of `prompt[:-1]` left the last prompt token (two with a 1-row
  final chunk) in the ring, overwritten after 128 generated tokens.
  `export_ocr_vision_reference.py` runs Baidu `deepencoder.py` in torch f32
  with the mlx-community weights (projector dequantized);
  `reference_unlimited_ocr.patch_mlx_vlm_to_baidu` patches mlx-vlm for the
  e2e, decoder, ocr2 decoder exporters and both corpus baselines. All
  Unlimited-OCR goldens re-exported.
- Evidence (M2 Pro Metal, 4-bit): old (mlx-vlm) vs Baidu golden rel RMS
  0.48..0.74 on the crop CLIP stages, 0.24..0.60 on projectors. Baidu's own
  bf16 torch run vs its f32 run: clip.final 0.042, projectors 0.025, small
  final features 0.074; the Rust port: stages <= 0.019, clip.final 0.059,
  projectors <= 0.041, final features 0.052 / 0.049. Vision bounds
  re-baselined to f32 truth (STAGE 0.03/0.05, HEAD 0.09/0.15, FINAL
  0.08/0.35). Patched mlx-vlm final features give the same 0.052 / 0.049
  (unpatched 0.28 / 0.44). Corpus gate PASS for v1 and v2, every page
  identical to the new baselines; v1 baselines unchanged, v2 baselines keep
  the text and move boxes by a few units, list items, heading labels.
  Hermetic, checkpoint and E2E tiers pass; prek, fmt, clippy clean.
- Changed (same day): v2 multi-page requests tile every page
  (`recognize_pages`, decision above). 10 scanned corpus pages (article x2,
  essay, notice, report, twice) on M2 Pro: gundam 41.4 s, base 23.2 s,
  byte-identical Markdown; both differ from page-by-page output only in
  heading levels. The c3ae82e2 KV quantization has no aimd counterpart
  (aimd has no KV-cache quantization option).
- Next action: commit.

### 2026-10-29 — Claude — WS-02 (Metal decode speed)

- Changed: the fused decode kernel scores one key per lane, using `float4`
  loads against q staged in threadgroup memory, and sums values with
  `simd_shuffle`-broadcast weights. New `dash_decode_prologue` (shim
  `mlx_dash_decode_prologue`, `DashRouter::decode_prologue`): one Metal
  launch for the q/k rope and the router index q/k; it falls back to the
  compiled mlx traces off Metal. CUDA keeps the mlx-op path (out of scope).
  `aimd-mlx/examples/bench_dash` times the decode ops; `bench_ocr --repo`
  selects the checkpoint.
- Evidence (M2 Pro, 4-bit): decode kernel 104.9 -> 49-51 µs per layer;
  prologue ~84 -> 16 µs. End-to-end v2 decode 5.7-6.3 -> 3.9-4.4 ms/token
  (v1 3.7-4.1), same tokens on two corpus pages. `mlx_dash` tests pass on
  Metal and CPU, including the new prologue parity test. The
  `unlimited_ocr` checkpoint tests pass (`decoder2_dash_golden`). Corpus
  gate PASS (similarity 1.000 on 4/4 pages). OmniDocBench subset: 740 ->
  477 s for 66 pages, sim/F1 0.857/0.903 -> 0.860/0.905, stop 65 -> 66/66.
  prek, fmt, clippy and the hermetic tier are clean.
- Skipped (each under ~1% of a step): a compiled `decode_ctx`, a fixed
  pending index-key buffer.
- Next action: commit.

### 2026-10-29 — Claude — WS-02 (decode routing, vision profile)

- Changed: the decode kernel's 1.5-entmax prior runs on one simdgroup
  (prefix sums via `simd_prefix_inclusive_sum`) instead of one thread with
  stack arrays; the kernel gate requires `min(topm, cap) <= 32`.
- Evidence (M2 Pro): `dash_decode` 49-52 -> 37-41 µs per layer; v2 decode
  3.9-4.4 -> 3.5-3.6 ms/token, same tokens; `mlx_dash` and `unlimited_ocr`
  checkpoint tests pass; corpus gate PASS; gates clean.
- Tried, not kept: two lanes per key (slower), key split over 2-4
  threadgroups plus merge (within noise), chunked global SAM attention
  (within noise; vision is GEMM-bound at ~3.5 TFLOPS).
- Next action: commit.

## Completion summary

Pending.
