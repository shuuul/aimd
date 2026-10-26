---
id: "005"
title: "Source anchors: OCR positions, ASR timestamps, structured JSON/SRT/VTT output"
status: Active
created: 2026-10-26
updated: 2026-10-26
coordinator: "shuuul"
---

# 005 — Source anchors: OCR positions, ASR timestamps, structured JSON/SRT/VTT output

## Context

aimd output today is flat Markdown. A reader (or an LLM citing its context)
cannot tell where a passage came from:

- OCR keeps only `## Page N` headings. Unlimited-OCR already emits a category
  and box per block (`<|det|>title [37, 26, 259, 53]<|/det|>Quarterly Report`),
  but `normalize_unlimited_ocr_markdown` throws the boxes away. PP-OCRv6 has
  a quad and score per line (`TextLine`) that `NativeOcr` drops.
  PaddleOCR-VL-1.6's `OCR:` task has no coordinates, but the checkpoint
  documents a `Spotting:` task that returns text with `<|LOC_n|>` location
  tokens (1000 of them in the tokenizer).
- Local ASR returns plain text. Qwen3-ASR has no timestamps; Qwen publishes
  `Qwen3-ForcedAligner-0.6B` (timestamps for arbitrary units, audio up to
  5 minutes, 11 languages), and mlx-community ships 4/5/6/8-bit and bf16
  conversions. It shares the Qwen3-ASR audio/LLM architecture.
- URL subtitles are normalized to SRT, then `strip_subtitle_formatting`
  drops the timing (kept only with `--raw-transcript`).

The user asked for anchors in the default output, structured output via
`--format json|srt`, OCR positions, and ForcedAligner timestamps.

## Goal and success criteria

- [ ] Default OCR Markdown carries an invisible anchor before every located
      block: `<!-- aimd:loc page=P bbox=x0,y0,x1,y1 kind=K -->` (normalized
      0..1, 3 decimals) for Unlimited-OCR blocks and PP-OCRv6 lines.
- [ ] Default local-ASR and URL-subtitle transcripts are paragraphs that
      start with `[hh:mm:ss] `; local ASR times come from Qwen3-ForcedAligner
      for supported languages and fall back to chunk start times otherwise.
- [ ] `--no-anchors` reproduces today's Markdown byte for byte and skips the
      aligner.
- [ ] `--format json` writes the result with `segments` (page/bbox/kind for
      OCR, start/end for transcripts); `--format srt|vtt` writes subtitles
      for transcript tasks and fails with `UnsupportedInput` otherwise.
- [ ] MCP `process_input` accepts `anchors` (default true) and
      `include_segments` (default false) and returns `segments` when asked.
- [ ] Aligner timestamps match the Python reference within 80 ms mean
      absolute error on the corpus clips; OCR engine markdown and the corpus
      gate stay unchanged (anchors live in `aimd-process`, not the engines).
- [ ] fmt, clippy `-D warnings`, prek, hermetic tier pass; docs updated.

## Scope and non-goals

In scope: the engine changes below, `aimd_core::anchors` (rendering,
sentence/paragraph grouping, SRT/VTT writers, SRT parsing),
`ProcessResult.segments`, process wiring for file and URL inputs, CLI
`--format` / `--no-anchors`, MCP fields, docs.

Not in scope: word-level karaoke output, speaker diarization, remote
OpenAI `verbose_json` timestamps (remote ASR gets chunk-level times only),
PaddleOCR-VL layout analysis (PP-DocLayout), anchors for Pandoc/PDF-text
conversions, user-selectable aligner precision.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-10-26 | Anchors are on by default; `--no-anchors` (MCP `anchors: false`) restores the old output exactly | User request; one switch keeps old consumers working | all |
| 2026-10-26 | OCR anchors are HTML comments on their own line before each block: `<!-- aimd:loc page=P bbox=x0,y0,x1,y1 kind=K -->`, page 1-based (1 for single images), bbox normalized to the page image, 3 decimals | Invisible in Markdown viewers, kept in `chunk_list` so LLMs can cite; normalized coordinates are resolution-independent | WS-01, WS-04 |
| 2026-10-26 | ASR anchors are a visible `[hh:mm:ss] ` prefix per paragraph | Proposed to and accepted by the user; times are useful to human readers | WS-01, WS-04 |
| 2026-10-26 | Paragraphs: split transcript text into sentences at `。！？!?` / `.` + space / `…`; start a new paragraph on a gap >= 2.0 s between sentences or once the paragraph reaches 300 characters | Readable blocks with stable anchors; no model needed | WS-01 |
| 2026-10-26 | ForcedAligner runs locally with `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` (not user-selectable); when it runs, ASR segments are 300 s (aligner limit) instead of 600 s | Timestamp accuracy over size (0.6B); 300 s keeps one aligner call per ASR segment | WS-02, WS-04 |
| 2026-10-26 | Language not supported by the aligner, alignment failure, or remote ASR -> chunk-level anchors (segment start/end), never an error | Anchors are a best-effort annotation; transcripts must not fail because of them | WS-04 |
| 2026-10-26 | PaddleOCR-VL keeps `OCR:` for Markdown (page-level anchors only); with `--format json` / `include_segments` it also runs `Spotting:` for line segments | Spotting loses tables/structure; running it only on request avoids doubling default cost | WS-03, WS-04 |
| 2026-10-26 | `ProcessResult.segments: Option<Vec<Segment>>`; MCP gains `anchors` (default true) and `include_segments` (default false) and returns `segments` only when asked | Explicit approval to extend the frozen MCP contract (additive fields only) | WS-01, WS-05 |
| 2026-10-26 | `--format json` writes the MCP response object with `include_segments=true`; `.json` / `.srt` / `.vtt` output suffixes | One JSON shape for CLI and MCP | WS-05 |
| 2026-10-26 | SRT/VTT cues: one per sentence, split at a comma/space when longer than 7 s or 84 Latin / 42 CJK characters; minimum cue 0.5 s | Common subtitle reading-speed limits | WS-01 |
| 2026-10-26 | Engine API contracts below are frozen before WS-02/WS-03 start | Parallel workers in separate worktrees | WS-02, WS-03, WS-04 |

Core contract (`aimd_core::model`):

```rust
/// One located piece of the source. `None` fields are omitted on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub text: String,
    pub kind: Option<String>,          // OCR category ("title", "text", "table", "line", ...)
    pub page: Option<usize>,           // 1-based
    pub bbox: Option<[f32; 4]>,        // x0, y0, x1, y1, normalized 0..1
    pub quad: Option<[[f32; 2]; 4]>,   // normalized; rotated lines only
    pub start: Option<f64>,            // seconds
    pub end: Option<f64>,
    pub score: Option<f32>,
}
// ProcessResult gains `pub segments: Option<Vec<Segment>>` (drops `Eq`).
// ProcessInput gains `pub anchors: bool` (serde default true).
```

Engine contracts:

```rust
// aimd_engine::models::qwen3_aligner::pipeline   (WS-02)
pub const ALIGNER_MAX_SECONDS: f32 = 300.0;
pub const ALIGNER_LANGUAGES: &[&str];  // canonical names as in tokenizer::ASR_LANGUAGES
pub struct ForcedAlignerPipeline { .. }
impl ForcedAlignerPipeline {
    pub fn load(dir: &Path) -> Result<Self, AlignError>;
    /// `text` is the ASR transcript of `wav` (16 kHz mono, <= 300 s);
    /// `language` a canonical name. Units follow the reference processor
    /// (characters for CJK, words otherwise) and concatenate to `text`
    /// modulo whitespace and dropped punctuation.
    pub fn align_path(&self, wav: &Path, text: &str, language: &str)
        -> Result<Vec<AlignedUnit>, AlignError>;
}
pub struct AlignedUnit { pub text: String, pub start: f32, pub end: f32 }
pub enum AlignError { UnsupportedLanguage(String), TooLong(f32), .. }

// aimd_engine::models::qwen3_asr::pipeline   (WS-02)
pub struct Transcription { pub text: String, pub language: Option<String> }
// Qwen3AsrPipeline::transcribe_detailed(&self, &Path, TranscriptionRequest)
//     -> Result<Transcription, PipelineError>; `language` is the canonical
//     name the model reported (or the forced one). `transcribe` stays.

// aimd_engine::models::unlimited_ocr::pipeline   (WS-03)
// OcrResult gains `pub blocks: Vec<LayoutBlock>`; empty when the output has
// no det markers. Image blocks are dropped like the normalizer does, and the
// block texts joined by "\n\n" equal `markdown`.
pub struct LayoutBlock { pub kind: String, pub bbox: [f32; 4] /* 0..1 */, pub text: String }

// aimd_engine::models::pp_ocrv6::pipeline   (WS-03)
// PpOcrOutput gains `pub width: u32, pub height: u32` (quads stay in pixels).

// aimd_engine::models::paddleocr_vl::pipeline   (WS-03)
#[derive(Default)] pub enum VlTask { #[default] Ocr, Spotting }
// VlRequest gains `pub task: VlTask`; VlOutput gains `pub spots: Vec<Spot>`
// (empty for Ocr). Spotting follows the checkpoint README preprocessing
// (x2 LANCZOS upscale when both sides < 1500, max_pixels 2048*28*28).
pub struct Spot { pub text: String, pub quad: [[f32; 2]; 4] /* 0..1 of the original image */ }
```

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | `aimd_core`: `Segment`, `ProcessResult.segments`, `ProcessInput.anchors`, `anchors` module (loc comment, `[hh:mm:ss]`, sentences, paragraphs, SRT/VTT writers, SRT cue parser) | coordinator | Completed | None | hermetic unit tests |
| WS-02 | `aimd-engine::models::qwen3_aligner` (Qwen3-ForcedAligner-0.6B on MLX, reuse Qwen3-ASR encoder/decoder code) + `Qwen3AsrPipeline::transcribe_detailed`; goldens via `scripts/export_aligner_reference.py` | agent ws02 | Pending | None | hermetic goldens; checkpoint parity; mean abs timestamp error vs reference <= 80 ms on corpus clips |
| WS-03 | OCR engine positions: Unlimited-OCR `blocks`, PP-OCRv6 `width`/`height`, PaddleOCR-VL `Spotting` task + `spots` | agent ws03 | Pending | None | hermetic parse tests; checkpoint test that `blocks` join to `markdown`; spotting text/quads vs reference |
| WS-04 | `aimd-process` + `aimd-url` wiring: OCR segments/anchors, ASR 300 s segments + alignment + paragraphs, URL subtitle cues and audio fallback, chunk-level fallback | coordinator | Pending | WS-01, WS-02/WS-03 API | hermetic tests with fake `Runtime`s |
| WS-05 | CLI `--format`, `--no-anchors`, output suffixes; MCP `anchors` / `include_segments`; README, AGENTS.md, help texts | coordinator | Pending | WS-01, WS-04 | CLI/MCP tests; smoke runs |
| WS-06 | E2E check on a real podcast clip and scanned PDF; performance notes; corpus gate unchanged | coordinator | Pending | WS-02..05 | `docs/performance.md`; `corpus_gate` PASS |

## Verification

```bash
AIMD_MLX_DEVICE=cpu cargo test --workspace
cargo test -p aimd-engine qwen3_aligner unlimited_ocr pp_ocrv6 paddleocr_vl   # checkpoint tier
AIMD_E2E=1 cargo test -p aimd-engine --lib qwen3_aligner
./target/release/aimd podcast.mp3 --format srt
./target/release/aimd scan.pdf --format json
cargo run --release -p aimd-engine --example corpus_gate
cargo xtask check-specs
```

## Documentation sync

- Durable product/developer docs: README (output formats, anchors,
  timestamps), `docs/performance.md` (aligner cost).
- Nearest local `AGENTS.md`: root `AGENTS.md` (output contract, crate map,
  golden sets, MCP fields).
- Parent/package guidance: none.
- Root guidance and roadmap: root `AGENTS.md`.

## Progress and handoff

Append entries rather than rewriting another worker's record.

- 2026-10-26 (coordinator, WS-01): Completed. `aimd_core::anchors` covers
  `loc_comment`, `quad_bbox`, `located_markdown`, `clock`, `sentence_spans`,
  `timed_sentences` (alignment units to sentences by alphanumeric matching),
  `spread_sentences` (chunk-level fallback), `timed_paragraphs`,
  `subtitle_cues`, `to_srt`/`to_vtt`, and `parse_srt` (rolling-caption dedupe).
  `Segment`, `ProcessResult.segments` and `ProcessInput.anchors` (default true)
  are in place; CLI/MCP set `anchors: true` until WS-05. Gates: clippy
  `-D warnings`, `AIMD_MLX_DEVICE=cpu cargo test --workspace` green.

## Completion summary

Pending.
