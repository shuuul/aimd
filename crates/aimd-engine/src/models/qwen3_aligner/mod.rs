//! Qwen3-ForcedAligner-0.6B on the MLX backend (spec 005 WS-02).
//!
//! The aligner is a Qwen3-ASR checkpoint whose LM head is a 5000-way time
//! classifier (80 ms bins, so up to 400 s): one causal forward over
//! `<|audio_start|>{<|audio_pad|> x N}<|audio_end|>unit<timestamp><timestamp>
//! unit<timestamp><timestamp>...` predicts, at every `<timestamp>` slot, the
//! start (first slot) and end (second slot) bin of the preceding unit. There
//! is no generation loop and no chat template.
//!
//! # Module layout
//!
//! - [`config`]: checkpoint config (`thinker_config` audio/text configs shared
//!   with Qwen3-ASR, plus `classify_num`, `timestamp_token_id`,
//!   `timestamp_segment_time`).
//! - [`units`]: the host-side processor: unit splitting per language, the
//!   prompt string, and timestamp repair/decoding (`fix_timestamp`).
//! - [`pipeline`]: [`pipeline::ForcedAlignerPipeline`], which reuses the
//!   Qwen3-ASR MLX audio tower and decoder from
//!   [`crate::models::qwen3_asr::pipeline`] (weights load the same way; the
//!   decoder body is shared, the untied quantized `lm_head` is the only new
//!   tensor).
//!
//! # Reference implementations
//!
//! - Graph: mlx-audio 0.6.1 `mlx_audio/stt/models/qwen3_asr/
//!   qwen3_forced_aligner.py` (`ForcedAlignerModel`) on the product
//!   checkpoint `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`; the goldens
//!   under `tests/golden/aligner/` come from it via
//!   `scripts/export_aligner_reference.py`.
//! - Processor: the official `qwen_asr.inference.qwen3_forced_aligner
//!   .Qwen3ForceAlignProcessor` (qwen-asr 0.0.6). It differs from mlx-audio's
//!   copy for Chinese (whitespace path, not `tokenize_chinese_mixed`) and
//!   Korean (soynlp `LTokenizer` scored with `korean_dict_jieba.dict`); the
//!   exporter patches mlx-audio to the official behaviour. Japanese needs the
//!   nagisa neural tagger and is not supported here.
//!
//! # Parity
//!
//! Given identical merged embeddings the decoder is bit-exact with the
//! reference. The tower matches to ~1e-5 (f32), which flips the bf16
//! rounding of a few audio embeddings; short clips stay exact (or one 80 ms
//! bin off), but over thousands of tokens the bf16 decoder amplifies the
//! flips, exactly as it does for the reference under 2e-6 tower noise.
//! `examples/aligner_parity.rs` gates the corpus clips and reports the
//! 60 s / 300 s probes.

pub mod config;
pub mod pipeline;
pub mod units;

#[cfg(test)]
pub(crate) mod test_support;
