//! Qwen3-ForcedAligner pipeline on MLX: wav + transcript -> timed units.
//!
//! Data flow (one forward, mirroring mlx-audio `ForcedAlignerModel.generate`):
//!
//! 1. wav -> 128-bin log-mel ([`crate::audio`]) -> the Qwen3-ASR audio tower
//!    ([`MlxTower`], shared code and weight layout) -> `[pads, 1024]` f32.
//! 2. Units ([`split_units`]) -> prompt ([`build_aligner_prompt`]) -> ids;
//!    every `<|audio_pad|>` embedding row is replaced by a tower row
//!    ([`MlxDecoder::merged_prompt_embeds`]).
//! 3. One cache-free causal pass of the 28-layer Qwen3 decoder
//!    ([`MlxDecoder::forward_full`]) and the untied 8-bit `lm_head`
//!    (`[5000, 1024]`) over the whole sequence, exactly the reference's
//!    `lm_head(model(inputs_embeds))`; the rows at the `<timestamp>`
//!    positions are gathered and argmaxed on the device (first index wins
//!    ties, like `mx.argmax`).
//! 4. Classes x 80 ms -> [`decode_timestamps`] (LIS repair) -> seconds.

use std::path::Path;

use aimd_mlx::{MlxArray, MlxError};
use aimd_weights::{WeightsError, load_checkpoint};

use crate::audio::{AudioError, MelConfig, load_wav, mel_spectrogram};
use crate::models::qwen3_aligner::config::{AlignerCheckpointConfig, parse_config};
use crate::models::qwen3_aligner::units::{build_aligner_prompt, decode_timestamps, split_units};
use crate::models::qwen3_asr::pipeline::{MlxDecoder, MlxQuantWeights, MlxTower, PipelineError};
use crate::tokenizer::{Tokenizer, TokenizerError, resolve_language};

/// Longest clip the aligner accepts (the model card's 5 minutes).
pub const ALIGNER_MAX_SECONDS: f32 = 300.0;

/// Languages the native aligner supports, as canonical
/// [`crate::tokenizer::ASR_LANGUAGES`] names: the checkpoint's
/// `support_languages` minus Japanese, whose reference unit splitting needs
/// the nagisa neural tagger.
pub const ALIGNER_LANGUAGES: &[&str] = &[
    "Chinese",
    "Cantonese",
    "English",
    "German",
    "Spanish",
    "French",
    "Italian",
    "Portuguese",
    "Russian",
    "Korean",
];

/// Sample rate of the accepted wav input.
const SAMPLE_RATE: f32 = 16_000.0;

/// One aligned unit (a CJK character or a word, punctuation dropped);
/// `start` / `end` in seconds from the start of the clip.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedUnit {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Aligner failures.
#[derive(Debug, thiserror::Error)]
pub enum AlignError {
    /// Not in [`ALIGNER_LANGUAGES`] (canonical name or code accepted).
    #[error("forced aligner does not support language {0:?}")]
    UnsupportedLanguage(String),
    /// Clip longer than [`ALIGNER_MAX_SECONDS`] (duration in seconds).
    #[error("audio is {0:.1} s, the forced aligner accepts at most {ALIGNER_MAX_SECONDS} s")]
    TooLong(f32),
    #[error("audio: {0}")]
    Audio(#[from] AudioError),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    /// Shared Qwen3-ASR graph failure (tower / decoder / layout).
    #[error("{0}")]
    Model(#[from] PipelineError),
}

/// Canonical aligner language for a canonical name or code.
pub fn aligner_language(language: &str) -> Result<&'static str, AlignError> {
    resolve_language(language)
        .and_then(|name| ALIGNER_LANGUAGES.iter().copied().find(|l| *l == name))
        .ok_or_else(|| AlignError::UnsupportedLanguage(language.to_string()))
}

/// A loaded Qwen3-ForcedAligner checkpoint.
pub struct ForcedAlignerPipeline {
    tokenizer: Tokenizer,
    tower: MlxTower,
    decoder: MlxDecoder,
    /// Time classifier `[classify_num, hidden]` (affine-quantized).
    lm_head: MlxQuantWeights,
    audio_token_id: u32,
    timestamp_token_id: u32,
    segment_ms: f64,
    output_dim: usize,
    mel_config: MelConfig,
}

impl std::fmt::Debug for ForcedAlignerPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForcedAlignerPipeline")
            .field("decoder", &self.decoder.cfg)
            .field("timestamp_token_id", &self.timestamp_token_id)
            .field("segment_ms", &self.segment_ms)
            .finish_non_exhaustive()
    }
}

/// Intermediate results of one alignment (checked stage by stage against
/// the goldens).
// Only `classes` is read outside the checkpoint-parity tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct AlignStages {
    /// Tower output `[pads, output_dim]` f32.
    pub(crate) features: Vec<f32>,
    pub(crate) prompt_ids: Vec<u32>,
    /// `lm_head` logits at the timestamp slots `[n_ts, classify_num]` bf16.
    pub(crate) ts_logits: MlxArray,
    /// Argmax class per timestamp slot.
    pub(crate) classes: Vec<u32>,
}

impl ForcedAlignerPipeline {
    /// Load a checkpoint snapshot directory
    /// (`mlx-community/Qwen3-ForcedAligner-0.6B-8bit`).
    pub fn load(dir: &Path) -> Result<Self, AlignError> {
        let store = load_checkpoint(dir)?;
        let raw = std::fs::read(dir.join("config.json"))
            .map_err(|e| AlignError::Config(format!("read config.json: {e}")))?;
        let cfg: AlignerCheckpointConfig =
            parse_config(&raw).map_err(|e| AlignError::Config(e.to_string()))?;
        let thinker = &cfg.thinker_config;
        let mel_config = MelConfig::default();
        if mel_config.n_mels != thinker.audio_config.num_mel_bins {
            return Err(AlignError::Config(format!(
                "mel config n_mels {} != checkpoint num_mel_bins {}",
                mel_config.n_mels, thinker.audio_config.num_mel_bins
            )));
        }
        let tokenizer = Tokenizer::from_checkpoint(dir)?;
        if tokenizer.token_id("<timestamp>")? != cfg.timestamp_token_id {
            return Err(AlignError::Config(
                "tokenizer <timestamp> id differs from config timestamp_token_id".to_string(),
            ));
        }
        let tower = MlxTower::from_store(&thinker.audio_config, &store)?;
        let decoder = MlxDecoder::from_store_body(&thinker.text_config, &store)?;
        let lm_head = MlxQuantWeights::from_store(&store, "lm_head")?;
        Ok(Self {
            tokenizer,
            tower,
            decoder,
            lm_head,
            audio_token_id: thinker.audio_token_id,
            timestamp_token_id: cfg.timestamp_token_id,
            segment_ms: cfg.timestamp_segment_time,
            output_dim: thinker.audio_config.output_dim,
            mel_config,
        })
    }

    /// Align `text` (the transcript of `wav`, 16 kHz mono, at most
    /// [`ALIGNER_MAX_SECONDS`]) in `language` (canonical name or code from
    /// [`ALIGNER_LANGUAGES`]). Units are CJK characters / words with
    /// punctuation dropped; an empty unit list returns `Ok(vec![])` without
    /// running the model.
    pub fn align_path(
        &self,
        wav: &Path,
        text: &str,
        language: &str,
    ) -> Result<Vec<AlignedUnit>, AlignError> {
        let language = aligner_language(language)?;
        let audio = load_wav(wav)?;
        let seconds = audio.samples.len() as f32 / SAMPLE_RATE;
        if seconds > ALIGNER_MAX_SECONDS {
            return Err(AlignError::TooLong(seconds));
        }
        let units = split_units(text, language);
        if units.is_empty() {
            return Ok(Vec::new());
        }
        let stages = self.stages(&audio.samples, &units)?;
        Ok(decode_timestamps(&units, &stages.classes, self.segment_ms))
    }

    /// Mel -> tower -> merged prompt -> decoder -> lm_head at the slots.
    pub(crate) fn stages(
        &self,
        samples: &[f32],
        units: &[String],
    ) -> Result<AlignStages, AlignError> {
        let mel = mel_spectrogram(samples, &self.mel_config)?;
        let (features, pads) = self.tower.forward(&mel.data, mel.frames)?;
        let prompt_ids = self.tokenizer.encode(&build_aligner_prompt(units, pads))?;
        let slots: Vec<u32> = prompt_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == self.timestamp_token_id)
            .map(|(i, _)| i as u32)
            .collect();
        if slots.len() != units.len() * 2 {
            return Err(PipelineError::Layout(format!(
                "{} timestamp slots for {} units",
                slots.len(),
                units.len()
            ))
            .into());
        }
        let embeds = self.decoder.merged_prompt_embeds(
            &prompt_ids,
            &features,
            self.audio_token_id,
            self.output_dim,
        )?;
        let hidden = self.decoder.forward_full(embeds)?;
        let seq = prompt_ids.len();
        let hidden = hidden.reshape(&[seq, self.decoder.cfg.hidden_size])?;
        let logits = self.lm_head.matmul(&hidden)?;
        let ts_logits = logits.take(&MlxArray::from_u32(&[slots.len()], &slots)?, 0)?;
        let classes = ts_logits.argmax(-1, false)?.to_u32_vec()?;
        aimd_mlx::clear_cache()?;
        Ok(AlignStages {
            features,
            prompt_ids,
            ts_logits,
            classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        CASES, e2e_enabled, golden_dir, read_bf16, read_f32, read_json, shared_pipeline,
    };
    use super::*;
    use crate::audio::load_wav;
    use crate::models::qwen3_aligner::units::split_units;
    use std::path::PathBuf;

    fn meta(case: &str) -> serde_json::Value {
        read_json(&golden_dir().join("cases").join(case).join("meta.json"))
    }

    fn u32s(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn items(v: &serde_json::Value) -> Vec<AlignedUnit> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|it| AlignedUnit {
                text: it["text"].as_str().unwrap().to_string(),
                start: it["start"].as_f64().unwrap() as f32,
                end: it["end"].as_f64().unwrap() as f32,
            })
            .collect()
    }

    fn strings(v: &serde_json::Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn aligner_language_gate() {
        assert_eq!(aligner_language("English").unwrap(), "English");
        assert_eq!(aligner_language("zh").unwrap(), "Chinese");
        assert_eq!(aligner_language("yue").unwrap(), "Cantonese");
        assert_eq!(aligner_language("korean").unwrap(), "Korean");
        for unsupported in ["Japanese", "ja", "Arabic", "xx", ""] {
            assert!(
                matches!(
                    aligner_language(unsupported),
                    Err(AlignError::UnsupportedLanguage(ref l)) if l == unsupported
                ),
                "{unsupported}"
            );
        }
        for lang in ALIGNER_LANGUAGES {
            assert_eq!(resolve_language(lang).as_deref(), Some(*lang));
        }
    }

    #[test]
    fn aligner_missing_checkpoint_reports_typed_error() {
        let err = ForcedAlignerPipeline::load(Path::new("/definitely/not/here-aimd-aligner"))
            .expect_err("missing checkpoint must fail");
        assert!(matches!(err, AlignError::Weights(_)), "{err}");
    }

    /// Hermetic timestamp decoding: argmax over the committed reference
    /// logits slices (first index wins ties) -> classes -> repaired seconds
    /// must reproduce the reference items exactly.
    #[test]
    fn aligner_decodes_reference_logits() {
        for case in CASES {
            let m = meta(case);
            let classify = m["classify_num"].as_u64().unwrap() as usize;
            let logits = read_bf16(&golden_dir().join("cases").join(case).join("ts_logits.bf16"));
            let classes: Vec<u32> = logits
                .chunks(classify)
                .map(|row| {
                    let mut best = 0;
                    for (i, v) in row.iter().enumerate() {
                        if *v > row[best] {
                            best = i;
                        }
                    }
                    best as u32
                })
                .collect();
            assert_eq!(classes, u32s(&m["ts_argmax"]), "{case}");
            let units = strings(&m["units"]);
            assert_eq!(
                split_units(m["text"].as_str().unwrap(), m["language"].as_str().unwrap()),
                units
            );
            assert_eq!(
                decode_timestamps(&units, &classes, 80.0),
                items(&m["items"]),
                "{case}"
            );
        }
    }

    /// Prompt ids against the reference (tokenizer only: runs whenever the
    /// checkpoint tokenizer is cached, on any device).
    #[test]
    fn aligner_prompt_ids_match_reference() {
        let Some(dir) = super::super::test_support::checkpoint_dir() else {
            eprintln!("skipping: aligner checkpoint not in HF cache");
            return;
        };
        let tokenizer = Tokenizer::from_checkpoint(&dir).unwrap();
        for case in CASES {
            let m = meta(case);
            let units = strings(&m["units"]);
            let pads = m["audio_pad_count"].as_u64().unwrap() as usize;
            let ids = tokenizer
                .encode(&build_aligner_prompt(&units, pads))
                .unwrap();
            assert_eq!(ids, u32s(&m["prompt_ids"]), "{case}");
        }
    }

    /// Checkpoint parity per stage on the committed clips: tower features,
    /// prompt ids, timestamp-slot logits (bf16), argmax classes, and the
    /// final units must match the mlx-audio reference.
    #[test]
    fn aligner_matches_reference_stages() {
        const FEATURE_ATOL: f64 = 1e-3;
        const FEATURE_RTOL: f64 = 1e-2;
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        for case in CASES {
            let dir = golden_dir().join("cases").join(case);
            let m = meta(case);
            let units = strings(&m["units"]);
            let wav = load_wav(&dir.join("audio.wav")).unwrap();
            let stages = pipeline.stages(&wav.samples, &units).unwrap();

            let reference = read_f32(&dir.join("audio_features.f32"));
            assert_eq!(stages.features.len(), reference.len(), "{case}");
            let mut max_abs = 0f64;
            let mut violations = 0;
            for (&a, &r) in stages.features.iter().zip(&reference) {
                let abs = (f64::from(a) - f64::from(r)).abs();
                max_abs = max_abs.max(abs);
                if abs > FEATURE_ATOL && abs / f64::from(r).abs() > FEATURE_RTOL {
                    violations += 1;
                }
            }
            assert_eq!(stages.prompt_ids, u32s(&m["prompt_ids"]), "{case}");

            let got = stages
                .ts_logits
                .astype(aimd_mlx::MlxDtype::F32)
                .and_then(|a| a.to_f32_vec())
                .unwrap();
            let want = read_bf16(&dir.join("ts_logits.bf16"));
            assert_eq!(got.len(), want.len(), "{case}");
            let logit_max_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "aligner {case}: tower max_abs={max_abs:.3e} violations={violations}, \
                 ts logits max_abs={logit_max_abs:.3e}"
            );
            assert_eq!(violations, 0, "{case}: tower output beyond the golden gate");
            assert!(logit_max_abs <= 0.5, "{case}: logits drifted");
            assert_eq!(stages.classes, u32s(&m["ts_argmax"]), "{case}");

            let aligned = pipeline
                .align_path(
                    &dir.join("audio.wav"),
                    m["text"].as_str().unwrap(),
                    m["language"].as_str().unwrap(),
                )
                .unwrap();
            assert_eq!(aligned, items(&m["items"]), "{case}");
        }
    }

    #[test]
    fn aligner_rejects_long_audio_and_unsupported_language() {
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let wav = golden_dir().join("cases/en01/audio.wav");
        assert!(matches!(
            pipeline.align_path(&wav, "hello", "Japanese"),
            Err(AlignError::UnsupportedLanguage(_))
        ));
        assert!(
            pipeline
                .align_path(&wav, "...", "English")
                .unwrap()
                .is_empty()
        );

        let path = std::env::temp_dir().join("aimd_aligner_too_long.wav");
        let samples = 301 * 16_000;
        let mut data = Vec::with_capacity(44 + samples * 2);
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + samples * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVEfmt ");
        for v in [16u32.to_le_bytes(), [1, 0, 1, 0], 16000u32.to_le_bytes()] {
            data.extend_from_slice(&v);
        }
        data.extend_from_slice(&32000u32.to_le_bytes());
        data.extend_from_slice(&[2, 0, 16, 0]);
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((samples * 2) as u32).to_le_bytes());
        data.resize(44 + samples * 2, 0);
        std::fs::write(&path, data).unwrap();
        let err = pipeline.align_path(&path, "hello", "English").unwrap_err();
        assert!(
            matches!(err, AlignError::TooLong(s) if (s - 301.0).abs() < 1e-3),
            "{err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Tier 3: a freshly `say`-rendered English sentence aligns to ordered,
    /// in-range spans covering every word.
    #[test]
    fn aligner_e2e_say_clip() {
        if !e2e_enabled() {
            eprintln!("skipping: tier-3 E2E (set AIMD_E2E=1)");
            return;
        }
        let Some(pipeline) = shared_pipeline() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("aimd_aligner_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let aiff = dir.join("clip.aiff");
        let wav: PathBuf = dir.join("clip.wav");
        let text = "Forced alignment turns a transcript into timed words.";
        let ok = std::process::Command::new("say")
            .arg("-o")
            .arg(&aiff)
            .arg(text)
            .status()
            .is_ok_and(|s| s.success())
            && std::process::Command::new("ffmpeg")
                .args(["-y", "-loglevel", "error", "-i"])
                .arg(&aiff)
                .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
                .arg(&wav)
                .status()
                .is_ok_and(|s| s.success());
        if !ok {
            eprintln!("skipping: say/ffmpeg unavailable");
            return;
        }
        let duration = load_wav(&wav).unwrap().samples.len() as f32 / 16_000.0;
        let units = pipeline.align_path(&wav, text, "en").unwrap();
        let words: Vec<&str> = units.iter().map(|u| u.text.as_str()).collect();
        assert_eq!(
            words,
            [
                "Forced",
                "alignment",
                "turns",
                "a",
                "transcript",
                "into",
                "timed",
                "words"
            ]
        );
        let mut last = 0.0;
        for u in &units {
            assert!(u.start >= last && u.end >= u.start, "{u:?}");
            assert!(u.end <= duration + 0.08, "{u:?} beyond {duration}");
            last = u.start;
        }
        assert!(units.last().unwrap().end > duration * 0.5);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
