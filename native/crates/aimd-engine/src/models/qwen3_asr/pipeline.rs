//! End-to-end Qwen3-ASR pipeline (WS-06 integration).
//!
//! Wires the wave-1/WS-06 pieces: wav -> mel ([WS-05]) -> audio tower
//! ([encoder]) -> prompt + token ids ([tokenizer]) -> Qwen3 decoder greedy
//! generation ([decoder]/[generate]) -> `parse_qwen_output`, with the
//! 4-bit -> 8-bit retry when the repetition-loop detector trips.
//!
//! All weights are consumed exactly as stored in the checkpoint (the
//! "checkpoint quantization is the contract" decision): the tower's bf16
//! tensors verbatim, decoder linears/embeddings through the bit-exact
//! affine dequantization.

use std::path::Path;

use crate::audio::{AudioError, MelConfig, load_wav, mel_spectrogram};
use crate::models::qwen3_asr::config::{Qwen3AsrCheckpointConfig, parse_config};
use crate::models::qwen3_asr::decoder::{DecoderError, Qwen3Decoder};
use crate::models::qwen3_asr::encoder::{AudioTower, EncoderError, frames_to_audio_pads};
use crate::models::qwen3_asr::generate::{
    GenerateError, GenerationOutcome, GenerationSettings, greedy_generate, parse_qwen_output,
};
use crate::tokenizer::{Tokenizer, TokenizerError, build_asr_prompt, resolve_language};

use aimd_tensor::{Shape, Tensor, TensorError};
use aimd_weights::{WeightsError, load_checkpoint};

/// Pipeline-level failures, chaining the module errors.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("audio: {0}")]
    Audio(#[from] AudioError),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("config mismatch: {0}")]
    ConfigMismatch(String),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("encoder: {0}")]
    Encoder(#[from] EncoderError),
    #[error("decoder: {0}")]
    Decoder(#[from] DecoderError),
    #[error("generate: {0}")]
    Generate(#[from] GenerateError),
    #[error("tensor: {0}")]
    Tensor(#[from] TensorError),
    #[error("unsupported language {0:?}")]
    UnsupportedLanguage(String),
    #[error("transcription was empty")]
    EmptyTranscription,
}

/// One transcription request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TranscriptionRequest<'a> {
    /// Short code (`"zh"`) or full name (`"Chinese"`); `None` = no hint.
    pub language: Option<&'a str>,
    /// Free-form biasing context injected as the system turn.
    pub context: Option<&'a str>,
}

fn resolved_language(input: &str) -> Result<String, PipelineError> {
    resolve_language(input).ok_or_else(|| PipelineError::UnsupportedLanguage(input.to_string()))
}

/// A loaded Qwen3-ASR checkpoint ready to transcribe.
pub struct Qwen3AsrPipeline {
    tower: AudioTower,
    decoder: Qwen3Decoder,
    tokenizer: Tokenizer,
    audio_token_id: u32,
    output_dim: usize,
    num_mel_bins: usize,
    mel_config: MelConfig,
}

impl Qwen3AsrPipeline {
    /// Load a checkpoint snapshot directory (4-bit or 8-bit — the retry is
    /// just a second [`Qwen3AsrPipeline`] on the other directory).
    pub fn load(checkpoint_dir: &Path) -> Result<Self, PipelineError> {
        let store = load_checkpoint(checkpoint_dir)?;
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| PipelineError::ConfigMismatch(format!("read config.json: {e}")))?;
        let cfg: Qwen3AsrCheckpointConfig = parse_config(&raw)?;
        let thinker = &cfg.thinker_config;
        let tokenizer = Tokenizer::from_checkpoint(checkpoint_dir)?;
        let tower = AudioTower::from_weight_store(&thinker.audio_config, &store)?;
        let decoder = Qwen3Decoder::load(checkpoint_dir)?;
        let mel_config = MelConfig::default();
        if mel_config.n_mels != thinker.audio_config.num_mel_bins {
            return Err(PipelineError::ConfigMismatch(format!(
                "mel config n_mels {} != checkpoint num_mel_bins {}",
                mel_config.n_mels, thinker.audio_config.num_mel_bins
            )));
        }
        Ok(Self {
            tower,
            decoder,
            tokenizer,
            audio_token_id: thinker.audio_token_id,
            output_dim: thinker.audio_config.output_dim,
            num_mel_bins: thinker.audio_config.num_mel_bins,
            mel_config,
        })
    }

    /// Transcribe one 16 kHz mono wav file.
    pub fn transcribe(
        &self,
        wav_path: &Path,
        request: TranscriptionRequest<'_>,
    ) -> Result<String, PipelineError> {
        let language = match request.language {
            None => None,
            Some(input) => Some(resolved_language(input)?),
        };
        let outcome = self.generate(wav_path, language.as_deref(), request.context)?;
        finish_transcription(outcome)
    }

    fn generate(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<GenerationOutcome, PipelineError> {
        let wav = load_wav(wav_path)?;
        let mel = mel_spectrogram(&wav.samples, &self.mel_config)?;
        // MelSpectrogram is [n_mels][frames] frequency-major; the tower
        // consumes [frames][n_mels] time-major.
        let mut mel_t = vec![0.0f32; mel.data.len()];
        for m in 0..mel.n_mels {
            for f in 0..mel.frames {
                mel_t[f * self.num_mel_bins + m] = mel.data[m * mel.frames + f];
            }
        }
        let tower_out = self.tower.forward(&mel_t, mel.frames)?;
        debug_assert_eq!(tower_out.pads, frames_to_audio_pads(mel.frames));
        let prompt = build_asr_prompt(context, language, tower_out.pads);
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let features = Tensor::from_f32_vec(
            Shape::from_dims(&[tower_out.pads, self.output_dim]),
            &tower_out.features,
        )?;
        Ok(greedy_generate(
            &self.decoder,
            &self.tokenizer,
            &prompt_ids,
            &features,
            self.audio_token_id,
            GenerationSettings::default(),
        )?)
    }
}

/// Turn a generation outcome into final transcription text.
fn finish_transcription(outcome: GenerationOutcome) -> Result<String, PipelineError> {
    let text = parse_qwen_output(&outcome.text).trim().to_string();
    if text.is_empty() {
        return Err(PipelineError::EmptyTranscription);
    }
    Ok(text)
}

/// Whether a 4-bit outcome warrants the 8-bit retry (pure, unit-testable).
pub fn should_retry(outcome: &GenerationOutcome) -> bool {
    outcome.loop_detected
}

/// Transcribe with the 4-bit -> 8-bit retry: run `primary`; when the
/// repetition-loop detector trips and a `retry` pipeline is supplied, rerun
/// on it. Retry failure falls back to the primary result (matching the
/// Python backend's best-effort semantics).
pub fn transcribe_with_retry(
    primary: &Qwen3AsrPipeline,
    retry: Option<&Qwen3AsrPipeline>,
    wav_path: &Path,
    request: TranscriptionRequest<'_>,
) -> Result<String, PipelineError> {
    let language = match request.language {
        None => None,
        Some(input) => Some(resolved_language(input)?),
    };
    let outcome = primary.generate(wav_path, language.as_deref(), request.context)?;
    if !should_retry(&outcome) {
        return finish_transcription(outcome);
    }
    match retry {
        Some(p8) => match p8.transcribe(wav_path, request) {
            Ok(text) => Ok(text),
            Err(_) => finish_transcription(outcome),
        },
        None => finish_transcription(outcome),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::models::qwen3_asr::generate::FinishReason;

    fn e2e_enabled() -> bool {
        std::env::var("AIMD_E2E").as_deref() == Ok("1")
    }

    fn checkpoint_dir(precision: &str) -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(format!("models--mlx-community--Qwen3-ASR-1.7B-{precision}"))
            .join("snapshots");
        let entries = std::fs::read_dir(snapshots).ok()?;
        entries
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("config.json").is_file())
    }

    #[test]
    fn pipeline_retry_decision_is_pure() {
        let healthy = GenerationOutcome {
            tokens: vec![1, 2],
            text: "hello".to_string(),
            finish_reason: FinishReason::Eos,
            loop_detected: false,
        };
        assert!(!should_retry(&healthy));
        let mut looping = healthy;
        looping.loop_detected = true;
        assert!(should_retry(&looping));
    }

    /// Full-stack smoke on a synthetic sine (noise input): the model emits
    /// EOS immediately, so the pipeline must report `EmptyTranscription` —
    /// proving every stage runs and composes. Gated behind AIMD_E2E=1
    /// because it loads the full checkpoint.
    #[test]
    fn pipeline_sine_fixture_end_to_end() {
        if !e2e_enabled() {
            eprintln!("skipping: set AIMD_E2E=1 to run the full-stack pipeline test");
            return;
        }
        let Some(dir) = checkpoint_dir("4bit") else {
            eprintln!("skipping: Qwen3-ASR-1.7B-4bit not in HF cache");
            return;
        };
        let pipe = Qwen3AsrPipeline::load(&dir).expect("load pipeline");
        // Same recipe as the tokenizer/mel fixtures: 1.5 s 440 Hz sine.
        let samples: Vec<f32> = (0..24000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let tmp = std::env::temp_dir().join("aimd_pipeline_sine.wav");
        write_s16_wav(&tmp, &samples);
        let err = pipe
            .transcribe(&tmp, TranscriptionRequest::default())
            .expect_err("noise input must yield EmptyTranscription");
        assert!(matches!(err, PipelineError::EmptyTranscription), "{err}");
    }

    /// Real-speech E2E via macOS TTS: non-empty transcription required.
    #[test]
    fn pipeline_transcribes_say_clip() {
        if !e2e_enabled() {
            eprintln!("skipping: set AIMD_E2E=1 to run the TTS pipeline test");
            return;
        }
        let Some(dir) = checkpoint_dir("4bit") else {
            eprintln!("skipping: Qwen3-ASR-1.7B-4bit not in HF cache");
            return;
        };
        let aiff = std::env::temp_dir().join("aimd_pipeline_say.aiff");
        let wav = std::env::temp_dir().join("aimd_pipeline_say.wav");
        let ok = |s: std::io::Result<std::process::ExitStatus>| {
            s.map(|st| st.success()).unwrap_or(false)
        };
        if !ok(std::process::Command::new("say")
            .args(["-o", aiff.to_str().unwrap()])
            .arg("Today's weather is great and context engineering just got simpler.")
            .status())
        {
            eprintln!("skipping: macOS `say` unavailable");
            return;
        }
        if !ok(std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&aiff)
            .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(&wav)
            .status())
        {
            eprintln!("skipping: ffmpeg unavailable");
            return;
        }
        let pipe = Qwen3AsrPipeline::load(&dir).expect("load pipeline");
        let text = pipe
            .transcribe(
                &wav,
                TranscriptionRequest {
                    language: Some("en"),
                    context: None,
                },
            )
            .expect("transcription");
        eprintln!("pipeline transcription: {text}");
        assert!(!text.is_empty());
    }

    fn write_s16_wav(path: &Path, samples: &[f32]) {
        let mut data = Vec::with_capacity(44 + samples.len() * 2);
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&((36 + samples.len() * 2) as u32).to_le_bytes());
        data.extend_from_slice(b"WAVEfmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&16000u32.to_le_bytes());
        data.extend_from_slice(&32000u32.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&((samples.len() * 2) as u32).to_le_bytes());
        for s in samples {
            let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            data.extend_from_slice(&pcm.to_le_bytes());
        }
        std::fs::write(path, data).expect("write wav");
    }
}
