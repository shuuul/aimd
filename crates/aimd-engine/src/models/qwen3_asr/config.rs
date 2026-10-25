//! Typed Qwen3-ASR checkpoint configuration (WS-06, shared).
//!
//! Parsed from `config.json` of `mlx-community/Qwen3-ASR-1.7B-{4bit,8bit}`:
//! the `thinker_config` object holds `audio_config` + `text_config` plus the
//! audio special-token ids. Unknown keys are ignored so future checkpoint
//! revisions do not break loading.

use serde::Deserialize;

/// Audio-tower (encoder) configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct AudioTowerConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub conv_chunksize: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    #[serde(default)]
    pub n_window_infer: Option<usize>,
    pub output_dim: usize,
    #[serde(default = "default_activation")]
    pub activation_function: String,
}

fn default_activation() -> String {
    "gelu".to_string()
}

impl AudioTowerConfig {
    /// Effective inference window (`n_window_infer` overrides `n_window`).
    pub fn infer_window(&self) -> usize {
        self.n_window_infer.unwrap_or(self.n_window)
    }
}

/// Qwen3 text-decoder configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct TextDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

fn default_rms_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_true() -> bool {
    true
}

/// Audio special-token ids from `thinker_config`.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct AudioTokenIds {
    pub audio_start_token_id: u32,
    pub audio_token_id: u32,
    pub audio_end_token_id: u32,
}

/// Full thinker configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioTowerConfig,
    pub text_config: TextDecoderConfig,
    pub audio_start_token_id: u32,
    pub audio_token_id: u32,
    pub audio_end_token_id: u32,
}

impl ThinkerConfig {
    pub fn audio_token_ids(&self) -> AudioTokenIds {
        AudioTokenIds {
            audio_start_token_id: self.audio_start_token_id,
            audio_token_id: self.audio_token_id,
            audio_end_token_id: self.audio_end_token_id,
        }
    }
}

/// Top-level checkpoint config (`thinker_config` + quantization metadata).
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3AsrCheckpointConfig {
    pub thinker_config: ThinkerConfig,
    #[serde(default)]
    pub quantization: Option<QuantizationInfo>,
}

/// Affine quantization descriptor (consumed by `aimd-weights`).
#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationInfo {
    pub group_size: u32,
    pub bits: u32,
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_mode() -> String {
    "affine".to_string()
}

/// Parse a checkpoint `config.json` from raw bytes.
pub fn parse_config(json: &[u8]) -> Result<Qwen3AsrCheckpointConfig, serde_json::Error> {
    serde_json::from_slice(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached_config() -> Option<Vec<u8>> {
        let home = std::env::var("HOME").ok()?;
        let pattern = format!(
            "{home}/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-4bit/snapshots/*/config.json"
        );
        let path = sorted_glob(&pattern)?;
        std::fs::read(path).ok()
    }

    fn sorted_glob(pattern: &str) -> Option<std::path::PathBuf> {
        let mut candidates: Vec<_> = glob_files(pattern);
        candidates.sort();
        candidates.into_iter().next()
    }

    fn glob_files(pattern: &str) -> Vec<std::path::PathBuf> {
        // Minimal glob for the single `*` component; avoids a dependency.
        let (dir, rest) = pattern.split_once("/*").unwrap();
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let candidate = entry.path().join(rest);
                if candidate.is_file() {
                    out.push(candidate);
                }
            }
        }
        out
    }

    #[test]
    fn config_parses_cached_checkpoint() {
        let Some(raw) = cached_config() else {
            eprintln!("skipping: Qwen3-ASR-1.7B-4bit config not in HF cache");
            return;
        };
        let cfg = parse_config(&raw).expect("parse checkpoint config");
        let t = &cfg.thinker_config;
        assert_eq!(t.audio_config.encoder_layers, 24);
        assert_eq!(t.audio_config.d_model, 1024);
        assert_eq!(t.audio_config.num_mel_bins, 128);
        assert_eq!(t.audio_config.infer_window(), 800);
        assert_eq!(t.text_config.num_hidden_layers, 28);
        assert_eq!(t.text_config.num_attention_heads, 16);
        assert_eq!(t.text_config.num_key_value_heads, 8);
        assert_eq!(t.text_config.head_dim, 128);
        assert!((t.text_config.rope_theta - 1e6).abs() < 1e-9);
        assert!(t.text_config.tie_word_embeddings);
        let q = cfg.quantization.expect("quantization present");
        assert_eq!((q.group_size, q.bits), (64, 4));
    }
}
