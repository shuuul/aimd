//! Typed Qwen3-ForcedAligner checkpoint configuration.
//!
//! `config.json` of `mlx-community/Qwen3-ForcedAligner-0.6B-*`: the
//! `thinker_config` carries the Qwen3-ASR `audio_config` / `text_config`
//! (parsed with the shared [`crate::models::qwen3_asr::config`] types) plus
//! `classify_num`; the timestamp token and bin width sit at the top level.

use serde::Deserialize;

use crate::models::qwen3_asr::config::{AudioTowerConfig, QuantizationInfo, TextDecoderConfig};

/// `thinker_config` of the aligner checkpoint.
#[derive(Clone, Debug, Deserialize)]
pub struct AlignerThinkerConfig {
    pub audio_config: AudioTowerConfig,
    pub text_config: TextDecoderConfig,
    pub audio_start_token_id: u32,
    pub audio_token_id: u32,
    pub audio_end_token_id: u32,
    /// Number of time classes of the `lm_head` (5000).
    #[serde(default = "default_classify_num")]
    pub classify_num: usize,
}

fn default_classify_num() -> usize {
    5000
}

fn default_timestamp_token_id() -> u32 {
    151705
}

fn default_segment_time() -> f64 {
    80.0
}

/// Top-level aligner checkpoint config.
#[derive(Clone, Debug, Deserialize)]
pub struct AlignerCheckpointConfig {
    pub thinker_config: AlignerThinkerConfig,
    /// `<timestamp>` token id (151705).
    #[serde(default = "default_timestamp_token_id")]
    pub timestamp_token_id: u32,
    /// Milliseconds per time class (80).
    #[serde(default = "default_segment_time")]
    pub timestamp_segment_time: f64,
    #[serde(default)]
    pub support_languages: Vec<String>,
    #[serde(default)]
    pub quantization: Option<QuantizationInfo>,
}

/// Parse an aligner checkpoint `config.json` from raw bytes.
pub fn parse_config(json: &[u8]) -> Result<AlignerCheckpointConfig, serde_json::Error> {
    serde_json::from_slice(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligner_config_parses_checkpoint_shape() {
        let raw = br#"{
            "support_languages": ["Chinese", "English"],
            "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
            "thinker_config": {
                "classify_num": 5000,
                "audio_start_token_id": 151669,
                "audio_token_id": 151676,
                "audio_end_token_id": 151670,
                "audio_config": {
                    "d_model": 1024, "encoder_layers": 24, "encoder_attention_heads": 16,
                    "encoder_ffn_dim": 4096, "num_mel_bins": 128, "conv_chunksize": 500,
                    "downsample_hidden_size": 480, "n_window": 50, "n_window_infer": 800,
                    "output_dim": 1024, "activation_function": "gelu"
                },
                "text_config": {
                    "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
                    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
                    "max_position_embeddings": 8192, "vocab_size": 152064,
                    "rms_norm_eps": 1e-6, "rope_theta": 1000000, "tie_word_embeddings": false
                }
            },
            "timestamp_segment_time": 80,
            "timestamp_token_id": 151705
        }"#;
        let cfg = parse_config(raw).unwrap();
        assert_eq!(cfg.thinker_config.classify_num, 5000);
        assert_eq!(cfg.timestamp_token_id, 151705);
        assert_eq!(cfg.timestamp_segment_time, 80.0);
        assert!(!cfg.thinker_config.text_config.tie_word_embeddings);
        assert_eq!(cfg.thinker_config.audio_config.output_dim, 1024);
        assert_eq!(cfg.quantization.unwrap().bits, 8);
    }
}
