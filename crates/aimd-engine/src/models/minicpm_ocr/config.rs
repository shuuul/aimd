//! Typed MiniCPM-OCR 1.0 checkpoint configuration.
//!
//! Parsed from the official `openbmb/MiniCPM-OCR-1.0` snapshot:
//! `config.json` (`vision_config`, the Qwen3.5 `text_config`, the image
//! token id and `downsample_mode`), `preprocessor_config.json` (slicing
//! bounds) and `generation_config.json` (EOS ids, shared with
//! PaddleOCR-VL's reader). Unknown keys are ignored.

use std::path::Path;

use serde::Deserialize;

pub use crate::models::paddleocr_vl::config::{ConfigFileError, load_eos_ids};

/// SigLIP NaViT vision tower (`vision_config`).
#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    #[serde(default = "default_channels")]
    pub num_channels: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_eps: f32,
}

/// `text_config.rope_parameters`.
#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
}

/// Qwen3.5 hybrid decoder (`text_config`).
#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub full_attention_interval: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl TextConfig {
    /// Whether layer `i` is a Gated DeltaNet (every `full_attention_interval`-th
    /// layer is full attention).
    pub fn is_linear(&self, i: usize) -> bool {
        !(i + 1).is_multiple_of(self.full_attention_interval)
    }

    /// Rotated width of the full-attention heads (64 of 256).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
}

/// `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct MiniCpmOcrConfig {
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    #[serde(default = "default_downsample")]
    pub downsample_mode: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// `preprocessor_config.json` (the slicing fields this port uses).
#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "default_scale")]
    pub scale_resolution: usize,
    #[serde(default = "default_slices")]
    pub max_slice_nums: usize,
    #[serde(default = "default_patch")]
    pub patch_size: usize,
    #[serde(default = "default_true")]
    pub slice_mode: bool,
    #[serde(default = "default_norm")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_norm")]
    pub image_std: Vec<f32>,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            scale_resolution: default_scale(),
            max_slice_nums: default_slices(),
            patch_size: default_patch(),
            slice_mode: true,
            image_mean: default_norm(),
            image_std: default_norm(),
        }
    }
}

fn default_channels() -> usize {
    3
}
fn default_eps() -> f32 {
    1e-6
}
fn default_partial_rotary() -> f64 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_downsample() -> String {
    "4x".to_string()
}
fn default_scale() -> usize {
    448
}
fn default_slices() -> usize {
    9
}
fn default_patch() -> usize {
    14
}
fn default_norm() -> Vec<f32> {
    vec![0.5; 3]
}

/// Parses `config.json` text.
pub fn parse_config(text: &str) -> Result<MiniCpmOcrConfig, serde_json::Error> {
    serde_json::from_str(text)
}

/// `preprocessor_config.json` from `dir`, or the checkpoint defaults when
/// the file is absent.
pub fn load_preprocessor_config(dir: &Path) -> Result<PreprocessorConfig, ConfigFileError> {
    let path = dir.join("preprocessor_config.json");
    if !path.is_file() {
        return Ok(PreprocessorConfig::default());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| ConfigFileError::Io(e.to_string()))?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
      "downsample_mode": "4x", "image_token_id": 248056, "tie_word_embeddings": true,
      "vision_config": {"hidden_size": 1152, "image_size": 980, "intermediate_size": 4304,
        "layer_norm_eps": 1e-06, "num_attention_heads": 16, "num_hidden_layers": 27,
        "patch_size": 14},
      "text_config": {"hidden_size": 1024, "intermediate_size": 3584, "num_hidden_layers": 24,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
        "rms_norm_eps": 1e-06, "vocab_size": 248100, "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "linear_value_head_dim": 128,
        "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
        "tie_word_embeddings": true}
    }"#;

    #[test]
    fn minicpm_ocr_config_parses_layer_pattern() {
        let cfg = parse_config(CONFIG).unwrap();
        let t = &cfg.text_config;
        let full: Vec<usize> = (0..t.num_hidden_layers)
            .filter(|&i| !t.is_linear(i))
            .collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23]);
        assert_eq!(t.rotary_dim(), 64);
        assert!(t.attn_output_gate && t.tie_word_embeddings);
        assert_eq!(
            cfg.vision_config.image_size / cfg.vision_config.patch_size,
            70
        );
        assert_eq!(cfg.downsample_mode, "4x");
    }
}
