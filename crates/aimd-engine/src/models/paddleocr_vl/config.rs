//! Typed PaddleOCR-VL-1.6 checkpoint configuration.
//!
//! Parsed from the official `PaddlePaddle/PaddleOCR-VL-1.6` snapshot:
//! `config.json` (ERNIE-4.5 text model at the top level plus
//! `vision_config`), `preprocessor_config.json` (image processor bounds)
//! and `generation_config.json` (EOS). Field names and defaults follow
//! `mlx_vlm/models/paddleocr_vl/config.py`; unknown keys are ignored.

use std::path::Path;

use serde::Deserialize;

/// ERNIE-4.5 text decoder (top-level keys of `config.json`).
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
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// `rope_scaling` (only `mrope_section` is used).
#[derive(Clone, Debug, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
}

/// SigLIP-style vision tower (`vision_config`).
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
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_merge")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_vision_act")]
    pub hidden_act: String,
}

fn default_channels() -> usize {
    3
}
fn default_ln_eps() -> f32 {
    1e-6
}
fn default_merge() -> usize {
    2
}
fn default_vision_act() -> String {
    "gelu_pytorch_tanh".to_string()
}

/// `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct PaddleOcrVlConfig {
    #[serde(flatten)]
    pub text: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub video_token_id: Option<u32>,
}

impl PaddleOcrVlConfig {
    /// `rope_scaling.mrope_section` (`[16, 24, 24]` for this checkpoint).
    pub fn mrope_section(&self) -> Vec<usize> {
        self.text
            .rope_scaling
            .as_ref()
            .and_then(|r| r.mrope_section.clone())
            .unwrap_or_else(|| vec![16, 24, 24])
    }
}

/// `preprocessor_config.json` (the image-processor fields this port uses).
#[derive(Clone, Debug, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "default_min_pixels")]
    pub min_pixels: usize,
    #[serde(default = "default_max_pixels")]
    pub max_pixels: usize,
    #[serde(default = "default_patch")]
    pub patch_size: usize,
    #[serde(default = "default_merge")]
    pub merge_size: usize,
    #[serde(default = "default_temporal")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_half3")]
    pub image_mean: Vec<f64>,
    #[serde(default = "default_half3")]
    pub image_std: Vec<f64>,
    #[serde(default = "default_resample")]
    pub resample: u32,
}

fn default_min_pixels() -> usize {
    112_896
}
fn default_max_pixels() -> usize {
    1_003_520
}
fn default_patch() -> usize {
    14
}
fn default_temporal() -> usize {
    1
}
fn default_half3() -> Vec<f64> {
    vec![0.5; 3]
}
fn default_resample() -> u32 {
    3
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            min_pixels: default_min_pixels(),
            max_pixels: default_max_pixels(),
            patch_size: default_patch(),
            merge_size: default_merge(),
            temporal_patch_size: default_temporal(),
            image_mean: default_half3(),
            image_std: default_half3(),
            resample: default_resample(),
        }
    }
}

/// `generation_config.json` `eos_token_id` (int or list).
#[derive(Clone, Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

/// Auxiliary config-file failures.
#[derive(Debug, thiserror::Error)]
pub enum ConfigFileError {
    #[error("io: {0}")]
    Io(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Parses `config.json` text.
pub fn parse_config(text: &str) -> Result<PaddleOcrVlConfig, serde_json::Error> {
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

/// EOS ids from `generation_config.json`; `None` when the file or key is
/// absent.
pub fn load_eos_ids(dir: &Path) -> Result<Option<Vec<u32>>, ConfigFileError> {
    let path = dir.join("generation_config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|e| ConfigFileError::Io(e.to_string()))?;
    let cfg: GenerationConfig = serde_json::from_str(&text)?;
    Ok(match cfg.eos_token_id {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| vec![v as u32]),
        Some(serde_json::Value::Array(a)) => Some(
            a.iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect(),
        ),
        _ => None,
    })
}
