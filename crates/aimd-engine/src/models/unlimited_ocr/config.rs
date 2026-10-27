//! Typed Unlimited-OCR checkpoint configuration (WS-07, shared).
//!
//! Parsed from `config.json` of `mlx-community/Unlimited-OCR-4bit`. The
//! language config mirrors `mlx_vlm/models/unlimited_ocr/config.py`
//! `TextConfig` defaults (rms_norm_eps 1e-6, rope_theta 10000.0) for keys
//! the checkpoint omits. Unknown keys are ignored.

use serde::Deserialize;

/// Per-variant geometry spec under `vision_config.width` — fields are
/// optional because the `clip-l-14-224` sibling variant lacks the SAM
/// keys; `deeplip_b_l` uses the `sam_vit_b` entry.
#[derive(Clone, Debug, Deserialize)]
pub struct VisionWidthSpec {
    #[serde(default)]
    pub downsample_channels: Option<Vec<usize>>,
    #[serde(default)]
    pub global_attn_indexes: Option<Vec<usize>>,
    #[serde(default)]
    pub heads: Option<usize>,
    #[serde(default)]
    pub layers: Option<usize>,
    #[serde(default)]
    pub width: Option<usize>,
}

/// SAM ViT-B vision tower geometry, resolved from the `sam_vit_b`
/// variant with SAM defaults for omitted keys.
#[derive(Clone, Debug)]
pub struct SamVitBWidth {
    pub downsample_channels: Vec<usize>,
    pub global_attn_indexes: Vec<usize>,
    pub heads: usize,
    pub layers: usize,
    pub width: usize,
}

impl Default for SamVitBWidth {
    fn default() -> Self {
        Self {
            downsample_channels: vec![512, 1024],
            global_attn_indexes: vec![2, 5, 8, 11],
            heads: 12,
            layers: 12,
            width: 768,
        }
    }
}

/// Vision tower config (top-level `vision_config`).
#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub image_size: usize,
    #[serde(default = "default_vision_model_name")]
    pub model_name: String,
    pub width: std::collections::HashMap<String, VisionWidthSpec>,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
}

impl VisionConfig {
    /// Resolved SAM ViT-B geometry for the `deeplip_b_l` tower.
    pub fn sam_vit_b(&self) -> SamVitBWidth {
        let spec = self.width.get("sam_vit_b");
        let d = SamVitBWidth::default();
        SamVitBWidth {
            downsample_channels: spec
                .and_then(|s| s.downsample_channels.clone())
                .unwrap_or(d.downsample_channels),
            global_attn_indexes: spec
                .and_then(|s| s.global_attn_indexes.clone())
                .unwrap_or(d.global_attn_indexes),
            heads: spec.and_then(|s| s.heads).unwrap_or(d.heads),
            layers: spec.and_then(|s| s.layers).unwrap_or(d.layers),
            width: spec.and_then(|s| s.width).unwrap_or(d.width),
        }
    }
}

fn default_vision_model_name() -> String {
    "deeplip_b_l".to_string()
}

fn default_mlp_ratio() -> f64 {
    3.7362
}

/// Projector config (`projector_config`): a single linear 2048 -> 1280.
#[derive(Clone, Debug, Deserialize)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub n_embed: usize,
    #[serde(default = "default_projector_type")]
    pub projector_type: String,
}

fn default_projector_type() -> String {
    "linear".to_string()
}

/// DeepSeek-V2-lite language decoder config.
#[derive(Clone, Debug, Deserialize)]
pub struct LanguageConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub v_head_dim: usize,
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    pub sliding_window: Option<usize>,
    pub sliding_window_size: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default = "default_n_group")]
    pub n_group: usize,
    #[serde(default = "default_topk_group")]
    pub topk_group: usize,
}

fn default_rms_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_topk_method() -> String {
    "greedy".to_string()
}

fn default_n_group() -> usize {
    1
}

fn default_topk_group() -> usize {
    1
}

impl LanguageConfig {
    /// Effective sliding window (`sliding_window_size` overrides
    /// `sliding_window`, matching the mlx-vlm reference).
    pub fn window(&self) -> Option<usize> {
        self.sliding_window_size.or(self.sliding_window)
    }
}

/// Top-level Unlimited-OCR checkpoint config.
#[derive(Clone, Debug, Deserialize)]
pub struct UnlimitedOcrConfig {
    #[serde(default)]
    pub tile_tag: String,
    #[serde(default = "default_global_view_pos")]
    pub global_view_pos: String,
    #[serde(default)]
    pub candidate_resolutions: Vec<(usize, usize)>,
    pub vision_config: VisionConfig,
    pub projector_config: ProjectorConfig,
    pub language_config: LanguageConfig,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default)]
    pub quantization: Option<QuantizationInfo>,
    /// Unlimited-OCR-2 DASH sparse attention (absent on v1 checkpoints).
    #[serde(default)]
    pub sparse_attention_config: Option<super::dash::SparseAttentionConfig>,
}

impl UnlimitedOcrConfig {
    /// Whether the decoder uses DASH attention (Unlimited-OCR-2).
    pub fn is_dash(&self) -> bool {
        self.sparse_attention_config
            .as_ref()
            .is_some_and(super::dash::SparseAttentionConfig::is_dash)
    }
}

/// Affine quantization descriptor (`quantization` block of `config.json`).
#[derive(Clone, Debug, Deserialize)]
pub struct QuantizationInfo {
    pub group_size: u32,
    pub bits: u32,
    #[serde(default = "default_quant_mode")]
    pub mode: String,
}

fn default_quant_mode() -> String {
    "affine".to_string()
}

fn default_global_view_pos() -> String {
    "head".to_string()
}

/// Parse a checkpoint `config.json` from raw bytes.
pub fn parse_config(json: &[u8]) -> Result<UnlimitedOcrConfig, serde_json::Error> {
    serde_json::from_slice(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached_config() -> Option<Vec<u8>> {
        let home = std::env::var("HOME").ok()?;
        let base = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join("models--mlx-community--Unlimited-OCR-4bit")
            .join("snapshots");
        let entry = std::fs::read_dir(base)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join("config.json").is_file())?;
        std::fs::read(entry.join("config.json")).ok()
    }

    #[test]
    fn ocr_config_parses_cached_checkpoint() {
        let Some(raw) = cached_config() else {
            eprintln!("skipping: Unlimited-OCR-4bit config not in HF cache");
            return;
        };
        let cfg = parse_config(&raw).expect("parse OCR config");
        let sam = cfg.vision_config.sam_vit_b();
        assert_eq!(cfg.vision_config.image_size, 1024);
        assert_eq!(sam.layers, 12);
        assert_eq!(sam.global_attn_indexes, vec![2, 5, 8, 11]);
        assert_eq!(sam.downsample_channels, vec![512, 1024]);
        assert_eq!(cfg.projector_config.input_dim, 2048);
        assert_eq!(cfg.projector_config.n_embed, 1280);
        let lc = &cfg.language_config;
        assert_eq!(lc.num_hidden_layers, 12);
        assert_eq!((lc.num_attention_heads, lc.num_key_value_heads), (10, 10));
        assert_eq!(lc.v_head_dim, 128);
        assert_eq!((lc.n_routed_experts, lc.n_shared_experts), (64, 2));
        assert_eq!(lc.num_experts_per_tok, 6);
        assert_eq!(lc.first_k_dense_replace, 1);
        assert_eq!(lc.window(), Some(128));
        assert_eq!(lc.vocab_size, 129280);
        let q = cfg.quantization.expect("quantization present");
        assert_eq!((q.group_size, q.bits), (64, 4));
    }
}
