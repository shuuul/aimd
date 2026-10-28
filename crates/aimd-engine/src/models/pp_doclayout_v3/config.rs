//! PP-DocLayoutV3 architecture constants, checked against the snapshot's
//! `config.json`.
//!
//! The port supports exactly the published `PaddlePaddle/PP-DocLayoutV3`
//! geometry (HGNetV2-L backbone, 3-level hybrid encoder, 6-layer deformable
//! decoder, 300 queries, 25 PaddleX labels); [`check_config`] rejects any
//! other checkpoint instead of silently mis-wiring it.

use std::path::Path;

use serde_json::Value;

/// Model input side (the SDK's fixed `cv2.resize((800, 800))`).
pub const INPUT_SIZE: usize = 800;
/// Hidden width of the encoder and decoder.
pub const D_MODEL: usize = 256;
/// Decoder queries (`num_queries`).
pub const NUM_QUERIES: usize = 300;
/// Detection classes (`PX_LABELS`).
pub const NUM_LABELS: usize = 25;
/// Attention heads in the AIFI layer and the decoder.
pub const NUM_HEADS: usize = 8;
/// Decoder layers.
pub const DECODER_LAYERS: usize = 6;
/// Deformable-attention feature levels and sampling points per level.
pub const NUM_LEVELS: usize = 3;
pub const NUM_POINTS: usize = 4;
/// FFN width of the AIFI and decoder layers.
pub const FFN_DIM: usize = 1024;
/// Mask prototypes (`num_prototypes`).
pub const NUM_PROTOTYPES: usize = 32;
/// Global-pointer head size.
pub const POINTER_HEAD: usize = 64;
/// LayerNorm and BatchNorm epsilon.
pub const EPS: f32 = 1e-5;

/// HGNetV2-L stage geometry.
pub struct StageSpec {
    pub in_channels: usize,
    pub mid_channels: usize,
    pub out_channels: usize,
    pub blocks: usize,
    pub layers: usize,
    pub downsample: bool,
    pub light: bool,
    pub kernel: usize,
}

/// HGNetV2 `arch="L"` (`stage_config` of transformers' `HGNetV2Config`).
pub const STAGES: [StageSpec; 4] = [
    StageSpec {
        in_channels: 48,
        mid_channels: 48,
        out_channels: 128,
        blocks: 1,
        layers: 6,
        downsample: false,
        light: false,
        kernel: 3,
    },
    StageSpec {
        in_channels: 128,
        mid_channels: 96,
        out_channels: 512,
        blocks: 1,
        layers: 6,
        downsample: true,
        light: false,
        kernel: 3,
    },
    StageSpec {
        in_channels: 512,
        mid_channels: 192,
        out_channels: 1024,
        blocks: 3,
        layers: 6,
        downsample: true,
        light: true,
        kernel: 5,
    },
    StageSpec {
        in_channels: 1024,
        mid_channels: 384,
        out_channels: 2048,
        blocks: 1,
        layers: 6,
        downsample: true,
        light: true,
        kernel: 5,
    },
];

/// Config failures.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("read {0}: {1}")]
    Io(String, String),
    #[error("parse config.json: {0}")]
    Parse(String),
    #[error("unsupported PP-DocLayoutV3 config: {0}")]
    Unsupported(String),
}

/// Checks `dir/config.json` against the geometry this port implements.
pub fn check_config(dir: &Path) -> Result<(), ConfigError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| ConfigError::Io(path.display().to_string(), e.to_string()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))?;
    let want_str = [
        ("model_type", "pp_doclayout_v3"),
        ("activation_function", "silu"),
        ("encoder_activation_function", "gelu"),
        ("decoder_activation_function", "relu"),
    ];
    for (key, want) in want_str {
        if v.get(key).and_then(Value::as_str) != Some(want) {
            return Err(ConfigError::Unsupported(format!("{key} != {want}")));
        }
    }
    let want_num = [
        ("d_model", D_MODEL),
        ("encoder_hidden_dim", D_MODEL),
        ("num_queries", NUM_QUERIES),
        ("decoder_layers", DECODER_LAYERS),
        ("decoder_attention_heads", NUM_HEADS),
        ("encoder_attention_heads", NUM_HEADS),
        ("decoder_n_points", NUM_POINTS),
        ("num_feature_levels", NUM_LEVELS),
        ("encoder_layers", 1),
        ("encoder_ffn_dim", FFN_DIM),
        ("decoder_ffn_dim", FFN_DIM),
        ("global_pointer_head_size", POINTER_HEAD),
        ("x4_feat_dim", 128),
    ];
    for (key, want) in want_num {
        if v.get(key).and_then(Value::as_u64) != Some(want as u64) {
            return Err(ConfigError::Unsupported(format!("{key} != {want}")));
        }
    }
    let arch = v
        .pointer("/backbone_config/arch")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if arch != "L" {
        return Err(ConfigError::Unsupported(format!(
            "backbone arch {arch:?} != \"L\""
        )));
    }
    if v.get("normalize_before").and_then(Value::as_bool) == Some(true)
        || v.get("learn_initial_query").and_then(Value::as_bool) == Some(true)
        || v.get("mask_enhanced").and_then(Value::as_bool) == Some(false)
    {
        return Err(ConfigError::Unsupported(
            "normalize_before / learn_initial_query / mask_enhanced variant".into(),
        ));
    }
    if v.get("id2label")
        .and_then(Value::as_object)
        .is_some_and(|m| m.len() != NUM_LABELS)
    {
        return Err(ConfigError::Unsupported(format!(
            "label count != {NUM_LABELS}"
        )));
    }
    Ok(())
}
