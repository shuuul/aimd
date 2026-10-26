//! Typed PP-OCRv6 checkpoint configs, parsed from the HF snapshot
//! `config.json` (+ the rec `preprocessor_config.json` character list) the
//! way mlx4ocr's `det_config_from_artifacts` / `rec_config_from_artifacts`
//! read them. Only the medium layout (branch stem, RepLKPAN + IntraCL neck,
//! LightSVTR rec encoder) is supported.

use std::path::Path;

use serde_json::Value;

/// Config parse failures.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("config: {0}")]
    Invalid(String),
}

/// One LCNetV4 block: `[kernel, in, out, stride, use_se]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockSpec {
    pub kernel: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// `(stride_h, stride_w)`.
    pub stride: [usize; 2],
    /// The config gave an integer stride (Python `stride == 1` is false
    /// for a `(1, 1)` tuple, which disables the residual path).
    pub scalar_stride: bool,
    pub use_se: bool,
}

impl BlockSpec {
    /// `in == out and stride == 1`: re-parameterised depthwise conv with a
    /// residual connection.
    pub fn rep_dw(&self) -> bool {
        self.in_channels == self.out_channels && self.scalar_stride && self.stride == [1, 1]
    }
}

/// Detection network config.
#[derive(Clone, Debug, PartialEq)]
pub struct DetConfig {
    pub stem_mid: usize,
    pub stem_out: usize,
    pub stages: Vec<Vec<BlockSpec>>,
    pub neck_out_channels: usize,
    pub head_kernel_list: [usize; 3],
}

/// Recognition network config.
#[derive(Clone, Debug, PartialEq)]
pub struct RecConfig {
    pub stem_mid: usize,
    pub stem_out: usize,
    pub stages: Vec<Vec<BlockSpec>>,
    pub hidden_size: usize,
    pub depth: usize,
    pub mlp_ratio: f64,
    pub num_classes: usize,
    pub local_kernel: usize,
    /// CTC table, index 0 = blank.
    pub characters: Vec<String>,
}

impl RecConfig {
    /// Backbone output channels (last block of the last stage).
    pub fn backbone_channels(&self) -> usize {
        self.stages
            .last()
            .and_then(|s| s.last())
            .map_or(0, |b| b.out_channels)
    }
}

fn read_json(path: &Path) -> Result<Value, ConfigError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError::Io(format!("{}: {e}", path.display())))?;
    Ok(serde_json::from_str(&text)?)
}

fn as_usize(v: &Value, what: &str) -> Result<usize, ConfigError> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| ConfigError::Invalid(format!("{what}: expected an integer, got {v}")))
}

fn parse_stages(raw: &Value, det: bool) -> Result<Vec<Vec<BlockSpec>>, ConfigError> {
    let stages = raw
        .as_array()
        .ok_or_else(|| ConfigError::Invalid("block_configs must be a list".into()))?;
    if stages.len() != 4 {
        return Err(ConfigError::Invalid(format!(
            "expected 4 backbone stages, got {}",
            stages.len()
        )));
    }
    let mut out = Vec::with_capacity(4);
    for stage in stages {
        let blocks = stage
            .as_array()
            .ok_or_else(|| ConfigError::Invalid("block stage must be a list".into()))?;
        let mut specs = Vec::with_capacity(blocks.len());
        for block in blocks {
            let b = block
                .as_array()
                .filter(|b| b.len() == 5)
                .ok_or_else(|| ConfigError::Invalid(format!("invalid block spec {block}")))?;
            let (stride, scalar_stride) = match &b[3] {
                Value::Array(s) if s.len() == 2 => {
                    let (h, w) = (as_usize(&s[0], "stride")?, as_usize(&s[1], "stride")?);
                    // the detection block uses `stride[0]` for both axes
                    if det { ([h, h], true) } else { ([h, w], false) }
                }
                v => {
                    let s = as_usize(v, "stride")?;
                    ([s, s], true)
                }
            };
            specs.push(BlockSpec {
                kernel: as_usize(&b[0], "kernel")?,
                in_channels: as_usize(&b[1], "in_channels")?,
                out_channels: as_usize(&b[2], "out_channels")?,
                stride,
                scalar_stride,
                use_se: b[4]
                    .as_bool()
                    .ok_or_else(|| ConfigError::Invalid("use_se must be a bool".into()))?,
            });
        }
        if specs.is_empty() {
            return Err(ConfigError::Invalid("empty backbone stage".into()));
        }
        out.push(specs);
    }
    Ok(out)
}

fn stem(backbone: &Value) -> Result<(usize, usize), ConfigError> {
    let stem_type = backbone
        .get("stem_type")
        .and_then(Value::as_str)
        .unwrap_or("small");
    if !matches!(stem_type, "large" | "branch") {
        return Err(ConfigError::Invalid(format!(
            "unsupported stem_type {stem_type:?} (only the medium branch stem is ported)"
        )));
    }
    let ch = backbone
        .get("stem_channels")
        .and_then(Value::as_array)
        .filter(|c| c.len() == 3)
        .ok_or_else(|| ConfigError::Invalid("stem_channels must hold 3 integers".into()))?;
    Ok((as_usize(&ch[1], "stem_mid")?, as_usize(&ch[2], "stem_out")?))
}

/// Parses the detection `config.json` in `dir`.
pub fn load_det_config(dir: &Path) -> Result<DetConfig, ConfigError> {
    let config = read_json(&dir.join("config.json"))?;
    let neck = config.get("neck_config").unwrap_or(&config);
    let backbone = neck
        .get("backbone_config")
        .ok_or_else(|| ConfigError::Invalid("missing backbone_config".into()))?;
    let (stem_mid, stem_out) = stem(backbone)?;
    let stages = parse_stages(
        backbone
            .get("block_configs")
            .ok_or_else(|| ConfigError::Invalid("missing block_configs".into()))?,
        true,
    )?;
    let intracl = neck
        .get("intraclass_block_number")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if intracl == 0 {
        return Err(ConfigError::Invalid(
            "only the RepLKPAN + IntraCL (medium) detection neck is ported".into(),
        ));
    }
    let neck_out_channels = as_usize(
        neck.get("neck_out_channels")
            .ok_or_else(|| ConfigError::Invalid("missing neck_out_channels".into()))?,
        "neck_out_channels",
    )?;
    let kl = match neck.get("kernel_list") {
        Some(Value::Array(k)) if k.len() == 3 => [
            as_usize(&k[0], "kernel_list")?,
            as_usize(&k[1], "kernel_list")?,
            as_usize(&k[2], "kernel_list")?,
        ],
        None => [3, 2, 2],
        Some(v) => return Err(ConfigError::Invalid(format!("invalid kernel_list {v}"))),
    };
    Ok(DetConfig {
        stem_mid,
        stem_out,
        stages,
        neck_out_channels,
        head_kernel_list: kl,
    })
}

/// Parses the recognition `config.json` and `preprocessor_config.json`
/// (`character_list`) in `dir`.
pub fn load_rec_config(dir: &Path) -> Result<RecConfig, ConfigError> {
    let config = read_json(&dir.join("config.json"))?;
    let backbone = config
        .get("backbone_config")
        .ok_or_else(|| ConfigError::Invalid("missing backbone_config".into()))?;
    let (stem_mid, stem_out) = stem(backbone)?;
    let stages = parse_stages(
        backbone
            .get("block_configs")
            .ok_or_else(|| ConfigError::Invalid("missing block_configs".into()))?,
        false,
    )?;
    let get = |k: &str| {
        config
            .get(k)
            .ok_or_else(|| ConfigError::Invalid(format!("missing {k}")))
    };
    let hidden_size = as_usize(get("hidden_size")?, "hidden_size")?;
    let depth = config
        .get("depth")
        .map_or(Ok(2), |v| as_usize(v, "depth"))?;
    let mlp_ratio = config
        .get("mlp_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(4.0);
    let num_classes = as_usize(get("head_out_channels")?, "head_out_channels")?;
    let local_kernel = match config.get("conv_kernel_size") {
        Some(Value::Array(k)) if k.len() == 2 => as_usize(&k[1], "conv_kernel_size")?,
        None => 7,
        Some(v) => {
            return Err(ConfigError::Invalid(format!(
                "invalid conv_kernel_size {v}"
            )));
        }
    };
    let pre = read_json(&dir.join("preprocessor_config.json"))?;
    let characters: Vec<String> = pre
        .get("character_list")
        .and_then(Value::as_array)
        .ok_or_else(|| ConfigError::Invalid("missing character_list".into()))?
        .iter()
        .map(|c| {
            c.as_str().map(str::to_string).ok_or_else(|| {
                ConfigError::Invalid("character_list entries must be strings".into())
            })
        })
        .collect::<Result<_, _>>()?;
    if characters.len() != num_classes {
        return Err(ConfigError::Invalid(format!(
            "character_list has {} entries, head_out_channels is {num_classes}",
            characters.len()
        )));
    }
    Ok(RecConfig {
        stem_mid,
        stem_out,
        stages,
        hidden_size,
        depth,
        mlp_ratio,
        num_classes,
        local_kernel,
        characters,
    })
}
