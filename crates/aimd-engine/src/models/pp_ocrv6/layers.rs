//! Checkpoint import and the shared PP-OCRv6 building blocks (conv + fused
//! batch norm, LCNetV4 blocks, the branch stem), op-for-op with mlx4ocr's
//! `models/common` and `models/det/backbone.py`.
//!
//! Paddle conv weights (`OIHW`) are transposed to MLX `OHWI` on the host
//! (pure data movement); batch norms are folded into the preceding conv at
//! load time with the same MLX ops as mlx4ocr's `fuse_conv_batch_norm`
//! (`scale = gamma * rsqrt(var + 1e-5)`, `W * scale`, `beta + scale * (b -
//! mean)`), so the fused parameters are bit-identical to the reference.

use aimd_mlx::{MlxArray, MlxDtype, MlxError, UnaryOp};
use aimd_weights::{WeightStore, WeightsError};

use super::config::BlockSpec;

/// Weight-import failures.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Mlx(#[from] MlxError),
    #[error(transparent)]
    Weights(#[from] WeightsError),
    #[error("checkpoint layout: {0}")]
    Layout(String),
}

/// Host copy of an f32 checkpoint tensor plus its shape.
pub(crate) fn host_f32(
    store: &WeightStore,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), LoadError> {
    let info = store
        .tensor_info(name)
        .ok_or_else(|| LoadError::Layout(format!("missing checkpoint tensor {name}")))?;
    if format!("{:?}", info.dtype) != "F32" {
        return Err(LoadError::Layout(format!(
            "{name}: expected F32, got {:?}",
            info.dtype
        )));
    }
    let shape = info.shape.clone();
    let bytes = store.tensor_bytes(name)?;
    let data = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok((data, shape))
}

/// Uploads an f32 tensor verbatim, checking its shape.
pub(crate) fn tensor(
    store: &WeightStore,
    name: &str,
    expected: &[usize],
) -> Result<MlxArray, LoadError> {
    let (data, shape) = host_f32(store, name)?;
    if shape != expected {
        return Err(LoadError::Layout(format!(
            "{name}: shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(MlxArray::from_f32(&shape, &data)?)
}

/// A conv weight `[O, I, KH, KW]` (Paddle) -> `[O, KH, KW, I]` (MLX).
fn conv_weight(store: &WeightStore, name: &str) -> Result<(MlxArray, [usize; 4]), LoadError> {
    let (data, shape) = host_f32(store, name)?;
    let [o, i, kh, kw] = <[usize; 4]>::try_from(shape.as_slice()).map_err(|_| {
        LoadError::Layout(format!(
            "{name}: rank-4 conv weight expected, got {shape:?}"
        ))
    })?;
    let mut out = vec![0f32; data.len()];
    for a in 0..o {
        for b in 0..i {
            for y in 0..kh {
                for x in 0..kw {
                    out[((a * kh + y) * kw + x) * i + b] = data[((a * i + b) * kh + y) * kw + x];
                }
            }
        }
    }
    Ok((MlxArray::from_f32(&[o, kh, kw, i], &out)?, [o, kh, kw, i]))
}

/// A transposed-conv weight `[I, O, KH, KW]` (Paddle) -> `[O, KH, KW, I]`.
fn conv_transpose_weight(
    store: &WeightStore,
    name: &str,
) -> Result<(MlxArray, [usize; 4]), LoadError> {
    let (data, shape) = host_f32(store, name)?;
    let [i, o, kh, kw] = <[usize; 4]>::try_from(shape.as_slice())
        .map_err(|_| LoadError::Layout(format!("{name}: rank-4 weight expected, got {shape:?}")))?;
    let mut out = vec![0f32; data.len()];
    for a in 0..i {
        for b in 0..o {
            for y in 0..kh {
                for x in 0..kw {
                    out[((b * kh + y) * kw + x) * i + a] = data[((a * o + b) * kh + y) * kw + x];
                }
            }
        }
    }
    Ok((MlxArray::from_f32(&[o, kh, kw, i], &out)?, [o, kh, kw, i]))
}

/// Folds the batch norm at `bn` (`.weight/.bias/.running_mean/.running_var`)
/// into `(weight, bias)` exactly like mlx4ocr `fuse_conv_batch_norm`.
fn fuse_bn(
    store: &WeightStore,
    weight: MlxArray,
    bias: Option<MlxArray>,
    bn: &str,
    out_channels: usize,
) -> Result<(MlxArray, MlxArray), LoadError> {
    let c = [out_channels];
    let gamma = tensor(store, &format!("{bn}.weight"), &c)?;
    let beta = tensor(store, &format!("{bn}.bias"), &c)?;
    let mean = tensor(store, &format!("{bn}.running_mean"), &c)?;
    let var = tensor(store, &format!("{bn}.running_var"), &c)?;
    let eps = MlxArray::scalar_f32(1e-5)?;
    let scale = gamma.multiply(&var.add(&eps)?.unary(UnaryOp::Rsqrt)?)?;
    let fused_w = weight.multiply(&scale.reshape(&[out_channels, 1, 1, 1])?)?;
    let conv_bias = match bias {
        Some(b) => b,
        None => MlxArray::zeros(&c, MlxDtype::F32)?,
    };
    let fused_b = beta.add(&scale.multiply(&conv_bias.subtract(&mean)?)?)?;
    MlxArray::eval_many(&[&fused_w, &fused_b])?;
    Ok((fused_w, fused_b))
}

/// `nn.Conv2d` / `nn.ConvTranspose2d` (optionally with a folded batch norm)
/// over NHWC: `conv(x, W) + b`.
pub(crate) struct Conv {
    w: MlxArray,
    b: Option<MlxArray>,
    stride: [usize; 2],
    padding: [usize; 2],
    groups: usize,
    transpose: bool,
}

/// How a conv's parameters are stored in the checkpoint.
pub(crate) enum ConvKind<'a> {
    /// `{prefix}.weight` (+ `{prefix}.bias` when `bias`).
    Plain { bias: bool },
    /// `{prefix}.{conv}.weight` (+ bias) followed by the batch norm at
    /// `{prefix}.{norm}`.
    Bn {
        conv: &'a str,
        norm: &'a str,
        bias: bool,
    },
}

impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn load_impl(
        store: &WeightStore,
        prefix: &str,
        kind: ConvKind<'_>,
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        transpose: bool,
    ) -> Result<Self, LoadError> {
        let (wname, bname) = match &kind {
            ConvKind::Plain { .. } => (format!("{prefix}.weight"), format!("{prefix}.bias")),
            ConvKind::Bn { conv, .. } => (
                format!("{prefix}.{conv}.weight"),
                format!("{prefix}.{conv}.bias"),
            ),
        };
        let has_bias = match kind {
            ConvKind::Plain { bias } | ConvKind::Bn { bias, .. } => bias,
        };
        let (w, shape) = if transpose {
            conv_transpose_weight(store, &wname)?
        } else {
            conv_weight(store, &wname)?
        };
        let out_channels = shape[0];
        let b = if has_bias {
            Some(tensor(store, &bname, &[out_channels])?)
        } else {
            None
        };
        let (w, b) = match kind {
            ConvKind::Plain { .. } => (w, b),
            ConvKind::Bn { norm, .. } => {
                let (w, b) = fuse_bn(store, w, b, &format!("{prefix}.{norm}"), out_channels)?;
                (w, Some(b))
            }
        };
        Ok(Self {
            w,
            b,
            stride,
            padding,
            groups,
            transpose,
        })
    }

    /// A 2-D convolution.
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        kind: ConvKind<'_>,
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
    ) -> Result<Self, LoadError> {
        Self::load_impl(store, prefix, kind, stride, padding, groups, false)
    }

    /// A transposed 2-D convolution (padding 0, groups 1).
    pub(crate) fn load_transpose(
        store: &WeightStore,
        prefix: &str,
        kind: ConvKind<'_>,
        stride: [usize; 2],
    ) -> Result<Self, LoadError> {
        Self::load_impl(store, prefix, kind, stride, [0, 0], 1, true)
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let y = if self.transpose {
            x.conv_transpose2d(&self.w, self.stride, self.padding, self.groups)?
        } else {
            x.conv2d_ex(&self.w, self.stride, self.padding, [1, 1], self.groups)?
        };
        match &self.b {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}

/// `nn.Linear` with bias: `addmm(b, x, W.T)`.
pub(crate) struct Linear {
    w: MlxArray,
    b: MlxArray,
}

impl Linear {
    /// From explicit `[out, in]` weight rows and bias.
    pub(crate) fn from_parts(
        w: &[f32],
        b: &[f32],
        out: usize,
        inp: usize,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            w: MlxArray::from_f32(&[out, inp], w)?,
            b: MlxArray::from_f32(&[out], b)?,
        })
    }

    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        out: usize,
        inp: usize,
    ) -> Result<Self, LoadError> {
        Ok(Self {
            w: tensor(store, &format!("{prefix}.weight"), &[out, inp])?,
            b: tensor(store, &format!("{prefix}.bias"), &[out])?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        self.b.addmm(x, &self.w.transpose(&[1, 0])?, 1.0, 1.0)
    }
}

fn scalar(v: f32) -> Result<MlxArray, MlxError> {
    MlxArray::scalar_f32(v)
}

/// `HardSigmoid`: `clip(maximum(x + 3, 0), 0, 6) / 6`.
pub(crate) fn hard_sigmoid(x: &MlxArray) -> Result<MlxArray, MlxError> {
    x.add(&scalar(3.0)?)?
        .maximum(&scalar(0.0)?)?
        .clip(0.0, 6.0)?
        .divide(&scalar(6.0)?)
}

/// `nn.ReLU` (`maximum(x, 0)`).
pub(crate) fn relu(x: &MlxArray) -> Result<MlxArray, MlxError> {
    x.maximum(&scalar(0.0)?)
}

/// LCNetV4 squeeze-excitation (`TokenSqueezeExcitation` / `SELayer`).
struct Se {
    conv1: Conv,
    conv2: Conv,
}

impl Se {
    fn load(store: &WeightStore, prefix: &str) -> Result<Self, LoadError> {
        let plain = || ConvKind::Plain { bias: true };
        Ok(Self {
            conv1: Conv::load(
                store,
                &format!("{prefix}.convolutions.0"),
                plain(),
                [1, 1],
                [0, 0],
                1,
            )?,
            conv2: Conv::load(
                store,
                &format!("{prefix}.convolutions.2"),
                plain(),
                [1, 1],
                [0, 0],
                1,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let pooled = x.mean_axes(&[1, 2], true)?;
        let gate = hard_sigmoid(&self.conv2.forward(&relu(&self.conv1.forward(&pooled)?)?)?)?;
        x.multiply(&gate)
    }
}

const BN: ConvKind<'static> = ConvKind::Bn {
    conv: "convolution",
    norm: "normalization",
    bias: false,
};

/// `LCNetV4Block` / `LCNetV4RecBlock`.
pub(crate) struct LcNetBlock {
    token_conv: Conv,
    se: Option<Se>,
    channel_conv1: Conv,
    channel_conv2: Conv,
    residual: bool,
}

impl LcNetBlock {
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        spec: &BlockSpec,
    ) -> Result<Self, LoadError> {
        let pad = (spec.kernel - 1) / 2;
        let c = spec.in_channels;
        let rep = spec.rep_dw();
        let token_conv = if rep {
            Conv::load(
                store,
                &format!("{prefix}.token_conv"),
                ConvKind::Plain { bias: true },
                [1, 1],
                [pad, pad],
                c,
            )?
        } else {
            Conv::load(
                store,
                &format!("{prefix}.token_conv"),
                BN,
                spec.stride,
                [pad, pad],
                c,
            )?
        };
        let se = if spec.use_se {
            Some(Se::load(
                store,
                &format!("{prefix}.token_squeeze_excitation"),
            )?)
        } else {
            None
        };
        Ok(Self {
            token_conv,
            se,
            channel_conv1: Conv::load(
                store,
                &format!("{prefix}.channel_conv1"),
                BN,
                [1, 1],
                [0, 0],
                1,
            )?,
            channel_conv2: Conv::load(
                store,
                &format!("{prefix}.channel_conv2"),
                BN,
                [1, 1],
                [0, 0],
                1,
            )?,
            residual: rep,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut x = self.token_conv.forward(x)?;
        if let Some(se) = &self.se {
            x = se.forward(&x)?;
        }
        let mixed = self
            .channel_conv2
            .forward(&self.channel_conv1.forward(&x)?.gelu()?)?;
        if self.residual {
            x.add(&mixed)
        } else {
            Ok(mixed)
        }
    }
}

/// Paddle `SAME` padding for a 2x2 stride-1 window: one zero row/column
/// after the data.
fn pad_same_2x2(x: &MlxArray) -> Result<MlxArray, MlxError> {
    x.pad(&[(0, 0), (0, 1), (0, 1), (0, 0)])
}

/// `StemBlock` (the "large"/branch stem).
pub(crate) struct Stem {
    stem1: Conv,
    stem2a: Conv,
    stem2b: Conv,
    stem3: Conv,
    stem4: Conv,
}

impl Stem {
    pub(crate) fn load(store: &WeightStore, prefix: &str) -> Result<Self, LoadError> {
        let p = |n: &str| format!("{prefix}.{n}");
        Ok(Self {
            stem1: Conv::load(store, &p("stem1"), BN, [2, 2], [1, 1], 1)?,
            stem2a: Conv::load(store, &p("stem2a"), BN, [1, 1], [0, 0], 1)?,
            stem2b: Conv::load(store, &p("stem2b"), BN, [1, 1], [0, 0], 1)?,
            stem3: Conv::load(store, &p("stem3"), BN, [2, 2], [1, 1], 1)?,
            stem4: Conv::load(store, &p("stem4"), BN, [1, 1], [0, 0], 1)?,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let x = relu(&self.stem1.forward(x)?)?;
        let a = relu(&self.stem2a.forward(&pad_same_2x2(&x)?)?)?;
        let branch = relu(&self.stem2b.forward(&pad_same_2x2(&a)?)?)?;
        // MaxPool2d(2, stride 1) over the SAME-padded input.
        let p = pad_same_2x2(&x)?;
        let s = p.shape()?;
        let (n, h, w, c) = (s[0], s[1] - 1, s[2] - 1, s[3]);
        let win =
            |dy: usize, dx: usize| p.slice(&[0, dy, dx, 0], &[n, dy + h, dx + w, c], &[1, 1, 1, 1]);
        let pooled = win(0, 0)?
            .maximum(&win(0, 1)?)?
            .maximum(&win(1, 0)?.maximum(&win(1, 1)?)?)?;
        let merged = MlxArray::concatenate(&[&pooled, &branch], 3)?;
        relu(&self.stem4.forward(&relu(&self.stem3.forward(&merged)?)?)?)
    }
}

/// The LCNetV4 backbone stages at `{prefix}.blocks.{s}.blocks.{b}`.
pub(crate) fn load_stages(
    store: &WeightStore,
    prefix: &str,
    stages: &[Vec<BlockSpec>],
) -> Result<Vec<Vec<LcNetBlock>>, LoadError> {
    stages
        .iter()
        .enumerate()
        .map(|(s, specs)| {
            specs
                .iter()
                .enumerate()
                .map(|(b, spec)| {
                    LcNetBlock::load(store, &format!("{prefix}.blocks.{s}.blocks.{b}"), spec)
                })
                .collect()
        })
        .collect()
}
