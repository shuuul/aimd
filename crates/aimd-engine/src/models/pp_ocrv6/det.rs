//! PP-OCRv6 medium DB text detector: PPLCNetV4 backbone -> RepLKPAN neck
//! with IntraCL blocks -> DB head (binarize branch), NHWC f32, op-for-op
//! with mlx4ocr `models/det/{backbone,neck,intracl,head}.py`.

use aimd_mlx::{Activation, MlxArray, MlxError};
use aimd_weights::WeightStore;

use super::config::DetConfig;
use super::layers::{Conv, ConvKind, LcNetBlock, LoadError, Stem, load_stages};

/// Nearest-neighbour upsampling by an integer factor (`mx.repeat` on the
/// H and W axes).
fn upsample(x: &MlxArray, s: usize) -> Result<MlxArray, MlxError> {
    if s == 1 {
        return x.try_clone();
    }
    let d = x.shape()?;
    let (n, h, w, c) = (d[0], d[1], d[2], d[3]);
    x.reshape(&[n, h, 1, w, 1, c])?
        .broadcast_to(&[n, h, s, w, s, c])?
        .reshape(&[n, h * s, w * s, c])
}

const PLAIN_BIAS: ConvKind<'static> = ConvKind::Plain { bias: true };

/// `IntraCLBlock` (reduce factor 2).
struct IntraCl {
    reduce: Conv,
    final_: Conv,
    /// `(symmetric, vertical, horizontal)` for the 7, 5, 3 stages.
    stages: [(Conv, Conv, Conv); 3],
}

impl IntraCl {
    fn load(store: &WeightStore, prefix: &str) -> Result<Self, LoadError> {
        let p = |n: &str| format!("{prefix}.{n}");
        let conv = |n: &str, pad: [usize; 2]| Conv::load(store, &p(n), PLAIN_BIAS, [1, 1], pad, 1);
        let stage = |ratio: &str, k: usize| -> Result<(Conv, Conv, Conv), LoadError> {
            let r = k / 2;
            Ok((
                conv(&format!("symmetric_conv_long_{ratio}"), [r, r])?,
                conv(&format!("vertical_long_to_small_conv_{ratio}"), [r, 0])?,
                conv(&format!("horizontal_small_to_long_conv_{ratio}"), [0, r])?,
            ))
        };
        Ok(Self {
            reduce: conv("conv_reduce_channel", [0, 0])?,
            final_: Conv::load(
                store,
                &p("conv_final"),
                ConvKind::Bn {
                    conv: "convolution",
                    norm: "norm",
                    bias: true,
                },
                [1, 1],
                [0, 0],
                1,
            )?,
            stages: [
                stage("longratio", 7)?,
                stage("midratio", 5)?,
                stage("shortratio", 3)?,
            ],
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut z = self.reduce.forward(x)?;
        for (sym, vert, horiz) in &self.stages {
            z = sym
                .forward(&z)?
                .add(&vert.forward(&z)?)?
                .add(&horiz.forward(&z)?)?;
        }
        self.final_.forward_act(&z, Activation::Relu, Some(x))
    }
}

/// `RepLKPAN` (medium neck).
struct RepLkPan {
    adjust: Vec<Conv>,
    project: Vec<Conv>,
    lateral: Vec<Conv>,
    intracl: Vec<IntraCl>,
    head: Vec<Conv>,
}

impl RepLkPan {
    fn load(store: &WeightStore) -> Result<Self, LoadError> {
        let p = |n: &str, i: usize| format!("model.neck.{n}.{i}");
        let mut s = Self {
            adjust: Vec::new(),
            project: Vec::new(),
            lateral: Vec::new(),
            intracl: Vec::new(),
            head: Vec::new(),
        };
        for i in 0..4 {
            s.adjust.push(Conv::load(
                store,
                &p("input_channel_adjustment_convolution", i),
                ConvKind::Plain { bias: false },
                [1, 1],
                [0, 0],
                1,
            )?);
            s.project.push(Conv::load(
                store,
                &p("input_feature_projection_convolution", i),
                PLAIN_BIAS,
                [1, 1],
                [4, 4],
                1,
            )?);
            s.lateral.push(Conv::load(
                store,
                &p("path_aggregation_lateral_convolution", i),
                PLAIN_BIAS,
                [1, 1],
                [4, 4],
                1,
            )?);
            s.intracl
                .push(IntraCl::load(store, &p("intraclass_blocks", i))?);
        }
        for i in 0..3 {
            s.head.push(Conv::load(
                store,
                &p("path_aggregation_head_convolution", i),
                ConvKind::Plain { bias: false },
                [2, 2],
                [1, 1],
                1,
            )?);
        }
        Ok(s)
    }

    fn forward(&self, f: &[MlxArray]) -> Result<MlxArray, MlxError> {
        let in2 = self.adjust[0].forward(&f[0])?;
        let in3 = self.adjust[1].forward(&f[1])?;
        let in4 = self.adjust[2].forward(&f[2])?;
        let in5 = self.adjust[3].forward(&f[3])?;
        let out4 = in4.add(&upsample(&in5, 2)?)?;
        let out3 = in3.add(&upsample(&out4, 2)?)?;
        let out2 = in2.add(&upsample(&out3, 2)?)?;
        let f2 = self.project[0].forward(&out2)?;
        let f3 = self.project[1].forward(&out3)?;
        let f4 = self.project[2].forward(&out4)?;
        let f5 = self.project[3].forward(&in5)?;
        let pan3 = f3.add(&self.head[0].forward(&f2)?)?;
        let pan4 = f4.add(&self.head[1].forward(&pan3)?)?;
        let pan5 = f5.add(&self.head[2].forward(&pan4)?)?;
        let p2 = self.intracl[0].forward(&self.lateral[0].forward(&f2)?)?;
        let p3 = self.intracl[1].forward(&self.lateral[1].forward(&pan3)?)?;
        let p4 = self.intracl[2].forward(&self.lateral[2].forward(&pan4)?)?;
        let p5 = self.intracl[3].forward(&self.lateral[3].forward(&pan5)?)?;
        let p5 = upsample(&p5, 8)?;
        let p4 = upsample(&p4, 4)?;
        let p3 = upsample(&p3, 2)?;
        MlxArray::concatenate(&[&p5, &p4, &p3, &p2], 3)
    }
}

/// `DBHead` (inference binarize branch).
struct DbHead {
    down: Conv,
    up: Conv,
    final_: Conv,
}

impl DbHead {
    fn load(store: &WeightStore, kernels: [usize; 3]) -> Result<Self, LoadError> {
        let k1 = kernels[0];
        Ok(Self {
            down: Conv::load(
                store,
                "head.conv_down",
                ConvKind::Bn {
                    conv: "convolution",
                    norm: "norm",
                    bias: false,
                },
                [1, 1],
                [k1 / 2, k1 / 2],
                1,
            )?,
            up: Conv::load_transpose(
                store,
                "head.conv_up",
                ConvKind::Bn {
                    conv: "convolution",
                    norm: "norm",
                    bias: true,
                },
                [2, 2],
            )?,
            final_: Conv::load_transpose(store, "head.conv_final", PLAIN_BIAS, [2, 2])?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let x = self.down.forward_act(x, Activation::Relu, None)?;
        let x = self.up.forward_act(&x, Activation::Relu, None)?;
        self.final_.forward(&x)?.sigmoid()
    }
}

/// The detection network.
pub struct DetModel {
    stem: Stem,
    stages: Vec<Vec<LcNetBlock>>,
    neck: RepLkPan,
    head: DbHead,
}

impl DetModel {
    /// Loads (and BN-fuses) the medium detector from its checkpoint.
    pub fn load(store: &WeightStore, config: &DetConfig) -> Result<Self, LoadError> {
        if config.neck_out_channels != 256 {
            return Err(LoadError::Layout(format!(
                "unexpected neck_out_channels {}",
                config.neck_out_channels
            )));
        }
        Ok(Self {
            stem: Stem::load(store, "model.backbone.encoder.convolution")?,
            stages: load_stages(store, "model.backbone.encoder", &config.stages)?,
            neck: RepLkPan::load(store)?,
            head: DbHead::load(store, config.head_kernel_list)?,
        })
    }

    /// Probability map `[1, H, W, 1]` for a normalized `[1, H, W, 3]` input.
    pub fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut x = self.stem.forward(x)?;
        let mut feats = Vec::with_capacity(4);
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
            }
            feats.push(x.try_clone()?);
        }
        let fused = self.neck.forward(&feats)?;
        self.head.forward(&fused)
    }
}
