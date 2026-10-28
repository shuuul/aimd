//! HGNetV2-L backbone (transformers `modeling_hgnet_v2.py`) over NHWC f32:
//! the branch stem (shared with PP-OCRv6), then four stages of
//! `HGNetV2BasicLayer`s. Frozen batch norms fold into the convs at load.
//! Returns the four stage outputs (strides 4, 8, 16, 32).

use aimd_mlx::{Activation, MlxArray, MlxError};
use aimd_weights::WeightStore;

use super::config::{STAGES, StageSpec};
use super::layers::{HG_BN, conv};
use crate::models::pp_ocrv6::layers::{Conv, LoadError, Stem};

const PREFIX: &str = "model.backbone.model";

/// One aggregated layer: a 3x3 conv + ReLU, or the light 1x1 (no act) +
/// depthwise kxk + ReLU pair.
enum Unit {
    Plain(Conv),
    Light(Conv, Conv),
}

impl Unit {
    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        match self {
            Self::Plain(c) => c.forward_act(x, Activation::Relu, None),
            Self::Light(c1, c2) => c2.forward_act(&c1.forward(x)?, Activation::Relu, None),
        }
    }
}

/// `HGNetV2BasicLayer`.
struct BasicLayer {
    units: Vec<Unit>,
    squeeze: Conv,
    excite: Conv,
    residual: bool,
}

impl BasicLayer {
    fn load(
        store: &WeightStore,
        prefix: &str,
        spec: &StageSpec,
        residual: bool,
    ) -> Result<Self, LoadError> {
        let units = (0..spec.layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                Ok(if spec.light {
                    Unit::Light(
                        conv(store, &format!("{p}.conv1"), HG_BN, 1, 1, 1)?,
                        conv(
                            store,
                            &format!("{p}.conv2"),
                            HG_BN,
                            spec.kernel,
                            1,
                            spec.mid_channels,
                        )?,
                    )
                } else {
                    Unit::Plain(conv(store, &p, HG_BN, spec.kernel, 1, 1)?)
                })
            })
            .collect::<Result<_, LoadError>>()?;
        Ok(Self {
            units,
            squeeze: conv(store, &format!("{prefix}.aggregation.0"), HG_BN, 1, 1, 1)?,
            excite: conv(store, &format!("{prefix}.aggregation.1"), HG_BN, 1, 1, 1)?,
            residual,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<MlxArray, MlxError> {
        let mut outs = vec![x.try_clone()?];
        for unit in &self.units {
            let h = unit.forward(outs.last().unwrap())?;
            outs.push(h);
        }
        let refs: Vec<&MlxArray> = outs.iter().collect();
        let cat = MlxArray::concatenate(&refs, 3)?;
        let h = self.squeeze.forward_act(&cat, Activation::Relu, None)?;
        let residual = self.residual.then_some(x);
        self.excite.forward_act(&h, Activation::Relu, residual)
    }
}

struct Stage {
    downsample: Option<Conv>,
    blocks: Vec<BasicLayer>,
}

/// The HGNetV2-L backbone.
pub(crate) struct Backbone {
    stem: Stem,
    stages: Vec<Stage>,
}

impl Backbone {
    pub(crate) fn load(store: &WeightStore) -> Result<Self, LoadError> {
        let stem = Stem::load(store, &format!("{PREFIX}.embedder"))?;
        let stages = STAGES
            .iter()
            .enumerate()
            .map(|(s, spec)| {
                let p = format!("{PREFIX}.encoder.stages.{s}");
                let downsample = if spec.downsample {
                    Some(conv(
                        store,
                        &format!("{p}.downsample"),
                        HG_BN,
                        3,
                        2,
                        spec.in_channels,
                    )?)
                } else {
                    None
                };
                let blocks = (0..spec.blocks)
                    .map(|b| BasicLayer::load(store, &format!("{p}.blocks.{b}"), spec, b != 0))
                    .collect::<Result<_, _>>()?;
                Ok(Stage { downsample, blocks })
            })
            .collect::<Result<_, LoadError>>()?;
        Ok(Self { stem, stages })
    }

    /// `[1, 800, 800, 3]` -> the four stage outputs (128 / 512 / 1024 /
    /// 2048 channels at 200 / 100 / 50 / 25).
    pub(crate) fn forward(&self, pixels: &MlxArray) -> Result<Vec<MlxArray>, MlxError> {
        let mut x = self.stem.forward(pixels)?;
        let mut outs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            if let Some(d) = &stage.downsample {
                x = d.forward(&x)?;
            }
            for block in &stage.blocks {
                x = block.forward(&x)?;
            }
            outs.push(x.try_clone()?);
        }
        Ok(outs)
    }
}
