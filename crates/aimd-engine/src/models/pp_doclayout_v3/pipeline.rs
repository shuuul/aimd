//! Page -> layout regions, as the MiniCPM-OCR SDK's
//! `PPDocLayoutV3TorchDetector._detect_one` does it: BGR + 800x800
//! `INTER_CUBIC` resize + `/ 255` ([`preprocess`]), the detector forward
//! pass ([`PpDocLayoutV3Model`]), then the SDK post-process
//! ([`postprocess`]) against the source size.
//!
//! There is no cancellation inside: one call is one page.

use std::path::Path;

use aimd_mlx::MlxError;
use aimd_weights::{WeightsError, load_checkpoint};

use super::config::{ConfigError, INPUT_SIZE, check_config};
use super::model::{PpDocLayoutV3Model, RawOutputs};
use super::postprocess::{LayoutRegion, postprocess};
use super::preprocess::preprocess;
use crate::models::pp_ocrv6::layers::LoadError;

/// Pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(#[from] ConfigError),
    #[error("weight import: {0}")]
    Load(#[from] LoadError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("empty image")]
    EmptyImage,
}

/// The loaded layout detector.
pub struct PpDocLayoutV3 {
    model: PpDocLayoutV3Model,
}

impl PpDocLayoutV3 {
    /// Loads a `PaddlePaddle/PP-DocLayoutV3_safetensors` snapshot directory
    /// (`config.json` + `model.safetensors`, f32).
    pub fn load(dir: &Path) -> Result<Self, PipelineError> {
        check_config(dir)?;
        let store = load_checkpoint(dir)?;
        Ok(Self {
            model: PpDocLayoutV3Model::load(&store)?,
        })
    }

    /// The raw detector outputs for an RGB page.
    pub fn raw_outputs(&self, rgb: &image::RgbImage) -> Result<RawOutputs, PipelineError> {
        if rgb.width() == 0 || rgb.height() == 0 {
            return Err(PipelineError::EmptyImage);
        }
        let (pixels, _) = preprocess(rgb);
        Ok(self.model.forward(&pixels, INPUT_SIZE)?)
    }

    /// Layout regions of an RGB page in reading order, boxes in the page's
    /// pixel coordinates.
    pub fn detect(&self, rgb: &image::RgbImage) -> Result<Vec<LayoutRegion>, PipelineError> {
        let raw = self.raw_outputs(rgb)?;
        Ok(postprocess(
            &raw.logits,
            &raw.boxes,
            &raw.order,
            rgb.width(),
            rgb.height(),
        ))
    }

    #[cfg(test)]
    pub(crate) fn model(&self) -> &PpDocLayoutV3Model {
        &self.model
    }
}
