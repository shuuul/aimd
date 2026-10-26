//! PaddleOCR-VL end-to-end recognition: image decode -> preprocessing ->
//! vision features -> `OCR:` / `Spotting:` prompt ids -> embedding splice -> greedy
//! generation -> `decode(skip_special_tokens=True).strip()`.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use aimd_mlx::{MlxArray, MlxError};
use aimd_weights::{WeightsError, load_checkpoint};

use super::config::{
    ConfigFileError, PaddleOcrVlConfig, PreprocessorConfig, load_eos_ids, load_preprocessor_config,
    parse_config,
};
use super::decoder::{DecoderError, PaddleDecoder};
use super::generate::{
    DEFAULT_MAX_TOKENS, FinishReason, GenerateError, GreedyStream, greedy_stream,
};
use super::preprocess::{
    IMAGE_PLACEHOLDER, OCR_TASK, PreparedImage, RopeIndex, build_prompt_text, image_token_count,
    preprocess_rgb8, rope_index,
};
pub use super::spotting::Spot;
use super::spotting::{
    SPOTTING_MAX_PIXELS, SPOTTING_TASK, parse_spotting_output, spotting_upscale,
    strip_location_tokens,
};
use super::vision::{PaddleVision, VisionError};
use super::weights::Loader;
use crate::models::unlimited_ocr::mlx_weights::LoadError;
use crate::models::unlimited_ocr::preprocess::PreprocessError;
use crate::tokenizer::{Tokenizer, TokenizerError};

/// Pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("io: {0}")]
    Io(String),
    #[error("image decode: {0}")]
    Image(String),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("weight import: {0}")]
    Load(#[from] LoadError),
    #[error("checkpoint config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("checkpoint config: {0}")]
    ConfigFile(#[from] ConfigFileError),
    #[error("config mismatch: {0}")]
    ConfigMismatch(String),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("vision: {0}")]
    Vision(#[from] VisionError),
    #[error("decoder: {0}")]
    Decoder(#[from] DecoderError),
    #[error("generate: {0}")]
    Generate(#[from] GenerateError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("preprocess: {0}")]
    Preprocess(#[from] PreprocessError),
    #[error("prompt layout: {0}")]
    Layout(String),
    #[error("PaddleOCR-VL produced empty content")]
    EmptyContent,
}

/// The task prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VlTask {
    /// Full-page `OCR:` (Markdown, no positions).
    #[default]
    Ocr,
    /// `Spotting:`: text lines with quadrilaterals (see [`super::spotting`]).
    Spotting,
}

impl VlTask {
    /// The prompt text after the image.
    pub fn prompt(self) -> &'static str {
        match self {
            VlTask::Ocr => OCR_TASK,
            VlTask::Spotting => SPOTTING_TASK,
        }
    }
}

/// One recognition request.
#[derive(Clone, Copy, Debug, Default)]
pub struct VlRequest {
    /// Generation budget; `None` = [`DEFAULT_MAX_TOKENS`] (4096).
    pub max_tokens: Option<usize>,
    /// Task prompt (default `OCR:`).
    pub task: VlTask,
}

/// Recognition result.
#[derive(Clone, Debug)]
pub struct VlOutput {
    /// `Ocr`: `decode(tokens, skip_special_tokens=True).strip()`.
    /// `Spotting`: the same decode with the `<|LOC_n|>` tokens removed
    /// (one text line per spotted item).
    pub markdown: String,
    /// Generated tokens (EOS excluded).
    pub tokens: usize,
    /// Spotted lines (quads normalized to the original image); empty for
    /// [`VlTask::Ocr`].
    pub spots: Vec<Spot>,
}

/// Prompt size and timings of one recognition.
#[derive(Clone, Debug)]
pub struct VlStats {
    pub prompt_tokens: usize,
    pub image_tokens: usize,
    /// Preprocessing + vision tower + embedding splice (lazy graph build
    /// included; the tower itself is evaluated inside `prefill`).
    pub vision: Duration,
    /// Prompt processing up to the first sampled token.
    pub prefill: Duration,
    /// Remaining decode steps.
    pub decode: Duration,
    pub finish_reason: FinishReason,
}

/// A prepared prompt: ids, image, mRoPE index.
pub(crate) struct Prompt {
    pub(crate) ids: Vec<u32>,
    pub(crate) image: PreparedImage,
    pub(crate) rope: RopeIndex,
    /// First index of the contiguous image-token run.
    pub(crate) image_start: usize,
}

/// Loaded PaddleOCR-VL-1.6 (vision + projector + decoder + tokenizer).
pub struct PaddleOcrVlPipeline {
    cfg: PaddleOcrVlConfig,
    pre_cfg: PreprocessorConfig,
    vision: PaddleVision,
    decoder: PaddleDecoder,
    tokenizer: Tokenizer,
    eos_ids: BTreeSet<u32>,
    quant_bits: Option<u32>,
}

impl PaddleOcrVlPipeline {
    /// Loads the official dense checkpoint snapshot (`PaddlePaddle/
    /// PaddleOCR-VL-1.6`). `quant_bits = None` keeps bf16; `Some(4 | 6 |
    /// 8)` quantizes at load like `mlx_vlm.convert` (affine, group 64).
    pub fn load(dir: &Path, quant_bits: Option<u32>) -> Result<Self, PipelineError> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| PipelineError::Io(format!("read config.json: {e}")))?;
        let cfg = parse_config(&text)?;
        let pre_cfg = load_preprocessor_config(dir)?;
        if pre_cfg.patch_size != cfg.vision_config.patch_size
            || pre_cfg.merge_size != cfg.vision_config.spatial_merge_size
        {
            return Err(PipelineError::ConfigMismatch(format!(
                "preprocessor patch/merge {}/{} vs vision {}/{}",
                pre_cfg.patch_size,
                pre_cfg.merge_size,
                cfg.vision_config.patch_size,
                cfg.vision_config.spatial_merge_size
            )));
        }
        let tokenizer = Tokenizer::from_checkpoint(dir)?;
        let ph = tokenizer.token_id(IMAGE_PLACEHOLDER)?;
        if ph != cfg.image_token_id {
            return Err(PipelineError::ConfigMismatch(format!(
                "tokenizer {IMAGE_PLACEHOLDER} id {ph} != config image_token_id {}",
                cfg.image_token_id
            )));
        }
        let eos_ids: BTreeSet<u32> = match load_eos_ids(dir)? {
            Some(ids) if !ids.is_empty() => ids.into_iter().collect(),
            _ => [tokenizer.token_id("</s>")?].into_iter().collect(),
        };
        let store = load_checkpoint(dir)?;
        let ld = Loader::new(&store, quant_bits)?;
        let vision = PaddleVision::load(&ld, &cfg.vision_config)?;
        let decoder = PaddleDecoder::load(&ld, &cfg)?;
        drop(store);
        Ok(Self {
            cfg,
            pre_cfg,
            vision,
            decoder,
            tokenizer,
            eos_ids,
            quant_bits,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn quant_bits(&self) -> Option<u32> {
        self.quant_bits
    }

    #[cfg(test)]
    pub(crate) fn vision(&self) -> &PaddleVision {
        &self.vision
    }

    #[cfg(test)]
    pub(crate) fn decoder(&self) -> &PaddleDecoder {
        &self.decoder
    }

    #[cfg(test)]
    pub(crate) fn eos_ids(&self) -> &BTreeSet<u32> {
        &self.eos_ids
    }

    /// Recognizes an image file (any format the `image` crate decodes).
    pub fn recognize_path(
        &self,
        path: &Path,
        request: VlRequest,
    ) -> Result<VlOutput, PipelineError> {
        Ok(self.recognize_path_with_stats(path, request)?.0)
    }

    /// [`Self::recognize_path`] plus prompt size and timings (benchmarks).
    pub fn recognize_path_with_stats(
        &self,
        path: &Path,
        request: VlRequest,
    ) -> Result<(VlOutput, VlStats), PipelineError> {
        let (rgb, w, h) = load_rgb8(path)?;
        self.recognize_rgb8_with_stats(&rgb, w, h, request)
    }

    /// Recognizes an RGB8 HWC image.
    pub fn recognize_rgb8(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: VlRequest,
    ) -> Result<VlOutput, PipelineError> {
        Ok(self
            .recognize_rgb8_with_stats(rgb, width, height, request)?
            .0)
    }

    pub(crate) fn recognize_rgb8_with_stats(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: VlRequest,
    ) -> Result<(VlOutput, VlStats), PipelineError> {
        let t0 = Instant::now();
        let (stream, prompt) = self.generate_rgb8(rgb, width, height, request, None)?;
        let total = t0.elapsed();
        let decoded = self.tokenizer.decode(&stream.tokens, true)?;
        let (markdown, spots) = match request.task {
            VlTask::Ocr => (decoded.trim().to_string(), Vec::new()),
            VlTask::Spotting => (
                strip_location_tokens(&decoded),
                parse_spotting_output(&decoded),
            ),
        };
        if markdown.is_empty() {
            return Err(PipelineError::EmptyContent);
        }
        let stats = VlStats {
            prompt_tokens: prompt.ids.len(),
            image_tokens: image_token_count(
                prompt.image.grid_thw,
                self.cfg.vision_config.spatial_merge_size,
            ),
            vision: total.saturating_sub(stream.prefill + stream.decode),
            prefill: stream.prefill,
            decode: stream.decode,
            finish_reason: stream.finish_reason,
        };
        Ok((
            VlOutput {
                markdown,
                tokens: stream.tokens.len(),
                spots,
            },
            stats,
        ))
    }

    /// Preprocessing + prompt ids + mRoPE index for one image. `Spotting`
    /// first applies the README upscale and raises `max_pixels`.
    pub(crate) fn prepare(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        task: VlTask,
    ) -> Result<Prompt, PipelineError> {
        let image = match task {
            VlTask::Ocr => preprocess_rgb8(rgb, width, height, &self.pre_cfg)?,
            VlTask::Spotting => {
                let cfg = PreprocessorConfig {
                    max_pixels: SPOTTING_MAX_PIXELS,
                    ..self.pre_cfg.clone()
                };
                match spotting_upscale(rgb, width, height)? {
                    Some((up, w, h)) => preprocess_rgb8(&up, w, h, &cfg)?,
                    None => preprocess_rgb8(rgb, width, height, &cfg)?,
                }
            }
        };
        let merge = self.cfg.vision_config.spatial_merge_size;
        let n_image = image_token_count(image.grid_thw, merge);
        let ids = self
            .tokenizer
            .encode(&build_prompt_text(task.prompt(), n_image))?;
        let img = self.cfg.image_token_id;
        let image_start = ids
            .iter()
            .position(|&t| t == img)
            .ok_or_else(|| PipelineError::Layout("no image tokens in the prompt".into()))?;
        let run = ids[image_start..].iter().take_while(|&&t| t == img).count();
        let total = ids.iter().filter(|&&t| t == img).count();
        if run != n_image || total != n_image {
            return Err(PipelineError::Layout(format!(
                "image run {run} / total {total}, expected one contiguous run of {n_image}"
            )));
        }
        let rope = rope_index(
            &ids,
            img,
            self.cfg.vision_start_token_id,
            &[image.grid_thw],
            merge,
        )?;
        Ok(Prompt {
            ids,
            image,
            rope,
            image_start,
        })
    }

    /// Image features `[n_image, hidden]` for a prepared prompt.
    pub(crate) fn image_features(&self, prompt: &Prompt) -> Result<MlxArray, PipelineError> {
        let p = self.cfg.vision_config.patch_size;
        let pixels = MlxArray::from_bf16(
            &[
                prompt.image.n_patches(),
                self.cfg.vision_config.num_channels,
                p,
                p,
            ],
            &prompt.image.pixels,
        )?;
        Ok(self.vision.forward(&pixels, prompt.image.grid_thw, None)?)
    }

    /// `embed_tokens(ids)` with the image run replaced by `features`.
    pub(crate) fn splice(
        &self,
        prompt: &Prompt,
        features: &MlxArray,
    ) -> Result<MlxArray, PipelineError> {
        let n = prompt.ids.len();
        let d = self.decoder.hidden_size();
        let n_img = features.dim(0)?;
        if features.shape()? != [n_img, d] {
            return Err(PipelineError::Layout(format!(
                "features {:?}",
                features.shape()?
            )));
        }
        let emb = self
            .decoder
            .embed_ids(&MlxArray::from_u32(&[1, n], &prompt.ids)?)?;
        let s = prompt.image_start;
        let e = s + n_img;
        let head = emb.slice(&[0, 0, 0], &[1, s, d], &[1, 1, 1])?;
        let tail = emb.slice(&[0, e, 0], &[1, n, d], &[1, 1, 1])?;
        let feats = features.astype(emb.dtype()?)?.reshape(&[1, n_img, d])?;
        Ok(MlxArray::concatenate(&[&head, &feats, &tail], 1)?)
    }

    /// Full generation; returns the token stream and the prompt.
    pub(crate) fn generate_rgb8(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: VlRequest,
        capture: Option<&mut Vec<MlxArray>>,
    ) -> Result<(GreedyStream, Prompt), PipelineError> {
        let prompt = self.prepare(rgb, width, height, request.task)?;
        let features = self.image_features(&prompt)?;
        let embeds = self.splice(&prompt, &features)?;
        let stream = greedy_stream(
            &self.decoder,
            &embeds,
            &prompt.rope,
            &self.eos_ids,
            request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            capture,
        )?;
        Ok((stream, prompt))
    }
}

/// Reads and decodes an image file to RGB8 HWC `(pixels, width, height)`.
pub fn load_rgb8(path: &Path) -> Result<(Vec<u8>, usize, usize), PipelineError> {
    let bytes = std::fs::read(path)
        .map_err(|e| PipelineError::Io(format!("read {}: {e}", path.display())))?;
    let rgb = image::load_from_memory(&bytes)
        .map_err(|e| PipelineError::Image(e.to_string()))?
        .to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw(), w, h))
}
