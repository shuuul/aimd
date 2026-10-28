//! MiniCPM-OCR end-to-end recognition of one image: slicing -> per-crop
//! vision features -> chat prompt with `<image>` / `<slice>` runs ->
//! embedding splice -> greedy generation ->
//! `decode(skip_special_tokens=True).strip()`.
//!
//! The splice reproduces mlx-vlm's `_apply_vision_delta`
//! (`embeds.at[idx].add(features - embeds[idx])`): each image row is
//! `e + bf16(f - e)`, which is not always `f` in bf16.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use aimd_mlx::{MlxArray, MlxDtype, MlxError};
use aimd_weights::{WeightsError, load_checkpoint};
use half::bf16;

use super::config::{
    ConfigFileError, MiniCpmOcrConfig, PreprocessorConfig, load_eos_ids, load_preprocessor_config,
    parse_config,
};
use super::decoder::{DecoderError, MiniCpmDecoder};
use super::dspark::{DSparkDrafter, DSparkError};
use super::generate::{
    DEFAULT_MAX_TOKENS, FinishReason, GenerateError, GreedyStream, NO_REPEAT_NGRAM_SIZE, SpecStats,
    greedy_stream, speculative_stream,
};
use super::preprocess::{
    Crop, IMAGE_PAD, build_prompt_text, crop_tokens, crops_rgb8, image_placeholder, patch_rows,
    pixel_lut,
};
use super::vision::{MiniCpmVision, VisionError};
use crate::models::paddleocr_vl::weights::Loader;
use crate::models::unlimited_ocr::mlx_weights::LoadError;
use crate::models::unlimited_ocr::preprocess::PreprocessError;
use crate::tokenizer::{Tokenizer, TokenizerError};

/// The SDK's plain text-recognition prompt.
pub const TEXT_TASK: &str = "Text Recognition:";

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
    #[error("DSpark draft: {0}")]
    DSpark(#[from] DSparkError),
    #[error("generate: {0}")]
    Generate(#[from] GenerateError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("preprocess: {0}")]
    Preprocess(#[from] PreprocessError),
    #[error("prompt layout: {0}")]
    Layout(String),
}

/// One recognition request.
#[derive(Clone, Copy, Debug)]
pub struct OcrRequest<'a> {
    /// Task prompt after the image (the SDK's per-region prompts).
    pub task: &'a str,
    /// Generation budget; `None` = [`DEFAULT_MAX_TOKENS`].
    pub max_tokens: Option<usize>,
    /// Slice cap; `None` = the checkpoint's `max_slice_nums` (9).
    pub max_slices: Option<usize>,
    /// The SDK's `no_repeat_ngram_size` guard (default on).
    pub no_repeat_ngram: bool,
    /// DSpark speculative decoding when a drafter is attached (default on).
    pub speculative: bool,
}

impl Default for OcrRequest<'_> {
    fn default() -> Self {
        Self {
            task: TEXT_TASK,
            max_tokens: None,
            max_slices: None,
            no_repeat_ngram: true,
            speculative: true,
        }
    }
}

/// Recognition result.
#[derive(Clone, Debug)]
pub struct OcrOutput {
    /// `decode(tokens, skip_special_tokens=True).strip()` (may be empty).
    pub text: String,
    /// Generated tokens (EOS excluded).
    pub tokens: usize,
    pub finish_reason: FinishReason,
}

/// Prompt size and timings of one recognition.
#[derive(Clone, Debug)]
pub struct OcrStats {
    pub prompt_tokens: usize,
    pub image_tokens: usize,
    pub crops: usize,
    /// Preprocessing + vision graph build + splice.
    pub vision: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    /// DSpark acceptance (`None` without speculative decoding).
    pub speculative: Option<SpecStats>,
}

/// A prepared prompt: ids, crops and the image-token spans.
pub(crate) struct Prompt {
    pub(crate) ids: Vec<u32>,
    pub(crate) crops: Vec<Crop>,
    /// `(start, end)` of each crop's `<|image_pad|>` run, crop order.
    pub(crate) spans: Vec<(usize, usize)>,
}

/// Loaded MiniCPM-OCR 1.0 (vision + merger + decoder + tokenizer).
pub struct MiniCpmOcrPipeline {
    pre_cfg: PreprocessorConfig,
    luts: [[bf16; 256]; 3],
    image_token_id: u32,
    vision: MiniCpmVision,
    decoder: MiniCpmDecoder,
    tokenizer: Tokenizer,
    eos_ids: BTreeSet<u32>,
    quant_bits: Option<u32>,
    drafter: Option<DSparkDrafter>,
}

impl MiniCpmOcrPipeline {
    /// Loads the official bf16 snapshot (`openbmb/MiniCPM-OCR-1.0`).
    /// `quant_bits = Some(4 | 6 | 8)` quantizes the decoder's attention
    /// and MLP projections like `mlx_vlm.convert` (affine, group 64).
    pub fn load(dir: &Path, quant_bits: Option<u32>) -> Result<Self, PipelineError> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| PipelineError::Io(format!("read config.json: {e}")))?;
        let cfg: MiniCpmOcrConfig = parse_config(&text)?;
        let pre_cfg = load_preprocessor_config(dir)?;
        if cfg.downsample_mode != "4x" || pre_cfg.patch_size != cfg.vision_config.patch_size {
            return Err(PipelineError::ConfigMismatch(format!(
                "downsample_mode {} / patch {} vs {}",
                cfg.downsample_mode, pre_cfg.patch_size, cfg.vision_config.patch_size
            )));
        }
        if pre_cfg.image_mean.len() != 3 || pre_cfg.image_std.len() != 3 {
            return Err(PipelineError::ConfigMismatch(
                "image_mean / image_std".into(),
            ));
        }
        let tokenizer = Tokenizer::from_checkpoint(dir)?;
        let pad = tokenizer.token_id(IMAGE_PAD)?;
        if pad != cfg.image_token_id {
            return Err(PipelineError::ConfigMismatch(format!(
                "tokenizer {IMAGE_PAD} id {pad} != config image_token_id {}",
                cfg.image_token_id
            )));
        }
        let eos_ids: BTreeSet<u32> = match load_eos_ids(dir)? {
            Some(ids) if !ids.is_empty() => ids.into_iter().collect(),
            _ => [tokenizer.token_id("<|im_end|>")?].into_iter().collect(),
        };
        let luts = std::array::from_fn(|c| pixel_lut(pre_cfg.image_mean[c], pre_cfg.image_std[c]));
        let store = load_checkpoint(dir)?;
        let dense = Loader::new(&store, None)?;
        let quant = Loader::new(&store, quant_bits)?;
        let vision = MiniCpmVision::load(&dense, &cfg.vision_config, cfg.text_config.hidden_size)?;
        let decoder = MiniCpmDecoder::load(&dense, &quant, &cfg)?;
        drop(store);
        Ok(Self {
            pre_cfg,
            luts,
            image_token_id: cfg.image_token_id,
            vision,
            decoder,
            tokenizer,
            eos_ids,
            quant_bits,
            drafter: None,
        })
    }

    /// Loads the DSpark draft snapshot (`openbmb/MiniCPM-OCR-1.0-DSpark`)
    /// for speculative decoding.
    pub fn attach_drafter(&mut self, dir: &Path) -> Result<(), PipelineError> {
        self.drafter = Some(DSparkDrafter::load(dir, &self.decoder)?);
        Ok(())
    }

    pub fn has_drafter(&self) -> bool {
        self.drafter.is_some()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn quant_bits(&self) -> Option<u32> {
        self.quant_bits
    }

    #[cfg(test)]
    pub(crate) fn vision(&self) -> &MiniCpmVision {
        &self.vision
    }

    #[cfg(test)]
    pub(crate) fn decoder(&self) -> &MiniCpmDecoder {
        &self.decoder
    }

    #[cfg(test)]
    pub(crate) fn eos_ids(&self) -> &BTreeSet<u32> {
        &self.eos_ids
    }

    #[cfg(test)]
    pub(crate) fn luts(&self) -> &[[bf16; 256]; 3] {
        &self.luts
    }

    /// Recognizes an image file.
    pub fn recognize_path(
        &self,
        path: &Path,
        request: OcrRequest<'_>,
    ) -> Result<OcrOutput, PipelineError> {
        let (rgb, w, h) = load_rgb8(path)?;
        Ok(self.recognize_rgb8_with_stats(&rgb, w, h, request)?.0)
    }

    /// Recognizes an RGB8 HWC image.
    pub fn recognize_rgb8(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: OcrRequest<'_>,
    ) -> Result<OcrOutput, PipelineError> {
        Ok(self
            .recognize_rgb8_with_stats(rgb, width, height, request)?
            .0)
    }

    /// [`Self::recognize_rgb8`] plus prompt size and timings.
    pub fn recognize_rgb8_with_stats(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: OcrRequest<'_>,
    ) -> Result<(OcrOutput, OcrStats), PipelineError> {
        let t0 = Instant::now();
        let (stream, prompt, spec) = self.generate_rgb8_spec(rgb, width, height, request)?;
        let total = t0.elapsed();
        let text = self
            .tokenizer
            .decode(&stream.tokens, true)?
            .trim()
            .to_string();
        let stats = OcrStats {
            prompt_tokens: prompt.ids.len(),
            image_tokens: prompt.spans.iter().map(|(s, e)| e - s).sum(),
            crops: prompt.crops.len(),
            vision: total.saturating_sub(stream.prefill + stream.decode),
            prefill: stream.prefill,
            decode: stream.decode,
            speculative: spec,
        };
        Ok((
            OcrOutput {
                text,
                tokens: stream.tokens.len(),
                finish_reason: stream.finish_reason,
            },
            stats,
        ))
    }

    /// Slicing + prompt ids + image spans for one image.
    pub(crate) fn prepare(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        task: &str,
        max_slices: Option<usize>,
    ) -> Result<Prompt, PipelineError> {
        let cfg = PreprocessorConfig {
            max_slice_nums: max_slices.unwrap_or(self.pre_cfg.max_slice_nums),
            ..self.pre_cfg.clone()
        };
        let (crops, plan) = crops_rgb8(rgb, width, height, &cfg)?;
        let patch = cfg.patch_size;
        let tokens: Vec<usize> = crops
            .iter()
            .map(|c| {
                let (gh, gw) = c.grid(patch);
                crop_tokens(gh, gw)
            })
            .collect();
        let text = build_prompt_text(&image_placeholder(&tokens, plan.grid), task);
        let ids = self.tokenizer.encode(&text)?;
        let pad = self.image_token_id;
        let mut spans = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < ids.len() {
            if ids[i] == pad {
                let s = i;
                while i < ids.len() && ids[i] == pad {
                    i += 1;
                }
                spans.push((s, i));
            } else {
                i += 1;
            }
        }
        let lens: Vec<usize> = spans.iter().map(|(s, e)| e - s).collect();
        if lens != tokens {
            return Err(PipelineError::Layout(format!(
                "image runs {lens:?}, expected {tokens:?}"
            )));
        }
        Ok(Prompt { ids, crops, spans })
    }

    /// Merged features of each crop, crop order.
    pub(crate) fn crop_features(&self, prompt: &Prompt) -> Result<Vec<MlxArray>, PipelineError> {
        let patch = self.pre_cfg.patch_size;
        prompt
            .crops
            .iter()
            .map(|c| {
                let (gh, gw) = c.grid(patch);
                let rows = patch_rows(c, patch, &self.luts);
                Ok(self.vision.forward(&rows, gh, gw, None)?)
            })
            .collect()
    }

    /// `embed_tokens(ids)` with each image span replaced by the image
    /// features (transformers' `masked_scatter`).
    pub(crate) fn splice(
        &self,
        prompt: &Prompt,
        features: &[MlxArray],
    ) -> Result<MlxArray, PipelineError> {
        let n = prompt.ids.len();
        let d = self.decoder.hidden_size();
        let emb = self
            .decoder
            .embed_ids(&MlxArray::from_u32(&[1, n], &prompt.ids)?)?;
        let dtype = emb.dtype()?;
        let mut parts = Vec::with_capacity(2 * features.len() + 1);
        let mut at = 0;
        for (&(s, e), f) in prompt.spans.iter().zip(features) {
            if f.shape()? != [e - s, d] {
                return Err(PipelineError::Layout(format!(
                    "features {:?} for span {s}..{e}",
                    f.shape()?
                )));
            }
            if s > at {
                parts.push(emb.slice(&[0, at, 0], &[1, s, d], &[1, 1, 1])?);
            }
            parts.push(f.astype(dtype)?.reshape(&[1, e - s, d])?);
            at = e;
        }
        if at < n {
            parts.push(emb.slice(&[0, at, 0], &[1, n, d], &[1, 1, 1])?);
        }
        let refs: Vec<&MlxArray> = parts.iter().collect();
        let out = MlxArray::concatenate(&refs, 1)?;
        debug_assert_eq!(dtype, MlxDtype::Bf16);
        Ok(out)
    }

    /// Full generation, speculative when a drafter is attached and
    /// `request.speculative`.
    pub(crate) fn generate_rgb8_spec(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: OcrRequest<'_>,
    ) -> Result<(GreedyStream, Prompt, Option<SpecStats>), PipelineError> {
        let Some(drafter) = self.drafter.as_ref().filter(|_| request.speculative) else {
            let (stream, prompt) = self.generate_rgb8(rgb, width, height, request, None)?;
            return Ok((stream, prompt, None));
        };
        let prompt = self.prepare(rgb, width, height, request.task, request.max_slices)?;
        let features = self.crop_features(&prompt)?;
        let embeds = self.splice(&prompt, &features)?;
        let (stream, stats) = speculative_stream(
            &self.decoder,
            drafter,
            &embeds,
            &self.eos_ids,
            request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            request.no_repeat_ngram.then_some(NO_REPEAT_NGRAM_SIZE),
        )?;
        Ok((stream, prompt, Some(stats)))
    }

    /// Plain greedy generation; returns the token stream and the prompt.
    pub(crate) fn generate_rgb8(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        request: OcrRequest<'_>,
        capture: Option<&mut Vec<MlxArray>>,
    ) -> Result<(GreedyStream, Prompt), PipelineError> {
        let prompt = self.prepare(rgb, width, height, request.task, request.max_slices)?;
        let features = self.crop_features(&prompt)?;
        let embeds = self.splice(&prompt, &features)?;
        let stream = greedy_stream(
            &self.decoder,
            &embeds,
            &self.eos_ids,
            request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            request.no_repeat_ngram.then_some(NO_REPEAT_NGRAM_SIZE),
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
