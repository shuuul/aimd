//! End-to-end PP-OCRv6 medium pipeline, mirroring mlx4ocr
//! `PP_OCRv6.predict` (non-compiled path, `quad` boxes):
//!
//! 1. decode to BGR8 (`PIL.Image.open(..).convert("RGB")` reversed). PNG
//!    and other lossless inputs decode identically; JPEG goes through the
//!    `image` crate's zune-jpeg, which differs from PIL's libjpeg-turbo by
//!    a few levels per pixel (IDCT / chroma upsampling rounding), so JPEG
//!    boxes and scores can drift slightly from mlx4ocr. Everything after
//!    decode is bit-exact with mlx4ocr on Metal;
//! 2. detection preprocess: pad tiny images to 32, `limit_type=min` resize
//!    with `limit_side_len=960` (mlx4ocr's pipeline default; the HF
//!    `preprocessor_config.json` says 736 but mlx4ocr never reads it),
//!    `max_side_limit=4000`, snap to multiples of 32, cv2 bilinear resize,
//!    ImageNet normalize (`x * scale + bias`, two f32 roundings);
//! 3. detector forward -> `[H, W]` probability map -> DB post-process
//!    ([`db::db_postprocess`]) -> reading order ([`db::reading_order`]);
//! 4. perspective crops ([`image_ops::rotate_crop`]);
//! 5. recognition in batches of 6 crops ordered by `np.argsort` of the
//!    width/height ratio: each batch shares `img_w = int(48 *
//!    max(320 / 48, ratios))`, crops are resized to `min(ceil(48 * ratio),
//!    img_w) x 48`, normalized `(x / 255 - 0.5) / 0.5` and zero-padded on
//!    the right;
//! 6. greedy CTC decode, drop lines scoring below 0.5, join the kept lines
//!    with `\n` (markdown = text + `\n`).
//!
//! There is no cancellation inside the pipeline: one call is one page.

use std::path::Path;

use aimd_mlx::{MlxArray, MlxError};
use aimd_weights::{WeightsError, load_checkpoint};

use super::config::{ConfigError, DetConfig, RecConfig, load_det_config, load_rec_config};
use super::ctc::{Decoded, ctc_decode, np_argsort};
use super::db::{DbParams, Detection, db_postprocess, reading_order};
use super::det::DetModel;
use super::image_ops::{BgrImage, resize_linear, rotate_crop};
use super::layers::LoadError;
use super::rec::RecModel;

/// Detection `limit_side_len` (mlx4ocr `PipelineConfig` default).
pub const DET_LIMIT_SIDE_LEN: usize = 960;
/// Detection `max_side_limit`.
pub const DET_MAX_SIDE_LIMIT: usize = 4000;
/// Recognition input height and base width (`rec_image_shape`).
pub const REC_IMAGE_H: usize = 48;
pub const REC_IMAGE_W: usize = 320;
/// Crops per recognition forward pass (`rec_batch_num`).
pub const REC_BATCH: usize = 6;
/// Lines scoring below this are dropped (`drop_score`).
pub const DROP_SCORE: f32 = 0.5;

/// Pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("io: {0}")]
    Io(String),
    #[error("image decode: {0}")]
    Image(String),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(#[from] ConfigError),
    #[error("weight import: {0}")]
    Load(#[from] LoadError),
    #[error("mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("no text recognized")]
    EmptyContent,
}

/// One recognized line in reading order.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLine {
    pub text: String,
    pub score: f32,
    /// Corners TL, TR, BR, BL in source pixels.
    pub quad: [[f32; 2]; 4],
}

/// OCR result for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct PpOcrOutput {
    pub lines: Vec<TextLine>,
    pub markdown: String,
}

/// The detector input after preprocessing.
#[derive(Clone, Debug)]
pub struct DetInput {
    /// Resized BGR image (`resize_w x resize_h`).
    pub resized: BgrImage,
    /// Normalized `[H, W, 3]` f32 values.
    pub normalized: Vec<f32>,
    /// Source size the boxes are scaled back to (after tiny-image padding).
    pub src_w: usize,
    pub src_h: usize,
}

/// Python `round()` on a float (half to even).
fn py_round(v: f64) -> f64 {
    v.round_ties_even()
}

/// Detector input size `(width, height)` for a `src_w x src_h` source
/// (after tiny-image padding): `limit_type=min` scaling, the max-side cap,
/// then rounding to multiples of 32.
pub fn det_resize_target(src_w: usize, src_h: usize) -> (usize, usize) {
    let limit = DET_LIMIT_SIDE_LEN as f64;
    let ratio = if src_h.min(src_w) < DET_LIMIT_SIDE_LEN {
        if src_h <= src_w {
            limit / src_h as f64
        } else {
            limit / src_w as f64
        }
    } else {
        1.0
    };
    let mut rh = (src_h as f64 * ratio) as usize;
    let mut rw = (src_w as f64 * ratio) as usize;
    if rh.max(rw) > DET_MAX_SIDE_LIMIT {
        let scale = DET_MAX_SIDE_LIMIT as f64 / rh.max(rw) as f64;
        rh = (rh as f64 * scale) as usize;
        rw = (rw as f64 * scale) as usize;
    }
    let rh = ((py_round(rh as f64 / 32.0) * 32.0) as usize).max(32);
    let rw = ((py_round(rw as f64 / 32.0) * 32.0) as usize).max(32);
    (rw, rh)
}

/// Source size after mlx4ocr's tiny-image padding (images whose width +
/// height is below 64 are padded to at least 32 x 32).
pub fn det_padded_size(width: usize, height: usize) -> (usize, usize) {
    if width + height < 64 {
        (width.max(32), height.max(32))
    } else {
        (width, height)
    }
}

/// mlx4ocr `resize_det_image` + `_normalize_image`.
pub fn det_preprocess(image: &BgrImage) -> DetInput {
    let padded;
    let image = if det_padded_size(image.width, image.height) != (image.width, image.height) {
        let (pw, ph) = det_padded_size(image.width, image.height);
        let mut p = BgrImage::zeros(pw, ph);
        for y in 0..image.height {
            let row = &image.data[y * image.width * 3..(y + 1) * image.width * 3];
            p.data[y * pw * 3..y * pw * 3 + row.len()].copy_from_slice(row);
        }
        padded = p;
        &padded
    } else {
        image
    };
    let (src_h, src_w) = (image.height, image.width);
    let (rw, rh) = det_resize_target(src_w, src_h);
    let resized = resize_linear(image, rw, rh);
    let scale: [f32; 3] = [
        (1.0 / (255.0 * 0.229)) as f32,
        (1.0 / (255.0 * 0.224)) as f32,
        (1.0 / (255.0 * 0.225)) as f32,
    ];
    let bias: [f32; 3] = [
        (-0.485 / 0.229) as f32,
        (-0.456 / 0.224) as f32,
        (-0.406 / 0.225) as f32,
    ];
    let normalized = resized
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let c = i % 3;
            let m = v as f32 * scale[c];
            m + bias[c]
        })
        .collect();
    DetInput {
        resized,
        normalized,
        src_w,
        src_h,
    }
}

/// Width/height ratio of a crop (`crop.shape[1] / float(crop.shape[0])`).
fn wh_ratio(crop: &BgrImage) -> f64 {
    crop.width as f64 / crop.height as f64
}

/// mlx4ocr `resize_norm_img`: the crop as a normalized, right-padded
/// `[48, img_w, 3]` f32 slab.
pub fn rec_preprocess(crop: &BgrImage, max_wh_ratio: f64) -> (Vec<f32>, usize) {
    let img_w = (REC_IMAGE_H as f64 * max_wh_ratio) as usize;
    let resized_w = ((REC_IMAGE_H as f64 * wh_ratio(crop)).ceil() as usize).min(img_w);
    let resized = resize_linear(crop, resized_w, REC_IMAGE_H);
    let mut out = vec![0f32; REC_IMAGE_H * img_w * 3];
    for y in 0..REC_IMAGE_H {
        for x in 0..resized_w {
            for c in 0..3 {
                let v = resized.at(x, y, c) as f32 / 255.0f32;
                out[(y * img_w + x) * 3 + c] = (v - 0.5f32) / 0.5f32;
            }
        }
    }
    (out, img_w)
}

/// Recognition batches: `(crop indices, max_wh_ratio)` in processing order.
pub fn rec_batches(crops: &[BgrImage]) -> Vec<(Vec<usize>, f64)> {
    let ratios: Vec<f64> = crops.iter().map(wh_ratio).collect();
    let order = np_argsort(&ratios);
    order
        .chunks(REC_BATCH)
        .map(|chunk| {
            let mut max_wh = REC_IMAGE_W as f64 / REC_IMAGE_H as f64;
            for &i in chunk {
                max_wh = max_wh.max(ratios[i]);
            }
            (chunk.to_vec(), max_wh)
        })
        .collect()
}

/// Loaded PP-OCRv6 medium det + rec.
pub struct PpOcrV6Pipeline {
    det: DetModel,
    rec: RecModel,
    rec_config: RecConfig,
    det_config: DetConfig,
    db: DbParams,
}

impl PpOcrV6Pipeline {
    /// Loads the detection and recognition safetensors snapshots.
    pub fn load(det_dir: &Path, rec_dir: &Path) -> Result<Self, PipelineError> {
        let det_config = load_det_config(det_dir)?;
        let rec_config = load_rec_config(rec_dir)?;
        let det_store = load_checkpoint(det_dir)?;
        let rec_store = load_checkpoint(rec_dir)?;
        let det = DetModel::load(&det_store, &det_config)?;
        let rec = RecModel::load(&rec_store, &rec_config)?;
        Ok(Self {
            det,
            rec,
            rec_config,
            det_config,
            db: DbParams::default(),
        })
    }

    /// The parsed detection config.
    pub fn det_config(&self) -> &DetConfig {
        &self.det_config
    }

    /// The parsed recognition config (incl. the CTC character table).
    pub fn rec_config(&self) -> &RecConfig {
        &self.rec_config
    }

    /// Detector probability map (`[H, W]` row-major) for a preprocessed
    /// input.
    pub fn det_prob_map(&self, input: &DetInput) -> Result<Vec<f32>, PipelineError> {
        let (w, h) = (input.resized.width, input.resized.height);
        let x = MlxArray::from_f32(&[1, h, w, 3], &input.normalized)?;
        let prob = self.det.forward(&x)?;
        prob.eval()?;
        Ok(prob.to_f32_vec()?)
    }

    /// Detection boxes in reading order.
    pub fn detect(&self, image: &BgrImage) -> Result<Vec<Detection>, PipelineError> {
        let input = det_preprocess(image);
        let prob = self.det_prob_map(&input)?;
        let dets = db_postprocess(
            &prob,
            input.resized.width,
            input.resized.height,
            input.src_w,
            input.src_h,
            &self.db,
        );
        let quads: Vec<_> = dets.iter().map(|d| d.quad).collect();
        Ok(reading_order(&quads)
            .into_iter()
            .map(|i| dets[i].clone())
            .collect())
    }

    /// Softmax `argmax` / `max` per step for one preprocessed batch
    /// (`[B, 48, W, 3]`), as `(indices, probs, steps)`.
    pub fn rec_step_maxima(
        &self,
        batch: &[f32],
        b: usize,
        w: usize,
    ) -> Result<(Vec<u32>, Vec<f32>, usize), PipelineError> {
        let x = MlxArray::from_f32(&[b, REC_IMAGE_H, w, 3], batch)?;
        let probs = self.rec.forward(&x)?;
        let idx = probs.argmax(-1, false)?;
        let max = probs.max_axes(&[-1], false)?;
        MlxArray::eval_many(&[&idx, &max])?;
        let steps = idx.shape()?[1];
        Ok((idx.to_u32_vec()?, max.to_f32_vec()?, steps))
    }

    /// Full softmax output for one preprocessed batch (tests / goldens).
    pub fn rec_probs(
        &self,
        batch: &[f32],
        b: usize,
        w: usize,
    ) -> Result<(Vec<f32>, Vec<usize>), PipelineError> {
        let x = MlxArray::from_f32(&[b, REC_IMAGE_H, w, 3], batch)?;
        let probs = self.rec.forward(&x)?;
        probs.eval()?;
        Ok((probs.to_f32_vec()?, probs.shape()?))
    }

    /// Recognizes crops (results in crop order).
    pub fn recognize(&self, crops: &[BgrImage]) -> Result<Vec<Decoded>, PipelineError> {
        let mut out: Vec<Option<Decoded>> = vec![None; crops.len()];
        for (indices, max_wh) in rec_batches(crops) {
            let mut batch = Vec::new();
            let mut img_w = 0;
            let mut valid = Vec::new();
            for &i in &indices {
                if crops[i].width == 0 || crops[i].height == 0 {
                    out[i] = Some(Decoded {
                        text: String::new(),
                        score: 0.0,
                    });
                    continue;
                }
                let (slab, w) = rec_preprocess(&crops[i], max_wh);
                img_w = w;
                batch.extend_from_slice(&slab);
                valid.push(i);
            }
            if valid.is_empty() {
                continue;
            }
            let (idx, max, steps) = self.rec_step_maxima(&batch, valid.len(), img_w)?;
            for (k, &i) in valid.iter().enumerate() {
                let r = k * steps..(k + 1) * steps;
                out[i] = Some(ctc_decode(
                    &idx[r.clone()],
                    &max[r],
                    &self.rec_config.characters,
                ));
            }
        }
        Ok(out.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// Runs det + rec on a decoded BGR image.
    pub fn recognize_image(&self, image: &BgrImage) -> Result<PpOcrOutput, PipelineError> {
        let dets = self.detect(image)?;
        let crops: Vec<BgrImage> = dets.iter().map(|d| rotate_crop(image, &d.quad)).collect();
        let decoded = self.recognize(&crops)?;
        let lines: Vec<TextLine> = dets
            .iter()
            .zip(decoded)
            .filter(|(_, r)| r.score >= DROP_SCORE)
            .map(|(d, r)| TextLine {
                text: r.text,
                score: r.score,
                quad: d.quad,
            })
            .collect();
        if lines.is_empty() {
            return Err(PipelineError::EmptyContent);
        }
        let text = lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let markdown = if text.is_empty() {
            String::new()
        } else {
            text + "\n"
        };
        Ok(PpOcrOutput { lines, markdown })
    }

    /// Decodes and recognizes the image at `path`.
    pub fn recognize_path(&self, path: &Path) -> Result<PpOcrOutput, PipelineError> {
        let bytes = std::fs::read(path)
            .map_err(|e| PipelineError::Io(format!("{}: {e}", path.display())))?;
        let image = BgrImage::decode(&bytes).map_err(|e| PipelineError::Image(e.to_string()))?;
        self.recognize_image(&image)
    }
}
