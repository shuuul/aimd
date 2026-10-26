//! End-to-end Unlimited-OCR pipeline (WS-07 integration).
//!
//! Wires the WS-07a/07b pieces exactly like mlx-vlm drives the checkpoint
//! from `src/aimd/plugins/ocr/models/mlx.py::_generate_unlimited_ocr_text`:
//!
//! 1. decode the image to RGB8 (PIL `convert("RGB")`) and run gundam
//!    preprocessing ([`preprocess_gundam`]: 1024 global view + 640 local
//!    crops when either side exceeds 640);
//! 2. vision towers + projector ([`OcrVision`]) -> `[n_image, 1280]`
//!    features already shaped in mlx-vlm order: local crops grid (with an
//!    `image_newline` column per row), then the global view (with its
//!    `image_newline` column), then one `view_separator` row;
//! 3. the prompt `<image>document parsing.` (mlx-vlm `SINGLE_IMAGE_TOKEN`
//!    chat format — no newline after `<image>`) tokenized as
//!    `[bos=0] ++ [<image> × n_image] ++ encode("document parsing.")`
//!    ([`build_ocr_prompt_ids`]). `n_image` is the processor's
//!    `tokenize_with_images` count ([`gundam_image_token_count`]), which
//!    must equal the vision row count — every `<image>` position is one
//!    feature row, newline/separator rows included, in one contiguous run
//!    starting at index 1;
//! 4. the feature rows replace the `<image>` embedding rows in order
//!    (`Model.get_input_embeddings`, via
//!    [`splice_image_features`]);
//! 5. greedy generation (temperature 0, `max_tokens` 8192) with the
//!    sliding-window no-repeat n-gram guard (n=35, window=128), EOS = id 1
//!    dropped from the output;
//! 6. `text.strip()` -> empty is a failure -> markdown normalization ->
//!    empty is a failure (the Python `ProcessingFailedError` contract).
//!
//! Token layout verified against the reference processor for the fixtures
//! in `tests/golden/ocr_e2e/` (dumped with mlx-vlm's
//! `UnlimitedOCRProcessor.process_one`): 560x180 -> crop `[1, 1]`, 273
//! image tokens at positions 1..=273, 277 prompt ids; 900x1160 -> crop
//! `[4, 5]` (20 crops), 2323 image tokens at 1..=2323, 2327 prompt ids;
//! both end with `[34030, 76466, 16]` ("document parsing.").

use std::path::Path;
use std::time::{Duration, Instant};

use aimd_mlx::{MlxArray, MlxDtype, MlxError};
use aimd_weights::{WeightsError, load_checkpoint};

use super::config::{UnlimitedOcrConfig, parse_config};
pub use super::generate::LayoutBlock;
use super::generate::{
    FinishReason, GenerateError, OCR_IMAGE_TOKEN_ID, OcrGenerationSettings, greedy_ocr_generate,
    normalize_unlimited_ocr_markdown, parse_unlimited_ocr_blocks, splice_image_features,
};
use super::mlx_decoder::{DecoderError, OcrDecoder};
use super::mlx_vision::{OcrVision, VisionError};
use super::preprocess::{GundamPreprocess, PreprocessError, preprocess_gundam};
use crate::tokenizer::{Tokenizer, TokenizerError, build_ocr_prompt_ids};

/// The single-image gundam prompt (`UNLIMITED_OCR_SINGLE_PROMPT`).
pub const UNLIMITED_OCR_SINGLE_PROMPT: &str = "document parsing.";

/// Gundam global view edge (`base_size`).
const BASE_SIZE: usize = 1024;
/// Gundam crop edge (`image_size`).
const IMAGE_SIZE: usize = 640;
/// Processor `patch_size`.
const PATCH_SIZE: usize = 16;
/// Processor `downsample_ratio`.
const DOWNSAMPLE_RATIO: usize = 4;

/// Pipeline-level failures, chaining the module errors.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("io: {0}")]
    Io(String),
    #[error("image decode: {0}")]
    Image(String),
    #[error("weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("checkpoint config: {0}")]
    Config(#[from] serde_json::Error),
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
    #[error("Unlimited-OCR produced empty content")]
    EmptyContent,
}

/// One recognition request (defaults = the Python adapter's settings).
#[derive(Clone, Copy, Debug)]
pub struct OcrRequest<'a> {
    /// User text after the `<image>` placeholder (`document parsing.`).
    pub prompt: &'a str,
    /// Generation knobs (max tokens 8192, guard n=35 window=128).
    pub settings: OcrGenerationSettings,
}

impl Default for OcrRequest<'_> {
    fn default() -> Self {
        Self {
            prompt: UNLIMITED_OCR_SINGLE_PROMPT,
            settings: OcrGenerationSettings::default(),
        }
    }
}

/// Recognition output for one image.
#[derive(Clone, Debug)]
pub struct OcrResult {
    /// Normalized markdown (`normalize_unlimited_ocr_markdown`).
    pub markdown: String,
    /// Located layout blocks parsed from the det protocol of `raw_text`
    /// (bbox normalized to the original image); their texts joined by
    /// `"\n\n"` equal `markdown`. Empty when the output has no det markers.
    pub blocks: Vec<LayoutBlock>,
    /// Raw decoded generation (special tokens kept, stripped).
    pub raw_text: String,
    pub finish_reason: FinishReason,
    /// Generated token ids (EOS excluded).
    pub tokens: Vec<u32>,
    /// Full prompt length (bos + image tokens + prompt text).
    pub prompt_tokens: usize,
    /// `<image>` positions (= vision feature rows).
    pub image_tokens: usize,
    /// `(width_crop_num, height_crop_num)`; `(1, 1)` = global view only.
    pub crop_ratio: (usize, usize),
    pub timings: OcrTimings,
}

/// Stage wall-clock timings.
#[derive(Clone, Copy, Debug, Default)]
pub struct OcrTimings {
    /// Image decode + gundam preprocessing.
    pub preprocess: Duration,
    /// SAM + CLIP towers + projector + token shaping.
    pub vision: Duration,
    /// Prefill + greedy decode + detokenize.
    pub generate: Duration,
    /// Chunked prefill up to the first sampled token's readback (includes
    /// any still-lazy vision work the prefill forces).
    pub prefill: Duration,
    /// Remaining decode steps.
    pub decode: Duration,
}

/// Number of `<image>` prompt positions the reference processor emits for
/// one gundam image (`UnlimitedOCRProcessor.tokenize_with_images`):
/// `(q_base + 1) * q_base + 1` for the global view plus
/// `(q * w + 1) * (q * h)` when crops exist (`w > 1 || h > 1`).
pub fn gundam_image_token_count(crop_ratio: (usize, usize)) -> usize {
    let q_base = (BASE_SIZE / PATCH_SIZE).div_ceil(DOWNSAMPLE_RATIO);
    let q = (IMAGE_SIZE / PATCH_SIZE).div_ceil(DOWNSAMPLE_RATIO);
    let (w, h) = crop_ratio;
    let mut n = (q_base + 1) * q_base + 1;
    if w > 1 || h > 1 {
        n += (q * w + 1) * (q * h);
    }
    n
}

/// Validates the prompt layout: exactly `n_image` `<image>` ids in a single
/// contiguous run at positions `1..=n_image` after `bos`. Returns the run
/// as `(first, last)`.
pub fn check_prompt_layout(
    prompt_ids: &[u32],
    image_token_id: u32,
    n_image: usize,
) -> Result<(usize, usize), PipelineError> {
    let positions: Vec<usize> = prompt_ids
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == image_token_id)
        .map(|(i, _)| i)
        .collect();
    let (Some(&first), Some(&last)) = (positions.first(), positions.last()) else {
        return Err(PipelineError::Layout("no <image> tokens in prompt".into()));
    };
    if positions.len() != n_image {
        return Err(PipelineError::Layout(format!(
            "{} <image> tokens in prompt, expected {n_image}",
            positions.len()
        )));
    }
    if last - first + 1 != n_image || first != 1 {
        return Err(PipelineError::Layout(format!(
            "<image> run must be contiguous at 1..={n_image}, got {first}..={last}"
        )));
    }
    Ok((first, last))
}

/// A loaded Unlimited-OCR checkpoint ready to recognize images.
pub struct UnlimitedOcrPipeline {
    vision: OcrVision,
    decoder: OcrDecoder,
    tokenizer: Tokenizer,
    image_token_id: u32,
}

impl UnlimitedOcrPipeline {
    /// Load a checkpoint snapshot directory (e.g. the
    /// `mlx-community/Unlimited-OCR-4bit` HF snapshot).
    pub fn load(checkpoint_dir: &Path) -> Result<Self, PipelineError> {
        let raw = std::fs::read(checkpoint_dir.join("config.json"))
            .map_err(|e| PipelineError::Io(format!("read config.json: {e}")))?;
        let cfg: UnlimitedOcrConfig = parse_config(&raw)?;
        let tokenizer = Tokenizer::from_checkpoint(checkpoint_dir)?;
        let image_token_id = tokenizer.token_id("<image>")?;
        if image_token_id != OCR_IMAGE_TOKEN_ID {
            return Err(PipelineError::ConfigMismatch(format!(
                "tokenizer <image> id {image_token_id} != {OCR_IMAGE_TOKEN_ID}"
            )));
        }
        let store = load_checkpoint(checkpoint_dir)?;
        let vision = OcrVision::from_store(&store, &cfg)?;
        let decoder = OcrDecoder::from_store(&store, &cfg)?;
        drop(store);
        if decoder.config().hidden_size != cfg.projector_config.n_embed {
            return Err(PipelineError::ConfigMismatch(format!(
                "decoder hidden {} != projector n_embed {}",
                decoder.config().hidden_size,
                cfg.projector_config.n_embed
            )));
        }
        Ok(Self {
            vision,
            decoder,
            tokenizer,
            image_token_id,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    #[cfg(test)]
    pub(crate) fn vision(&self) -> &OcrVision {
        &self.vision
    }

    #[cfg(test)]
    pub(crate) fn decoder(&self) -> &OcrDecoder {
        &self.decoder
    }

    /// Recognize an image file (PNG/JPEG/... as supported by `image`).
    pub fn recognize_path(
        &self,
        path: &Path,
        request: OcrRequest<'_>,
    ) -> Result<OcrResult, PipelineError> {
        let bytes = std::fs::read(path)
            .map_err(|e| PipelineError::Io(format!("read {}: {e}", path.display())))?;
        self.recognize(&bytes, request)
    }

    /// Recognize encoded image bytes.
    pub fn recognize(
        &self,
        image_bytes: &[u8],
        request: OcrRequest<'_>,
    ) -> Result<OcrResult, PipelineError> {
        let t0 = Instant::now();
        let prep = decode_and_preprocess(image_bytes)?;
        let preprocess = t0.elapsed();
        self.recognize_preprocessed(&prep, request, preprocess)
    }

    /// Prompt ids for a preprocessed image (exposed for layout checks).
    pub fn prompt_ids(
        &self,
        crop_ratio: (usize, usize),
        prompt: &str,
    ) -> Result<Vec<u32>, PipelineError> {
        let n_image = gundam_image_token_count(crop_ratio);
        let ids = build_ocr_prompt_ids(&self.tokenizer, prompt, n_image)?;
        check_prompt_layout(&ids, self.image_token_id, n_image)?;
        Ok(ids)
    }

    /// Recognize an already-preprocessed gundam input.
    pub fn recognize_preprocessed(
        &self,
        prep: &GundamPreprocess,
        request: OcrRequest<'_>,
        preprocess: Duration,
    ) -> Result<OcrResult, PipelineError> {
        let crop_ratio = if prep.crops.is_empty() {
            (1, 1)
        } else {
            prep.crop_ratio
        };
        let t1 = Instant::now();
        let features = self.vision.features_from_preprocessed(prep)?;
        features.eval()?;
        let vision = t1.elapsed();
        let mut out = self.recognize_with_features(crop_ratio, &features, request)?;
        out.timings.preprocess = preprocess;
        out.timings.vision = vision;
        Ok(out)
    }

    /// Vision features for a preprocessed input (`[n_image, 1280]` bf16,
    /// evaluated).
    pub fn vision_features(&self, prep: &GundamPreprocess) -> Result<MlxArray, PipelineError> {
        let feats = self.vision.features_from_preprocessed(prep)?;
        feats.eval()?;
        Ok(feats)
    }

    /// Decoder half of the pipeline over precomputed vision features
    /// `[n_image, hidden]` (any float dtype; cast to bf16) for an image whose gundam
    /// crop grid is `crop_ratio`: prompt layout -> embedding splice ->
    /// guarded greedy generation -> strip + normalization.
    pub fn recognize_with_features(
        &self,
        crop_ratio: (usize, usize),
        features: &MlxArray,
        request: OcrRequest<'_>,
    ) -> Result<OcrResult, PipelineError> {
        let prompt_ids = self.prompt_ids(crop_ratio, request.prompt)?;
        let n_image = gundam_image_token_count(crop_ratio);
        let rows = features.shape()?.first().copied().unwrap_or(0);
        if rows != n_image {
            return Err(PipelineError::Layout(format!(
                "vision produced {rows} feature rows, prompt has {n_image} <image> tokens"
            )));
        }

        let t2 = Instant::now();
        let embeds =
            splice_image_features(&self.decoder, &prompt_ids, features, self.image_token_id)?;
        let (generation, gen_timings) = greedy_ocr_generate(
            &self.decoder,
            &self.tokenizer,
            &prompt_ids,
            &embeds,
            request.settings,
        )?;
        let generate = t2.elapsed();

        let raw_text = generation.text.trim().to_string();
        if raw_text.is_empty() {
            return Err(PipelineError::EmptyContent);
        }
        let markdown = normalize_unlimited_ocr_markdown(&raw_text);
        if markdown.is_empty() {
            return Err(PipelineError::EmptyContent);
        }
        let blocks = parse_unlimited_ocr_blocks(&raw_text);
        Ok(OcrResult {
            markdown,
            blocks,
            raw_text,
            finish_reason: generation.finish_reason,
            tokens: generation.tokens,
            prompt_tokens: prompt_ids.len(),
            image_tokens: n_image,
            crop_ratio,
            timings: OcrTimings {
                generate,
                prefill: gen_timings.prefill,
                decode: gen_timings.decode,
                ..OcrTimings::default()
            },
        })
    }
}

/// Load a raw little-endian bf16 feature dump `[rows, dim]` (the
/// `tests/golden/ocr_e2e/*_features.bf16` reference vision features) as a
/// bf16 MLX array.
pub fn load_bf16_features(path: &Path, dim: usize) -> Result<MlxArray, PipelineError> {
    let raw = std::fs::read(path)
        .map_err(|e| PipelineError::Io(format!("read {}: {e}", path.display())))?;
    if dim == 0 || raw.len() % (2 * dim) != 0 {
        return Err(PipelineError::Layout(format!(
            "{} bytes is not a whole number of bf16 [_, {dim}] rows",
            raw.len()
        )));
    }
    let rows = raw.len() / (2 * dim);
    Ok(MlxArray::from_bytes(&raw, MlxDtype::Bf16, &[rows, dim])?)
}

/// Decode encoded image bytes to RGB8 (PIL `convert("RGB")`) and run the
/// gundam preprocessing.
pub fn decode_and_preprocess(image_bytes: &[u8]) -> Result<GundamPreprocess, PipelineError> {
    let decoded =
        image::load_from_memory(image_bytes).map_err(|e| PipelineError::Image(e.to_string()))?;
    let rgb = decoded.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok(preprocess_gundam(rgb.as_raw(), w, h)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::unlimited_ocr::test_support::{
        Parity, checkpoint_dir, compare_rows, e2e_enabled, mlx_rows_f32, shared_pipeline,
    };
    use std::path::PathBuf;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ocr_e2e")
    }

    fn load_reference(name: &str) -> serde_json::Value {
        let raw = std::fs::read(fixture_dir().join(format!("{name}.json"))).expect("fixture json");
        serde_json::from_slice(&raw).expect("parse fixture")
    }

    /// `difflib.SequenceMatcher`-style similarity proxy: 1 - normalized
    /// char-level Levenshtein distance.
    fn similarity(a: &str, b: &str) -> f64 {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for (i, ca) in a.iter().enumerate() {
            cur[0] = i + 1;
            for (j, cb) in b.iter().enumerate() {
                let sub = prev[j] + usize::from(ca != cb);
                cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        1.0 - prev[b.len()] as f64 / a.len().max(b.len()) as f64
    }

    #[test]
    fn pipeline_image_token_count_formula() {
        // Global view only: (16 + 1) * 16 + 1.
        assert_eq!(gundam_image_token_count((1, 1)), 273);
        // 2x1 crops: 273 + (20 + 1) * 10.
        assert_eq!(gundam_image_token_count((2, 1)), 273 + 210);
        // Reference processor dump for the 900x1160 fixture: [4, 5].
        assert_eq!(gundam_image_token_count((4, 5)), 2323);
        // Tokenizer golden: ocr_document_parsing (640x640) = 273 pads.
        assert_eq!(gundam_image_token_count((1, 1)), 273);
    }

    #[test]
    fn pipeline_prompt_layout_checks() {
        let n = 5;
        let mut ids = vec![0u32];
        ids.extend(std::iter::repeat_n(OCR_IMAGE_TOKEN_ID, n));
        ids.extend([34030, 76466, 16]);
        assert_eq!(
            check_prompt_layout(&ids, OCR_IMAGE_TOKEN_ID, n).unwrap(),
            (1, 5)
        );
        // Wrong count.
        assert!(check_prompt_layout(&ids, OCR_IMAGE_TOKEN_ID, n + 1).is_err());
        // Split run.
        let mut split = ids.clone();
        split[3] = 42;
        split.push(OCR_IMAGE_TOKEN_ID);
        assert!(check_prompt_layout(&split, OCR_IMAGE_TOKEN_ID, n).is_err());
        // No bos before the run.
        assert!(check_prompt_layout(&ids[1..], OCR_IMAGE_TOKEN_ID, n).is_err());
    }

    /// Preprocessing + token-count layout against the mlx-vlm processor dump
    /// (`images_spatial_crop`, crop count, image-token count/positions,
    /// prompt length) — no checkpoint needed.
    #[test]
    fn pipeline_fixture_layout_matches_reference() {
        for name in ["small_text", "page_text"] {
            let r = load_reference(name);
            let bytes = std::fs::read(fixture_dir().join(r["image"].as_str().unwrap())).unwrap();
            let prep = decode_and_preprocess(&bytes).expect("preprocess");
            let crop = if prep.crops.is_empty() {
                (1, 1)
            } else {
                prep.crop_ratio
            };
            let want_crop = r["images_spatial_crop"].as_array().unwrap();
            assert_eq!(
                [crop.0 as u64, crop.1 as u64],
                [
                    want_crop[0].as_u64().unwrap(),
                    want_crop[1].as_u64().unwrap()
                ],
                "{name} crop ratio"
            );
            assert_eq!(
                prep.crops.len() as u64,
                r["num_crops"].as_u64().unwrap(),
                "{name}"
            );
            let n = gundam_image_token_count(crop);
            assert_eq!(n as u64, r["image_token_count"].as_u64().unwrap(), "{name}");
            assert_eq!(r["image_token_first"].as_u64().unwrap(), 1, "{name}");
            assert_eq!(r["image_token_last"].as_u64().unwrap(), n as u64, "{name}");
            assert!(r["image_tokens_contiguous"].as_bool().unwrap(), "{name}");
            let tail = r["prompt_ids_tail"].as_array().unwrap().len();
            assert_eq!(
                (1 + n + tail) as u64,
                r["prompt_ids_len"].as_u64().unwrap(),
                "{name}"
            );
            assert_eq!(r["formatted_prompt"], "<image>document parsing.");
        }
    }

    /// The reference generations' det boxes parse into blocks that join to
    /// the reference markdown and cover every ink pixel of the fixture
    /// (global-view-only 560x180 and the 4x5 gundam-crop 900x1160 page):
    /// evidence for the 0..999 whole-image grid.
    #[test]
    fn pipeline_reference_blocks_cover_fixture_text() {
        for (name, want_blocks) in [("small_text", 3), ("page_text", 7)] {
            let r = load_reference(name);
            let blocks = parse_unlimited_ocr_blocks(r["raw_text"].as_str().unwrap());
            assert_eq!(blocks.len(), want_blocks, "{name}");
            let joined: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
            assert_eq!(
                joined.join("\n\n"),
                r["markdown"].as_str().unwrap(),
                "{name}"
            );

            let bytes = std::fs::read(fixture_dir().join(r["image"].as_str().unwrap())).unwrap();
            let img = image::load_from_memory(&bytes).unwrap().to_luma8();
            let (w, h) = (img.width() as f32, img.height() as f32);
            // Antialiased glyph edges may poke a pixel or two past a box.
            let margin = 3.0;
            let (mut ink, mut outside) = (0usize, 0usize);
            for (x, y, p) in img.enumerate_pixels() {
                if p.0[0] >= 128 {
                    continue;
                }
                ink += 1;
                let (x, y) = (x as f32 + 0.5, y as f32 + 0.5);
                let covered = blocks.iter().any(|b| {
                    x >= b.bbox[0] * w - margin
                        && x <= b.bbox[2] * w + margin
                        && y >= b.bbox[1] * h - margin
                        && y <= b.bbox[3] * h + margin
                });
                outside += usize::from(!covered);
            }
            eprintln!("{name}: {outside}/{ink} ink pixels outside the blocks");
            assert!(ink > 0 && outside == 0, "{name}: {outside}/{ink} outside");
        }
    }

    /// Tokenizer-backed prompt ids vs the reference processor's input_ids
    /// (head, tail, length). Needs only the tokenizer files.
    #[test]
    fn pipeline_prompt_ids_match_reference() {
        let Some(dir) = checkpoint_dir() else {
            eprintln!("skipping: Unlimited-OCR-4bit not in HF cache");
            return;
        };
        let tok = Tokenizer::from_checkpoint(&dir).expect("tokenizer");
        for name in ["small_text", "page_text"] {
            let r = load_reference(name);
            let n = gundam_image_token_count(reference_crop(&r));
            let ids = build_ocr_prompt_ids(&tok, UNLIMITED_OCR_SINGLE_PROMPT, n).unwrap();
            check_prompt_layout(&ids, OCR_IMAGE_TOKEN_ID, n).unwrap();
            let as_u32 = |v: &serde_json::Value| -> Vec<u32> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u32)
                    .collect()
            };
            assert_eq!(
                ids.len() as u64,
                r["prompt_ids_len"].as_u64().unwrap(),
                "{name}"
            );
            assert_eq!(ids[..3], as_u32(&r["prompt_ids_head"])[..], "{name}");
            assert_eq!(ids[1 + n..], as_u32(&r["prompt_ids_tail"])[..], "{name}");
        }
    }

    fn reference_crop(r: &serde_json::Value) -> (usize, usize) {
        let c = r["images_spatial_crop"].as_array().unwrap();
        (
            c[0].as_u64().unwrap() as usize,
            c[1].as_u64().unwrap() as usize,
        )
    }

    fn reference_tokens(r: &serde_json::Value) -> Vec<u32> {
        r["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .filter(|&t| t != 1)
            .collect()
    }

    fn same_prefix(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// Our vision features for the small fixture vs the mlx-vlm reference
    /// rows spliced by `Model.get_input_embeddings` (no generation).
    #[test]
    fn pipeline_vision_matches_reference_features() {
        let Some(pipe) = shared_pipeline() else {
            return;
        };
        let r = load_reference("small_text");
        let bytes = std::fs::read(fixture_dir().join(r["image"].as_str().unwrap())).unwrap();
        let prep = decode_and_preprocess(&bytes).unwrap();
        let ours = pipe.vision_features(&prep).unwrap();
        let want =
            load_bf16_features(&fixture_dir().join("small_text_features.bf16"), 1280).unwrap();
        let (got, cols) = mlx_rows_f32(&ours);
        let (want, _) = mlx_rows_f32(&want);
        let rows: Vec<usize> = (0..want.len() / cols).collect();
        assert_eq!(got.len(), want.len(), "feature rows");
        let mut fails = Vec::new();
        compare_rows(
            "small_text vision features",
            &got,
            cols,
            &rows,
            &want,
            Parity {
                rel_rms: 0.03,
                max_frac: 0.08,
            },
            &mut fails,
        );
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    /// Full-stack OCR of the rendered fixtures vs the mlx-vlm reference
    /// output: finish = stop, markdown similarity, token agreement. One
    /// generation per fixture, gated behind AIMD_E2E=1
    /// (`AIMD_OCR_E2E_FIXTURES=small_text,page_text` narrows the set).
    /// `AIMD_OCR_REF_FEATURES=1` additionally decodes small_text from the
    /// reference vision features (isolates the decoder from vision drift).
    #[test]
    fn pipeline_ocr_fixture_end_to_end() {
        if !e2e_enabled() {
            eprintln!("skipping: set AIMD_E2E=1 to run the OCR pipeline test");
            return;
        }
        let Some(pipe) = shared_pipeline() else {
            return;
        };
        let names: Vec<String> = std::env::var("AIMD_OCR_E2E_FIXTURES")
            .map(|s| s.split(',').map(str::to_string).collect())
            .unwrap_or_else(|_| vec!["small_text".to_string(), "page_text".to_string()]);
        for name in &names {
            let r = load_reference(name);
            let path = fixture_dir().join(r["image"].as_str().unwrap());
            let out = pipe
                .recognize_path(&path, OcrRequest::default())
                .expect("recognize");
            let want = r["markdown"].as_str().unwrap();
            let sim = similarity(&out.markdown, want);
            let ref_tokens = reference_tokens(&r);
            eprintln!(
                "{name}: finish={} sim={sim:.4} tokens {}/{} (identical prefix {}) timings={:?}",
                out.finish_reason,
                out.tokens.len(),
                ref_tokens.len(),
                same_prefix(&out.tokens, &ref_tokens),
                out.timings,
            );
            let k = same_prefix(&out.tokens, &ref_tokens);
            if k < ref_tokens.len() {
                let tok = pipe.tokenizer();
                let show = |t: &[u32]| tok.decode(&t[k..(k + 6).min(t.len())], false).unwrap();
                eprintln!(
                    "{name}: first divergence at {k}: ours {:?} vs ref {:?}",
                    show(&out.tokens),
                    show(&ref_tokens)
                );
            }
            assert_eq!(out.finish_reason, FinishReason::Stop, "{name}");
            let joined: Vec<&str> = out.blocks.iter().map(|b| b.text.as_str()).collect();
            assert_eq!(joined.join("\n\n"), out.markdown, "{name}: blocks");
            let want_blocks = parse_unlimited_ocr_blocks(r["raw_text"].as_str().unwrap());
            assert_eq!(out.blocks.len(), want_blocks.len(), "{name}: block count");
            for (got, want) in out.blocks.iter().zip(&want_blocks) {
                assert_eq!(got.kind, want.kind, "{name}");
                let drift = (0..4)
                    .map(|i| (got.bbox[i] - want.bbox[i]).abs())
                    .fold(0f32, f32::max);
                assert!(drift <= 0.02, "{name}: bbox {:?} vs {:?}", got, want);
            }
            assert_eq!(
                out.image_tokens as u64,
                r["image_token_count"].as_u64().unwrap()
            );
            assert!(
                sim >= 0.98,
                "{name}: similarity {sim:.4} < 0.98\n{}",
                out.markdown
            );
        }
        if std::env::var("AIMD_OCR_REF_FEATURES").as_deref() == Ok("1") {
            let r = load_reference("small_text");
            let feats =
                load_bf16_features(&fixture_dir().join("small_text_features.bf16"), 1280).unwrap();
            let out = pipe
                .recognize_with_features(reference_crop(&r), &feats, OcrRequest::default())
                .expect("recognize");
            let ref_tokens = reference_tokens(&r);
            let sim = similarity(&out.markdown, r["markdown"].as_str().unwrap());
            eprintln!(
                "small_text (reference features): finish={} sim={sim:.4} tokens {}/{} (identical prefix {})",
                out.finish_reason,
                out.tokens.len(),
                ref_tokens.len(),
                same_prefix(&out.tokens, &ref_tokens),
            );
            assert_eq!(out.finish_reason, FinishReason::Stop);
            assert!(sim >= 0.98, "reference features: similarity {sim:.4}");
        }
    }

    #[test]
    fn pipeline_similarity_helper() {
        assert_eq!(similarity("abc", "abc"), 1.0);
        assert!((similarity("abcd", "abce") - 0.75).abs() < 1e-9);
        assert_eq!(similarity("", ""), 1.0);
    }
}
