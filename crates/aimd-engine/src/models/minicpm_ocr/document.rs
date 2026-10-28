//! One page through the full MiniCPM-OCR pipeline: PP-DocLayoutV3 regions,
//! the SDK region driver ([`super::regions`]) and the recogniser
//! ([`MiniCpmOcrPipeline`], DSpark when a drafter is attached).

use std::path::Path;

use image::RgbImage;

use super::generate::FinishReason;
use super::pipeline::{BatchImage, MiniCpmOcrPipeline, OcrRequest, PipelineError};
use super::regions::{PageOutput, RegionOptions, recognize_page_batch};
use crate::models::pp_doclayout_v3::pipeline::{PipelineError as LayoutError, PpDocLayoutV3};

#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    #[error("{0}")]
    Recognizer(#[from] PipelineError),
    #[error("layout: {0}")]
    Layout(#[from] LayoutError),
    #[error("no text recognized")]
    EmptyContent,
}

/// A recognised page and its pixel size (block boxes are page pixels).
pub struct DocumentPage {
    pub width: u32,
    pub height: u32,
    pub page: PageOutput,
    /// Recogniser calls that hit their token budget instead of EOS.
    pub truncated_regions: usize,
}

/// PP-DocLayoutV3 + MiniCPM-OCR.
pub struct MiniCpmDocument {
    layout: PpDocLayoutV3,
    ocr: MiniCpmOcrPipeline,
}

impl MiniCpmDocument {
    pub fn new(layout: PpDocLayoutV3, ocr: MiniCpmOcrPipeline) -> Self {
        Self { layout, ocr }
    }

    pub fn recognizer(&self) -> &MiniCpmOcrPipeline {
        &self.ocr
    }

    /// Decodes `path` and recognises it as one page.
    pub fn recognize_path(&self, path: &Path) -> Result<DocumentPage, DocumentError> {
        let bytes = std::fs::read(path)
            .map_err(|e| PipelineError::Io(format!("read {}: {e}", path.display())))?;
        let rgb = image::load_from_memory(&bytes)
            .map_err(|e| PipelineError::Image(e.to_string()))?
            .to_rgb8();
        self.recognize(&rgb)
    }

    /// Layout regions in reading order, every region that reaches the VLM
    /// recognised in length-sorted batches (the SDK's per-label prompt,
    /// token budget and slice cap), then the SDK's page assembly. Image
    /// regions emit nothing.
    pub fn recognize(&self, rgb: &RgbImage) -> Result<DocumentPage, DocumentError> {
        let layout = self.layout.detect(rgb)?;
        let mut truncated_regions = 0;
        let page = recognize_page_batch(&layout, rgb, &RegionOptions::default(), |todo| {
            let images: Vec<BatchImage<'_>> = todo
                .iter()
                .map(|(job, recipe)| {
                    let img = &job.image;
                    (
                        img.as_raw().as_slice(),
                        img.width() as usize,
                        img.height() as usize,
                        OcrRequest {
                            task: recipe.prompt,
                            max_tokens: Some(recipe.max_new_tokens),
                            max_slices: Some(recipe.slice_nums),
                            ..OcrRequest::default()
                        },
                    )
                })
                .collect();
            let outs = self.ocr.recognize_batch_rgb8(&images)?;
            Ok::<_, PipelineError>(
                outs.into_iter()
                    .map(|(o, _)| {
                        truncated_regions += usize::from(o.finish_reason != FinishReason::Stop);
                        o.text
                    })
                    .collect(),
            )
        })?;
        if page.markdown.trim().is_empty() {
            return Err(DocumentError::EmptyContent);
        }
        Ok(DocumentPage {
            width: rgb.width(),
            height: rgb.height(),
            page,
            truncated_regions,
        })
    }
}
