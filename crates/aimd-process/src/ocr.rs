//! OCR for images and PDFs (ports `aimd.plugins.ocr._plugin` and
//! `aimd.plugins.ocr.backends`). PDF pages are rendered with `mutool draw`
//! at 144 DPI RGB (PyMuPDF `Matrix(2.0, 2.0)`, `alpha=False`).

use std::path::{Path, PathBuf};

use aimd_core::ext::{IMAGE_FILE_EXTENSIONS, suffix_lower};
use aimd_core::remote::resolve_remote_backend_with;
use aimd_core::{
    AimdError, ProcessHooks, ProcessInput, RemoteBackendConfig, RemoteModality, Result, Segment,
    anchors, log_warn,
};

use crate::models::{OcrFamily, ensure_native_ocr, ocr_family, resolve_mlx_ocr_model};
use crate::tempdir::TempDir;
use crate::{LocalModality, LocatedOcr, OcrFailure, Runtime, remote};

/// PyMuPDF `Matrix(2.0, 2.0)` = 2 x 72 DPI.
pub const RENDER_DPI: u32 = 144;

/// One recognized page (`page_index` is `None` for single images).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrPage {
    pub page_index: Option<usize>,
    pub text: String,
}

/// OCR output: the input stem and the assembled Markdown.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrOutput {
    pub title: String,
    pub pages: Vec<OcrPage>,
    pub markdown: String,
    /// Located blocks/lines (or one page-level segment per page) when
    /// `ProcessInput::wants_positions`.
    pub segments: Option<Vec<Segment>>,
}

/// The whole page as a box.
const FULL_PAGE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Anchored page Markdown and its segments (`page` is 1-based): layout
/// blocks get one anchor each; otherwise the page gets a page-level
/// anchor, and line positions (if any) are segments only.
pub fn locate_page(page: usize, ocr: LocatedOcr) -> (String, Vec<Segment>) {
    let at_page = |s: Segment| Segment {
        page: Some(page),
        ..s
    };
    let lines: Vec<Segment> = ocr
        .lines
        .unwrap_or_default()
        .into_iter()
        .map(at_page)
        .collect();
    match ocr.blocks.filter(|b| !b.is_empty()) {
        Some(blocks) => {
            let mut blocks: Vec<Segment> = blocks.into_iter().map(at_page).collect();
            let text = anchors::located_markdown(&blocks);
            blocks.extend(lines);
            (text, blocks)
        }
        None => {
            let text = ocr.markdown.trim().to_string();
            let anchored = format!("{}\n{text}", anchors::loc_comment(page, FULL_PAGE, "page"));
            let segments = if lines.is_empty() {
                vec![Segment {
                    text,
                    kind: Some("page".into()),
                    page: Some(page),
                    bbox: Some(FULL_PAGE),
                    ..Segment::default()
                }]
            } else {
                lines
            };
            (anchored, segments)
        }
    }
}

enum Engine {
    Remote(RemoteBackendConfig),
    Local(String),
}

/// `_page_to_markdown`.
pub fn page_to_markdown(page: &OcrPage, total_pages: usize) -> String {
    let text = page.text.trim();
    match page.page_index {
        Some(index) if total_pages > 1 => format!("## Page {}\n\n{text}", index + 1)
            .trim()
            .to_string(),
        _ => text.to_string(),
    }
}

/// `_ocr_result_to_markdown`: non-empty page chunks joined by blank lines.
pub fn pages_to_markdown(pages: &[OcrPage]) -> Result<String> {
    let chunks: Vec<String> = pages
        .iter()
        .map(|p| page_to_markdown(p, pages.len()))
        .filter(|c| !c.is_empty())
        .collect();
    if chunks.is_empty() {
        return Err(AimdError::failed("OCR returned empty content"));
    }
    Ok(chunks.join("\n\n"))
}

/// Validate, recognize, and assemble one OCR input.
pub fn recognize_file(
    rt: &dyn Runtime,
    path: &Path,
    request: &ProcessInput,
    hooks: &dyn ProcessHooks,
) -> Result<OcrOutput> {
    if !path.exists() {
        return Err(AimdError::not_found(format!(
            "Input file not found: {}",
            path.display()
        )));
    }
    if request.start.is_some_and(|s| s < 0) {
        return Err(AimdError::failed(
            "OCR start page must be greater than or equal to 0.",
        ));
    }
    if request.end.is_some_and(|e| e < 0) {
        return Err(AimdError::failed(
            "OCR end page must be greater than or equal to 0.",
        ));
    }
    if let (Some(s), Some(e)) = (request.start, request.end)
        && s > e
    {
        return Err(AimdError::failed(
            "OCR start page must be less than or equal to end page.",
        ));
    }
    if hooks.is_cancelled() {
        return Err(AimdError::cancelled("OCR cancelled before model inference"));
    }
    let remote_config = resolve_remote_backend_with(
        RemoteModality::Ocr,
        request.ocr_base_url.as_deref(),
        request.ocr_model.as_deref(),
        request.ocr_api_key.as_deref(),
        &|key| rt.env(key),
    )?;
    let engine = match remote_config {
        Some(config) => {
            if request.precision.is_some() {
                log_warn!(
                    "Ignoring OCR precision for remote inference; server-side weights already \
                     define their precision."
                );
            }
            Engine::Remote(config)
        }
        None => {
            let model_id =
                resolve_mlx_ocr_model(request.model.as_deref(), request.precision.as_deref())?;
            ensure_native_ocr(&model_id)?;
            if ocr_family(&model_id) == OcrFamily::PpOcrV6 && request.precision.is_some() {
                log_warn!("Ignoring OCR precision for PP-OCRv6; its checkpoints are dense f32.");
            }
            rt.preflight_local(LocalModality::Ocr)?;
            Engine::Local(model_id)
        }
    };

    let suffix = suffix_lower(path);
    let spotting = request.include_segments;
    let anchored = request.anchors;
    let mut segments = request.wants_positions().then(Vec::new);
    // Page text, anchored when requested (`page` is 1-based).
    let mut place = |page: usize, ocr: LocatedOcr| match &mut segments {
        Some(all) => {
            let plain = (!anchored).then(|| ocr.markdown.clone());
            let (text, located) = locate_page(page, ocr);
            all.extend(located);
            plain.unwrap_or(text)
        }
        None => ocr.markdown,
    };
    let pages = if IMAGE_FILE_EXTENSIONS.contains(&suffix.as_str()) {
        let ocr = recognize_one(rt, &engine, path, false, spotting)?;
        hooks.progress("ocr", Some(1), Some(1), Some("Processed OCR page 1 of 1"));
        vec![OcrPage {
            page_index: None,
            text: place(1, ocr).trim().to_string(),
        }]
    } else if suffix == ".pdf" {
        let rendered = render_pdf_pages(
            rt,
            path,
            request.start,
            request.end,
            request.temp_dir.as_deref(),
        )?;
        let total = rendered.pages.len();
        let mut pages = Vec::with_capacity(total);
        for (i, (page_index, image)) in rendered.pages.iter().enumerate() {
            if i > 0 && hooks.is_cancelled() {
                return Err(AimdError::cancelled("OCR cancelled between pages"));
            }
            let ocr = recognize_one(rt, &engine, image, true, spotting)?;
            let text = place(page_index + 1, ocr);
            hooks.progress(
                "ocr",
                Some(i as u64 + 1),
                Some(total as u64),
                Some(&format!("Processed OCR page {} of {total}", i + 1)),
            );
            pages.push(OcrPage {
                page_index: Some(*page_index),
                text,
            });
        }
        pages
    } else {
        return Err(AimdError::failed(match engine {
            Engine::Remote(_) => "Remote OCR supports image files and PDFs only.",
            Engine::Local(_) => "MLX VLM OCR supports image files and PDFs only.",
        }));
    };
    if !pages.iter().any(|p| !p.text.trim().is_empty()) {
        return Err(AimdError::failed("OCR returned empty content"));
    }
    let markdown = pages_to_markdown(&pages)?;
    Ok(OcrOutput {
        title: crate::file_stem(path),
        pages,
        markdown,
        segments,
    })
}

fn recognize_one(
    rt: &dyn Runtime,
    engine: &Engine,
    image: &Path,
    multi_page: bool,
    spotting: bool,
) -> Result<LocatedOcr> {
    match engine {
        Engine::Remote(config) => Ok(LocatedOcr {
            markdown: remote::recognize_image(config, image, multi_page)?,
            ..LocatedOcr::default()
        }),
        Engine::Local(model_id) => {
            let model = rt.load_ocr(model_id)?;
            let family = ocr_family(model_id).name();
            match model.recognize_located(image, spotting) {
                Ok(ocr) if !ocr.markdown.trim().is_empty() => Ok(ocr),
                Ok(_) | Err(OcrFailure::Empty) => Err(AimdError::failed(format!(
                    "{family} produced empty content"
                ))),
                Err(OcrFailure::Other(e)) => {
                    Err(AimdError::failed(format!("{family} inference failed: {e}")))
                }
            }
        }
    }
}

/// Rendered page images (0-based page index, PNG path) in a temp dir.
#[derive(Debug)]
pub struct RenderedPages {
    pub pages: Vec<(usize, PathBuf)>,
    _dir: TempDir,
}

/// Render the requested 0-based inclusive range with `mutool draw`.
pub fn render_pdf_pages(
    rt: &dyn Runtime,
    input: &Path,
    start: Option<i64>,
    end: Option<i64>,
    temp_dir: Option<&Path>,
) -> Result<RenderedPages> {
    let page_count = i64::from(rt.pdf_page_count(input)?);
    if page_count == 0 {
        return Err(AimdError::failed(format!(
            "PDF has no pages: {}",
            input.display()
        )));
    }
    let first = start.unwrap_or(0);
    let last = end.unwrap_or(page_count - 1);
    if first < 0 || last < first || last >= page_count {
        return Err(AimdError::failed(format!(
            "Invalid OCR page range {first}-{last} for {} with {page_count} pages.",
            input.display()
        )));
    }
    let Some(mutool) = rt.which("mutool") else {
        return Err(AimdError::backend_unavailable(
            "PDF OCR requires mutool (MuPDF). Install it with `brew install mupdf`, then retry.",
        ));
    };
    let dir = TempDir::new(temp_dir, "aimd-mlx-vlm-pdf-")?;
    let template = dir.path().join("page_%04d.png");
    let args = vec![
        "draw".to_string(),
        "-q".to_string(),
        "-o".to_string(),
        template.to_string_lossy().into_owned(),
        "-F".to_string(),
        "png".to_string(),
        "-r".to_string(),
        RENDER_DPI.to_string(),
        "-c".to_string(),
        "rgb".to_string(),
        input.to_string_lossy().into_owned(),
        format!("{}-{}", first + 1, last + 1),
    ];
    let out = rt
        .run(&mutool, &args, None)
        .map_err(|e| AimdError::failed(format!("mutool draw failed: {e}")))?;
    if !out.success() {
        return Err(AimdError::failed(format!(
            "mutool draw failed for {}: {}",
            input.display(),
            out.stderr_text().trim()
        )));
    }
    let mut pages = Vec::new();
    for index in first..=last {
        let image = dir.path().join(format!("page_{:04}.png", index + 1));
        if !image.is_file() {
            return Err(AimdError::failed(format!(
                "mutool draw did not render page {} of {}",
                index + 1,
                input.display()
            )));
        }
        pages.push((index as usize, image));
    }
    Ok(RenderedPages { pages, _dir: dir })
}
