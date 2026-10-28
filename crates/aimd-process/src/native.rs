//! The real [`Runtime`]: system processes, pdf-inspector, and per-process
//! cached MLX pipelines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

use aimd_core::{AimdError, Result, Segment, anchors, log_info, log_warn};
use aimd_engine::models::minicpm_ocr::document::{DocumentError as MiniCpmError, MiniCpmDocument};
use aimd_engine::models::minicpm_ocr::dspark::DSPARK_REPO;
use aimd_engine::models::minicpm_ocr::pipeline::MiniCpmOcrPipeline;
use aimd_engine::models::paddleocr_vl::pipeline::{
    PaddleOcrVlPipeline, PipelineError as VlPipelineError, VlRequest, VlTask, load_rgb8,
};
use aimd_engine::models::pp_doclayout_v3::pipeline::PpDocLayoutV3;
use aimd_engine::models::pp_ocrv6::pipeline::{PipelineError as PpPipelineError, PpOcrV6Pipeline};
use aimd_engine::models::qwen3_aligner::pipeline::ForcedAlignerPipeline;
use aimd_engine::models::qwen3_asr::pipeline::{
    PipelineError as AsrPipelineError, Qwen3AsrPipeline, Transcription, TranscriptionRequest,
};
use aimd_engine::models::unlimited_ocr::pipeline::{
    OcrRequest, PipelineError as OcrPipelineError, UnlimitedOcrPipeline,
};

use crate::models::{
    MINICPM_OCR_REPO, OcrFamily, PADDLEOCR_VL_REPO, PP_DOCLAYOUT_V3_REPO, ocr_family,
};
use crate::{
    AlignedTranscript, AsrFailure, AsrModel, CommandOutput, LocalModality, LocatedOcr, OcrFailure,
    OcrModel, Runtime,
};

/// Native runtime (real system, cached pipelines).
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeRuntime;

impl Runtime for NativeRuntime {
    fn url_transcript(
        &self,
        request: &aimd_core::ProcessInput,
        hooks: &dyn aimd_core::ProcessHooks,
    ) -> Result<crate::UrlOutput> {
        crate::convert_url(self, request, hooks)
    }

    fn preflight_local(&self, modality: LocalModality) -> Result<()> {
        let what = match modality {
            LocalModality::Asr => ("ASR", "AIMD_ASR_BASE_URL"),
            LocalModality::Ocr => ("OCR", "AIMD_OCR_BASE_URL"),
        };
        match aimd_mlx::device_is_cpu() {
            Ok(false) => Ok(()),
            Ok(true) => Err(AimdError::backend_unavailable(format!(
                "Local {} inference requires the MLX Metal GPU device, but the CPU device is \
                 active (AIMD_MLX_DEVICE=cpu or no Metal GPU). Run on Apple Silicon with Metal \
                 or set {} for a remote backend.",
                what.0, what.1
            ))),
            Err(e) => Err(AimdError::backend_unavailable(format!(
                "MLX is unavailable: {e}"
            ))),
        }
    }

    fn load_asr(&self, model_id: &str) -> Result<Arc<dyn AsrModel>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<NativeAsr>>>> = OnceLock::new();
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(model) = cache.get(model_id) {
            return Ok(model.clone());
        }
        let dir = crate::hf::resolve_checkpoint(model_id, &|k| self.env(k))?;
        log_info!(
            "Loading Qwen3-ASR checkpoint {model_id} from {}",
            dir.display()
        );
        let pipeline = Qwen3AsrPipeline::load(&dir).map_err(|e| {
            AimdError::backend_unavailable(format!(
                "Unable to load MLX ASR model '{model_id}': {e}"
            ))
        })?;
        let model = Arc::new(NativeAsr(pipeline));
        cache.insert(model_id.to_string(), model.clone());
        Ok(model)
    }

    fn load_ocr(&self, model_id: &str) -> Result<Arc<dyn OcrModel>> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<NativeOcr>>>> = OnceLock::new();
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(model) = cache.get(model_id) {
            return Ok(model.clone());
        }
        let family = ocr_family(model_id);
        let load_err = |e: String| {
            AimdError::backend_unavailable(format!(
                "Unable to load MLX VLM OCR model '{model_id}': {e}"
            ))
        };
        let model = match family {
            OcrFamily::UnlimitedOcr | OcrFamily::UnlimitedOcr2 => {
                let dir = crate::hf::resolve_checkpoint(model_id, &|k| self.env(k))?;
                log_info!(
                    "Loading {} checkpoint {model_id} from {}",
                    family.name(),
                    dir.display()
                );
                let pipeline =
                    UnlimitedOcrPipeline::load(&dir).map_err(|e| load_err(e.to_string()))?;
                Arc::new(NativeOcr::Unlimited(pipeline))
            }
            OcrFamily::PaddleOcrVl { quant_bits } => {
                let dir = crate::hf::resolve_checkpoint(PADDLEOCR_VL_REPO, &|k| self.env(k))?;
                log_info!(
                    "Loading {} checkpoint {model_id} from {}",
                    family.name(),
                    dir.display()
                );
                let pipeline = PaddleOcrVlPipeline::load(&dir, quant_bits)
                    .map_err(|e| load_err(e.to_string()))?;
                Arc::new(NativeOcr::PaddleVl(pipeline))
            }
            OcrFamily::PpOcrV6 => {
                let det =
                    crate::hf::resolve_checkpoint(&format!("{model_id}_det_safetensors"), &|k| {
                        self.env(k)
                    })?;
                let rec =
                    crate::hf::resolve_checkpoint(&format!("{model_id}_rec_safetensors"), &|k| {
                        self.env(k)
                    })?;
                log_info!(
                    "Loading {} checkpoints {model_id} from {} and {}",
                    family.name(),
                    det.display(),
                    rec.display()
                );
                let pipeline =
                    PpOcrV6Pipeline::load(&det, &rec).map_err(|e| load_err(e.to_string()))?;
                Arc::new(NativeOcr::PpOcr(pipeline))
            }
            OcrFamily::MiniCpmOcr { quant_bits } => {
                let resolve = |repo: &str| crate::hf::resolve_checkpoint(repo, &|k| self.env(k));
                let (dir, draft, layout) = (
                    resolve(MINICPM_OCR_REPO)?,
                    resolve(DSPARK_REPO)?,
                    resolve(PP_DOCLAYOUT_V3_REPO)?,
                );
                log_info!(
                    "Loading {} checkpoint {model_id} from {} (DSpark draft {}, layout {})",
                    family.name(),
                    dir.display(),
                    draft.display(),
                    layout.display()
                );
                let mut ocr = MiniCpmOcrPipeline::load(&dir, quant_bits)
                    .map_err(|e| load_err(e.to_string()))?;
                ocr.attach_drafter(&draft)
                    .map_err(|e| load_err(e.to_string()))?;
                let layout = PpDocLayoutV3::load(&layout).map_err(|e| load_err(e.to_string()))?;
                Arc::new(NativeOcr::MiniCpm(Box::new(MiniCpmDocument::new(
                    layout, ocr,
                ))))
            }
        };
        cache.insert(model_id.to_string(), model.clone());
        Ok(model)
    }
}

struct NativeAsr(Qwen3AsrPipeline);

/// Forced-aligner checkpoint for transcript timestamps (spec 005).
pub const ALIGNER_MODEL: &str = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit";

/// The process-wide aligner, loaded on first use; `None` (logged once)
/// when it cannot be loaded, so transcripts fall back to segment times.
fn aligner() -> Option<&'static ForcedAlignerPipeline> {
    static ALIGNER: OnceLock<Option<ForcedAlignerPipeline>> = OnceLock::new();
    ALIGNER
        .get_or_init(|| {
            let rt = NativeRuntime;
            let loaded = crate::hf::resolve_checkpoint(ALIGNER_MODEL, &|k| rt.env(k))
                .map_err(|e| e.message)
                .and_then(|dir| {
                    log_info!(
                        "Loading forced aligner {ALIGNER_MODEL} from {}",
                        dir.display()
                    );
                    ForcedAlignerPipeline::load(&dir).map_err(|e| e.to_string())
                });
            match loaded {
                Ok(pipeline) => Some(pipeline),
                Err(e) => {
                    log_warn!(
                        "Forced aligner unavailable ({e}); transcript times are segment-level."
                    );
                    None
                }
            }
        })
        .as_ref()
}

impl NativeAsr {
    fn detailed(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<Transcription, AsrFailure> {
        let request = TranscriptionRequest {
            language,
            context: context.filter(|c| !c.is_empty()),
        };
        match self.0.transcribe_detailed(wav_path, request) {
            Ok(t) => Ok(t),
            Err(AsrPipelineError::EmptyTranscription) => Err(AsrFailure::Empty),
            Err(e) => Err(AsrFailure::Other(e.to_string())),
        }
    }
}

impl AsrModel for NativeAsr {
    fn transcribe(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<String, AsrFailure> {
        self.detailed(wav_path, language, context).map(|t| t.text)
    }

    fn transcribe_aligned(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<AlignedTranscript, AsrFailure> {
        let t = self.detailed(wav_path, language, context)?;
        // The forced language, else the one Qwen3-ASR detected.
        let units = match language.or(t.language.as_deref()) {
            None => {
                log_info!("No ASR language detected; transcript times are segment-level.");
                None
            }
            Some(lang) => aligner().and_then(|a| match a.align_path(wav_path, &t.text, lang) {
                Ok(units) if !units.is_empty() => Some(
                    units
                        .into_iter()
                        .map(|u| (u.text, f64::from(u.start), f64::from(u.end)))
                        .collect(),
                ),
                Ok(_) => None,
                Err(e) => {
                    log_info!("Forced alignment skipped ({e}); using segment-level times.");
                    None
                }
            }),
        };
        Ok(AlignedTranscript {
            text: t.text,
            units,
        })
    }
}

enum NativeOcr {
    Unlimited(UnlimitedOcrPipeline),
    PaddleVl(PaddleOcrVlPipeline),
    PpOcr(PpOcrV6Pipeline),
    MiniCpm(Box<MiniCpmDocument>),
}

impl OcrModel for NativeOcr {
    fn recognize(&self, image_path: &Path) -> std::result::Result<String, OcrFailure> {
        self.recognize_located(image_path, false)
            .map(|ocr| ocr.markdown)
    }

    fn pages_per_request(&self) -> usize {
        match self {
            Self::Unlimited(p) => p.max_pages(),
            _ => 1,
        }
    }

    fn recognize_pages(&self, pages: &[&[u8]]) -> std::result::Result<Vec<LocatedOcr>, OcrFailure> {
        let Self::Unlimited(p) = self else {
            return Err(OcrFailure::Other(
                "multi-page OCR requests are not supported".into(),
            ));
        };
        let result = match p.recognize_pages(pages, OcrRequest::default()) {
            Ok(r) => r,
            Err(OcrPipelineError::EmptyContent) => return Err(OcrFailure::Empty),
            Err(e) => return Err(OcrFailure::Other(e.to_string())),
        };
        // Items belong to the page of their first block (a chain that
        // continues onto later pages stays whole on its first page).
        let mut out = vec![Vec::new(); pages.len()];
        for item in result.items {
            let page = item.page.clamp(1, pages.len()) - 1;
            out[page].push(Segment {
                text: item.text,
                kind: Some(item.kind),
                bbox: Some(item.bbox.unwrap_or([0.0, 0.0, 1.0, 1.0])),
                ..Segment::default()
            });
        }
        Ok(out
            .into_iter()
            .map(|blocks: Vec<Segment>| {
                let texts: Vec<&str> = blocks.iter().map(|b| b.text.as_str()).collect();
                LocatedOcr {
                    markdown: texts.join("\n\n"),
                    blocks: Some(blocks),
                    lines: None,
                }
            })
            .collect())
    }

    fn recognize_located(
        &self,
        image_path: &Path,
        spotting: bool,
    ) -> std::result::Result<LocatedOcr, OcrFailure> {
        let segment = |text: &str, kind: &str| Segment {
            text: text.to_string(),
            kind: Some(kind.into()),
            ..Segment::default()
        };
        match self {
            Self::Unlimited(p) => match p.recognize_path(image_path, OcrRequest::default()) {
                Ok(result) => Ok(LocatedOcr {
                    blocks: Some(
                        result
                            .blocks
                            .iter()
                            .map(|b| Segment {
                                bbox: Some(b.bbox),
                                ..segment(&b.text, &b.kind)
                            })
                            .collect(),
                    ),
                    markdown: result.markdown,
                    lines: None,
                }),
                Err(OcrPipelineError::EmptyContent) => Err(OcrFailure::Empty),
                Err(e) => Err(OcrFailure::Other(e.to_string())),
            },
            Self::PaddleVl(p) => {
                // Both passes share one decode of the page image.
                let (rgb, w, h) =
                    load_rgb8(image_path).map_err(|e| OcrFailure::Other(e.to_string()))?;
                let run = |task| match p.recognize_rgb8(
                    &rgb,
                    w,
                    h,
                    VlRequest {
                        task,
                        ..VlRequest::default()
                    },
                ) {
                    Ok(result) => Ok(result),
                    Err(VlPipelineError::EmptyContent) => Err(OcrFailure::Empty),
                    Err(e) => Err(OcrFailure::Other(e.to_string())),
                };
                let markdown = run(VlTask::Ocr)?.markdown;
                // Spotting loses tables/structure, so it only adds line
                // positions; an empty spotting pass is not an error.
                let lines = match spotting {
                    true => match run(VlTask::Spotting) {
                        Ok(out) => Some(
                            out.spots
                                .iter()
                                .map(|s| Segment {
                                    quad: Some(s.quad),
                                    bbox: Some(anchors::quad_bbox(&s.quad)),
                                    ..segment(&s.text, "line")
                                })
                                .collect(),
                        ),
                        Err(OcrFailure::Empty) => None,
                        Err(e) => return Err(e),
                    },
                    false => None,
                };
                Ok(LocatedOcr {
                    markdown,
                    blocks: None,
                    lines,
                })
            }
            Self::PpOcr(p) => match p.recognize_path(image_path) {
                Ok(result) => {
                    let (w, h) = (result.width.max(1) as f32, result.height.max(1) as f32);
                    let lines = result
                        .lines
                        .iter()
                        .map(|l| {
                            let quad = l
                                .quad
                                .map(|[x, y]| [(x / w).clamp(0.0, 1.0), (y / h).clamp(0.0, 1.0)]);
                            Segment {
                                quad: Some(quad),
                                bbox: Some(anchors::quad_bbox(&quad)),
                                score: Some(l.score),
                                ..segment(&l.text, "line")
                            }
                        })
                        .collect();
                    Ok(LocatedOcr {
                        markdown: result.markdown,
                        blocks: None,
                        lines: Some(lines),
                    })
                }
                Err(PpPipelineError::EmptyContent) => Err(OcrFailure::Empty),
                Err(e) => Err(OcrFailure::Other(e.to_string())),
            },
            Self::MiniCpm(d) => match d.recognize_path(image_path) {
                Ok(out) => {
                    let (w, h) = (out.width.max(1) as f32, out.height.max(1) as f32);
                    let blocks = out
                        .page
                        .blocks
                        .iter()
                        .filter_map(|b| {
                            let text = b.markdown.as_deref()?;
                            let [x0, y0, x1, y1] = b.bbox.map(|v| v as f32);
                            let bbox = [x0 / w, y0 / h, x1 / w, y1 / h].map(|v| v.clamp(0.0, 1.0));
                            Some(Segment {
                                bbox: Some(bbox),
                                ..segment(text, b.label)
                            })
                        })
                        .collect();
                    Ok(LocatedOcr {
                        markdown: out.page.markdown,
                        blocks: Some(blocks),
                        lines: None,
                    })
                }
                Err(MiniCpmError::EmptyContent) => Err(OcrFailure::Empty),
                Err(e) => Err(OcrFailure::Other(e.to_string())),
            },
        }
    }
}

/// `shutil.which` over a `PATH` string.
pub(crate) fn which_in_path(program: &str, path: Option<&str>) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let is_exec = |p: &Path| {
        std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    };
    if program.contains('/') {
        let p = PathBuf::from(program);
        return is_exec(&p).then_some(p);
    }
    std::env::split_paths(path?)
        .map(|dir| dir.join(program))
        .find(|p| is_exec(p))
}

pub(crate) fn run_command(
    program: &Path,
    args: &[String],
    cwd: Option<&Path>,
) -> std::io::Result<CommandOutput> {
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null());
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let out = command.output()?;
    Ok(CommandOutput {
        status: out.status.code(),
        stdout: out.stdout,
        stderr: out.stderr,
    })
}

/// `_pdf_requires_ocr`: probe failures keep the convert route.
pub(crate) fn pdf_requires_ocr(path: &Path) -> bool {
    match pdf_inspector::process_pdf(path) {
        Ok(result) => {
            result.has_encoding_issues || result.markdown.as_deref().unwrap_or("").trim().is_empty()
        }
        Err(_) => false,
    }
}

pub(crate) fn pdf_page_count(path: &Path) -> Result<u32> {
    pdf_inspector::detect_pdf(path)
        .map(|r| r.page_count)
        .map_err(|e| AimdError::failed(format!("Cannot open PDF {}: {e}", path.display())))
}
