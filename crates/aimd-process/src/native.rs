//! The real [`Runtime`]: system processes, pdf-inspector, and per-process
//! cached MLX pipelines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

use aimd_core::{AimdError, Result, log_info};
use aimd_engine::models::paddleocr_vl::pipeline::{
    PaddleOcrVlPipeline, PipelineError as VlPipelineError, VlRequest,
};
use aimd_engine::models::pp_ocrv6::pipeline::{PipelineError as PpPipelineError, PpOcrV6Pipeline};
use aimd_engine::models::qwen3_asr::pipeline::{
    PipelineError as AsrPipelineError, Qwen3AsrPipeline, TranscriptionRequest,
};
use aimd_engine::models::unlimited_ocr::pipeline::{
    OcrRequest, PipelineError as OcrPipelineError, UnlimitedOcrPipeline,
};

use crate::models::{OcrFamily, PADDLEOCR_VL_REPO, ocr_family};
use crate::{AsrFailure, AsrModel, CommandOutput, LocalModality, OcrFailure, OcrModel, Runtime};

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
            OcrFamily::UnlimitedOcr => {
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
        };
        cache.insert(model_id.to_string(), model.clone());
        Ok(model)
    }
}

struct NativeAsr(Qwen3AsrPipeline);

impl AsrModel for NativeAsr {
    fn transcribe(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<String, AsrFailure> {
        let request = TranscriptionRequest {
            language,
            context: context.filter(|c| !c.is_empty()),
        };
        match self.0.transcribe(wav_path, request) {
            Ok(text) => Ok(text),
            Err(AsrPipelineError::EmptyTranscription) => Err(AsrFailure::Empty),
            Err(e) => Err(AsrFailure::Other(e.to_string())),
        }
    }
}

enum NativeOcr {
    Unlimited(UnlimitedOcrPipeline),
    PaddleVl(PaddleOcrVlPipeline),
    PpOcr(PpOcrV6Pipeline),
}

impl OcrModel for NativeOcr {
    fn recognize(&self, image_path: &Path) -> std::result::Result<String, OcrFailure> {
        match self {
            Self::Unlimited(p) => match p.recognize_path(image_path, OcrRequest::default()) {
                Ok(result) => Ok(result.markdown),
                Err(OcrPipelineError::EmptyContent) => Err(OcrFailure::Empty),
                Err(e) => Err(OcrFailure::Other(e.to_string())),
            },
            Self::PaddleVl(p) => match p.recognize_path(image_path, VlRequest::default()) {
                Ok(result) => Ok(result.markdown),
                Err(VlPipelineError::EmptyContent) => Err(OcrFailure::Empty),
                Err(e) => Err(OcrFailure::Other(e.to_string())),
            },
            Self::PpOcr(p) => match p.recognize_path(image_path) {
                Ok(result) => Ok(result.markdown),
                Err(PpPipelineError::EmptyContent) => Err(OcrFailure::Empty),
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
