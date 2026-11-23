//! Real system runtime and PDF probing.

use std::path::Path;
use std::sync::Arc;

use crate::{AimdError, AsrModel, LocalModality, OcrModel, Result, Runtime};

#[cfg(feature = "native")]
mod mlx;

/// Native runtime (real system, cached pipelines).
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeRuntime;

impl crate::System for NativeRuntime {}

impl Runtime for NativeRuntime {
    fn url_transcript(
        &self,
        request: &crate::ProcessInput,
        hooks: &dyn crate::ProcessHooks,
    ) -> Result<crate::UrlOutput> {
        crate::runtime::convert_url(self, request, hooks)
    }

    fn preflight_local(&self, modality: LocalModality) -> Result<()> {
        #[cfg(feature = "native")]
        return self.preflight_mlx(modality);
        #[cfg(not(feature = "native"))]
        Err(native_unavailable(modality))
    }

    fn load_asr(&self, _model_id: &str) -> Result<Arc<dyn AsrModel>> {
        #[cfg(feature = "native")]
        return self.load_mlx_asr(_model_id);
        #[cfg(not(feature = "native"))]
        Err(native_unavailable(LocalModality::Asr))
    }

    fn load_ocr(&self, _model_id: &str) -> Result<Arc<dyn OcrModel>> {
        #[cfg(feature = "native")]
        return self.load_mlx_ocr(_model_id);
        #[cfg(not(feature = "native"))]
        Err(native_unavailable(LocalModality::Ocr))
    }
}

#[cfg(not(feature = "native"))]
fn native_unavailable(modality: LocalModality) -> AimdError {
    let (name, setting) = match modality {
        LocalModality::Asr => ("ASR", "AIMD_ASR_BASE_URL"),
        LocalModality::Ocr => ("OCR", "AIMD_OCR_BASE_URL"),
    };
    AimdError::backend_unavailable(format!(
        "Local {name} inference requires the native feature. Set {setting} for a remote backend."
    ))
}

#[cfg(all(test, not(feature = "native")))]
mod tests {
    use crate::error::ErrorKind;
    use crate::util::tempdir::TempDir;
    use crate::{NoHooks, ProcessInput, process_input_with};

    use super::*;

    #[test]
    fn local_inference_requires_native_support() {
        struct RemoteOnly;
        impl crate::System for RemoteOnly {
            fn env_os(&self, _: &str) -> Option<std::ffi::OsString> {
                None
            }
        }
        impl Runtime for RemoteOnly {
            fn preflight_local(&self, modality: LocalModality) -> Result<()> {
                NativeRuntime.preflight_local(modality)
            }
            fn load_asr(&self, model_id: &str) -> Result<Arc<dyn AsrModel>> {
                NativeRuntime.load_asr(model_id)
            }
            fn load_ocr(&self, model_id: &str) -> Result<Arc<dyn OcrModel>> {
                NativeRuntime.load_ocr(model_id)
            }
        }

        let dir = TempDir::new(None, "aimd-no-native-").unwrap();
        for name in ["clip.wav", "page.png"] {
            let path = dir.path().join(name);
            std::fs::write(&path, []).unwrap();
            let request = ProcessInput::new(path.to_string_lossy());
            let err = process_input_with(&request, &NoHooks, &RemoteOnly).unwrap_err();
            assert_eq!(err.kind, ErrorKind::BackendUnavailable);
            assert!(err.message.contains("requires the native feature"));
        }
    }
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
