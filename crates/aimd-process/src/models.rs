//! Model alias + precision -> `mlx-community/...` checkpoint resolution
//! (ports `resolve_mlx_asr_model` / `resolve_mlx_vlm_model`) plus the
//! native-engine support gates.

use aimd_core::precision::SUPPORTED_PRECISIONS;
use aimd_core::pyfmt::{repr, strip};
use aimd_core::{AimdError, Precision, Result, normalize_precision};
use aimd_engine::tokenizer::{ASR_LANGUAGES, resolve_language};

/// `MLX_AUDIO_MODELS` (display order).
pub const MLX_AUDIO_MODELS: [&str; 8] = [
    "mlx-community/Qwen3-ASR-1.7B-4bit",
    "mlx-community/Qwen3-ASR-1.7B-6bit",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/Qwen3-ASR-1.7B-bf16",
    "mlx-community/Qwen3-ASR-0.6B-4bit",
    "mlx-community/Qwen3-ASR-0.6B-6bit",
    "mlx-community/Qwen3-ASR-0.6B-8bit",
    "mlx-community/Qwen3-ASR-0.6B-bf16",
];

/// `TRANSFORMERS_ASR_MODELS` (CUDA-only in Python).
pub const TRANSFORMERS_ASR_MODELS: [&str; 4] = [
    "Qwen/Qwen3-ASR-1.7B-hf",
    "Qwen/Qwen3-ASR-0.6B-hf",
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B",
];

const QWEN3_ASR_MODEL_ALIASES: [(&str, &str); 4] = [
    ("qwen3-asr-1.7b", "1.7B"),
    ("qwen3-asr-0.6b", "0.6B"),
    ("qwen3_asr_1_7b", "1.7B"),
    ("qwen3_asr_0_6b", "0.6B"),
];

/// Quantized OCR families the native engine resolves to
/// `mlx-community/<repo>-<precision>` (GLM-OCR was removed: it is an unknown
/// model like any other).
const MLX_OCR_FAMILY_REPOS: [(&str, &str); 1] = [("unlimited-ocr", "Unlimited-OCR")];

/// PP-OCRv6 medium: the model key names the `_det_safetensors` /
/// `_rec_safetensors` repo pair. Dense f32, so precision does not apply.
pub const PP_OCRV6_MODEL: &str = "PaddlePaddle/PP-OCRv6_medium";
const PP_OCRV6_ALIASES: [&str; 2] = ["pp-ocrv6", "pp-ocrv6-medium"];

/// PaddleOCR-VL-1.6: always the official bf16 checkpoint. A quantized run is
/// keyed `<repo>:<precision>` (a colon cannot appear in a Hub repo id) and
/// quantizes at load.
pub const PADDLEOCR_VL_REPO: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const PADDLEOCR_VL_ALIASES: [&str; 2] = ["paddleocr-vl", "paddleocr-vl-1.6"];

/// Native OCR family behind a resolved model key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrFamily {
    /// `mlx-community/Unlimited-OCR-*`.
    UnlimitedOcr,
    /// [`PP_OCRV6_MODEL`].
    PpOcrV6,
    /// [`PADDLEOCR_VL_REPO`]; `quant_bits` `None` = dense bf16.
    PaddleOcrVl { quant_bits: Option<u32> },
}

impl OcrFamily {
    /// Display name for logs and error messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::UnlimitedOcr => "Unlimited-OCR",
            Self::PpOcrV6 => "PP-OCRv6",
            Self::PaddleOcrVl { .. } => "PaddleOCR-VL",
        }
    }
}

/// Family of a model key produced by [`resolve_mlx_ocr_model`].
pub fn ocr_family(model_id: &str) -> OcrFamily {
    if model_id == PP_OCRV6_MODEL {
        return OcrFamily::PpOcrV6;
    }
    if let Some(rest) = model_id.strip_prefix(PADDLEOCR_VL_REPO) {
        let quant_bits = rest
            .strip_prefix(':')
            .and_then(|p| p.strip_suffix("bit"))
            .and_then(|b| b.parse().ok());
        return OcrFamily::PaddleOcrVl { quant_bits };
    }
    OcrFamily::UnlimitedOcr
}

/// Whether Python would pick the Transformers ASR backend for `model`.
pub fn is_transformers_asr_model(model: Option<&str>) -> bool {
    let m = model.unwrap_or("");
    TRANSFORMERS_ASR_MODELS.contains(&m) || m.starts_with("Qwen/Qwen3-ASR-")
}

/// `resolve_mlx_asr_model` (verbatim Python semantics and messages).
pub fn resolve_mlx_asr_model(model: Option<&str>, precision: Option<&str>) -> Result<String> {
    let normalized = normalize_precision(precision)?;
    let requested = strip(model.unwrap_or(""));
    let size = if requested.is_empty() {
        "1.7B"
    } else if MLX_AUDIO_MODELS.contains(&requested) {
        let embedded = requested.rsplit('-').next().unwrap_or("");
        if let Some(p) = normalized
            && p.as_str() != embedded
        {
            return Err(AimdError::failed(format!(
                "Precision {} conflicts with explicit MLX ASR model ID {}. Drop --precision \
                 or use the alias form.",
                repr(p.as_str()),
                repr(requested)
            )));
        }
        return Ok(requested.to_string());
    } else {
        let lower = requested.to_lowercase();
        match QWEN3_ASR_MODEL_ALIASES.iter().find(|(a, _)| *a == lower) {
            Some((_, size)) => size,
            None => {
                return Err(AimdError::failed(format!(
                    "Unsupported MLX ASR model {}. Supported aliases: qwen3-asr-1.7b, \
                     qwen3-asr-0.6b. Supported model IDs: {}",
                    repr(requested),
                    MLX_AUDIO_MODELS.join(", ")
                )));
            }
        }
    };
    let precision = normalized.map_or("4bit", Precision::as_str);
    Ok(format!("mlx-community/Qwen3-ASR-{size}-{precision}"))
}

/// `resolve_mlx_vlm_model` plus the PP-OCRv6 and PaddleOCR-VL aliases.
/// Returns a model key; see [`ocr_family`].
pub fn resolve_mlx_ocr_model(model: Option<&str>, precision: Option<&str>) -> Result<String> {
    let normalized = normalize_precision(precision)?;
    let requested = strip(model.unwrap_or("unlimited-ocr"))
        .to_lowercase()
        .replace(' ', "_")
        .replace('_', "-");
    let mut candidate = ["mlx-community/", "paddlepaddle/"]
        .iter()
        .find_map(|prefix| requested.strip_prefix(prefix))
        .unwrap_or(&requested)
        .to_string();
    let mut embedded: Option<&str> = None;
    for known in SUPPORTED_PRECISIONS {
        if let Some(rest) = candidate.strip_suffix(&format!("-{known}")) {
            embedded = Some(known);
            candidate = rest.to_string();
            break;
        }
    }
    let model_repr = model.map_or_else(|| "None".to_string(), repr);
    if let (Some(p), Some(e)) = (normalized, embedded)
        && p.as_str() != e
    {
        return Err(AimdError::failed(format!(
            "Precision {} conflicts with explicit MLX OCR model {model_repr}. Drop the precision \
             argument or use the alias form.",
            repr(p.as_str())
        )));
    }
    let precision = normalized.map(Precision::as_str).or(embedded);
    if PP_OCRV6_ALIASES.contains(&candidate.as_str()) && embedded.is_none() {
        return Ok(PP_OCRV6_MODEL.to_string());
    }
    if PADDLEOCR_VL_ALIASES.contains(&candidate.as_str()) {
        return Ok(match precision {
            None | Some("bf16") => PADDLEOCR_VL_REPO.to_string(),
            Some(p) => format!("{PADDLEOCR_VL_REPO}:{p}"),
        });
    }
    let Some((_, repo)) = MLX_OCR_FAMILY_REPOS.iter().find(|(f, _)| *f == candidate) else {
        return Err(AimdError::failed(format!(
            "Unsupported MLX VLM OCR model {model_repr}. Supported models: unlimited-ocr \
             (default), pp-ocrv6, paddleocr-vl, or an explicit \
             mlx-community/Unlimited-OCR-{{4bit,6bit,8bit,bf16}} Hugging Face model ID."
        )));
    };
    Ok(format!(
        "mlx-community/{repo}-{}",
        precision.unwrap_or("4bit")
    ))
}

/// Native-engine gate for a resolved ASR checkpoint: the decoder needs an
/// affine-quantized checkpoint.
pub fn ensure_native_asr(model_id: &str) -> Result<()> {
    if model_id.ends_with("-bf16") {
        return Err(AimdError::backend_unavailable(format!(
            "The native MLX engine does not support bf16 (dense) Qwen3-ASR checkpoints yet \
             ({model_id}). Use --precision 4bit, 6bit, or 8bit."
        )));
    }
    Ok(())
}

/// Native-engine gate for a resolved OCR model key: Unlimited-OCR runs
/// quantized checkpoints only; PP-OCRv6 and PaddleOCR-VL load dense weights.
pub fn ensure_native_ocr(model_id: &str) -> Result<()> {
    if ocr_family(model_id) == OcrFamily::UnlimitedOcr && model_id.ends_with("-bf16") {
        return Err(AimdError::backend_unavailable(format!(
            "The native MLX engine does not support bf16 (dense) Unlimited-OCR checkpoints yet \
             ({model_id}). Use --precision 4bit, 6bit, or 8bit."
        )));
    }
    Ok(())
}

/// `get_8bit_fallback_model`.
pub fn eight_bit_fallback(model_id: &str) -> Option<String> {
    model_id
        .contains("-4bit")
        .then(|| model_id.replace("-4bit", "-8bit"))
}

/// Qwen3-ASR language hint: `None` (or blank) -> `None`, which lets the
/// model detect the language; codes and full names map to the engine's
/// canonical name.
pub fn resolve_asr_language(language: Option<&str>) -> Result<Option<String>> {
    let Some(language) = language.filter(|l| !l.trim().is_empty()) else {
        return Ok(None);
    };
    if let Some(name) = resolve_language(language) {
        return Ok(Some(name));
    }
    let codes: Vec<&str> = ASR_LANGUAGES.iter().map(|(c, _)| *c).collect();
    Err(AimdError::unsupported(format!(
        "Unsupported language for mlx backend: '{language}'. Supported: {}",
        aimd_core::pyfmt::repr_list(&codes)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aimd_core::ErrorKind;

    #[test]
    fn asr_aliases_and_precision() {
        // test_asr_model_precision.py scenarios
        assert_eq!(
            resolve_mlx_asr_model(None, None).unwrap(),
            "mlx-community/Qwen3-ASR-1.7B-4bit"
        );
        assert_eq!(
            resolve_mlx_asr_model(Some("qwen3-asr-0.6b"), Some("8-bit")).unwrap(),
            "mlx-community/Qwen3-ASR-0.6B-8bit"
        );
        assert_eq!(
            resolve_mlx_asr_model(Some("QWEN3_ASR_1_7B"), Some("6bit")).unwrap(),
            "mlx-community/Qwen3-ASR-1.7B-6bit"
        );
        assert_eq!(
            resolve_mlx_asr_model(Some("mlx-community/Qwen3-ASR-0.6B-8bit"), Some("8bit")).unwrap(),
            "mlx-community/Qwen3-ASR-0.6B-8bit"
        );
        let err = resolve_mlx_asr_model(Some("mlx-community/Qwen3-ASR-1.7B-8bit"), Some("4bit"))
            .unwrap_err();
        assert_eq!(
            err.message,
            "Precision '4bit' conflicts with explicit MLX ASR model ID \
             'mlx-community/Qwen3-ASR-1.7B-8bit'. Drop --precision or use the alias form."
        );
        let err = resolve_mlx_asr_model(Some("whisper"), None).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ProcessingFailed);
        assert!(err.message.starts_with(
            "Unsupported MLX ASR model 'whisper'. Supported aliases: qwen3-asr-1.7b, \
             qwen3-asr-0.6b. Supported model IDs: mlx-community/Qwen3-ASR-1.7B-4bit, "
        ));
    }

    #[test]
    fn paddle_ocr_families() {
        for alias in [
            "pp-ocrv6",
            "PP-OCRv6_medium",
            "PaddlePaddle/PP-OCRv6_medium",
        ] {
            assert_eq!(
                resolve_mlx_ocr_model(Some(alias), None).unwrap(),
                PP_OCRV6_MODEL
            );
        }
        // Precision is ignored (with a warning in `ocr`), not an error.
        let key = resolve_mlx_ocr_model(Some("pp-ocrv6"), Some("8bit")).unwrap();
        assert_eq!(key, PP_OCRV6_MODEL);
        assert_eq!(ocr_family(&key), OcrFamily::PpOcrV6);
        assert!(ensure_native_ocr(&key).is_ok());

        let cases = [
            (Some("paddleocr-vl"), None, PADDLEOCR_VL_REPO, None),
            (
                Some("PaddlePaddle/PaddleOCR-VL-1.6"),
                Some("bf16"),
                PADDLEOCR_VL_REPO,
                None,
            ),
            (
                Some("paddleocr-vl-1.6"),
                Some("4bit"),
                "PaddlePaddle/PaddleOCR-VL-1.6:4bit",
                Some(4),
            ),
            (
                Some("paddleocr_vl"),
                Some("8-bit"),
                "PaddlePaddle/PaddleOCR-VL-1.6:8bit",
                Some(8),
            ),
        ];
        for (model, precision, want, bits) in cases {
            let key = resolve_mlx_ocr_model(model, precision).unwrap();
            assert_eq!(key, want, "{model:?} {precision:?}");
            assert_eq!(
                ocr_family(&key),
                OcrFamily::PaddleOcrVl { quant_bits: bits }
            );
            assert!(ensure_native_ocr(&key).is_ok());
        }
        assert_eq!(
            ocr_family("mlx-community/Unlimited-OCR-4bit"),
            OcrFamily::UnlimitedOcr
        );
    }

    #[test]
    fn ocr_aliases_and_precision() {
        assert_eq!(
            resolve_mlx_ocr_model(None, None).unwrap(),
            "mlx-community/Unlimited-OCR-4bit"
        );
        assert_eq!(
            resolve_mlx_ocr_model(Some("unlimited_ocr"), Some("8bit")).unwrap(),
            "mlx-community/Unlimited-OCR-8bit"
        );
        assert_eq!(
            resolve_mlx_ocr_model(Some("mlx-community/Unlimited-OCR-6bit"), None).unwrap(),
            "mlx-community/Unlimited-OCR-6bit"
        );
        for removed in [
            "glm-ocr",
            "GLM OCR",
            "mlx-community/GLM-OCR-4bit",
            "zai-org/GLM-OCR",
        ] {
            let err = resolve_mlx_ocr_model(Some(removed), None).unwrap_err();
            assert_eq!(err.kind, ErrorKind::ProcessingFailed, "{removed}");
            assert!(err.message.starts_with("Unsupported MLX VLM OCR model"));
            assert!(err.message.ends_with(
                "Supported models: unlimited-ocr (default), pp-ocrv6, paddleocr-vl, or an explicit \
                 mlx-community/Unlimited-OCR-{4bit,6bit,8bit,bf16} Hugging Face model ID."
            ));
        }
        let err = resolve_mlx_ocr_model(Some("mlx-community/Unlimited-OCR-8bit"), Some("4bit"))
            .unwrap_err();
        assert_eq!(
            err.message,
            "Precision '4bit' conflicts with explicit MLX OCR model \
             'mlx-community/Unlimited-OCR-8bit'. Drop the precision argument or use the alias form."
        );
        let err = resolve_mlx_ocr_model(Some("got-ocr"), None).unwrap_err();
        assert!(
            err.message
                .starts_with("Unsupported MLX VLM OCR model 'got-ocr'.")
        );
    }

    #[test]
    fn native_gates() {
        assert_eq!(
            ensure_native_asr("mlx-community/Qwen3-ASR-1.7B-bf16")
                .unwrap_err()
                .kind,
            ErrorKind::BackendUnavailable
        );
        assert!(ensure_native_asr("mlx-community/Qwen3-ASR-1.7B-6bit").is_ok());
        assert_eq!(
            ensure_native_ocr("mlx-community/Unlimited-OCR-bf16")
                .unwrap_err()
                .kind,
            ErrorKind::BackendUnavailable
        );
        assert!(ensure_native_ocr("mlx-community/Unlimited-OCR-8bit").is_ok());
        assert!(is_transformers_asr_model(Some("Qwen/Qwen3-ASR-1.7B-hf")));
        assert!(!is_transformers_asr_model(Some("qwen3-asr-1.7b")));
        assert_eq!(
            eight_bit_fallback("mlx-community/Qwen3-ASR-1.7B-4bit").as_deref(),
            Some("mlx-community/Qwen3-ASR-1.7B-8bit")
        );
        assert_eq!(
            eight_bit_fallback("mlx-community/Qwen3-ASR-1.7B-8bit"),
            None
        );
    }

    #[test]
    fn languages() {
        assert_eq!(resolve_asr_language(None).unwrap(), None);
        assert_eq!(resolve_asr_language(Some("  ")).unwrap(), None);
        assert_eq!(
            resolve_asr_language(Some("EN")).unwrap().unwrap(),
            "English"
        );
        assert_eq!(
            resolve_asr_language(Some("japanese")).unwrap().unwrap(),
            "Japanese"
        );
        assert_eq!(
            resolve_asr_language(Some("yue")).unwrap().unwrap(),
            "Cantonese"
        );
        let err = resolve_asr_language(Some("xx")).unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnsupportedInput);
        assert!(
            err.message.starts_with(
                "Unsupported language for mlx backend: 'xx'. Supported: ['zh', 'en', 'ja', "
            ),
            "{}",
            err.message
        );
        assert!(err.message.ends_with("'mk', 'ro']"), "{}", err.message);
    }
}
