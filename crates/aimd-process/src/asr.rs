//! Local/remote audio transcription (ports `aimd.plugins.asr._plugin`).

use std::path::Path;

use aimd_core::ext::{
    AUDIO_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS, is_audio_extension, suffix_lower,
};
use aimd_core::remote::resolve_remote_backend_with;
use aimd_core::{
    AimdError, ProcessHooks, ProcessInput, RemoteBackendConfig, RemoteModality, Result, log_info,
    log_warn,
};
use aimd_engine::models::qwen3_asr::generate::detect_repetition_loop;

use crate::models::{
    eight_bit_fallback, ensure_native_asr, is_transformers_asr_model, resolve_asr_language,
    resolve_mlx_asr_model,
};
use crate::{AsrFailure, LocalModality, Runtime, audio, remote};

/// Python segments inputs longer than 10 minutes.
pub const SEGMENT_SECONDS: f64 = 600.0;

/// ASR options shared by local files and the URL audio fallback.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub model: Option<String>,
    pub precision: Option<String>,
    pub temp_dir: Option<std::path::PathBuf>,
    pub asr_base_url: Option<String>,
    pub asr_model: Option<String>,
    pub asr_api_key: Option<String>,
}

impl AsrOptions {
    pub fn from_request(request: &ProcessInput) -> Self {
        Self {
            language: request.language.clone(),
            model: request.model.clone(),
            precision: request.precision.clone(),
            temp_dir: request.temp_dir.clone(),
            asr_base_url: request.asr_base_url.clone(),
            asr_model: request.asr_model.clone(),
            asr_api_key: request.asr_api_key.clone(),
        }
    }
}

enum Engine {
    Remote(RemoteBackendConfig),
    /// `language: None` lets Qwen3-ASR detect the language.
    Local {
        model_id: String,
        language: Option<String>,
    },
}

/// `transcribe_file` for a [`ProcessInput`].
pub fn transcribe_file(
    rt: &dyn Runtime,
    file_path: &Path,
    request: &ProcessInput,
    context: Option<&str>,
    hooks: &dyn ProcessHooks,
) -> Result<String> {
    transcribe_with_options(
        rt,
        file_path,
        &AsrOptions::from_request(request),
        context,
        hooks,
    )
}

/// Transcribe one audio/video file (segmenting inputs over 10 minutes).
pub fn transcribe_with_options(
    rt: &dyn Runtime,
    file_path: &Path,
    options: &AsrOptions,
    context: Option<&str>,
    hooks: &dyn ProcessHooks,
) -> Result<String> {
    if !file_path.exists() {
        return Err(AimdError::not_found(format!(
            "Audio/video file not found: {}",
            file_path.display()
        )));
    }
    let ext = suffix_lower(file_path);
    if !is_audio_extension(&ext) {
        let mut all: Vec<&str> = AUDIO_FILE_EXTENSIONS
            .iter()
            .chain(VIDEO_FILE_EXTENSIONS)
            .copied()
            .collect();
        all.sort_unstable();
        return Err(AimdError::unsupported(format!(
            "Unsupported file format: {ext}. Supported formats: {}",
            all.join(", ")
        )));
    }
    if VIDEO_FILE_EXTENSIONS.contains(&ext.as_str()) {
        log_info!(
            "Processing video file: {} (audio will be extracted)",
            file_path.display()
        );
    } else {
        log_info!("Processing audio file: {}", file_path.display());
    }

    let remote_config = resolve_remote_backend_with(
        RemoteModality::Asr,
        options.asr_base_url.as_deref(),
        options.asr_model.as_deref(),
        options.asr_api_key.as_deref(),
        &|key| rt.env(key),
    )?;
    let engine = match remote_config {
        Some(config) => {
            log_info!("Using transcription backend: remote");
            if options.precision.is_some() {
                log_warn!(
                    "Ignoring ASR precision for remote inference; server-side weights already \
                     define their precision."
                );
            }
            Engine::Remote(config)
        }
        None => {
            if is_transformers_asr_model(options.model.as_deref()) {
                return Err(AimdError::backend_unavailable(format!(
                    "Transformers ASR model {} is not supported by the native engine. Use \
                     qwen3-asr-1.7b or qwen3-asr-0.6b with --precision.",
                    aimd_core::pyfmt::repr(options.model.as_deref().unwrap_or(""))
                )));
            }
            log_info!("Using transcription backend: mlx");
            let model_id =
                resolve_mlx_asr_model(options.model.as_deref(), options.precision.as_deref())?;
            ensure_native_asr(&model_id)?;
            let language = resolve_asr_language(options.language.as_deref())?;
            rt.preflight_local(LocalModality::Asr)?;
            Engine::Local { model_id, language }
        }
    };
    let temp_dir = options.temp_dir.as_deref();

    let duration = match audio::get_audio_duration(rt, file_path) {
        Ok(d) => {
            log_info!("Audio/video duration is {d:.2} seconds");
            d
        }
        Err(e) => {
            log_warn!(
                "Could not determine audio/video duration: {e}. Processing without segmentation."
            );
            0.0
        }
    };

    if duration <= SEGMENT_SECONDS {
        return transcribe_segment_with_fallback(
            rt, &engine, options, file_path, temp_dir, context,
        );
    }
    log_info!("Audio/video duration exceeds 10 minutes. Segmenting for transcription...");
    let segments = audio::segment_audio(rt, file_path, SEGMENT_SECONDS, temp_dir)?;
    let total = segments.paths.len();
    let mut results = Vec::new();
    for (i, segment) in segments.paths.iter().enumerate() {
        if hooks.is_cancelled() {
            return Err(AimdError::cancelled(
                "Transcription cancelled between segments",
            ));
        }
        hooks.progress(
            "transcribing",
            Some(i as u64 + 1),
            Some(total as u64),
            Some(&format!("Transcribing segment {} of {total}", i + 1)),
        );
        log_info!(
            "Transcribing segment {}/{total}: {}",
            i + 1,
            segment
                .file_name()
                .map(|n| n.to_string_lossy())
                .unwrap_or_default()
        );
        let text =
            transcribe_segment_with_fallback(rt, &engine, options, segment, temp_dir, context)?;
        if !text.is_empty() {
            results.push(text);
        }
    }
    let joined = aimd_core::pyfmt::strip(&results.join(" ")).to_string();
    if joined.is_empty() {
        return Err(AimdError::failed(
            "ASR produced empty transcription for all segments",
        ));
    }
    Ok(joined)
}

/// `_transcribe_segment_with_fallback`: empty -> `""`; a repetition loop on
/// a 4-bit local model retries once with the 8-bit checkpoint.
fn transcribe_segment_with_fallback(
    rt: &dyn Runtime,
    engine: &Engine,
    options: &AsrOptions,
    path: &Path,
    temp_dir: Option<&Path>,
    context: Option<&str>,
) -> Result<String> {
    let text = match engine {
        Engine::Remote(config) => {
            match transcribe_remote(rt, config, options, path, temp_dir, context) {
                Err(e) if e.message.to_lowercase().contains("empty transcription") => {
                    return Ok(String::new());
                }
                other => other?,
            }
        }
        Engine::Local { model_id, language } => {
            match transcribe_local(rt, model_id, language.as_deref(), path, temp_dir, context)? {
                Some(text) => text,
                None => return Ok(String::new()),
            }
        }
    };
    if !detect_repetition_loop(&text) {
        return Ok(text);
    }
    match engine {
        Engine::Local { model_id, language } if eight_bit_fallback(model_id).is_some() => {
            let fallback = eight_bit_fallback(model_id).unwrap_or_default();
            log_warn!(
                "Repetition loop detected with 4-bit model ({model_id}). Switching to 8-bit \
                 fallback model: {fallback}"
            );
            ensure_native_asr(&fallback)?;
            Ok(
                transcribe_local(rt, &fallback, language.as_deref(), path, temp_dir, context)?
                    .unwrap_or_default(),
            )
        }
        _ => {
            log_warn!("Repetition loop detected, but no 8-bit fallback model is available.");
            Ok(text)
        }
    }
}

/// Local Qwen3-ASR on a 16 kHz mono wav (`None` = empty transcription).
fn transcribe_local(
    rt: &dyn Runtime,
    model_id: &str,
    language: Option<&str>,
    path: &Path,
    temp_dir: Option<&Path>,
    context: Option<&str>,
) -> Result<Option<String>> {
    let model = rt.load_asr(model_id)?;
    log_info!(
        "Transcribing with native Qwen3-ASR model: {model_id}, language: {}",
        language.unwrap_or("auto")
    );
    let converted;
    let wav_path = if audio::is_native_wav(path) {
        path
    } else {
        converted = audio::convert_to_wav(rt, path, temp_dir)?;
        converted.path()
    };
    match model.transcribe(wav_path, language, context) {
        Ok(text) => {
            let text = text.trim().to_string();
            if text.is_empty() {
                return Ok(None);
            }
            log_info!(
                "Successfully transcribed {} characters with the native engine",
                text.chars().count()
            );
            Ok(Some(text))
        }
        Err(AsrFailure::Empty) => Ok(None),
        Err(AsrFailure::Other(e)) => Err(AimdError::failed(format!(
            "Qwen3-ASR transcription failed: {e}"
        ))),
    }
}

fn transcribe_remote(
    rt: &dyn Runtime,
    config: &RemoteBackendConfig,
    options: &AsrOptions,
    path: &Path,
    temp_dir: Option<&Path>,
    context: Option<&str>,
) -> Result<String> {
    let converted = audio::convert_to_wav_if_needed(rt, path, temp_dir)?;
    let request_path = converted.as_ref().map_or(path, |c| c.path());
    remote::transcribe(config, request_path, options.language.as_deref(), context)
}
