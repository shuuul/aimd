//! Opt-in OpenAI-compatible remote backend configuration (ports
//! `aimd.core.remote`).

use crate::error::{AimdError, Result};
use crate::pyfmt::{repr, strip};
use crate::url::parse_url;

/// Remote inference modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteModality {
    Asr,
    Ocr,
}

impl RemoteModality {
    /// Environment prefix (`AIMD_ASR` / `AIMD_OCR`).
    pub fn env_prefix(self) -> &'static str {
        match self {
            RemoteModality::Asr => "AIMD_ASR",
            RemoteModality::Ocr => "AIMD_OCR",
        }
    }

    /// Served model name used when neither the request nor the
    /// environment names one.
    pub fn default_model(self) -> &'static str {
        match self {
            RemoteModality::Asr => "Qwen3-ASR-1.7B",
            RemoteModality::Ocr => "Unlimited-OCR",
        }
    }
}

/// Resolved OpenAI-compatible backend settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteBackendConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

/// Resolve explicit settings over `AIMD_{ASR,OCR}_{BASE_URL,MODEL,API_KEY}`
/// from the process environment. `None` = use the local backend.
pub fn resolve_remote_backend(
    modality: RemoteModality,
    base_url: Option<&str>,
    model: Option<&str>,
    api_key: Option<&str>,
) -> Result<Option<RemoteBackendConfig>> {
    resolve_remote_backend_with(modality, base_url, model, api_key, &|key| {
        std::env::var(key).ok()
    })
}

/// [`resolve_remote_backend`] over an injectable environment lookup.
/// Empty strings count as unset (Python `or` semantics).
pub fn resolve_remote_backend_with(
    modality: RemoteModality,
    base_url: Option<&str>,
    model: Option<&str>,
    api_key: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Option<RemoteBackendConfig>> {
    let prefix = modality.env_prefix();
    let pick = |explicit: Option<&str>, suffix: &str| -> Option<String> {
        explicit
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .or_else(|| env(&format!("{prefix}_{suffix}")).filter(|v| !v.is_empty()))
    };
    let Some(url) = pick(base_url, "BASE_URL") else {
        return Ok(None);
    };
    Ok(Some(RemoteBackendConfig {
        base_url: normalize_base_url(&url)?,
        model: pick(model, "MODEL").unwrap_or_else(|| modality.default_model().to_string()),
        api_key: pick(api_key, "API_KEY").unwrap_or_else(|| "not-needed".to_string()),
    }))
}

/// Strip trailing slashes and add `/v1` to bare server URLs.
pub fn normalize_base_url(value: &str) -> Result<String> {
    let mut base = strip(value).trim_end_matches('/').to_string();
    let valid = parse_url(&base)
        .filter(|u| (u.scheme == "http" || u.scheme == "https") && !u.netloc.is_empty());
    let Some(parsed) = valid else {
        return Err(AimdError::backend_unavailable(format!(
            "Invalid remote backend URL {}; expected an http(s) URL.",
            repr(value)
        )));
    };
    if parsed.path.is_empty() || parsed.path == "/" {
        base.push_str("/v1");
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn explicit_over_env_and_normalizes_root_url() {
        // test_remote_backends.py::test_remote_config_explicit_over_env_and_normalizes_root_url
        let env = env_of(&[
            ("AIMD_ASR_BASE_URL", "http://env.example/v1"),
            ("AIMD_ASR_MODEL", "env-model"),
            ("AIMD_ASR_API_KEY", "env-key"),
        ]);
        let config = resolve_remote_backend_with(
            RemoteModality::Asr,
            Some("http://explicit.example/"),
            Some("explicit-model"),
            Some("explicit-key"),
            &env,
        )
        .unwrap();
        assert_eq!(
            config,
            Some(RemoteBackendConfig {
                base_url: "http://explicit.example/v1".into(),
                model: "explicit-model".into(),
                api_key: "explicit-key".into(),
            })
        );
    }

    #[test]
    fn env_values_and_defaults() {
        let env = env_of(&[("AIMD_OCR_BASE_URL", "https://ocr.example/api/")]);
        let config = resolve_remote_backend_with(RemoteModality::Ocr, None, None, None, &env)
            .unwrap()
            .unwrap();
        assert_eq!(config.base_url, "https://ocr.example/api");
        assert_eq!(config.model, "Unlimited-OCR");
        assert_eq!(config.api_key, "not-needed");
    }

    #[test]
    fn unset_keeps_local_backend() {
        // test_remote_backends.py::test_remote_config_unset_keeps_local_backend
        let env = env_of(&[]);
        assert_eq!(
            resolve_remote_backend_with(RemoteModality::Ocr, None, None, None, &env).unwrap(),
            None
        );
        assert_eq!(
            resolve_remote_backend_with(RemoteModality::Ocr, Some(""), None, None, &env).unwrap(),
            None
        );
    }

    #[test]
    fn invalid_url_is_backend_unavailable() {
        let err = normalize_base_url("ftp://x").unwrap_err();
        assert_eq!(err.kind, ErrorKind::BackendUnavailable);
        assert_eq!(
            err.message,
            "Invalid remote backend URL 'ftp://x'; expected an http(s) URL."
        );
    }
}
