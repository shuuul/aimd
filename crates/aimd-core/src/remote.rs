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
    /// Whether `model` is the modality default rather than a value the user
    /// supplied. Clients may substitute a served model for a defaulted name,
    /// but never for an explicit choice.
    pub model_is_default: bool,
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
    let chosen = pick(model, "MODEL");
    Ok(Some(RemoteBackendConfig {
        base_url: normalize_base_url(&url)?,
        model: chosen
            .clone()
            .unwrap_or_else(|| modality.default_model().to_string()),
        model_is_default: chosen.is_none(),
        api_key: pick(api_key, "API_KEY").unwrap_or_else(|| "not-needed".to_string()),
    }))
}

/// Whether `value` looks like a scheme-less `host:port` authority
/// (`192.168.100.114:8888`, `host:8888/v1`, `[::1]:8888`), which is accepted
/// and defaulted to `http://`. A scheme-less value without a port is not an
/// authority (it could be a typo'd path), so it keeps the invalid-URL error.
fn is_schemeless_authority(value: &str) -> bool {
    if value.contains("://") || value.is_empty() || value.starts_with('/') {
        return false;
    }
    let authority_end = value.find(['/', '?', '#']).unwrap_or(value.len());
    let authority = &value[..authority_end];
    if authority.is_empty() || authority.contains(char::is_whitespace) {
        return false;
    }
    // A bracketed IPv6 literal must close before the port separator.
    if let Some(rest) = authority.strip_prefix('[') {
        let Some((_, tail)) = rest.split_once(']') else {
            return false;
        };
        return tail.strip_prefix(':').is_some_and(valid_port);
    }
    let Some((host, port)) = authority.rsplit_once(':') else {
        return false;
    };
    // Reject `user:pass@host` style values and a second colon (`a:b:80`).
    !host.is_empty() && !host.contains(':') && !host.contains('@') && valid_port(port)
}

/// Non-empty ASCII digits that fit in a `u16` port.
fn valid_port(port: &str) -> bool {
    !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()) && port.parse::<u16>().is_ok()
}

/// Strip trailing slashes and add `/v1` to bare server URLs. A scheme-less
/// `host:port` is accepted and read as `http://host:port`; any other
/// scheme-less value is rejected.
pub fn normalize_base_url(value: &str) -> Result<String> {
    let trimmed = strip(value).trim_end_matches('/').to_string();
    let owned;
    let candidate = if is_schemeless_authority(&trimmed) {
        owned = format!("http://{trimmed}");
        owned.as_str()
    } else {
        trimmed.as_str()
    };
    let valid = parse_url(candidate)
        .filter(|u| (u.scheme == "http" || u.scheme == "https") && !u.netloc.is_empty());
    let Some(parsed) = valid else {
        return Err(AimdError::backend_unavailable(format!(
            "Invalid remote backend URL {}; expected an http(s) URL or a bare host:port.",
            repr(value)
        )));
    };
    let mut base = candidate.to_string();
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
                model_is_default: false,
            })
        );
    }

    #[test]
    fn defaulted_model_is_flagged_and_env_model_is_not() {
        // No model anywhere -> the modality default, flagged as defaulted.
        let env = env_of(&[("AIMD_OCR_BASE_URL", "http://ocr.example")]);
        let config = resolve_remote_backend_with(RemoteModality::Ocr, None, None, None, &env)
            .unwrap()
            .unwrap();
        assert_eq!(config.model, "Unlimited-OCR");
        assert!(config.model_is_default);

        // An env-supplied model counts as an explicit choice.
        let env = env_of(&[
            ("AIMD_OCR_BASE_URL", "http://ocr.example"),
            ("AIMD_OCR_MODEL", "Unlimited-OCR-2"),
        ]);
        let config = resolve_remote_backend_with(RemoteModality::Ocr, None, None, None, &env)
            .unwrap()
            .unwrap();
        assert_eq!(config.model, "Unlimited-OCR-2");
        assert!(!config.model_is_default);

        // An empty env value counts as unset (Python `or` semantics).
        let env = env_of(&[
            ("AIMD_OCR_BASE_URL", "http://ocr.example"),
            ("AIMD_OCR_MODEL", ""),
        ]);
        let config = resolve_remote_backend_with(RemoteModality::Ocr, None, None, None, &env)
            .unwrap()
            .unwrap();
        assert!(config.model_is_default);
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
            "Invalid remote backend URL 'ftp://x'; expected an http(s) URL or a bare host:port."
        );
    }

    #[test]
    fn schemeless_host_port_defaults_to_http_and_gains_v1() {
        // The LAN endpoint this was reported against.
        assert_eq!(
            normalize_base_url("192.168.100.114:8888").unwrap(),
            "http://192.168.100.114:8888/v1"
        );
        // Surrounding whitespace and trailing slashes are still stripped.
        assert_eq!(
            normalize_base_url("  192.168.100.114:8888/  ").unwrap(),
            "http://192.168.100.114:8888/v1"
        );
        // An explicit path is preserved and does not gain `/v1`.
        assert_eq!(
            normalize_base_url("192.168.100.114:8888/api").unwrap(),
            "http://192.168.100.114:8888/api"
        );
        assert_eq!(
            normalize_base_url("localhost:8888").unwrap(),
            "http://localhost:8888/v1"
        );
        // Bracketed IPv6 literal.
        assert_eq!(
            normalize_base_url("[::1]:8888").unwrap(),
            "http://[::1]:8888/v1"
        );
    }

    #[test]
    fn explicit_scheme_is_never_rewritten() {
        // `https` stays `https`; only scheme-less values default to `http`.
        assert_eq!(
            normalize_base_url("https://192.168.100.114:8888").unwrap(),
            "https://192.168.100.114:8888/v1"
        );
        assert_eq!(
            normalize_base_url("HTTP://host:8888").unwrap(),
            "HTTP://host:8888/v1"
        );
    }

    #[test]
    fn non_authority_schemeless_values_stay_invalid() {
        for value in [
            "192.168.100.114", // no port: could be a path typo
            "host",
            "/tmp/audio.wav",
            "192.168.100.114:99999", // port out of range
            "192.168.100.114:",      // empty port
            "192.168.100.114:8a8",   // non-numeric port
            "user:pass@host:8888",
            "a:b:80",
            "http://", // scheme but no netloc
            "[::1]8888",
        ] {
            let err = normalize_base_url(value).unwrap_err();
            assert_eq!(err.kind, ErrorKind::BackendUnavailable, "{value}");
        }
    }
}
