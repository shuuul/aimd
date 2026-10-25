//! Shared-filesystem allow-root policy for the local HTTP sidecar (ports
//! `aimd.interfaces.api.paths`; hand-written per the dependency policy).

use std::fmt;
use std::path::{Path, PathBuf};

use crate::schemas::ProcessRequest;
use crate::util::{expand_user, resolve_lenient, url_scheme};

/// A request escaped its configured local roots (HTTP 403).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathAccessError(pub String);

impl fmt::Display for PathAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PathAccessError {}

/// Normalize local request paths and constrain them to canonical roots.
/// No roots means every local path is allowed (operator-run API).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PathPolicy {
    pub roots: Vec<PathBuf>,
}

impl PathPolicy {
    /// Canonicalize every root (`resolve(strict=True)`; each must be an
    /// existing directory).
    pub fn from_roots<P: AsRef<Path>>(roots: &[P]) -> Result<Self, PathAccessError> {
        let mut canonical = Vec::with_capacity(roots.len());
        for root in roots {
            let expanded = expand_user(root.as_ref());
            let resolved = std::fs::canonicalize(&expanded).map_err(|e| {
                PathAccessError(format!(
                    "Allowed root cannot be resolved: {}: {e}",
                    expanded.display()
                ))
            })?;
            if !resolved.is_dir() {
                return Err(PathAccessError(format!(
                    "Allowed root is not a directory: {}",
                    resolved.display()
                )));
            }
            canonical.push(resolved);
        }
        Ok(Self { roots: canonical })
    }

    /// Roots from the path-separated `AIMD_ALLOWED_ROOTS` value.
    pub fn from_env_value(value: Option<&str>) -> Result<Self, PathAccessError> {
        let roots: Vec<&str> = value
            .unwrap_or("")
            .split(':')
            .filter(|v| !v.is_empty())
            .collect();
        Self::from_roots(&roots)
    }

    /// Return a request whose local paths are absolute and allow-root
    /// checked. URLs must be `http(s)`; other URI schemes are rejected.
    pub fn normalize_request(
        &self,
        request: &ProcessRequest,
    ) -> Result<ProcessRequest, PathAccessError> {
        let mut out = request.clone();
        if request.has_blob() {
            // Blob paths are sidecar-owned and resolved later.
        } else if let Some(source) = request.input_source.as_deref().filter(|s| !s.is_empty()) {
            let scheme = url_scheme(source);
            if matches!(scheme.as_deref(), Some("http" | "https")) {
                // URLs pass through unchanged.
            } else if source.contains("://") {
                return Err(PathAccessError(
                    "Only HTTP(S) URLs or local filesystem paths are allowed".into(),
                ));
            } else {
                out.input_source = Some(path_string(&self.normalize(source, "input")?));
            }
        } else {
            return Err(PathAccessError(
                "Exactly one of input_source or blob_id must be set".into(),
            ));
        }
        for (name, slot) in [
            ("output_file", &mut out.output_file),
            ("save_original", &mut out.save_original),
            ("cookies", &mut out.cookies),
        ] {
            if let Some(value) = slot.as_deref() {
                *slot = Some(path_string(&self.normalize(value, name)?));
            }
        }
        Ok(out)
    }

    /// Resolve symlinks and traversal before enforcing configured roots.
    pub fn normalize(
        &self,
        value: impl AsRef<Path>,
        purpose: &str,
    ) -> Result<PathBuf, PathAccessError> {
        let resolved = resolve_lenient(&expand_user(value.as_ref()));
        if !self.roots.is_empty() && !self.roots.iter().any(|root| resolved.starts_with(root)) {
            return Err(PathAccessError(format!(
                "{purpose} path is outside configured allowed roots"
            )));
        }
        Ok(resolved)
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
