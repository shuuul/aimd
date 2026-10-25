//! `yt-dlp` CLI invocation (replaces `aimd.plugins.url.ydl`'s library client).
//!
//! Every call passes `--ignore-config` and `--no-warnings`; YouTube gets the
//! same JS runtimes (`deno` + `node`) and the `chrome` impersonation target
//! when yt-dlp reports it available, as the Python options did.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::ErrorKind as IoErrorKind;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use aimd_core::{AimdError, Result};

use crate::cookies::CookieSource;

/// Install hint for a missing yt-dlp.
pub const YT_DLP_INSTALL_HINT: &str = "install it with `brew install yt-dlp`";

/// yt-dlp's networking default is 20s; YouTube timedtext conversion of long
/// ASR tracks can stall past that, so subtitle fetches wait longer.
pub const SUBTITLE_SOCKET_TIMEOUT_SECS: &str = "40";

/// Same-URL retries for a subtitle fetch (3 attempts total, as in Python),
/// with 0.5s then 1.5s backoff.
pub const SUBTITLE_RETRIES: &str = "2";
pub const SUBTITLE_RETRY_SLEEP: &str = "http:linear=0.5:1.5:1";

/// Lower-cased stderr fragments that mean the installed extractor is stale.
const STALE_EXTRACTOR_SIGNATURES: &[&str] = &[
    "please report this issue on",
    "confirm you are on the latest version",
    "nsig extraction failed",
    "signature extraction failed",
    "unable to extract",
    "failed to extract any player response",
    "only images are available for download",
    "unable to download api page",
];

/// Outcome of one yt-dlp run.
#[derive(Debug, Clone)]
pub struct Run {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

impl Run {
    /// The error text yt-dlp reported: its `ERROR:` lines, else the whole
    /// stderr, else the exit status. Mirrors `str(DownloadError)`.
    pub fn error_message(&self) -> String {
        let errors: Vec<&str> = self
            .stderr
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("ERROR:") || l.starts_with("yt-dlp: error:"))
            .collect();
        if !errors.is_empty() {
            return errors.join("\n");
        }
        let trimmed = self.stderr.trim();
        if trimmed.is_empty() {
            "yt-dlp exited with an error and no message".to_string()
        } else {
            trimmed.to_string()
        }
    }
}

/// Appends the `brew upgrade yt-dlp` hint when `message` matches a known
/// stale-extractor signature.
pub fn with_stale_hint(message: &str) -> String {
    let lower = message.to_lowercase();
    if STALE_EXTRACTOR_SIGNATURES.iter().any(|s| lower.contains(s)) {
        format!(
            "{message} (yt-dlp may be out of date for this site; run `brew upgrade yt-dlp` and retry)"
        )
    } else {
        message.to_string()
    }
}

/// How to launch yt-dlp: a program plus leading arguments (tests run a fake
/// script through `sh`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct YtDlp {
    pub command: Vec<OsString>,
}

impl YtDlp {
    /// `AIMD_YT_DLP` (path to the executable) or `yt-dlp` on `PATH`.
    pub fn from_env() -> Self {
        let program = std::env::var_os("AIMD_YT_DLP")
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| OsString::from("yt-dlp"));
        Self {
            command: vec![program],
        }
    }

    pub fn run(&self, args: &[OsString]) -> Result<Run> {
        let (program, prefix) = self
            .command
            .split_first()
            .ok_or_else(|| AimdError::backend_unavailable("yt-dlp command is empty"))?;
        let output = Command::new(program)
            .args(prefix)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|err| {
                if err.kind() == IoErrorKind::NotFound {
                    AimdError::backend_unavailable(format!(
                        "yt-dlp is required for URL extraction but was not found ({}); {YT_DLP_INSTALL_HINT}",
                        program.to_string_lossy()
                    ))
                } else {
                    AimdError::backend_unavailable(format!(
                        "failed to run yt-dlp ({}): {err}",
                        program.to_string_lossy()
                    ))
                }
            })?;
        Ok(Run {
            success: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Whether `--impersonate chrome` works (cached per command, like the
    /// Python `lru_cache`).
    pub fn impersonation_available(&self) -> bool {
        static CACHE: OnceLock<Mutex<HashMap<YtDlp, bool>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(hit) = cache.lock().map(|c| c.get(self).copied()).ok().flatten() {
            return hit;
        }
        let available = self
            .run(&[
                "--ignore-config".into(),
                "--list-impersonate-targets".into(),
            ])
            .map(|run| {
                run.success
                    && run.stdout.lines().chain(run.stderr.lines()).any(|line| {
                        let l = line.trim_start().to_lowercase();
                        l.starts_with("chrome") && !l.contains("unavailable")
                    })
            })
            .unwrap_or(false);
        if let Ok(mut c) = cache.lock() {
            c.insert(self.clone(), available);
        }
        available
    }

    /// Options shared by metadata, subtitle and audio runs (`_base_ydl_opts`).
    pub fn base_args(&self, platform: &str) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec!["--ignore-config".into(), "--no-warnings".into()];
        if platform == "youtube" {
            for a in [
                "--no-js-runtimes",
                "--js-runtimes",
                "deno",
                "--js-runtimes",
                "node",
            ] {
                args.push(a.into());
            }
            if self.impersonation_available() {
                args.push("--impersonate".into());
                args.push("chrome".into());
            }
        }
        args
    }
}

/// `apply_cookie_source`.
pub fn cookie_args(source: &CookieSource) -> Vec<OsString> {
    match source {
        CookieSource::File(path) => vec!["--cookies".into(), path.clone().into_os_string()],
        CookieSource::Browser { spec, .. } => {
            vec!["--cookies-from-browser".into(), spec.to_yt_dlp_arg().into()]
        }
        CookieSource::NoCookie => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_hint_only_for_known_signatures() {
        assert!(
            with_stale_hint("ERROR: [youtube] x: nsig extraction failed")
                .contains("brew upgrade yt-dlp")
        );
        assert!(!with_stale_hint("ERROR: Video unavailable").contains("brew upgrade"));
    }

    #[test]
    fn error_message_prefers_error_lines() {
        let run = Run {
            success: false,
            stdout: String::new(),
            stderr: "[info] noise\nERROR: [youtube] abc: Private video\n".into(),
        };
        assert_eq!(run.error_message(), "ERROR: [youtube] abc: Private video");
    }
}
