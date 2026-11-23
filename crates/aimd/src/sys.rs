//! Environment and subprocess access shared by the source adapters and runtime.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Captured subprocess result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandOutput {
    /// Exit code (`None` when killed by a signal).
    pub status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl CommandOutput {
    pub fn success(&self) -> bool {
        self.status == Some(0)
    }

    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    pub fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

/// System calls that source adapters can use without loading models.
pub trait System: Send + Sync {
    fn env_os(&self, key: &str) -> Option<OsString> {
        std::env::var_os(key)
    }

    fn env(&self, key: &str) -> Option<String> {
        self.env_os(key)?.into_string().ok()
    }

    fn which(&self, program: &str) -> Option<PathBuf> {
        use std::os::unix::fs::PermissionsExt;

        let is_exec = |p: &Path| {
            std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        };
        if program.contains('/') {
            let path = PathBuf::from(program);
            return is_exec(&path).then_some(path);
        }
        std::env::split_paths(&self.env("PATH")?)
            .map(|dir| dir.join(program))
            .find(|path| is_exec(path))
    }

    fn run(
        &self,
        program: &Path,
        args: &[OsString],
        cwd: Option<&Path>,
    ) -> std::io::Result<CommandOutput> {
        let mut command = Command::new(program);
        command.args(args).stdin(Stdio::null());
        if let Some(cwd) = cwd {
            command.current_dir(cwd);
        }
        let output = command.output()?;
        Ok(CommandOutput {
            status: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

pub(crate) struct NativeSystem;

impl System for NativeSystem {}

/// `AIMD_TEMP_DIR` (created when set) for MCP requests.
pub fn get_request_temp_dir() -> std::io::Result<Option<PathBuf>> {
    let Some(dir) = NativeSystem
        .env_os("AIMD_TEMP_DIR")
        .filter(|dir| !dir.is_empty())
    else {
        return Ok(None);
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir)?;
    Ok(Some(dir))
}
