//! Shared contract for every aimd crate: request/result models, domain
//! errors, and cooperative cancellation/progress hooks.
//!
//! Ports `aimd.core.{models,errors,precision}`. Pipelines (`aimd-doc`,
//! `aimd-url`, `aimd-process`) return [`AimdError`]; interfaces map
//! [`ErrorKind::status_code`] onto HTTP/MCP/exit codes.

pub mod error;
pub mod hooks;
pub mod model;

pub use error::{AimdError, ErrorKind, Result};
pub use hooks::{NoHooks, ProcessHooks};
pub use model::{InputRoute, ProcessInput, ProcessResult, SourceKind, TaskType, TextContext};
