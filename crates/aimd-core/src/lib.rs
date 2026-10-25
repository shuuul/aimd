//! Shared contract for every aimd crate: request/result models, domain
//! errors, cooperative cancellation/progress hooks, and the small
//! interface-independent helpers every pipeline needs.
//!
//! Ports `aimd.core.{models,errors,precision,remote}`, the Markdown
//! title/chunk shaping of `aimd.core.process`, the extension tables of the
//! ASR/OCR/doc plugins, and `aimd.interfaces.output`. Pipelines (`aimd-doc`,
//! `aimd-url`, `aimd-process`) return [`AimdError`]; interfaces map
//! [`ErrorKind::status_code`] onto HTTP/MCP/exit codes.

pub mod error;
pub mod ext;
pub mod hooks;
pub mod log;
pub mod model;
pub mod output;
pub mod precision;
pub mod pyfmt;
pub mod remote;
pub mod text;
pub mod url;

pub use error::{AimdError, ErrorKind, Result};
pub use hooks::{NoHooks, ProcessHooks};
pub use model::{InputRoute, ProcessInput, ProcessResult, SourceKind, TaskType, TextContext};
pub use precision::{Precision, normalize_precision};
pub use remote::{RemoteBackendConfig, RemoteModality, resolve_remote_backend};
pub use text::{MAX_CHUNK_SIZE, extract_title_from_content, text_context_from_markdown};
pub use url::{detect_platform, is_url};
