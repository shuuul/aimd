//! Convert documents, URLs, audio and images to Markdown and chunked text.

mod anchors;
mod asr;
mod audio;
mod doc;
mod error;
mod ext;
#[cfg(any(feature = "native", test))]
mod hf;
mod hooks;
mod http;
pub mod log;
mod model;
mod models;
mod native;
mod ocr;
pub mod output;
mod precision;
mod remote;
mod remote_config;
mod route;
mod runtime;
mod sys;
mod text;
mod url;
#[doc(hidden)]
pub mod util;

pub use error::{AimdError, ErrorKind, Result};
pub use hooks::{NoHooks, ProcessHooks};
pub use model::*;
pub use native::NativeRuntime;
pub use precision::{Precision, QUANTIZED_PRECISIONS, normalize_precision};
pub use remote_config::{RemoteBackendConfig, RemoteModality, resolve_remote_backend};
pub use route::{ensure_supported_input, get_input_route};
pub use runtime::{
    AlignedTranscript, AsrFailure, AsrModel, DocOutput, LocalModality, LocatedOcr, OcrFailure,
    OcrModel, Runtime, UrlOutput, process_input, process_input_with,
};
pub use sys::{CommandOutput, System, get_request_temp_dir};
pub use text::{MAX_CHUNK_SIZE, extract_title_from_content, text_context_from_markdown};
pub use url::parse::{detect_platform, is_url};
