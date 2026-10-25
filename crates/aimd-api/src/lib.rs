//! `aimd-api` (WS-19): the local HTTP sidecar. Ports
//! `aimd.interfaces.api` — `/healthz`, `/readyz`, `/v1/process`, the
//! `/v1/jobs` lifecycle with SSE events, `/v1/blobs`, bearer auth,
//! loopback-only binding, and canonical `AIMD_ALLOWED_ROOTS` checks.
//! JSON shapes, status codes, and error bodies match the FastAPI server.

pub mod app;
pub mod blobs;
mod docs;
pub mod jobs;
mod openapi;
pub mod paths;
pub mod schemas;
pub mod server;
pub mod sse;
pub mod util;
pub mod validate;

pub use app::{App, AppConfig, AppState, Processor, RuntimeProcessor, create_app};

#[cfg(test)]
mod tests;
