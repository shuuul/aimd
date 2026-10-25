//! Hermetic ports of tests/test_api*.py (mock processors/runtimes, no
//! network, no sleeps: synchronization goes through job event cursors and
//! channels).

mod api;
mod blobs;
mod jobs;
mod sidecar;
mod support;
