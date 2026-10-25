//! `aimd-api` binary: the local HTTP sidecar (see `docs/sidecar.md`).
//! Configuration is environment-only, like the Python server:
//! `AIMD_HTTP_HOST`, `AIMD_HTTP_PORT`, `AIMD_API_TOKEN`,
//! `AIMD_ALLOWED_ROOTS`, `AIMD_BLOB_DIR`, `AIMD_TEMP_DIR`.

fn main() {
    if std::env::args().skip(1).any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: aimd-api\n\nRun the aimd HTTP API (loopback-only).\n\nEnvironment:\n  \
             AIMD_HTTP_HOST      loopback bind address (default 127.0.0.1)\n  \
             AIMD_HTTP_PORT      bind port (default 8000)\n  \
             AIMD_API_TOKEN      require `Authorization: Bearer <token>` on every endpoint\n  \
             AIMD_ALLOWED_ROOTS  path-separated allow-list for local paths\n  \
             AIMD_BLOB_DIR       sidecar-owned blob directory (default: a temp dir)\n  \
             AIMD_TEMP_DIR       temporary downloads/transcoding directory"
        );
        return;
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("aimd-api: failed to start the async runtime: {e}");
            std::process::exit(1);
        }
    };
    let code = runtime.block_on(aimd_api::server::run_from_env());
    // Blocking workers may still be inside uninterruptible model calls;
    // exit without waiting for them (the parent owns the deadline).
    std::process::exit(code);
}
