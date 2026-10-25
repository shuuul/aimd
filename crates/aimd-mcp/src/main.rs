//! `aimd-mcp`: MCP stdio server (ports `aimd.interfaces.mcp.app:main`).

use std::sync::Arc;

use aimd_mcp::AimdMcp;
use aimd_process::NativeRuntime;
use rmcp::{ServiceExt, transport::stdio};

#[tokio::main]
async fn main() {
    let server = AimdMcp::new(Arc::new(NativeRuntime));
    let service = match server.serve(stdio()).await {
        Ok(service) => service,
        Err(err) => {
            eprintln!("aimd-mcp: failed to start: {err}");
            std::process::exit(1);
        }
    };
    if let Err(err) = service.waiting().await {
        eprintln!("aimd-mcp: {err}");
        std::process::exit(1);
    }
}
