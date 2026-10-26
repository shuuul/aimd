//! `aimd-mcp`: MCP stdio server (ports `aimd.interfaces.mcp.app:main`).

use std::sync::Arc;

use aimd_mcp::AimdMcp;
use aimd_process::NativeRuntime;
use rmcp::{ServiceExt, transport::stdio};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("aimd-mcp {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "Usage: aimd-mcp\n\nRun the aimd MCP server over stdio (tools: healthz, process_input).\n\n\
             Register it with an MCP client as a stdio command; it reads JSON-RPC on stdin."
        );
        return;
    }
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
