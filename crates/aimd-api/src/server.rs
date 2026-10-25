//! Process entry: environment, loopback-only bind validation, serving, and
//! graceful shutdown (ports `aimd.interfaces.api.main`).

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use aimd_core::log_info;
use tokio::net::TcpListener;

use crate::app::{AppConfig, AppState, RuntimeProcessor, create_app};

/// Cooperative-cancellation grace period for active jobs at shutdown.
pub const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Reject non-loopback or invalid sidecar bind settings.
pub fn validate_sidecar_bind(host: &str, port: i64) -> Result<(), String> {
    if host != "localhost" {
        let address: IpAddr = host
            .parse()
            .map_err(|_| "AIMD_HTTP_HOST must be a loopback IP or localhost".to_string())?;
        let loopback = match address {
            IpAddr::V4(v4) => v4.is_loopback(),
            IpAddr::V6(v6) => {
                v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
            }
        };
        if !loopback {
            return Err("AIMD_HTTP_HOST must be loopback-only".into());
        }
    }
    if !(1..=65535).contains(&port) {
        return Err("AIMD_HTTP_PORT must be between 1 and 65535".into());
    }
    Ok(())
}

/// `AIMD_HTTP_HOST` (default `127.0.0.1`) and `AIMD_HTTP_PORT` (default
/// `8000`), validated.
pub fn bind_from_env(env: &dyn Fn(&str) -> Option<String>) -> Result<(String, u16), String> {
    let host = env("AIMD_HTTP_HOST").unwrap_or_else(|| "127.0.0.1".into());
    let raw_port = env("AIMD_HTTP_PORT").unwrap_or_else(|| "8000".into());
    let port: i64 = raw_port.trim().parse().map_err(|_| {
        format!("invalid literal for int() with base 10: '{raw_port}' (AIMD_HTTP_PORT)")
    })?;
    validate_sidecar_bind(&host, port)?;
    Ok((host, port as u16))
}

/// Resolve once SIGINT/SIGTERM arrives.
async fn termination_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = term => {},
    }
}

/// Serve `router` on `listener` until `shutdown` resolves; then stop
/// active jobs cooperatively (grace period, then abort) and drain
/// connections for at most another grace period.
pub async fn serve(
    listener: TcpListener,
    app: crate::app::App,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    grace: Duration,
) -> std::io::Result<()> {
    let state: Arc<AppState> = app.state.clone();
    let (drained_tx, drained_rx) = tokio::sync::oneshot::channel::<()>();
    let jobs_state = state.clone();
    let graceful = async move {
        shutdown.await;
        log_info!("Shutting down: cancelling active jobs");
        jobs_state.jobs.shutdown(grace).await;
        let _ = drained_tx.send(());
    };
    let server = axum::serve(listener, app.router).with_graceful_shutdown(graceful);
    let server = tokio::spawn(server.into_future());
    let mut server = server;
    tokio::select! {
        result = &mut server => return result.unwrap_or(Ok(())),
        _ = drained_rx => {}
    }
    match tokio::time::timeout(grace, &mut server).await {
        Ok(result) => result.unwrap_or(Ok(())),
        Err(_) => Ok(()),
    }
}

/// `aimd-api` main: returns the process exit code.
pub async fn run_from_env() -> i32 {
    let env = |k: &str| std::env::var(k).ok();
    let (host, port) = match bind_from_env(&env) {
        Ok(v) => v,
        Err(message) => {
            eprintln!("aimd-api: {message}");
            return 1;
        }
    };
    let app = match create_app(AppConfig::new(Arc::new(RuntimeProcessor::native()))) {
        Ok(app) => app,
        Err(message) => {
            eprintln!("aimd-api: {message}");
            return 1;
        }
    };
    let listener = match TcpListener::bind((host.as_str(), port)).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("aimd-api: could not bind {host}:{port}: {e}");
            return 1;
        }
    };
    let shown = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    log_info!("aimd-api running on http://{shown}:{port} (Press CTRL+C to quit)");
    match serve(listener, app, termination_signal(), SHUTDOWN_GRACE).await {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("aimd-api: {e}");
            1
        }
    }
}
