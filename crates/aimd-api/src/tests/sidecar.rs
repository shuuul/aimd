//! tests/test_api_sidecar.py

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aimd_core::{ProcessHooks, ProcessInput, TaskType};
use axum::body::Body;
use serde_json::json;
use tokio::sync::oneshot;

use super::support::*;
use crate::jobs::{JobContext, JobFailure, JobManager};
use crate::paths::PathPolicy;
use crate::schemas::{CancellationStatus, JobState, ProcessArtifact, ProcessRequest};
use crate::server::{bind_from_env, serve, validate_sidecar_bind};

type Outcome = Result<ProcessArtifact, JobFailure>;

#[test]
fn path_policy_normalizes_paths_inside_allowed_root() {
    let tmp = TempDir::new("norm");
    let root = tmp.child("root");
    let policy = PathPolicy::from_roots(&[&root]).unwrap();
    let request = ProcessRequest {
        output_file: Some(root.join("output/result.md").to_string_lossy().into()),
        ..ProcessRequest::input(root.join("folder/../source.md").to_string_lossy())
    };
    let request = policy.normalize_request(&request).unwrap();
    assert_eq!(
        request.input_source.as_deref(),
        Some(root.join("source.md").to_str().unwrap())
    );
    assert_eq!(
        request.output_file.as_deref(),
        Some(root.join("output/result.md").to_str().unwrap())
    );
}

#[test]
fn path_policy_rejects_traversal_and_symlink_escape() {
    let tmp = TempDir::new("escape");
    let root = tmp.child("root");
    let outside = tmp.child("outside");
    let secret = outside.join("secret.md");
    std::fs::write(&secret, "secret").unwrap();
    std::os::unix::fs::symlink(&secret, root.join("escape.md")).unwrap();
    let policy = PathPolicy::from_roots(&[&root]).unwrap();

    let err = policy
        .normalize(root.join("../outside/new.md"), "output")
        .unwrap_err();
    assert!(err.0.contains("outside configured allowed roots"));
    let err = policy
        .normalize(root.join("escape.md"), "input")
        .unwrap_err();
    assert_eq!(err.0, "input path is outside configured allowed roots");
    let err = policy
        .normalize_request(&ProcessRequest::input("file:///etc/passwd"))
        .unwrap_err();
    assert!(err.0.starts_with("Only HTTP(S)"));
    // A symlinked directory inside the root that points back inside is fine.
    std::os::unix::fs::symlink(&root, root.join("self")).unwrap();
    assert_eq!(
        policy.normalize(root.join("self/a.md"), "input").unwrap(),
        root.join("a.md")
    );
    // Missing roots and non-directories are rejected at startup.
    assert!(PathPolicy::from_roots(&[tmp.path().join("missing")]).is_err());
    let err = PathPolicy::from_roots(&[&secret]).unwrap_err();
    assert!(err.0.starts_with("Allowed root is not a directory"));
}

#[tokio::test]
async fn api_enforces_allowed_roots_before_processing() {
    let tmp = TempDir::new("roots");
    let root = tmp.child("root");
    let source = root.join("source.md");
    std::fs::write(&source, "source").unwrap();
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = seen.clone();
    let processor = Arc::new(move |input: &ProcessInput, _: &dyn ProcessHooks| {
        sink.lock().unwrap().push(input.input_source.clone());
        Ok(result(TaskType::Convert, "source", &["body"], "body"))
    });
    let app = builder(processor).roots(&[&root]).build();
    let accepted = app
        .post_json("/v1/process", json!({"input_source": source}))
        .await;
    let rejected_input = app
        .post_json(
            "/v1/process",
            json!({"input_source": tmp.path().join("outside.md")}),
        )
        .await;
    let rejected_output = app
        .post_json(
            "/v1/jobs",
            json!({"input_source": source, "output_file": tmp.path().join("outside.md")}),
        )
        .await;
    assert_eq!(accepted.status, 200);
    assert_eq!(seen.lock().unwrap().as_slice(), [source.to_str().unwrap()]);
    assert_eq!(rejected_input.status, 403);
    assert_eq!(
        rejected_input.json(),
        json!({"detail": "input path is outside configured allowed roots"})
    );
    assert_eq!(rejected_output.status, 403);
    assert_eq!(
        rejected_output.json(),
        json!({"detail": "output_file path is outside configured allowed roots"})
    );
}

#[tokio::test]
async fn sidecar_bearer_auth_protects_health_jobs_and_openapi() {
    let tmp = TempDir::new("auth");
    let root = tmp.child("root");
    let app = builder(fixed(result(TaskType::Convert, "t", &[], "")))
        .roots(&[&root])
        .token("desktop-secret")
        .build();
    assert_eq!(app.get("/healthz").await.status, 401);
    assert_eq!(app.get("/readyz").await.status, 401);
    assert_eq!(app.get("/openapi.json").await.status, 401);
    assert_eq!(app.get("/no-such-route").await.status, 401);
    let wrong = app
        .request(
            "GET",
            "/healthz",
            &[("authorization", "Bearer wrong-secret")],
            Body::empty(),
        )
        .await;
    assert_eq!(wrong.status, 401);
    assert_eq!(wrong.json(), json!({"detail": "Unauthorized"}));
    let unauth_job = app
        .post_json("/v1/jobs", json!({"input_source": root.join("source.md")}))
        .await;
    assert_eq!(unauth_job.status, 401);

    let auth = [("authorization", "Bearer desktop-secret")];
    for uri in ["/healthz", "/readyz"] {
        let r = app.request("GET", uri, &auth, Body::empty()).await;
        assert_eq!(r.json(), json!({"status": "ok"}));
    }
    let openapi = app
        .request("GET", "/openapi.json", &auth, Body::empty())
        .await
        .json();
    assert_eq!(
        openapi["components"]["securitySchemes"]["bearerAuth"],
        json!({"type": "http", "scheme": "bearer"})
    );
    assert_eq!(openapi["security"], json!([{"bearerAuth": []}]));
}

#[test]
fn sidecar_bind_accepts_loopback() {
    for host in ["127.0.0.1", "::1", "localhost", "127.1.2.3"] {
        validate_sidecar_bind(host, 49152).unwrap();
    }
}

#[test]
fn sidecar_bind_rejects_non_loopback() {
    for host in ["0.0.0.0", "192.168.1.10", "example.com", "::"] {
        let err = validate_sidecar_bind(host, 8000).unwrap_err();
        assert!(err.contains("loopback"), "{host}: {err}");
    }
}

#[test]
fn sidecar_bind_rejects_invalid_port() {
    for port in [0, 65536] {
        let err = validate_sidecar_bind("127.0.0.1", port).unwrap_err();
        assert!(err.contains("between 1 and 65535"));
    }
}

#[test]
fn bind_reads_environment_defaults() {
    assert_eq!(
        bind_from_env(&|_| None).unwrap(),
        ("127.0.0.1".to_string(), 8000)
    );
    let env = |k: &str| match k {
        "AIMD_HTTP_HOST" => Some("::1".to_string()),
        "AIMD_HTTP_PORT" => Some("49152".to_string()),
        _ => None,
    };
    assert_eq!(bind_from_env(&env).unwrap(), ("::1".to_string(), 49152));
    let bad = |k: &str| (k == "AIMD_HTTP_PORT").then(|| "http".to_string());
    assert!(bind_from_env(&bad).is_err());
    let lan = |k: &str| (k == "AIMD_HTTP_HOST").then(|| "0.0.0.0".to_string());
    assert!(bind_from_env(&lan).unwrap_err().contains("loopback-only"));
}

fn started_runner(cooperative: bool) -> (oneshot::Receiver<()>, impl crate::jobs::JobRunner) {
    let (tx, rx) = oneshot::channel::<()>();
    let tx = Mutex::new(Some(tx));
    let runner = move |_req: ProcessRequest, ctx: JobContext| {
        let tx = tx.lock().unwrap().take();
        async move {
            let _ = tx.unwrap().send(());
            if cooperative {
                ctx.cancelled().await;
                Err(JobFailure::Cancelled) as Outcome
            } else {
                std::future::pending::<()>().await;
                Ok(artifact("never")) as Outcome
            }
        }
    };
    (rx, runner)
}

#[tokio::test]
async fn job_manager_shutdown_cooperatively_stops_active_work() {
    let (started, runner) = started_runner(true);
    let manager = JobManager::with_runner(runner);
    let created = manager.create(ProcessRequest::input("source.md"));
    started.await.unwrap();
    manager.shutdown(Duration::from_secs(1)).await;
    let snapshot = manager.get(&created.job_id).unwrap();
    assert_eq!(snapshot.state, JobState::Cancelled);
    assert_eq!(snapshot.cancellation_status, CancellationStatus::Cancelled);
}

#[tokio::test]
async fn job_manager_shutdown_forces_task_after_grace_period() {
    let (started, runner) = started_runner(false);
    let manager = JobManager::with_runner(runner);
    let created = manager.create(ProcessRequest::input("source.md"));
    started.await.unwrap();
    manager.shutdown(Duration::ZERO).await;
    let snapshot = manager.get(&created.job_id).unwrap();
    assert_eq!(snapshot.state, JobState::Cancelled);
    let events = manager.events(&created.job_id, 0).unwrap().collect().await;
    assert_eq!(
        events.last().unwrap().message.as_deref(),
        Some("Job task cancelled")
    );
}

/// Real socket: an SSE stream follows live progress and ends when the
/// server shuts down (jobs cancelled cooperatively first).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn served_sse_stream_follows_job_and_shutdown() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let (started_tx, started_rx) = oneshot::channel::<()>();
    let started_tx = Mutex::new(Some(started_tx));
    let processor = Arc::new(move |_: &ProcessInput, hooks: &dyn ProcessHooks| {
        hooks.progress(
            "transcribing",
            Some(1),
            Some(3),
            Some("Transcribing segment 1 of 3"),
        );
        if let Some(tx) = started_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
        while !hooks.is_cancelled() {
            std::thread::park_timeout(Duration::from_millis(1));
        }
        Err(aimd_core::AimdError::cancelled("stopped"))
    });
    let test_app = builder(processor).build();
    let state = test_app.app.state.clone();
    let snapshot = state.jobs.create(ProcessRequest {
        task_type: Some(TaskType::Transcript),
        ..ProcessRequest::input("/tmp/clip.wav")
    });
    started_rx.await.unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let app = crate::app::App {
        router: test_app.app.router.clone(),
        state: state.clone(),
    };
    let server = tokio::spawn(serve(
        listener,
        app,
        async {
            let _ = stop_rx.await;
        },
        Duration::from_secs(5),
    ));

    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!(
        "GET /v1/jobs/{}/events HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        snapshot.job_id
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    // Read until the progress frame (sequence 3) arrives, then shut down.
    let mut received = Vec::new();
    let mut buf = [0u8; 4096];
    while !String::from_utf8_lossy(&received).contains("id: 3\n") {
        let n = stream.read(&mut buf).await.unwrap();
        assert!(n > 0, "stream closed early");
        received.extend_from_slice(&buf[..n]);
    }
    stop_tx.send(()).unwrap();
    stream.read_to_end(&mut received).await.unwrap();
    server.await.unwrap().unwrap();

    let text = String::from_utf8_lossy(&received);
    assert!(text.starts_with("HTTP/1.1 200 OK"));
    assert!(text.contains("content-type: text/event-stream; charset=utf-8"));
    assert!(text.contains("Transcribing segment 1 of 3"));
    assert!(text.contains("Job stopped at a cancellation checkpoint"));
    let final_state = state.jobs.get(&snapshot.job_id).unwrap();
    assert_eq!(final_state.state, JobState::Cancelled);
    assert_eq!(
        final_state.stage,
        Some(crate::schemas::JobStage::Transcribing)
    );
}

#[test]
fn sidecar_contract_documentation_covers_runtime_boundary() {
    let contract = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/sidecar.md"),
    )
    .unwrap();
    for required in [
        "AIMD_ALLOWED_ROOTS",
        "AIMD_API_TOKEN",
        "AIMD_HTTP_PORT",
        "GET /readyz",
        "Last-Event-ID",
        "completed_after_request",
        "caller-owned `output_dir`",
        "POST /v1/blobs",
        "blob_id",
    ] {
        assert!(contract.contains(required), "{required}");
    }
}
