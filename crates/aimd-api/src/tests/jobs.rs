//! tests/test_api_jobs.py

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aimd_core::{AimdError, ProcessHooks, ProcessInput, TaskType};
use serde_json::json;
use tokio::sync::{Notify, oneshot};

use super::support::*;
use crate::jobs::{JobContext, JobFailure, JobManager};
use crate::schemas::{CancellationStatus, JobStage, JobState, ProcessArtifact, ProcessRequest};

type Outcome = Result<ProcessArtifact, JobFailure>;

fn convert(source: &str) -> ProcessRequest {
    ProcessRequest {
        task_type: Some(TaskType::Convert),
        ..ProcessRequest::input(source)
    }
}

async fn wait_terminal(manager: &JobManager, id: &str) -> crate::schemas::JobSnapshot {
    manager.events(id, 0).unwrap().collect().await;
    manager.get(id).unwrap()
}

#[tokio::test]
async fn job_lifecycle_and_resumable_events() {
    let manager = JobManager::with_runner(|_req: ProcessRequest, ctx: JobContext| async move {
        ctx.report_progress(
            Some(JobStage::Converting),
            Some(1),
            Some(2),
            Some("Converting chapter 1 of 2"),
        );
        ctx.report_progress(
            Some(JobStage::Converting),
            Some(2),
            Some(2),
            Some("Converting chapter 2 of 2"),
        );
        Ok(artifact("# Result\n")) as Outcome
    });
    let created = manager.create(convert("document.md"));
    assert_eq!(created.state, JobState::Queued);

    let terminal = wait_terminal(&manager, &created.job_id).await;
    assert_eq!(terminal.state, JobState::Completed);
    assert_eq!(terminal.stage, Some(JobStage::Converting));
    assert_eq!((terminal.current, terminal.total), (Some(2), Some(2)));
    assert_eq!(terminal.artifact, Some(artifact("# Result\n")));

    let all = manager.events(&created.job_id, 0).unwrap().collect().await;
    let states: Vec<JobState> = all.iter().map(|e| e.state).collect();
    use JobState::*;
    assert_eq!(states, [Queued, Running, Running, Running, Completed]);
    let seqs: Vec<u64> = all.iter().map(|e| e.sequence).collect();
    assert_eq!(seqs, [1, 2, 3, 4, 5]);
    assert_eq!((all[2].current, all[2].total), (Some(1), Some(2)));
    assert_eq!(all[3].message.as_deref(), Some("Converting chapter 2 of 2"));

    let resumed = manager.events(&created.job_id, 1).unwrap().collect().await;
    let seqs: Vec<u64> = resumed.iter().map(|e| e.sequence).collect();
    assert_eq!(seqs, [2, 3, 4, 5]);
}

#[tokio::test]
async fn event_consumer_can_disconnect_and_resume() {
    let release = Arc::new(Notify::new());
    let gate = release.clone();
    let manager = JobManager::with_runner(move |_req: ProcessRequest, _ctx: JobContext| {
        let gate = gate.clone();
        async move {
            gate.notified().await;
            Ok(artifact("# Result\n")) as Outcome
        }
    });
    let created = manager.create(ProcessRequest::input("document.md"));
    let mut stream = manager.events(&created.job_id, 0).unwrap();
    assert_eq!(stream.next().await.unwrap().sequence, 1);
    drop(stream);

    release.notify_one();
    wait_terminal(&manager, &created.job_id).await;
    let resumed = manager.events(&created.job_id, 1).unwrap().collect().await;
    let seqs: Vec<u64> = resumed.iter().map(|e| e.sequence).collect();
    assert_eq!(seqs, [2, 3]);
}

#[tokio::test]
async fn job_failure_is_stored_and_emitted() {
    let manager = JobManager::with_runner(|_req: ProcessRequest, _ctx: JobContext| async {
        Err(JobFailure::Failed("processor failed".into())) as Outcome
    });
    let created = manager.create(ProcessRequest::input("broken.md"));
    let terminal = wait_terminal(&manager, &created.job_id).await;
    assert_eq!(terminal.state, JobState::Failed);
    assert_eq!(terminal.error.as_deref(), Some("processor failed"));
    assert_eq!(terminal.artifact, None);
    let events = manager.events(&created.job_id, 0).unwrap().collect().await;
    let last = events.last().unwrap();
    assert_eq!(last.state, JobState::Failed);
    assert_eq!(last.message.as_deref(), Some("Processing failed"));
}

#[tokio::test]
async fn runner_panic_fails_the_job() {
    let manager = JobManager::with_runner(|_req: ProcessRequest, _ctx: JobContext| async {
        if true {
            panic!("worker exploded");
        }
        Ok(artifact("never")) as Outcome
    });
    let created = manager.create(ProcessRequest::input("broken.md"));
    let terminal = wait_terminal(&manager, &created.job_id).await;
    assert_eq!(terminal.state, JobState::Failed);
    assert_eq!(terminal.error.as_deref(), Some("worker exploded"));
}

#[tokio::test]
async fn cancel_before_start_never_calls_runner() {
    let called = Arc::new(AtomicBool::new(false));
    let flag = called.clone();
    let manager = JobManager::with_runner(move |_req: ProcessRequest, _ctx: JobContext| {
        flag.store(true, Ordering::SeqCst);
        async { Ok(artifact("x")) as Outcome }
    });
    // Current-thread runtime: the job task has not run yet.
    let created = manager.create(ProcessRequest::input("document.md"));
    let cancelled = manager.cancel(&created.job_id).unwrap();
    tokio::task::yield_now().await;
    assert_eq!(cancelled.state, JobState::Cancelled);
    assert_eq!(cancelled.cancellation_status, CancellationStatus::Cancelled);
    let events = manager.events(&created.job_id, 0).unwrap().collect().await;
    let states: Vec<JobState> = events.iter().map(|e| e.state).collect();
    assert_eq!(states, [JobState::Queued, JobState::Cancelled]);
    assert!(!called.load(Ordering::SeqCst));
}

#[tokio::test]
async fn cancel_during_cooperative_work_stops_at_checkpoint() {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let started_tx = Mutex::new(Some(started_tx));
    let manager = JobManager::with_runner(move |_req: ProcessRequest, ctx: JobContext| {
        let tx = started_tx.lock().unwrap().take();
        async move {
            let _ = tx.unwrap().send(());
            ctx.cancelled().await;
            Err(JobFailure::Cancelled) as Outcome
        }
    });
    let created = manager.create(ProcessRequest::input("audio.wav"));
    started_rx.await.unwrap();
    let requested = manager.cancel(&created.job_id).unwrap();
    assert_eq!(requested.state, JobState::Running);
    assert_eq!(requested.cancellation_status, CancellationStatus::Requested);
    let terminal = wait_terminal(&manager, &created.job_id).await;
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(terminal.cancellation_status, CancellationStatus::Cancelled);
    assert_eq!(terminal.artifact, None);
}

#[tokio::test]
async fn cancel_during_uninterruptible_work_retains_completed_artifact() {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let started_tx = Mutex::new(Some(started_tx));
    let release = Arc::new(Notify::new());
    let gate = release.clone();
    let manager = JobManager::with_runner(move |_req: ProcessRequest, _ctx: JobContext| {
        let tx = started_tx.lock().unwrap().take();
        let gate = gate.clone();
        async move {
            let _ = tx.unwrap().send(());
            gate.notified().await;
            Ok(artifact("exact output")) as Outcome
        }
    });
    let created = manager.create(ProcessRequest::input("audio.wav"));
    started_rx.await.unwrap();
    let requested = manager.cancel(&created.job_id).unwrap();
    assert_eq!(requested.state, JobState::Running);
    assert_eq!(requested.cancellation_status, CancellationStatus::Requested);
    release.notify_one();
    let terminal = wait_terminal(&manager, &created.job_id).await;
    assert_eq!(terminal.state, JobState::Completed);
    assert_eq!(
        terminal.cancellation_status,
        CancellationStatus::CompletedAfterRequest
    );
    assert_eq!(terminal.artifact, Some(artifact("exact output")));
}

#[tokio::test]
async fn blocking_processor_uses_thread_safe_cancel_and_progress_hooks() {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let started_tx = Mutex::new(Some(started_tx));
    let processor = Arc::new(move |_input: &ProcessInput, hooks: &dyn ProcessHooks| {
        hooks.progress(
            "converting",
            Some(1),
            Some(2),
            Some("Completed blocking conversion step 1 of 2"),
        );
        if let Some(tx) = started_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
        // A pipeline checkpoint loop on the blocking pool.
        while !hooks.is_cancelled() {
            std::thread::park_timeout(std::time::Duration::from_millis(1));
        }
        Err(AimdError::cancelled(
            "cancelled at blocking worker checkpoint",
        ))
    });
    let app = builder(processor).build();
    let created = app
        .post_json(
            "/v1/jobs",
            json!({"input_source": "document.epub", "task_type": "convert"}),
        )
        .await
        .json();
    let id = created["job_id"].as_str().unwrap();
    started_rx.await.unwrap();
    let requested = app.delete(&format!("/v1/jobs/{id}")).await;
    assert_eq!(requested.status, 202);
    assert_eq!(requested.json()["cancellation_status"], "requested");
    let terminal = app.wait_terminal(id).await;
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(terminal.cancellation_status, CancellationStatus::Cancelled);
    let events = app.app.state.jobs.events(id, 0).unwrap().collect().await;
    let progress = events.iter().find(|e| e.current == Some(1)).unwrap();
    assert_eq!(
        (progress.stage, progress.current, progress.total),
        (Some(JobStage::Converting), Some(1), Some(2))
    );
}

#[tokio::test]
async fn terminal_history_is_bounded() {
    let manager = JobManager::new(
        |_req: ProcessRequest, _ctx: JobContext| async { Ok(artifact("x")) as Outcome },
        2,
    );
    let mut ids = Vec::new();
    for source in ["one.md", "two.md", "three.md"] {
        let created = manager.create(ProcessRequest::input(source));
        wait_terminal(&manager, &created.job_id).await;
        ids.push(created.job_id);
    }
    assert!(manager.get(&ids[0]).is_err());
    assert_eq!(manager.get(&ids[1]).unwrap().state, JobState::Completed);
    assert_eq!(manager.get(&ids[2]).unwrap().state, JobState::Completed);
}

#[tokio::test]
async fn job_api_lifecycle_sse_and_unknown_job() {
    let markdown = "\n# Exact\n";
    let app = builder(fixed(result(
        TaskType::Convert,
        "Exact",
        &["lossy"],
        markdown,
    )))
    .build();
    let created = app
        .post_json(
            "/v1/jobs",
            json!({"input_source": "document.md", "task_type": "convert"}),
        )
        .await;
    assert_eq!(created.status, 202);
    let created = created.json();
    assert_eq!(created["state"], "queued");
    let id = created["job_id"].as_str().unwrap();
    assert_eq!(created["status_url"], format!("/v1/jobs/{id}"));
    assert_eq!(created["events_url"], format!("/v1/jobs/{id}/events"));
    app.wait_terminal(id).await;

    let snapshot = app
        .get(created["status_url"].as_str().unwrap())
        .await
        .json();
    assert_eq!(snapshot["state"], "completed");
    assert_eq!(snapshot["artifact"]["markdown"], markdown);

    let events = app
        .request(
            "GET",
            created["events_url"].as_str().unwrap(),
            &[("last-event-id", "1")],
            axum::body::Body::empty(),
        )
        .await;
    assert_eq!(events.status, 200);
    assert_eq!(
        events.headers["content-type"],
        "text/event-stream; charset=utf-8"
    );
    assert_eq!(events.headers["cache-control"], "no-cache");
    assert_eq!(events.headers["x-accel-buffering"], "no");
    let frames: Vec<&str> = events.text.split_terminator("\n\n").collect();
    assert_eq!(frames.len(), 2);
    assert!(frames[0].starts_with("id: 2\nevent: job\ndata: {"));
    let parsed: Vec<serde_json::Value> = events
        .text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let seqs: Vec<u64> = parsed
        .iter()
        .map(|e| e["sequence"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, [2, 3]);
    assert_eq!(parsed[1]["state"], "completed");

    for (method, uri) in [
        ("GET", "/v1/jobs/unknown"),
        ("DELETE", "/v1/jobs/unknown"),
        ("GET", "/v1/jobs/unknown/events"),
    ] {
        let r = app
            .request(method, uri, &[], axum::body::Body::empty())
            .await;
        assert_eq!(r.status, 404);
        assert_eq!(r.json(), json!({"detail": "Job not found"}));
    }
}
