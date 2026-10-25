//! tests/test_api_blobs.py

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use aimd_core::{ProcessHooks, ProcessInput, TaskType};
use serde_json::json;

use super::support::*;
use crate::paths::PathPolicy;
use crate::schemas::{JobState, ProcessRequest};
use crate::util::uuid4;

type Seen = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

fn recording() -> (Seen, Arc<dyn crate::Processor>) {
    let seen: Seen = Arc::default();
    let sink = seen.clone();
    let processor = Arc::new(move |input: &ProcessInput, _: &dyn ProcessHooks| {
        let bytes = std::fs::read(&input.input_source).unwrap_or_default();
        sink.lock()
            .unwrap()
            .push((input.input_source.clone(), bytes));
        Ok(result(TaskType::Convert, "notes", &["body"], "body"))
    });
    (seen, processor)
}

#[tokio::test]
async fn path_input_source_still_processes() {
    let tmp = TempDir::new("paths");
    let root = tmp.child("root");
    let source = root.join("source.md");
    std::fs::write(&source, "source").unwrap();
    let (seen, processor) = recording();
    let app = builder(processor).roots(&[&root]).build();
    let r = app
        .post_json("/v1/process", json!({"input_source": source}))
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(seen.lock().unwrap()[0].0, source.to_str().unwrap());
}

#[tokio::test]
async fn upload_blob_and_process_uses_stored_bytes() {
    let tmp = TempDir::new("upload");
    let root = tmp.child("root");
    let blob_dir = tmp.path().join("blobs");
    let payload = b"# hello from blob\n";
    let (seen, processor) = recording();
    let app = builder(processor)
        .roots(&[&root])
        .blob_dir(&blob_dir)
        .build();
    let upload = app
        .upload(&[("file", Some("notes.md"), payload)], None)
        .await;
    assert_eq!(upload.status, 200);
    let body = upload.json();
    assert_eq!(body["bytes"], payload.len());
    assert_eq!(body["filename"], "notes.md");
    let blob_id = body["blob_id"].as_str().unwrap().to_string();
    let processed = app
        .post_json("/v1/process", json!({"blob_id": blob_id}))
        .await;
    assert_eq!(processed.status, 200);
    assert_eq!(processed.json()["markdown"], "body");
    let job = app
        .post_json(
            "/v1/jobs",
            json!({"blob_id": blob_id, "task_type": "convert"}),
        )
        .await;
    assert_eq!(job.status, 202);
    app.wait_terminal(job.json()["job_id"].as_str().unwrap())
        .await;

    let seen = seen.lock().unwrap();
    let resolved = std::fs::canonicalize(&seen[0].0).unwrap();
    assert!(resolved.starts_with(std::fs::canonicalize(&blob_dir).unwrap()));
    assert_eq!(seen[0].1, payload);
    assert_eq!(resolved.file_name().unwrap(), "notes.md");
}

#[tokio::test]
async fn upload_honours_explicit_blob_id_and_sanitizes_names() {
    let tmp = TempDir::new("upload-id");
    let app = builder(recording().1).blob_dir(tmp.path()).build();
    let r = app
        .upload(
            &[
                ("file", Some("../a b?.md"), b"xy"),
                ("blob_id", None, b"{6F1C2D38-9A27-4C49-8A0C-54D4F1C1E0AA}"),
            ],
            None,
        )
        .await;
    assert_eq!(
        r.text,
        r#"{"blob_id":"6f1c2d38-9a27-4c49-8a0c-54d4f1c1e0aa","bytes":2,"filename":"../a b?.md"}"#
    );
    let stored = tmp.path().join("6f1c2d38-9a27-4c49-8a0c-54d4f1c1e0aa");
    assert_eq!(std::fs::read(stored.join("a_b_.md")).unwrap(), b"xy");
    assert_eq!(
        std::fs::read_to_string(stored.join("meta.json")).unwrap(),
        r#"{"filename":"a_b_.md"}"#
    );

    let bad = app
        .upload(
            &[("file", Some("a.md"), b"x"), ("blob_id", None, b"zz")],
            None,
        )
        .await;
    assert_eq!(
        (bad.status.as_u16(), bad.json()),
        (422, json!({"detail": "blob_id must be a UUID"}))
    );
    let missing = app.upload(&[("blob_id", None, b"")], None).await;
    assert_eq!(
        missing.text,
        r#"{"detail":[{"type":"missing","loc":["body","file"],"msg":"Field required","input":null}]}"#
    );
    // Nothing but the stored blob remains (staging files are cleaned up).
    let entries: Vec<PathBuf> = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(entries, vec![stored]);

    let unknown = app
        .post_json("/v1/process", json!({"blob_id": uuid4()}))
        .await;
    assert_eq!(
        (unknown.status.as_u16(), unknown.json()),
        (404, json!({"detail": "Blob not found"}))
    );
    let invalid = app
        .post_json("/v1/process", json!({"blob_id": "nope"}))
        .await;
    assert_eq!(invalid.status, 422);
    assert_eq!(invalid.json(), json!({"detail": "blob_id must be a UUID"}));
}

#[tokio::test]
async fn job_blob_path_stays_inside_sidecar_blob_dir() {
    let tmp = TempDir::new("jobblob");
    let root = tmp.child("root");
    let blob_dir = tmp.path().join("blobs");
    let (seen, processor) = recording();
    let app = builder(processor)
        .roots(&[&root])
        .blob_dir(&blob_dir)
        .build();
    let blob_id = app
        .upload(&[("file", Some("talk.md"), b"talk")], None)
        .await
        .json()["blob_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = app.post_json("/v1/jobs", json!({"blob_id": blob_id})).await;
    assert_eq!(created.status, 202);
    let snapshot = app
        .wait_terminal(created.json()["job_id"].as_str().unwrap())
        .await;
    assert_eq!(snapshot.state, JobState::Completed);
    assert_eq!(
        snapshot.artifact.unwrap().source_uri,
        format!("blob:{blob_id}")
    );
    let seen = seen.lock().unwrap();
    let resolved = std::fs::canonicalize(&seen[0].0).unwrap();
    assert!(resolved.starts_with(std::fs::canonicalize(&blob_dir).unwrap()));
}

#[tokio::test]
async fn missing_both_and_both_set_are_unprocessable() {
    let tmp = TempDir::new("both");
    let root = tmp.child("root");
    let app = builder(recording().1).roots(&[&root]).build();
    let missing = app.post_json("/v1/process", json!({})).await;
    let both = app
        .post_json(
            "/v1/process",
            json!({"input_source": root.join("a.md"), "blob_id": uuid4()}),
        )
        .await;
    assert_eq!(missing.status, 422);
    assert_eq!(both.status, 422);
}

#[tokio::test]
async fn blobs_requires_bearer_when_token_set() {
    let tmp = TempDir::new("blobauth");
    let root = tmp.child("root");
    let app = builder(recording().1)
        .roots(&[&root])
        .token("desktop-secret")
        .build();
    let unauth = app.upload(&[("file", Some("notes.md"), b"x")], None).await;
    let authorized = app
        .upload(&[("file", Some("notes.md"), b"x")], Some("desktop-secret"))
        .await;
    assert_eq!(unauth.status, 401);
    assert_eq!(unauth.json(), json!({"detail": "Unauthorized"}));
    assert_eq!(authorized.status, 200);
    assert_eq!(authorized.json()["bytes"], 1);
}

#[test]
fn path_policy_skips_blob_id_input() {
    let tmp = TempDir::new("policyblob");
    let root = tmp.child("root");
    let policy = PathPolicy::from_roots(&[&root]).unwrap();
    let blob_id = uuid4();
    let request = policy
        .normalize_request(&ProcessRequest::blob(blob_id.clone()))
        .unwrap();
    assert_eq!(request.blob_id, Some(blob_id));
    assert_eq!(request.input_source, None);
}
