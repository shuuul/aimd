//! tests/test_api.py

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use aimd_core::{AimdError, ProcessHooks, ProcessInput, TaskType};
use aimd_process::{AsrModel, LocalModality, OcrModel, Runtime};
use serde_json::json;

use super::support::*;
use crate::app::RuntimeProcessor;
use crate::util::Timestamp;

fn failing(err: AimdError) -> Arc<dyn crate::Processor> {
    Arc::new(move |_: &ProcessInput, _: &dyn ProcessHooks| Err(err.clone()))
}

#[tokio::test]
async fn healthz() {
    let app = builder(fixed(result(TaskType::Convert, "t", &[], ""))).build();
    let r = app.get("/healthz").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.json(), json!({"status": "ok"}));
    assert_eq!(r.headers["content-type"], "application/json");
    assert_eq!(app.get("/readyz").await.json(), json!({"status": "ok"}));
}

#[tokio::test]
async fn process_rejects_unknown_input() {
    let app = builder(failing(AimdError::unsupported("Unsupported input source"))).build();
    let r = app
        .post_json("/v1/process", json!({"input_source": "x"}))
        .await;
    assert_eq!(r.status, 400);
    assert_eq!(r.json(), json!({"detail": "Unsupported input source"}));
}

#[tokio::test]
async fn process_transcript_success_with_output_file() {
    let tmp = TempDir::new("out");
    let mut res = result(
        TaskType::Transcript,
        "mock-title",
        &["hello transcript"],
        "# Raw title\n\nhello transcript",
    );
    res.asset_base_uri = Some("https://example.com/watch".into());
    let app = builder(fixed(res)).build();
    let output_file = tmp.path().join("out.md");
    let r = app
        .post_json(
            "/v1/process",
            json!({"input_source": "input.mp3", "output_file": output_file}),
        )
        .await;
    assert_eq!(r.status, 200);
    assert_eq!(
        r.json(),
        json!({
            "task_type": "transcript",
            "title": "mock-title",
            "markdown": "# Raw title\n\nhello transcript",
            "asset_base_uri": "https://example.com/watch",
            "chunk_list": ["hello transcript"],
            "split_header_level": null,
            "platform": null,
            "output_file": output_file,
            "output_dir": null,
        })
    );
    // Field order matches the pydantic model.
    assert!(
        r.text
            .starts_with(r#"{"task_type":"transcript","title":"mock-title","markdown""#)
    );
    assert_eq!(
        std::fs::read_to_string(&output_file).unwrap(),
        "# Raw title\n\nhello transcript"
    );
}

#[tokio::test]
async fn process_empty_transcript_output_is_processing_failure() {
    let tmp = TempDir::new("empty");
    let app = builder(fixed(result(TaskType::Transcript, "empty", &[], ""))).build();
    let output_file = tmp.path().join("out.md");
    let r = app
        .post_json(
            "/v1/process",
            json!({"input_source": "input.mp3", "output_file": output_file}),
        )
        .await;
    assert_eq!(r.status, 500);
    assert_eq!(r.json()["detail"], "Transcription returned empty content");
    assert!(!output_file.exists());
}

#[tokio::test]
async fn process_forwards_model_precision_and_remote_settings() {
    let seen: Arc<Mutex<Option<ProcessInput>>> = Arc::default();
    let sink = seen.clone();
    let processor = Arc::new(move |input: &ProcessInput, _: &dyn ProcessHooks| {
        *sink.lock().unwrap() = Some(input.clone());
        Ok(result(TaskType::Transcript, "t", &["x"], "x"))
    });
    let app = builder(processor).build();
    let r = app
        .post_json(
            "/v1/process",
            json!({
                "input_source": "input.mp3",
                "model": "qwen3-asr-1.7b",
                "precision": "4bit",
                "asr_base_url": "http://asr.example/v1",
                "ocr_base_url": "http://ocr.example/v1",
                "context": "Kubernetes",
                "metadata_context": false,
                "language": "en",
                "start": 1,
                "end": "2",
                "cookies_from_browser": "chrome",
                "raw_transcript": true,
            }),
        )
        .await;
    assert_eq!(r.status, 200);
    let input = seen.lock().unwrap().clone().unwrap();
    assert_eq!(input.model.as_deref(), Some("qwen3-asr-1.7b"));
    assert_eq!(input.precision.as_deref(), Some("4bit"));
    assert_eq!(input.asr_base_url.as_deref(), Some("http://asr.example/v1"));
    assert_eq!(input.ocr_base_url.as_deref(), Some("http://ocr.example/v1"));
    assert_eq!(input.context.as_deref(), Some("Kubernetes"));
    assert!(!input.metadata_context && input.raw_transcript);
    assert_eq!((input.start, input.end), (Some(1), Some(2)));
    assert_eq!(input.language.as_deref(), Some("en"));
    assert_eq!(input.cookies_from_browser.as_deref(), Some("chrome"));
    // Relative paths are made absolute by the path policy.
    assert!(PathBuf::from(&input.input_source).is_absolute());
}

#[tokio::test]
async fn process_maps_domain_errors_to_http_status() {
    for (err, code) in [
        (AimdError::backend_unavailable("unavailable"), 422),
        (AimdError::failed("boom"), 500),
        (AimdError::not_found("Input file not found: x"), 404),
        (AimdError::cancelled("stopped"), 409),
    ] {
        let message = err.message.clone();
        let app = builder(failing(err)).build();
        let r = app
            .post_json("/v1/process", json!({"input_source": "audio.wav"}))
            .await;
        assert_eq!(r.status, code);
        assert_eq!(r.json(), json!({ "detail": message }));
    }
}

/// Runtime whose local ASR preflight fails (no environment, no models).
struct NoBackendRuntime;

impl Runtime for NoBackendRuntime {
    fn env(&self, _key: &str) -> Option<String> {
        None
    }
    fn preflight_local(&self, _modality: LocalModality) -> aimd_core::Result<()> {
        Err(AimdError::backend_unavailable(
            "ASR backend unavailable on this host",
        ))
    }
    fn load_asr(&self, _model_id: &str) -> aimd_core::Result<Arc<dyn AsrModel>> {
        unreachable!("preflight fails first")
    }
    fn load_ocr(&self, _model_id: &str) -> aimd_core::Result<Arc<dyn OcrModel>> {
        unreachable!("preflight fails first")
    }
}

#[tokio::test]
async fn process_preserves_backend_unavailable_from_processor() {
    let tmp = TempDir::new("asr");
    let audio = tmp.path().join("audio.wav");
    std::fs::write(&audio, "x").unwrap();
    let processor = Arc::new(RuntimeProcessor::new(Arc::new(NoBackendRuntime)));
    let app = builder(processor).build();
    let r = app
        .post_json("/v1/process", json!({"input_source": audio}))
        .await;
    assert_eq!(r.status, 422);
    assert!(
        r.json()["detail"]
            .as_str()
            .unwrap()
            .contains("ASR backend unavailable")
    );
    // Real routing: a missing local file maps to 404.
    let r = app
        .post_json(
            "/v1/process",
            json!({"input_source": tmp.path().join("missing.md")}),
        )
        .await;
    assert_eq!(r.status, 404);
}

#[tokio::test]
async fn job_contract_is_documented_in_openapi() {
    let app = builder(fixed(result(TaskType::Convert, "t", &[], ""))).build();
    let schema = app.get("/openapi.json").await.json();
    let components = &schema["components"]["schemas"];
    for name in [
        "JobCreated",
        "JobEvent",
        "JobSnapshot",
        "ProcessArtifact",
        "ProcessRequest",
    ] {
        assert!(components[name].is_object(), "{name}");
    }
    assert!(
        components["ProcessArtifact"]["example"]["markdown"]
            .as_str()
            .unwrap()
            .ends_with('\n')
    );
    assert_eq!(components["JobEvent"]["example"]["current"], 2);
    let events = &schema["paths"]["/v1/jobs/{job_id}/events"]["get"];
    assert!(events["parameters"].as_array().unwrap().iter().any(|p| {
        p["name"] == "Last-Event-ID" && p["in"] == "header" && p["required"] == false
    }));
    assert_eq!(
        events["responses"]["200"]["content"]["text/event-stream"]["schema"],
        json!({"$ref": "#/components/schemas/JobEvent"})
    );
    assert_eq!(schema["info"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(
        components["ProcessRequest"]["properties"]["model"]["description"],
        aimd_core::output::MODEL_HELP_TEXT
    );
    assert!(schema.get("security").is_none());
}

#[tokio::test]
async fn job_schema_freezes_cancellation_and_progress_contract() {
    let app = builder(fixed(result(TaskType::Convert, "t", &[], ""))).build();
    let schema = app.get("/openapi.json").await.json();
    let event = &schema["components"]["schemas"]["JobEvent"]["properties"];
    assert_eq!(
        event["state"]["enum"],
        json!(["queued", "running", "completed", "failed", "cancelled"])
    );
    assert_eq!(
        event["stage"]["anyOf"][0]["enum"],
        json!([
            "downloading",
            "extracting",
            "transcribing",
            "ocr",
            "converting",
            "saving"
        ])
    );
    assert_eq!(
        event["cancellation_status"]["enum"],
        json!(["none", "requested", "cancelled", "completed_after_request"])
    );
    assert_eq!(event["current"]["anyOf"][0]["minimum"], 0);
    assert_eq!(event["total"]["anyOf"][0]["minimum"], 0);
}

#[tokio::test]
async fn job_artifact_preserves_plain_markdown_exactly() {
    let tmp = TempDir::new("plain");
    let markdown = "\n# Exact title\n\nBody without a final newline";
    let mut res = result(TaskType::Convert, "Exact title", &["lossy body"], markdown);
    let base = format!("{}/", aimd_core::url::path_to_file_uri(tmp.path()));
    res.asset_base_uri = Some(base.clone());
    let app = builder(fixed(res)).build();
    let output_file = tmp.path().join("saved.md");
    let source = tmp.path().join("source.txt");
    let created = app
        .post_json(
            "/v1/jobs",
            json!({"input_source": source, "output_file": output_file}),
        )
        .await
        .json();
    let snapshot = app.wait_terminal(created["job_id"].as_str().unwrap()).await;
    let artifact = snapshot.artifact.unwrap();
    assert_eq!(artifact.markdown, markdown);
    assert_eq!(std::fs::read_to_string(&output_file).unwrap(), markdown);
    assert_eq!(
        artifact.source_uri,
        aimd_core::url::path_to_file_uri(&source)
    );
    assert_eq!(artifact.asset_base_uri.as_deref(), Some(base.as_str()));
    assert_eq!(
        artifact.output_file.as_deref(),
        Some(output_file.to_str().unwrap())
    );
    // Output requested -> the terminal stage is `saving`.
    assert_eq!(snapshot.stage, Some(crate::schemas::JobStage::Saving));
}

#[tokio::test]
async fn job_artifact_preserves_asset_output_directory() {
    let tmp = TempDir::new("assets");
    let output_dir = tmp.child("document");
    let markdown = "# Document\n\n![figure](images/figure.png)\n";
    std::fs::write(output_dir.join("document.md"), markdown).unwrap();
    let images = output_dir.join("images");
    std::fs::create_dir_all(&images).unwrap();
    std::fs::write(images.join("figure.png"), b"image").unwrap();
    let mut res = result(TaskType::Convert, "Document", &["lossy"], markdown);
    let base = format!("{}/", aimd_core::url::path_to_file_uri(&output_dir));
    res.asset_base_uri = Some(base.clone());
    res.output_dir = Some(output_dir.clone());
    let app = builder(fixed(res)).build();
    let ignored = tmp.path().join("ignored.md");
    let created = app
        .post_json(
            "/v1/jobs",
            json!({"input_source": tmp.path().join("source.docx"), "output_file": ignored}),
        )
        .await
        .json();
    let snapshot = app.wait_terminal(created["job_id"].as_str().unwrap()).await;
    let artifact = snapshot.artifact.unwrap();
    assert_eq!(artifact.markdown, markdown);
    assert_eq!(
        artifact.output_dir.as_deref(),
        Some(output_dir.to_str().unwrap())
    );
    assert_eq!(artifact.output_file, None);
    assert_eq!(artifact.asset_base_uri.as_deref(), Some(base.as_str()));
    assert!(!ignored.exists());
    assert_eq!(
        std::fs::read_to_string(output_dir.join("document.md")).unwrap(),
        markdown
    );
    assert_eq!(std::fs::read(images.join("figure.png")).unwrap(), b"image");
}

#[tokio::test]
async fn validation_errors_and_fallbacks_match_fastapi() {
    let app = builder(fixed(result(TaskType::Convert, "t", &[], ""))).build();
    let r = app
        .post_json(
            "/v1/process",
            json!({"input_source": "a", "task_type": "foo"}),
        )
        .await;
    assert_eq!(r.status, 422);
    assert_eq!(
        r.text,
        r#"{"detail":[{"type":"literal_error","loc":["body","task_type"],"msg":"Input should be 'transcript', 'convert' or 'ocr'","input":"foo","ctx":{"expected":"'transcript', 'convert' or 'ocr'"}}]}"#
    );
    let r = app
        .post_json("/v1/process", json!({"input_source": "file:///etc/passwd"}))
        .await;
    assert_eq!(r.status, 403);
    assert_eq!(
        r.json(),
        json!({"detail": "Only HTTP(S) URLs or local filesystem paths are allowed"})
    );
    let r = app.get("/nope").await;
    assert_eq!(
        (r.status.as_u16(), r.json()),
        (404, json!({"detail": "Not Found"}))
    );
    let r = app.get("/v1/process").await;
    assert_eq!(
        (r.status.as_u16(), r.json()),
        (405, json!({"detail": "Method Not Allowed"}))
    );
    assert!(r.headers["allow"].to_str().unwrap().contains("POST"));
    let r = app
        .request(
            "GET",
            "/v1/jobs/x/events",
            &[("last-event-id", "abc")],
            axum::body::Body::empty(),
        )
        .await;
    assert_eq!(
        r.text,
        r#"{"detail":[{"type":"int_parsing","loc":["header","Last-Event-ID"],"msg":"Input should be a valid integer, unable to parse string as an integer","input":"abc"}]}"#
    );
}

#[tokio::test]
async fn snapshot_json_matches_python_shape() {
    let app = builder(fixed(result(TaskType::Convert, "t", &["x"], "x"))).build();
    let created = app
        .post_json("/v1/jobs", json!({"input_source": "https://example.com/x"}))
        .await;
    assert_eq!(created.status, 202);
    let id = created.json()["job_id"].as_str().unwrap().to_string();
    app.wait_terminal(&id).await;
    let r = app.get(&format!("/v1/jobs/{id}")).await;
    let expected_prefix = format!(
        concat!(
            r#"{{"job_id":"{}","state":"completed","request":{{"input_source":"https://example.com/x","blob_id":null,"task_type":null,"output_file":null,"model":null,"precision":null,"#,
            r#""asr_base_url":null,"asr_model":null,"asr_api_key":null,"ocr_base_url":null,"ocr_model":null,"ocr_api_key":null,"context":null,"metadata_context":true,"#,
            r#""language":null,"start":null,"end":null,"save_original":null,"cookies":null,"cookies_from_browser":null,"raw_transcript":false}},"#,
            r#""artifact":{{"task_type":"convert","title":"t","markdown":"x","source_uri":"https://example.com/x","asset_base_uri":null,"platform":null,"output_file":null,"output_dir":null}},"#,
            r#""error":null,"stage":"downloading","current":null,"total":null,"cancellation_status":"none","created_at":""#
        ),
        id
    );
    assert!(r.text.starts_with(&expected_prefix), "{}", r.text);
    let created_at = r.json()["created_at"].as_str().unwrap().to_string();
    assert!(created_at.ends_with('Z') && created_at.contains('T'));
    assert!(Timestamp::now().to_string() >= created_at);
}
