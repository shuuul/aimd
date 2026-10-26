//! Ports `tests/test_mcp_server.py` over a mock runtime and an in-memory
//! JSON-RPC transport. No checkpoints, network, or subprocesses.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aimd_core::{AimdError, ProcessHooks, ProcessInput};
use aimd_process::{AsrModel, LocalModality, OcrFailure, OcrModel, Runtime, UrlOutput};
use rmcp::ServiceExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::*;

#[derive(Default)]
struct MockRuntime {
    url_markdown: String,
    seen: Mutex<Vec<ProcessInput>>,
}

struct MockOcr;

impl OcrModel for MockOcr {
    fn recognize(&self, _image: &Path) -> Result<String, OcrFailure> {
        Ok("ocr".to_string())
    }
}

impl Runtime for MockRuntime {
    fn preflight_local(&self, _modality: LocalModality) -> aimd_core::Result<()> {
        Ok(())
    }
    fn load_asr(&self, _model_id: &str) -> aimd_core::Result<Arc<dyn AsrModel>> {
        Err(AimdError::backend_unavailable("no asr in tests"))
    }
    fn load_ocr(&self, _model_id: &str) -> aimd_core::Result<Arc<dyn OcrModel>> {
        Ok(Arc::new(MockOcr))
    }
    fn url_transcript(
        &self,
        request: &ProcessInput,
        _hooks: &dyn ProcessHooks,
    ) -> aimd_core::Result<UrlOutput> {
        self.seen.lock().unwrap().push(request.clone());
        Ok(UrlOutput {
            markdown: self.url_markdown.clone(),
            title: "mock-title".into(),
            platform: "generic".into(),
            segments: None,
        })
    }
}

fn server(url_markdown: &str) -> (AimdMcp, Arc<MockRuntime>) {
    let rt = Arc::new(MockRuntime {
        url_markdown: url_markdown.into(),
        ..Default::default()
    });
    (AimdMcp::new(rt.clone()), rt)
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aimd-mcp-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const URL: &str = "https://example.com/watch";

#[tokio::test]
async fn healthz_reports_ok() {
    let (s, _) = server("");
    assert_eq!(
        s.call("healthz", Map::new()).await.unwrap(),
        json!({"status": "ok"})
    );
}

#[tokio::test]
async fn process_input_transcript_persists_output() {
    let dir = scratch("persist");
    let out = dir.join("out.md");
    let (s, _) = server("# Raw\n\nhello");
    let result = s
        .call(
            "process_input",
            args(json!({"input_source": URL, "output_file": out})),
        )
        .await
        .unwrap();
    assert_eq!(result["task_type"], "transcript");
    assert_eq!(result["markdown"], "# Raw\n\nhello");
    assert_eq!(result["asset_base_uri"], URL);
    assert!(result["output_file"].is_string());
    assert_eq!(std::fs::read_to_string(&out).unwrap(), result["markdown"]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn empty_transcript_output_is_processing_failure() {
    let dir = scratch("empty");
    let out = dir.join("out.md");
    let (s, _) = server("");
    let err = s
        .call(
            "process_input",
            args(json!({"input_source": URL, "output_file": out})),
        )
        .await
        .unwrap_err();
    assert!(
        err.0.contains("Transcription returned empty content"),
        "{}",
        err.0
    );
    assert!(!out.exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn accepts_valid_task_type() {
    let dir = scratch("ocr");
    let png = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../aimd-engine/tests/golden/ocr_e2e/page_text.png");
    let (s, _) = server("");
    let result = s
        .call(
            "process_input",
            args(json!({
                "input_source": png,
                "task_type": "ocr",
                "output_file": dir.join("out.md"),
            })),
        )
        .await
        .unwrap();
    assert_eq!(result["task_type"], "ocr");
    assert_eq!(result["markdown"], "ocr");
    assert_eq!(std::fs::read_to_string(dir.join("out.md")).unwrap(), "ocr");
    assert!(result.get("segments").is_none());
    // include_segments returns the page segment without anchoring.
    let result = s
        .call(
            "process_input",
            args(json!({"input_source": png, "task_type": "ocr", "include_segments": true})),
        )
        .await
        .unwrap();
    assert_eq!(result["markdown"], "ocr");
    assert_eq!(
        result["segments"],
        json!([{"text": "ocr", "kind": "page", "page": 1, "bbox": [0.0, 0.0, 1.0, 1.0]}])
    );
    let result = s
        .call(
            "process_input",
            args(json!({"input_source": png, "task_type": "ocr", "anchors": true, "include_segments": true})),
        )
        .await
        .unwrap();
    assert_eq!(
        result["markdown"],
        "<!-- aimd:loc page=1 bbox=0.000,0.000,1.000,1.000 kind=page -->\nocr"
    );
    assert_eq!(
        result["segments"],
        json!([{"text": "ocr", "kind": "page", "page": 1, "bbox": [0.0, 0.0, 1.0, 1.0]}])
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn rejects_invalid_task_type() {
    let (s, _) = server("x");
    let err = s
        .call(
            "process_input",
            args(json!({"input_source": "input.mp3", "task_type": "nope"})),
        )
        .await
        .unwrap_err();
    assert!(
        err.0
            .starts_with("Error executing tool process_input: 1 validation error")
    );
    assert!(
        err.0
            .contains("Input should be 'transcript', 'convert' or 'ocr'")
    );
}

#[tokio::test]
async fn missing_and_mistyped_arguments_are_validation_errors() {
    let (s, _) = server("x");
    let err = s
        .call("process_input", args(json!({"start": "1"})))
        .await
        .unwrap_err();
    assert!(err.0.contains("2 validation errors"), "{}", err.0);
    assert!(err.0.contains("input_source\n  Field required"));
    assert!(err.0.contains("start\n  Input should be a valid integer"));
}

#[tokio::test]
async fn forwards_precision_remote_and_flags() {
    let (s, rt) = server("x");
    s.call(
        "process_input",
        args(json!({
            "input_source": URL,
            "model": "qwen3-asr-1.7b",
            "precision": "4bit",
            "context": "names",
            "metadata_context": false,
            "raw_transcript": true,
            "asr_base_url": "http://asr.example/v1",
            "ocr_base_url": "http://ocr.example/v1",
            "cookies": "",
        })),
    )
    .await
    .unwrap();
    let seen = rt.seen.lock().unwrap();
    let r = &seen[0];
    assert_eq!(r.model.as_deref(), Some("qwen3-asr-1.7b"));
    assert_eq!(r.precision.as_deref(), Some("4bit"));
    assert_eq!(r.context.as_deref(), Some("names"));
    assert!(!r.metadata_context && r.raw_transcript);
    assert_eq!(r.asr_base_url.as_deref(), Some("http://asr.example/v1"));
    assert_eq!(r.ocr_base_url.as_deref(), Some("http://ocr.example/v1"));
    assert_eq!(r.cookies, None);
}

#[tokio::test]
async fn defaults_match_python_signature() {
    let (s, rt) = server("x");
    s.call("process_input", args(json!({"input_source": URL})))
        .await
        .unwrap();
    let r = rt.seen.lock().unwrap()[0].clone();
    assert!(r.metadata_context && !r.raw_transcript);
    assert_eq!(r.task_type, None);
}

#[test]
fn schema_exposes_precision_and_task_enum() {
    let tools = tools();
    let process = tools.iter().find(|t| t.name == "process_input").unwrap();
    let props = process.input_schema.get("properties").unwrap();
    assert!(props.get("precision").is_some());
    assert_eq!(
        props["task_type"]["anyOf"][0]["enum"],
        json!(["transcript", "convert", "ocr"])
    );
    assert_eq!(process.input_schema["required"], json!(["input_source"]));
    assert_eq!(props.as_object().unwrap().len(), 22);
}

#[test]
fn text_content_keeps_python_key_order() {
    let value = result_payload(
        &aimd_core::ProcessResult {
            task_type: aimd_core::TaskType::Ocr,
            text_context: aimd_core::TextContext {
                title: "t".into(),
                chunk_list: vec!["a".into()],
                split_header_level: None,
            },
            markdown: "a".into(),
            asset_base_uri: None,
            output_dir: None,
            platform: None,
            segments: None,
        },
        None,
        None,
        true,
    );
    let text = aimd_core::output::payload_text(&value);
    let positions: Vec<usize> = aimd_core::output::PAYLOAD_KEY_ORDER
        .iter()
        .map(|k| text.find(&format!("\"{k}\"")).unwrap())
        .collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]));
    assert!(text.contains("\"chunk_list\": [\n    \"a\"\n  ]"));
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    assert_eq!(value["segments"], json!([]));
}

/// Full JSON-RPC round trip over an in-memory duplex transport.
#[tokio::test]
async fn wire_initialize_list_and_call() {
    let (s, _) = server("hello");
    let (client_io, server_io) = tokio::io::duplex(1 << 16);
    let (server_read, server_write) = tokio::io::split(server_io);
    let serving = tokio::spawn(async move {
        let running = s.serve((server_read, server_write)).await.unwrap();
        let _ = running.waiting().await;
    });
    let (client_read, mut client_write) = tokio::io::split(client_io);
    let mut lines = BufReader::new(client_read).lines();
    let mut send = async |msg: Value| {
        client_write
            .write_all(format!("{msg}\n").as_bytes())
            .await
            .unwrap();
    };

    send(
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},
        "clientInfo":{"name":"t","version":"0"}}}),
    )
    .await;
    let init: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
    assert_eq!(init["result"]["serverInfo"]["name"], "aimd");
    assert!(init["result"]["capabilities"]["tools"].is_object());

    send(json!({"jsonrpc":"2.0","method":"notifications/initialized"})).await;
    send(json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})).await;
    let list: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["healthz", "process_input"]);

    send(
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
        "name":"process_input","arguments":{"input_source":URL}}}),
    )
    .await;
    let call: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
    assert_eq!(call["result"]["isError"], false);
    assert_eq!(call["result"]["structuredContent"]["markdown"], "hello");
    let text = call["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(text).unwrap(),
        call["result"]["structuredContent"]
    );

    send(
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
        "name":"process_input","arguments":{"input_source":URL,"task_type":"nope"}}}),
    )
    .await;
    let bad: Value = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
    assert_eq!(bad["result"]["isError"], true);
    assert!(bad["result"].get("structuredContent").is_none());

    drop(send);
    drop(client_write);
    serving.abort();
}
