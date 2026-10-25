use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aimd_core::{ProcessHooks, ProcessInput, ProcessResult, TaskType, TextContext};
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use crate::app::{App, AppConfig, Processor, create_app};
use crate::schemas::{JobSnapshot, ProcessArtifact};

/// Self-deleting canonical scratch directory.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "aimd-api-test-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(std::fs::canonicalize(&dir).unwrap())
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn child(&self, name: &str) -> PathBuf {
        let p = self.0.join(name);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn result(task_type: TaskType, title: &str, chunks: &[&str], markdown: &str) -> ProcessResult {
    ProcessResult {
        task_type,
        text_context: TextContext {
            title: title.into(),
            chunk_list: chunks.iter().map(|s| s.to_string()).collect(),
            split_header_level: None,
        },
        markdown: markdown.into(),
        asset_base_uri: None,
        output_dir: None,
        platform: None,
    }
}

pub fn artifact(markdown: &str) -> ProcessArtifact {
    ProcessArtifact {
        task_type: TaskType::Convert,
        title: "Result".into(),
        markdown: markdown.into(),
        source_uri: "file:///tmp/source.md".into(),
        asset_base_uri: None,
        platform: None,
        output_file: None,
        output_dir: None,
    }
}

/// Processor returning a fixed result.
pub fn fixed(result: ProcessResult) -> Arc<dyn Processor> {
    Arc::new(move |_: &ProcessInput, _: &dyn ProcessHooks| Ok(result.clone()))
}

pub struct Builder {
    pub config: AppConfig,
    _blobs: Option<TempDir>,
}

/// Hermetic app: no roots, no token, private blob dir (env never read).
pub fn builder(processor: Arc<dyn Processor>) -> Builder {
    let blobs = TempDir::new("blobs");
    let mut config = AppConfig::new(processor);
    config.allowed_roots = Some(Vec::new());
    config.bearer_token = Some(String::new());
    config.blob_dir = Some(blobs.path().to_path_buf());
    Builder {
        config,
        _blobs: Some(blobs),
    }
}

impl Builder {
    pub fn roots(mut self, roots: &[&Path]) -> Self {
        self.config.allowed_roots = Some(roots.iter().map(|p| p.to_path_buf()).collect());
        self
    }
    pub fn token(mut self, token: &str) -> Self {
        self.config.bearer_token = Some(token.into());
        self
    }
    pub fn blob_dir(mut self, dir: &Path) -> Self {
        self.config.blob_dir = Some(dir.to_path_buf());
        self
    }
    pub fn build(self) -> TestApp {
        TestApp {
            app: create_app(self.config).unwrap(),
            _blobs: self._blobs,
        }
    }
}

pub struct TestApp {
    pub app: App,
    _blobs: Option<TempDir>,
}

pub struct Reply {
    pub status: StatusCode,
    pub headers: axum::http::HeaderMap,
    pub text: String,
}

impl Reply {
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.text).unwrap_or_else(|e| panic!("{e}: {}", self.text))
    }
}

pub async fn send(router: &Router, request: Request<Body>) -> Reply {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    Reply {
        status,
        headers,
        text: String::from_utf8_lossy(&bytes).into_owned(),
    }
}

impl TestApp {
    pub async fn get(&self, uri: &str) -> Reply {
        self.request("GET", uri, &[], Body::empty()).await
    }

    pub async fn delete(&self, uri: &str) -> Reply {
        self.request("DELETE", uri, &[], Body::empty()).await
    }

    pub async fn post_json(&self, uri: &str, body: serde_json::Value) -> Reply {
        self.request(
            "POST",
            uri,
            &[("content-type", "application/json")],
            Body::from(body.to_string()),
        )
        .await
    }

    pub async fn request(
        &self,
        method: &str,
        uri: &str,
        headers: &[(&str, &str)],
        body: Body,
    ) -> Reply {
        let mut builder = Request::builder().method(method).uri(uri);
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        send(&self.app.router, builder.body(body).unwrap()).await
    }

    /// Wait for a job to reach a terminal state (event cursor; no polling).
    pub async fn wait_terminal(&self, job_id: &str) -> JobSnapshot {
        self.app
            .state
            .jobs
            .events(job_id, 0)
            .unwrap()
            .collect()
            .await;
        self.app.state.jobs.get(job_id).unwrap()
    }
}

/// `multipart/form-data` body: `(name, filename, bytes)` parts.
pub fn multipart(parts: &[(&str, Option<&str>, &[u8])]) -> (String, Vec<u8>) {
    let boundary = "aimd-test-boundary-7MA4YWxkTrZu0gW";
    let mut body = Vec::new();
    for (name, filename, data) in parts {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match filename {
            Some(f) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\n\
                     Content-Type: text/markdown\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

impl TestApp {
    pub async fn upload(&self, parts: &[(&str, Option<&str>, &[u8])], auth: Option<&str>) -> Reply {
        let (ct, body) = multipart(parts);
        let mut headers = vec![("content-type", ct.as_str())];
        let bearer;
        if let Some(token) = auth {
            bearer = format!("Bearer {token}");
            headers.push(("authorization", bearer.as_str()));
        }
        self.request("POST", "/v1/blobs", &headers, Body::from(body))
            .await
    }
}
