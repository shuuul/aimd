//! axum application (ports `aimd.interfaces.api.app.create_app`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aimd_core::output::{PersistError, get_request_temp_dir, persist_result_output_if_requested};
use aimd_core::url::path_to_file_uri;
use aimd_core::{AimdError, ProcessHooks, ProcessInput, ProcessResult, log_error, log_warn};
use aimd_process::Runtime;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Multipart, Path as UrlPath, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Serialize;
use tokio::io::AsyncWriteExt;

use crate::blobs::{BlobError, BlobStore};
use crate::jobs::{JobContext, JobFailure, JobManager, panic_message};
use crate::paths::PathPolicy;
use crate::schemas::{
    BlobUploadResponse, HealthResponse, JobCreated, ProcessArtifact, ProcessRequest,
    ProcessResponse,
};
use crate::sse::{ChannelBody, frame_job_event};
use crate::util::{constant_time_eq, expand_user, resolve_lenient, url_scheme};
use crate::validate::{ErrorItem, LocPart, coerce_int_str, detail_body, parse_process_request};
use crate::{docs, openapi};

/// The processing seam: a blocking `ProcessInput -> ProcessResult` call.
/// Production uses [`RuntimeProcessor`]; tests inject fakes.
pub trait Processor: Send + Sync + 'static {
    fn process(
        &self,
        input: &ProcessInput,
        hooks: &dyn ProcessHooks,
    ) -> aimd_core::Result<ProcessResult>;
}

impl<F> Processor for F
where
    F: Fn(&ProcessInput, &dyn ProcessHooks) -> aimd_core::Result<ProcessResult>
        + Send
        + Sync
        + 'static,
{
    fn process(
        &self,
        input: &ProcessInput,
        hooks: &dyn ProcessHooks,
    ) -> aimd_core::Result<ProcessResult> {
        self(input, hooks)
    }
}

/// `aimd_process::process_input_with` over an injected [`Runtime`].
pub struct RuntimeProcessor {
    pub runtime: Arc<dyn Runtime>,
}

impl RuntimeProcessor {
    pub fn new(runtime: Arc<dyn Runtime>) -> Self {
        Self { runtime }
    }

    /// The real system runtime.
    pub fn native() -> Self {
        Self::new(Arc::new(aimd_process::NativeRuntime))
    }
}

impl Processor for RuntimeProcessor {
    fn process(
        &self,
        input: &ProcessInput,
        hooks: &dyn ProcessHooks,
    ) -> aimd_core::Result<ProcessResult> {
        aimd_process::process_input_with(input, hooks, &*self.runtime)
    }
}

/// `create_app` keyword arguments. `None` reads the matching environment
/// variable like the Python factory (`AIMD_ALLOWED_ROOTS`,
/// `AIMD_API_TOKEN`, `AIMD_BLOB_DIR`).
pub struct AppConfig {
    pub processor: Arc<dyn Processor>,
    pub allowed_roots: Option<Vec<PathBuf>>,
    pub bearer_token: Option<String>,
    pub blob_dir: Option<PathBuf>,
    pub max_terminal_jobs: usize,
}

impl AppConfig {
    pub fn new(processor: Arc<dyn Processor>) -> Self {
        Self {
            processor,
            allowed_roots: None,
            bearer_token: None,
            blob_dir: None,
            max_terminal_jobs: 64,
        }
    }
}

/// Shared handler state.
pub struct AppState {
    pub processor: Arc<dyn Processor>,
    pub blobs: Arc<BlobStore>,
    pub jobs: Arc<JobManager>,
    pub policy: PathPolicy,
    pub token: Option<String>,
    openapi: String,
}

/// Built application: the router plus its state (for shutdown).
pub struct App {
    pub router: Router,
    pub state: Arc<AppState>,
}

/// Build the application. Fails on an invalid allowed root or an
/// unusable blob directory (Python raises at import/startup too).
pub fn create_app(config: AppConfig) -> Result<App, String> {
    let blobs = match &config.blob_dir {
        Some(dir) => BlobStore::new(dir),
        None => BlobStore::from_env_value(std::env::var("AIMD_BLOB_DIR").ok().as_deref()),
    }
    .map_err(|e| format!("Unable to create blob directory: {e}"))?;
    let blobs = Arc::new(blobs);
    let policy = match &config.allowed_roots {
        Some(roots) => PathPolicy::from_roots(roots),
        None => PathPolicy::from_env_value(std::env::var("AIMD_ALLOWED_ROOTS").ok().as_deref()),
    }
    .map_err(|e| e.0)?;
    let token = config
        .bearer_token
        .clone()
        .or_else(|| std::env::var("AIMD_API_TOKEN").ok())
        .filter(|t| !t.is_empty());

    let processor = config.processor.clone();
    let runner_blobs = blobs.clone();
    let jobs = JobManager::new(
        move |request: ProcessRequest, ctx: JobContext| {
            run_job(request, ctx, processor.clone(), runner_blobs.clone())
        },
        config.max_terminal_jobs,
    );
    let state = Arc::new(AppState {
        processor: config.processor,
        blobs,
        jobs,
        policy,
        openapi: openapi::document(token.is_some()),
        token,
    });
    let router = Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(health))
        .route("/v1/blobs", post(upload_blob))
        .route("/v1/process", post(process))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/{job_id}", get(get_job).delete(cancel_job))
        .route("/v1/jobs/{job_id}/events", get(job_events))
        .route("/openapi.json", get(openapi_json))
        .route("/docs", get(|| async { html(docs::SWAGGER_UI) }))
        .route("/redoc", get(|| async { html(docs::REDOC) }))
        .fallback(|| async { detail(StatusCode::NOT_FOUND, "Not Found") })
        .method_not_allowed_fallback(|| async {
            detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed")
        })
        .layer(DefaultBodyLimit::disable())
        .layer(middleware::from_fn_with_state(state.clone(), authenticate))
        .with_state(state.clone());
    Ok(App { router, state })
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response {
    let body = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        body,
    )
        .into_response()
}

/// `HTTPException(status_code, detail=message)` body.
pub fn detail(status: StatusCode, message: &str) -> Response {
    #[derive(Serialize)]
    struct Detail<'a> {
        detail: &'a str,
    }
    json_response(status, &Detail { detail: message })
}

fn status(code: u16) -> StatusCode {
    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn validation(items: &[ErrorItem]) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        detail_body(items),
    )
        .into_response()
}

fn html(body: &'static str) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        body,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

async fn authenticate(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(token) = &state.token {
        let supplied = request
            .headers()
            .get(header::AUTHORIZATION)
            .map(|v| v.as_bytes())
            .unwrap_or(b"");
        let expected = format!("Bearer {token}");
        if !constant_time_eq(supplied, expected.as_bytes()) {
            return detail(StatusCode::UNAUTHORIZED, "Unauthorized");
        }
    }
    next.run(request).await
}

// ---------------------------------------------------------------------------
// Request preparation and processing
// ---------------------------------------------------------------------------

impl AppState {
    /// Allow-root normalization plus blob existence checks.
    fn prepare_request(&self, request: &ProcessRequest) -> Result<ProcessRequest, Box<Response>> {
        let request = self
            .policy
            .normalize_request(request)
            .map_err(|e| Box::new(detail(StatusCode::FORBIDDEN, &e.0)))?;
        if let Some(blob_id) = request.blob_id.as_deref().filter(|b| !b.is_empty()) {
            match self.blobs.resolve(blob_id) {
                Ok(_) => {}
                Err(BlobError::InvalidId) => {
                    return Err(Box::new(detail(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "blob_id must be a UUID",
                    )));
                }
                Err(_) => return Err(Box::new(detail(StatusCode::NOT_FOUND, "Blob not found"))),
            }
        }
        Ok(request)
    }
}

fn resolved_input_source(request: &ProcessRequest, blobs: &BlobStore) -> Result<String, BlobError> {
    if let Some(blob_id) = request.blob_id.as_deref().filter(|b| !b.is_empty()) {
        return Ok(blobs.resolve(blob_id)?.path.to_string_lossy().into_owned());
    }
    Ok(request.input_source.clone().unwrap_or_default())
}

/// `_build_process_input`.
pub fn build_process_input(
    request: &ProcessRequest,
    temp_dir: Option<PathBuf>,
    blobs: &BlobStore,
) -> Result<ProcessInput, BlobError> {
    Ok(ProcessInput {
        input_source: resolved_input_source(request, blobs)?,
        task_type: request.task_type,
        model: request.model.clone(),
        language: request.language.clone(),
        start: request.start,
        end: request.end,
        save_original: request
            .save_original
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from),
        cookies: request
            .cookies
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from),
        cookies_from_browser: request.cookies_from_browser.clone(),
        temp_dir,
        raw_transcript: request.raw_transcript,
        precision: request.precision.clone(),
        context: request.context.clone(),
        metadata_context: request.metadata_context,
        asr_base_url: request.asr_base_url.clone(),
        asr_model: request.asr_model.clone(),
        asr_api_key: request.asr_api_key.clone(),
        ocr_base_url: request.ocr_base_url.clone(),
        ocr_model: request.ocr_model.clone(),
        ocr_api_key: request.ocr_api_key.clone(),
    })
}

/// `_source_uri`: `blob:<id>`, the URL itself, or a canonical file URI.
pub fn source_uri(request: &ProcessRequest) -> String {
    if let Some(blob_id) = request.blob_id.as_deref().filter(|b| !b.is_empty()) {
        return format!("blob:{blob_id}");
    }
    let input = request.input_source.as_deref().unwrap_or("");
    if matches!(url_scheme(input).as_deref(), Some("http" | "https")) {
        return input.to_string();
    }
    path_to_file_uri(&resolve_lenient(&expand_user(Path::new(input))))
}

fn output_path(request: &ProcessRequest) -> Option<PathBuf> {
    request.output_file.as_deref().map(PathBuf::from)
}

const IGNORED_OUTPUT_WARNING: &str =
    "Ignoring output_file for document asset conversions; output is a directory.";

/// A failed synchronous request: domain errors keep their status.
enum SyncFailure {
    Aimd(AimdError),
    Other(String),
}

impl From<PersistError> for SyncFailure {
    fn from(e: PersistError) -> Self {
        match e {
            PersistError::Aimd(e) => SyncFailure::Aimd(e),
            PersistError::Io(e) => SyncFailure::Other(e.to_string()),
        }
    }
}

fn process_sync(
    state: &AppState,
    request: &ProcessRequest,
) -> Result<ProcessResponse, SyncFailure> {
    let temp_dir = get_request_temp_dir().map_err(|e| SyncFailure::Other(e.to_string()))?;
    let input = build_process_input(request, temp_dir, &state.blobs)
        .map_err(|e| SyncFailure::Other(e.to_string()))?;
    let result = state
        .processor
        .process(&input, &aimd_core::NoHooks)
        .map_err(SyncFailure::Aimd)?;
    let persisted = persist_result_output_if_requested(&result, output_path(request).as_deref())?;
    if persisted.ignored_output_file {
        log_warn!("{IGNORED_OUTPUT_WARNING}");
    }
    Ok(ProcessResponse {
        task_type: result.task_type,
        title: result.text_context.title,
        markdown: result.markdown,
        asset_base_uri: result.asset_base_uri,
        chunk_list: result.text_context.chunk_list,
        split_header_level: result.text_context.split_header_level,
        platform: result.platform,
        output_file: persisted.output_file,
        output_dir: persisted.output_dir,
    })
}

/// `_run_job`: blocking processing on the blocking pool with job hooks.
async fn run_job(
    request: ProcessRequest,
    ctx: JobContext,
    processor: Arc<dyn Processor>,
    blobs: Arc<BlobStore>,
) -> Result<ProcessArtifact, JobFailure> {
    let temp_dir = get_request_temp_dir().map_err(|e| JobFailure::Failed(e.to_string()))?;
    if ctx.is_cancelled() {
        return Err(JobFailure::Cancelled);
    }
    let input = build_process_input(&request, temp_dir, &blobs)
        .map_err(|e| JobFailure::Failed(e.to_string()))?;
    let hooks = ctx.clone();
    let result = tokio::task::spawn_blocking(move || processor.process(&input, &hooks))
        .await
        .map_err(|e| {
            JobFailure::Failed(if e.is_panic() {
                panic_message(e.into_panic())
            } else {
                "processing task was cancelled".to_string()
            })
        })??;
    let persisted = persist_result_output_if_requested(&result, output_path(&request).as_deref())
        .map_err(|e| JobFailure::Failed(e.to_string()))?;
    if persisted.ignored_output_file {
        log_warn!("{IGNORED_OUTPUT_WARNING}");
    }
    Ok(ProcessArtifact {
        task_type: result.task_type,
        title: result.text_context.title,
        markdown: result.markdown,
        source_uri: source_uri(&request),
        asset_base_uri: result.asset_base_uri,
        platform: result.platform,
        output_file: persisted.output_file,
        output_dir: persisted.output_dir,
    })
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health() -> Response {
    json_response(StatusCode::OK, &HealthResponse::default())
}

async fn openapi_json(State(state): State<Arc<AppState>>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        state.openapi.clone(),
    )
        .into_response()
}

fn parse_body(headers: &HeaderMap, body: &Bytes) -> Result<ProcessRequest, Box<Response>> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok());
    parse_process_request(content_type, body).map_err(|items| Box::new(validation(&items)))
}

async fn process(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let request = match parse_body(&headers, &body).and_then(|r| state.prepare_request(&r)) {
        Ok(r) => r,
        Err(response) => return *response,
    };
    let worker = state.clone();
    let outcome = tokio::task::spawn_blocking(move || process_sync(&worker, &request)).await;
    match outcome {
        Ok(Ok(response)) => json_response(StatusCode::OK, &response),
        Ok(Err(SyncFailure::Aimd(e))) => detail(status(e.status_code()), &e.message),
        Ok(Err(SyncFailure::Other(message))) => {
            log_error!("HTTP processing failed: {message}");
            detail(StatusCode::INTERNAL_SERVER_ERROR, &message)
        }
        Err(join) => {
            let message = if join.is_panic() {
                panic_message(join.into_panic())
            } else {
                "processing task was cancelled".to_string()
            };
            log_error!("HTTP processing failed: {message}");
            detail(StatusCode::INTERNAL_SERVER_ERROR, &message)
        }
    }
}

async fn upload_blob(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let missing_file = || validation(&[ErrorItem::missing(vec!["body".into(), "file".into()])]);
    let parse_error = || {
        detail(
            StatusCode::BAD_REQUEST,
            "There was an error parsing the body",
        )
    };
    let Ok(mut multipart) = multipart else {
        return missing_file();
    };
    let mut staged: Option<(PathBuf, u64, String)> = None;
    let mut text_file: Option<String> = None;
    let mut blob_id: Option<String> = None;
    let cleanup = |staged: &Option<(PathBuf, u64, String)>| {
        if let Some((path, _, _)) = staged {
            let _ = std::fs::remove_file(path);
        }
    };
    loop {
        let mut field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(_) => {
                cleanup(&staged);
                return parse_error();
            }
        };
        match field.name() {
            Some("file") if staged.is_none() => {
                let filename = field.file_name().map(str::to_owned).unwrap_or_default();
                if filename.is_empty() {
                    match field.text().await {
                        Ok(text) => text_file = Some(text),
                        Err(_) => return parse_error(),
                    }
                    continue;
                }
                let path = state.blobs.staging_path();
                let mut file = match tokio::fs::File::create(&path).await {
                    Ok(f) => f,
                    Err(e) => return detail(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
                };
                let mut size = 0u64;
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            size += chunk.len() as u64;
                            if let Err(e) = file.write_all(&chunk).await {
                                let _ = std::fs::remove_file(&path);
                                return detail(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
                            }
                        }
                        Ok(None) => break,
                        Err(_) => {
                            let _ = std::fs::remove_file(&path);
                            return parse_error();
                        }
                    }
                }
                if let Err(e) = file.flush().await {
                    let _ = std::fs::remove_file(&path);
                    return detail(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string());
                }
                staged = Some((path, size, filename));
            }
            Some("blob_id") => match field.text().await {
                Ok(text) => blob_id = Some(text),
                Err(_) => {
                    cleanup(&staged);
                    return parse_error();
                }
            },
            _ => while let Ok(Some(_)) = field.chunk().await {},
        }
    }
    let Some((path, size, filename)) = staged else {
        if let Some(text) = text_file {
            return validation(&[ErrorItem::new(
                "value_error",
                vec!["body".into(), "file".into()],
                "Value error, Expected UploadFile, received: <class 'str'>",
                serde_json::Value::String(text),
            )
            .with_ctx(serde_json::json!({"error": {}}))]);
        }
        return missing_file();
    };
    match state
        .blobs
        .put_staged(&path, size, Some(&filename), blob_id.as_deref())
    {
        Ok(stored) => json_response(
            StatusCode::OK,
            &BlobUploadResponse {
                blob_id: stored.blob_id,
                bytes: stored.size,
                filename: Some(filename),
            },
        ),
        Err(BlobError::InvalidId) => {
            let _ = std::fs::remove_file(&path);
            detail(StatusCode::UNPROCESSABLE_ENTITY, "blob_id must be a UUID")
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            detail(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string())
        }
    }
}

async fn create_job(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request = match parse_body(&headers, &body).and_then(|r| state.prepare_request(&r)) {
        Ok(r) => r,
        Err(response) => return *response,
    };
    let snapshot = state.jobs.create(request);
    json_response(
        StatusCode::ACCEPTED,
        &JobCreated {
            status_url: format!("/v1/jobs/{}", snapshot.job_id),
            events_url: format!("/v1/jobs/{}/events", snapshot.job_id),
            job_id: snapshot.job_id,
            state: snapshot.state,
        },
    )
}

async fn get_job(State(state): State<Arc<AppState>>, UrlPath(job_id): UrlPath<String>) -> Response {
    match state.jobs.get(&job_id) {
        Ok(snapshot) => json_response(StatusCode::OK, &snapshot),
        Err(_) => detail(StatusCode::NOT_FOUND, "Job not found"),
    }
}

async fn cancel_job(
    State(state): State<Arc<AppState>>,
    UrlPath(job_id): UrlPath<String>,
) -> Response {
    match state.jobs.cancel(&job_id) {
        Ok(snapshot) => json_response(StatusCode::ACCEPTED, &snapshot),
        Err(_) => detail(StatusCode::NOT_FOUND, "Job not found"),
    }
}

async fn job_events(
    State(state): State<Arc<AppState>>,
    UrlPath(job_id): UrlPath<String>,
    headers: HeaderMap,
) -> Response {
    let mut after = 0i64;
    if let Some(raw) = headers.get("last-event-id") {
        let text = String::from_utf8_lossy(raw.as_bytes()).into_owned();
        match coerce_int_str(&text) {
            Ok(v) => after = v,
            Err((kind, msg)) => {
                return validation(&[ErrorItem::new(
                    kind,
                    vec![LocPart::from("header"), LocPart::from("Last-Event-ID")],
                    msg,
                    serde_json::Value::String(text),
                )]);
            }
        }
    }
    let Ok(mut cursor) = state.jobs.events(&job_id, after) else {
        return detail(StatusCode::NOT_FOUND, "Job not found");
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(16);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                event = cursor.next() => {
                    let Some(event) = event else { break };
                    if tx.send(Bytes::from(frame_job_event(&event))).await.is_err() {
                        break;
                    }
                }
                () = tx.closed() => break,
            }
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::new(ChannelBody::new(rx)))
        .unwrap_or_else(|_| detail(StatusCode::INTERNAL_SERVER_ERROR, "stream failed"))
}
