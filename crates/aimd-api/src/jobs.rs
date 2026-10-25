//! In-memory job lifecycle and resumable event storage for the local
//! sidecar (ports `aimd.interfaces.api.jobs`).
//!
//! Each job runs as a tokio task. Cancellation is cooperative: `cancel`
//! raises a flag the runner observes at safe checkpoints (through
//! [`JobContext`], which also implements [`ProcessHooks`] for blocking
//! pipelines). A runner that finishes after a request keeps its artifact
//! (`completed_after_request`).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use aimd_core::{AimdError, ErrorKind, ProcessHooks, TaskType};
use tokio::sync::watch;
use tokio::task::{AbortHandle, JoinHandle};

use crate::schemas::{
    CancellationStatus, JobEvent, JobSnapshot, JobStage, JobState, ProcessArtifact, ProcessRequest,
};
use crate::util::{Timestamp, uuid4};

/// Why a runner did not produce an artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobFailure {
    /// Stopped at a cooperative cancellation checkpoint.
    Cancelled,
    /// Any other failure (`str(exc)`).
    Failed(String),
}

impl From<AimdError> for JobFailure {
    fn from(e: AimdError) -> Self {
        if e.kind == ErrorKind::Cancelled {
            JobFailure::Cancelled
        } else {
            JobFailure::Failed(e.message)
        }
    }
}

pub type JobFuture = Pin<Box<dyn Future<Output = Result<ProcessArtifact, JobFailure>> + Send>>;

/// Executes one job request.
pub trait JobRunner: Send + Sync + 'static {
    fn run(&self, request: ProcessRequest, ctx: JobContext) -> JobFuture;
}

impl<F, Fut> JobRunner for F
where
    F: Fn(ProcessRequest, JobContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ProcessArtifact, JobFailure>> + Send + 'static,
{
    fn run(&self, request: ProcessRequest, ctx: JobContext) -> JobFuture {
        Box::pin(self(request, ctx))
    }
}

/// The job id is unknown or expired.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobNotFound(pub String);

struct JobRecord {
    request: ProcessRequest,
    state: JobState,
    stage: Option<JobStage>,
    artifact: Option<ProcessArtifact>,
    error: Option<String>,
    cancellation_status: CancellationStatus,
    current: Option<u64>,
    total: Option<u64>,
    created_at: Timestamp,
    updated_at: Timestamp,
    events: Vec<JobEvent>,
}

struct JobCell {
    job_id: String,
    /// Insertion order (stable tie-break when pruning).
    seq: u64,
    record: Mutex<JobRecord>,
    /// Bumped on every emitted event.
    events_tx: watch::Sender<u64>,
    cancel_tx: watch::Sender<bool>,
    task: Mutex<Option<JoinHandle<()>>>,
    runner_abort: Mutex<Option<AbortHandle>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl JobCell {
    fn lock(&self) -> MutexGuard<'_, JobRecord> {
        lock(&self.record)
    }

    /// Append an event built from the record's current fields.
    fn emit(&self, record: &mut JobRecord, message: &str) {
        record.updated_at = Timestamp::now();
        let event = JobEvent {
            job_id: self.job_id.clone(),
            state: record.state,
            stage: record.stage,
            current: record.current,
            total: record.total,
            message: Some(message.to_string()),
            cancellation_status: record.cancellation_status,
            sequence: record.events.len() as u64 + 1,
            created_at: record.updated_at,
        };
        record.events.push(event);
        self.events_tx.send_modify(|v| *v += 1);
    }

    fn snapshot(&self) -> JobSnapshot {
        let r = self.lock();
        JobSnapshot {
            job_id: self.job_id.clone(),
            state: r.state,
            request: r.request.clone(),
            artifact: r.artifact.clone(),
            error: r.error.clone(),
            stage: r.stage,
            current: r.current,
            total: r.total,
            cancellation_status: r.cancellation_status,
            created_at: r.created_at,
            updated_at: r.updated_at,
        }
    }
}

/// Cancellation flag and progress sink handed to a runner.
#[derive(Clone)]
pub struct JobContext {
    cell: Arc<JobCell>,
    cancel_rx: watch::Receiver<bool>,
}

impl JobContext {
    /// `true` once cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        *self.cancel_rx.borrow()
    }

    /// Resolve once cancellation is requested.
    pub async fn cancelled(&self) {
        let mut rx = self.cancel_rx.clone();
        let _ = rx.wait_for(|v| *v).await;
    }

    /// Record progress (ignored unless the job is running). `stage: None`
    /// keeps the current stage.
    pub fn report_progress(
        &self,
        stage: Option<JobStage>,
        current: Option<u64>,
        total: Option<u64>,
        message: Option<&str>,
    ) {
        let mut r = self.cell.lock();
        if r.state != JobState::Running {
            return;
        }
        if let Some(stage) = stage {
            r.stage = Some(stage);
        }
        r.current = current;
        r.total = total;
        self.cell
            .emit(&mut r, message.unwrap_or("Processing progress"));
    }
}

impl ProcessHooks for JobContext {
    fn is_cancelled(&self) -> bool {
        JobContext::is_cancelled(self)
    }

    fn progress(
        &self,
        stage: &str,
        current: Option<u64>,
        total: Option<u64>,
        message: Option<&str>,
    ) {
        self.report_progress(JobStage::from_hook(stage), current, total, message);
    }
}

/// Resumable event cursor over one job (`JobManager.events`).
pub struct EventCursor {
    cell: Arc<JobCell>,
    cursor: u64,
}

impl EventCursor {
    /// Next event after the cursor; `None` once the job is terminal and
    /// every event was delivered.
    pub async fn next(&mut self) -> Option<JobEvent> {
        loop {
            let mut rx = self.cell.events_tx.subscribe();
            {
                let r = self.cell.lock();
                if let Some(event) = r.events.iter().find(|e| e.sequence > self.cursor) {
                    self.cursor = event.sequence;
                    return Some(event.clone());
                }
                if r.state.is_terminal() {
                    return None;
                }
            }
            if rx.changed().await.is_err() {
                return None;
            }
        }
    }

    /// Drain every remaining event (tests and terminal jobs).
    pub async fn collect(mut self) -> Vec<JobEvent> {
        let mut out = Vec::new();
        while let Some(e) = self.next().await {
            out.push(e);
        }
        out
    }
}

struct Jobs {
    next_seq: u64,
    map: HashMap<String, Arc<JobCell>>,
}

/// Own local jobs, bounded terminal history, and resumable event streams.
pub struct JobManager {
    runner: Arc<dyn JobRunner>,
    max_terminal_jobs: usize,
    jobs: Mutex<Jobs>,
}

impl JobManager {
    pub fn new(runner: impl JobRunner, max_terminal_jobs: usize) -> Arc<Self> {
        assert!(
            max_terminal_jobs >= 1,
            "max_terminal_jobs must be at least 1"
        );
        Arc::new(Self {
            runner: Arc::new(runner),
            max_terminal_jobs,
            jobs: Mutex::new(Jobs {
                next_seq: 0,
                map: HashMap::new(),
            }),
        })
    }

    /// Default history bound (64 terminal jobs).
    pub fn with_runner(runner: impl JobRunner) -> Arc<Self> {
        Self::new(runner, 64)
    }

    fn cell(&self, job_id: &str) -> Result<Arc<JobCell>, JobNotFound> {
        lock(&self.jobs)
            .map
            .get(job_id)
            .cloned()
            .ok_or_else(|| JobNotFound(job_id.to_string()))
    }

    /// Queue a job and start its task (requires a tokio runtime).
    pub fn create(self: &Arc<Self>, request: ProcessRequest) -> JobSnapshot {
        self.prune_terminal_jobs();
        let now = Timestamp::now();
        let cell = {
            let mut jobs = lock(&self.jobs);
            jobs.next_seq += 1;
            let cell = Arc::new(JobCell {
                job_id: uuid4(),
                seq: jobs.next_seq,
                record: Mutex::new(JobRecord {
                    request,
                    state: JobState::Queued,
                    stage: None,
                    artifact: None,
                    error: None,
                    cancellation_status: CancellationStatus::None,
                    current: None,
                    total: None,
                    created_at: now,
                    updated_at: now,
                    events: Vec::new(),
                }),
                events_tx: watch::Sender::new(0),
                cancel_tx: watch::Sender::new(false),
                task: Mutex::new(None),
                runner_abort: Mutex::new(None),
            });
            jobs.map.insert(cell.job_id.clone(), cell.clone());
            cell
        };
        {
            let mut r = cell.lock();
            cell.emit(&mut r, "Job queued");
        }
        let handle = tokio::spawn(Self::run(self.clone(), cell.clone()));
        *lock(&cell.task) = Some(handle);
        cell.snapshot()
    }

    pub fn get(&self, job_id: &str) -> Result<JobSnapshot, JobNotFound> {
        Ok(self.cell(job_id)?.snapshot())
    }

    /// Ids of every retained job (oldest first).
    pub fn job_ids(&self) -> Vec<String> {
        let jobs = lock(&self.jobs);
        let mut cells: Vec<&Arc<JobCell>> = jobs.map.values().collect();
        cells.sort_by_key(|c| c.seq);
        cells.iter().map(|c| c.job_id.clone()).collect()
    }

    pub fn cancel(&self, job_id: &str) -> Result<JobSnapshot, JobNotFound> {
        let cell = self.cell(job_id)?;
        {
            let mut r = cell.lock();
            if r.state.is_terminal() {
                drop(r);
                return Ok(cell.snapshot());
            }
            cell.cancel_tx.send_replace(true);
            r.cancellation_status = CancellationStatus::Requested;
            if r.state == JobState::Queued {
                r.state = JobState::Cancelled;
                r.cancellation_status = CancellationStatus::Cancelled;
                cell.emit(&mut r, "Job cancelled before processing started");
            } else {
                cell.emit(
                    &mut r,
                    "Cancellation requested; waiting for a safe processor checkpoint",
                );
            }
        }
        Ok(cell.snapshot())
    }

    /// Request cooperative stops, then abort runners that exceed the grace
    /// period (their jobs end `cancelled`).
    pub async fn shutdown(&self, timeout: Duration) {
        let active: Vec<Arc<JobCell>> = lock(&self.jobs)
            .map
            .values()
            .filter(|c| !c.lock().state.is_terminal())
            .cloned()
            .collect();
        for cell in &active {
            let _ = self.cancel(&cell.job_id);
        }
        let mut handles: Vec<Option<JoinHandle<()>>> =
            active.iter().map(|c| lock(&c.task).take()).collect();
        let wait_all = async {
            for slot in handles.iter_mut() {
                if let Some(h) = slot.as_mut() {
                    let _ = h.await;
                    *slot = None;
                }
            }
        };
        let timed_out = tokio::time::timeout(timeout, wait_all).await.is_err();
        if timed_out {
            for cell in &active {
                if let Some(abort) = lock(&cell.runner_abort).as_ref() {
                    abort.abort();
                }
            }
            for h in handles.into_iter().flatten() {
                let _ = h.await;
            }
        }
    }

    /// Events after `after_sequence` (negative values start from 0).
    pub fn events(&self, job_id: &str, after_sequence: i64) -> Result<EventCursor, JobNotFound> {
        Ok(EventCursor {
            cell: self.cell(job_id)?,
            cursor: after_sequence.max(0) as u64,
        })
    }

    async fn run(self: Arc<Self>, cell: Arc<JobCell>) {
        let request = {
            let mut r = cell.lock();
            if r.state == JobState::Cancelled {
                return;
            }
            r.state = JobState::Running;
            r.stage = Some(initial_stage(&r.request));
            cell.emit(&mut r, "Processing started");
            r.request.clone()
        };
        let ctx = JobContext {
            cell: cell.clone(),
            cancel_rx: cell.cancel_tx.subscribe(),
        };
        let inner = tokio::spawn(self.runner.run(request, ctx));
        *lock(&cell.runner_abort) = Some(inner.abort_handle());
        let outcome = match inner.await {
            Ok(result) => Some(result),
            Err(e) if e.is_panic() => Some(Err(JobFailure::Failed(panic_message(e.into_panic())))),
            Err(_) => None,
        };
        {
            let mut r = cell.lock();
            match outcome {
                Some(Ok(artifact)) => {
                    r.state = JobState::Completed;
                    if *cell.cancel_tx.borrow() {
                        r.cancellation_status = CancellationStatus::CompletedAfterRequest;
                    }
                    if artifact.output_file.is_some() {
                        r.stage = Some(JobStage::Saving);
                    }
                    r.artifact = Some(artifact);
                    cell.emit(&mut r, "Processing completed");
                }
                Some(Err(JobFailure::Cancelled)) => {
                    r.state = JobState::Cancelled;
                    r.cancellation_status = CancellationStatus::Cancelled;
                    cell.emit(&mut r, "Job stopped at a cancellation checkpoint");
                }
                Some(Err(JobFailure::Failed(message))) => {
                    r.state = JobState::Failed;
                    r.error = Some(message);
                    cell.emit(&mut r, "Processing failed");
                }
                None => {
                    if r.state != JobState::Cancelled {
                        r.state = JobState::Cancelled;
                        r.cancellation_status = CancellationStatus::Cancelled;
                        cell.emit(&mut r, "Job task cancelled");
                    }
                }
            }
        }
        self.prune_terminal_jobs();
    }

    fn prune_terminal_jobs(&self) {
        let mut jobs = lock(&self.jobs);
        let mut terminal: Vec<(Timestamp, u64, String)> = jobs
            .map
            .values()
            .filter_map(|c| {
                let r = c.lock();
                r.state
                    .is_terminal()
                    .then(|| (r.updated_at, c.seq, c.job_id.clone()))
            })
            .collect();
        if terminal.len() <= self.max_terminal_jobs {
            return;
        }
        terminal.sort();
        let excess = terminal.len() - self.max_terminal_jobs;
        for (_, _, id) in terminal.into_iter().take(excess) {
            jobs.map.remove(&id);
        }
    }
}

/// Stage reported when processing starts.
pub fn initial_stage(request: &ProcessRequest) -> JobStage {
    let source = request.input_source.as_deref().unwrap_or("");
    if !request.has_blob() && (source.starts_with("http://") || source.starts_with("https://")) {
        return JobStage::Downloading;
    }
    match request.task_type {
        Some(TaskType::Transcript) => JobStage::Transcribing,
        Some(TaskType::Ocr) => JobStage::Ocr,
        Some(TaskType::Convert) => JobStage::Converting,
        None => JobStage::Extracting,
    }
}

/// Text of a panic payload.
pub fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "processing panicked".to_string()
    }
}
