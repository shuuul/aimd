//! Hermetic scenarios ported from `test_file_processing`, `test_capabilities`,
//! `test_remote_backends`, `test_asr_*`, and `test_ocr`: every system seam
//! is mocked through [`Runtime`]; remote backends hit a local TCP server.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aimd_core::log::{Level, capture};
use aimd_core::{AimdError, ErrorKind, NoHooks, ProcessHooks, ProcessInput, SourceKind, TaskType};

use crate::tempdir::TempDir;
use crate::*;

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

type Responder = dyn Fn(&str, &[String]) -> std::io::Result<CommandOutput> + Send + Sync;

/// One recorded ASR call: wav path, language, context.
type AsrCall = (PathBuf, Option<String>, Option<String>);

#[derive(Default)]
struct ScriptedAsr {
    outputs: Mutex<VecDeque<std::result::Result<String, AsrFailure>>>,
    calls: Mutex<Vec<AsrCall>>,
}

impl ScriptedAsr {
    fn new(outputs: Vec<std::result::Result<String, AsrFailure>>) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs.into()),
            calls: Mutex::default(),
        })
    }
}

/// Fixed transcript with word alignment units.
struct AlignedAsr;

impl AsrModel for AlignedAsr {
    fn transcribe(
        &self,
        _: &Path,
        _: Option<&str>,
        _: Option<&str>,
    ) -> std::result::Result<String, AsrFailure> {
        Ok("Hello there. Bye now.".into())
    }

    fn transcribe_aligned(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<crate::AlignedTranscript, AsrFailure> {
        let words = [("Hello", 1.0), ("there", 1.5), ("Bye", 5.0), ("now", 5.5)];
        Ok(crate::AlignedTranscript {
            text: self.transcribe(wav_path, language, context)?,
            units: Some(
                words
                    .iter()
                    .map(|&(w, t)| (w.to_string(), t, t + 0.4))
                    .collect(),
            ),
        })
    }
}

impl AsrModel for ScriptedAsr {
    fn transcribe(
        &self,
        wav_path: &Path,
        language: Option<&str>,
        context: Option<&str>,
    ) -> std::result::Result<String, AsrFailure> {
        self.calls.lock().unwrap().push((
            wav_path.to_path_buf(),
            language.map(str::to_string),
            context.map(str::to_string),
        ));
        self.outputs
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Err(AsrFailure::Other("no scripted output".into())))
    }
}

#[derive(Default)]
struct ScriptedOcr {
    outputs: Mutex<VecDeque<std::result::Result<String, OcrFailure>>>,
    calls: Mutex<Vec<PathBuf>>,
    /// `pages_per_request`; above 1, PDFs go through `recognize_pages`.
    window: usize,
    /// Page count of each `recognize_pages` request.
    requests: Mutex<Vec<usize>>,
}

impl OcrModel for ScriptedOcr {
    fn recognize(&self, image_path: &Path) -> std::result::Result<String, OcrFailure> {
        self.calls.lock().unwrap().push(image_path.to_path_buf());
        self.outputs
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Err(OcrFailure::Other("no scripted output".into())))
    }

    fn pages_per_request(&self) -> usize {
        self.window
    }

    fn recognize_pages(&self, pages: &[&[u8]]) -> std::result::Result<Vec<LocatedOcr>, OcrFailure> {
        self.requests.lock().unwrap().push(pages.len());
        let mut outputs = self.outputs.lock().unwrap();
        pages
            .iter()
            .map(|_| {
                let markdown = outputs
                    .pop_front()
                    .unwrap_or(Err(OcrFailure::Other("no scripted output".into())))?;
                Ok(LocatedOcr {
                    markdown,
                    ..LocatedOcr::default()
                })
            })
            .collect()
    }
}

/// `(input_source, model, precision)` seen by `url_transcript`.
type UrlRequest = (String, Option<String>, Option<String>);

struct Mock {
    env: HashMap<String, String>,
    tools: HashMap<String, PathBuf>,
    responder: Box<Responder>,
    runs: Mutex<Vec<(String, Vec<String>)>>,
    asr: HashMap<String, Arc<dyn AsrModel>>,
    ocr: HashMap<String, Arc<ScriptedOcr>>,
    loads: Mutex<Vec<String>>,
    preflight: Option<AimdError>,
    pdf_ocr: bool,
    pdf_pages: u32,
    doc: Option<DocOutput>,
    doc_calls: Mutex<Vec<(PathBuf, Option<PathBuf>)>>,
    url: Option<UrlOutput>,
    url_requests: Mutex<Vec<UrlRequest>>,
}

impl Default for Mock {
    fn default() -> Self {
        Self {
            env: HashMap::new(),
            tools: HashMap::new(),
            responder: Box::new(|program, _| {
                Err(std::io::Error::other(format!(
                    "unexpected run of {program}"
                )))
            }),
            runs: Mutex::default(),
            asr: HashMap::new(),
            ocr: HashMap::new(),
            loads: Mutex::default(),
            preflight: None,
            pdf_ocr: false,
            pdf_pages: 0,
            doc: None,
            doc_calls: Mutex::default(),
            url: None,
            url_requests: Mutex::default(),
        }
    }
}

impl Mock {
    fn tool(mut self, name: &str) -> Self {
        self.tools
            .insert(name.into(), PathBuf::from(format!("/mock/bin/{name}")));
        self
    }

    fn respond(
        mut self,
        f: impl Fn(&str, &[String]) -> std::io::Result<CommandOutput> + Send + Sync + 'static,
    ) -> Self {
        self.responder = Box::new(f);
        self
    }

    fn with_asr(mut self, id: &str, model: Arc<dyn AsrModel>) -> Self {
        self.asr.insert(id.into(), model);
        self
    }

    fn with_ocr(self, id: &str, outputs: Vec<std::result::Result<String, OcrFailure>>) -> Self {
        self.with_ocr_window(id, 1, outputs)
    }

    fn with_ocr_window(
        mut self,
        id: &str,
        window: usize,
        outputs: Vec<std::result::Result<String, OcrFailure>>,
    ) -> Self {
        self.ocr.insert(
            id.into(),
            Arc::new(ScriptedOcr {
                outputs: Mutex::new(outputs.into()),
                calls: Mutex::default(),
                window,
                requests: Mutex::default(),
            }),
        );
        self
    }

    fn runs(&self) -> Vec<(String, Vec<String>)> {
        self.runs.lock().unwrap().clone()
    }
}

impl Runtime for Mock {
    fn env(&self, key: &str) -> Option<String> {
        self.env.get(key).cloned()
    }

    fn which(&self, program: &str) -> Option<PathBuf> {
        self.tools.get(program).cloned()
    }

    fn run(
        &self,
        program: &Path,
        args: &[String],
        _cwd: Option<&Path>,
    ) -> std::io::Result<CommandOutput> {
        let name = program.file_name().unwrap().to_string_lossy().into_owned();
        self.runs
            .lock()
            .unwrap()
            .push((name.clone(), args.to_vec()));
        (self.responder)(&name, args)
    }

    fn preflight_local(&self, _modality: LocalModality) -> Result<()> {
        match &self.preflight {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    fn load_asr(&self, model_id: &str) -> Result<Arc<dyn AsrModel>> {
        self.loads.lock().unwrap().push(model_id.into());
        self.asr
            .get(model_id)
            .cloned()
            .map(|m| m as Arc<dyn AsrModel>)
            .ok_or_else(|| AimdError::backend_unavailable(format!("no mock ASR {model_id}")))
    }

    fn load_ocr(&self, model_id: &str) -> Result<Arc<dyn OcrModel>> {
        self.loads.lock().unwrap().push(model_id.into());
        self.ocr
            .get(model_id)
            .cloned()
            .map(|m| m as Arc<dyn OcrModel>)
            .ok_or_else(|| AimdError::backend_unavailable(format!("no mock OCR {model_id}")))
    }

    fn pdf_requires_ocr(&self, _path: &Path) -> bool {
        self.pdf_ocr
    }

    fn pdf_page_count(&self, _path: &Path) -> Result<u32> {
        Ok(self.pdf_pages)
    }

    fn convert_document(
        &self,
        path: &Path,
        output_dir: Option<&Path>,
        _temp_dir: Option<&Path>,
        _hooks: &dyn ProcessHooks,
    ) -> Result<DocOutput> {
        self.doc_calls
            .lock()
            .unwrap()
            .push((path.to_path_buf(), output_dir.map(Path::to_path_buf)));
        self.doc
            .clone()
            .ok_or_else(|| AimdError::failed("no mock doc"))
    }

    fn url_transcript(
        &self,
        request: &ProcessInput,
        _hooks: &dyn ProcessHooks,
    ) -> Result<UrlOutput> {
        self.url_requests.lock().unwrap().push((
            request.input_source.clone(),
            request.model.clone(),
            request.precision.clone(),
        ));
        self.url
            .clone()
            .ok_or_else(|| AimdError::backend_unavailable("no mock url"))
    }
}

/// Hooks recording progress; cancels once `cancel_after` progress events
/// were seen (or immediately with `Some(0)`).
type ProgressEvent = (String, Option<u64>, Option<u64>, Option<String>);

#[derive(Default)]
struct RecHooks {
    cancel_after: Option<usize>,
    events: Mutex<Vec<ProgressEvent>>,
}

impl ProcessHooks for RecHooks {
    fn is_cancelled(&self) -> bool {
        self.cancel_after
            .is_some_and(|n| self.events.lock().unwrap().len() >= n)
    }

    fn progress(
        &self,
        stage: &str,
        current: Option<u64>,
        total: Option<u64>,
        message: Option<&str>,
    ) {
        self.events.lock().unwrap().push((
            stage.into(),
            current,
            total,
            message.map(str::to_string),
        ));
    }
}

const ASR_4BIT: &str = "mlx-community/Qwen3-ASR-1.7B-4bit";
const ASR_8BIT: &str = "mlx-community/Qwen3-ASR-1.7B-8bit";
/// The default OCR model.
const OCR_4BIT: &str = "mlx-community/Unlimited-OCR-2-4bit";
const OCR2_4BIT: &str = OCR_4BIT;

fn ok_output(stdout: &str) -> std::io::Result<CommandOutput> {
    Ok(CommandOutput {
        status: Some(0),
        stdout: stdout.as_bytes().to_vec(),
        stderr: Vec::new(),
    })
}

/// 0.1 s of 16 kHz mono s16 silence (accepted by the native wav loader).
fn write_wav(path: &Path) {
    let samples = 1600u32;
    let data_len = samples * 2;
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&16000u32.to_le_bytes());
    b.extend_from_slice(&32000u32.to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    b.resize(b.len() + data_len as usize, 0);
    std::fs::write(path, b).unwrap();
}

/// Minimal one-page-per-entry PDF; `None` pages have no text.
fn write_pdf(path: &Path, pages: &[Option<&str>]) {
    let mut objects: Vec<String> = Vec::new();
    let n = pages.len();
    // 1 catalog, 2 pages, 3 font, then (page, content) pairs.
    objects.push("<< /Type /Catalog /Pages 2 0 R >>".into());
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
    objects.push(format!(
        "<< /Type /Pages /Kids [{}] /Count {n} >>",
        kids.join(" ")
    ));
    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into());
    for (i, text) in pages.iter().enumerate() {
        let content_id = 5 + 2 * i;
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ));
        let stream = match text {
            Some(t) => format!("BT /F1 24 Tf 72 700 Td ({t}) Tj ET"),
            None => String::new(),
        };
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ));
    }
    let mut out = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.push_str(&format!("{} 0 obj\n{obj}\nendobj\n", i + 1));
    }
    let xref = out.len();
    out.push_str(&format!(
        "xref\n0 {}\n0000000000 65535 f \n",
        objects.len() + 1
    ));
    for off in offsets {
        out.push_str(&format!("{off:010} 00000 n \n"));
    }
    out.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    ));
    std::fs::write(path, out).unwrap();
}

/// A default request (anchors off); anchored tests opt in.
fn request(path: &Path) -> ProcessInput {
    ProcessInput::new(path.to_string_lossy())
}

fn warnings(logs: &[(Level, String)]) -> Vec<String> {
    logs.iter()
        .filter(|(l, _)| *l == Level::Warning)
        .map(|(_, m)| m.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Local TCP mock HTTP server (one request per connection, no sleeps)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Recorded {
    request_line: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Recorded {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Serve `responses` (status, body) in order; returns the base URL and the
/// recorded requests (joined once every response was sent).
fn serve(responses: Vec<(u16, String)>) -> (String, std::thread::JoinHandle<Vec<Recorded>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || {
        let mut recorded = Vec::new();
        for (status, body) in responses {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut headers = Vec::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end();
                if line.is_empty() {
                    break;
                }
                let (k, v) = line.split_once(':').unwrap();
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
            let len: usize = headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .map_or(0, |(_, v)| v.parse().unwrap());
            let mut req_body = vec![0u8; len];
            reader.read_exact(&mut req_body).unwrap();
            recorded.push(Recorded {
                request_line: request_line.trim_end().to_string(),
                headers,
                body: req_body,
            });
            let mut stream = stream;
            write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            stream.flush().unwrap();
        }
        recorded
    });
    (format!("http://{addr}"), handle)
}

/// A port with nothing listening.
fn closed_port_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}")
}

// ---------------------------------------------------------------------------
// Routing (test_file_processing / test_capabilities)
// ---------------------------------------------------------------------------

#[test]
fn routes_by_source_kind_and_requested_task() {
    let dir = TempDir::new(None, "aimd-route-").unwrap();
    let touch = |name: &str| {
        let p = dir.path().join(name);
        std::fs::write(&p, b"x").unwrap();
        p.to_string_lossy().into_owned()
    };
    let (wav, png, pdf, epub, mkv, doc) = (
        touch("a.wav"),
        touch("a.PNG"),
        touch("a.pdf"),
        touch("a.epub"),
        touch("a.mkv"),
        touch("a.doc"),
    );
    let text_pdf = Mock::default();
    let scanned = Mock {
        pdf_ocr: true,
        ..Mock::default()
    };
    let r = |src: &str, task: Option<TaskType>, rt: &dyn Runtime| {
        let route = get_input_route(src, task, rt);
        (route.source_kind, route.task_type)
    };
    use SourceKind::*;
    use TaskType::*;
    assert_eq!(
        r("https://youtu.be/x", None, &text_pdf),
        (Url, Some(Transcript))
    );
    assert_eq!(
        r("https://youtu.be/x", Some(Convert), &text_pdf),
        (Url, None)
    );
    assert_eq!(r(&wav, None, &text_pdf), (AudioFile, Some(Transcript)));
    assert_eq!(r(&wav, Some(Convert), &text_pdf), (AudioFile, None));
    assert_eq!(r(&mkv, None, &text_pdf), (VideoFile, Some(Transcript)));
    assert_eq!(r(&png, None, &text_pdf), (ImageFile, Some(Ocr)));
    assert_eq!(r(&pdf, None, &text_pdf), (DocumentFile, Some(Convert)));
    assert_eq!(r(&pdf, None, &scanned), (DocumentFile, Some(Ocr)));
    assert_eq!(r(&pdf, Some(Ocr), &text_pdf), (DocumentFile, Some(Ocr)));
    assert_eq!(r(&epub, None, &text_pdf), (DocumentFile, Some(Convert)));
    assert_eq!(r(&epub, Some(Ocr), &text_pdf), (Unknown, None));
    assert_eq!(r(&epub, Some(Transcript), &text_pdf), (DocumentFile, None));
    assert_eq!(r(&doc, None, &text_pdf), (Unknown, None));

    let err = ensure_supported_input(&wav, Some(Ocr), &text_pdf).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedInput);
    assert_eq!(
        err.message,
        "Unsupported OCR input. OCR supports image files (.png, .jpg, .jpeg, .webp, .tif, .tiff) \
         and PDF files."
    );
    let err = ensure_supported_input("missing.wav", None, &text_pdf).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InputNotFound);
    assert_eq!(err.message, "Input file not found: missing.wav");
    let err = ensure_supported_input("nofile", None, &text_pdf).unwrap_err();
    assert_eq!(
        err.message,
        "Unsupported input source. Supported inputs: audio/video files, video URLs, supported \
         document files, and OCR-capable images/PDFs."
    );
    let err = route::parse_task(Some("summarize")).unwrap_err();
    assert_eq!(
        err.message,
        "Unsupported task. Supported tasks: transcript, convert, ocr."
    );
}

#[test]
fn pdf_inspector_probe_routes_scanned_pdfs_to_ocr() {
    let dir = TempDir::new(None, "aimd-pdf-").unwrap();
    let text = dir.path().join("text.pdf");
    let blank = dir.path().join("blank.pdf");
    write_pdf(&text, &[Some("Hello native aimd world")]);
    write_pdf(&blank, &[None, None]);
    assert!(!native::pdf_requires_ocr(&text));
    assert!(native::pdf_requires_ocr(&blank));
    assert_eq!(native::pdf_page_count(&blank).unwrap(), 2);
    // Probe failures keep the convert route.
    let junk = dir.path().join("junk.pdf");
    std::fs::write(&junk, b"not a pdf").unwrap();
    assert!(!native::pdf_requires_ocr(&junk));
    assert_eq!(
        native::pdf_page_count(&junk).unwrap_err().kind,
        ErrorKind::ProcessingFailed
    );
}

// ---------------------------------------------------------------------------
// Document conversion and result shaping
// ---------------------------------------------------------------------------

#[test]
fn asset_documents_get_output_dir_and_directory_uri() {
    let dir = TempDir::new(None, "aimd-doc-").unwrap();
    let docx = dir.path().join("My Report.docx");
    std::fs::write(&docx, b"x").unwrap();
    let rt = Mock {
        doc: Some(DocOutput {
            markdown: "# Heading\n\nBody".into(),
            title: None,
        }),
        ..Mock::default()
    };
    let result = process_input_with(&request(&docx), &NoHooks, &rt).unwrap();
    let expected_dir = dir.path().join("My Report");
    assert_eq!(result.task_type, TaskType::Convert);
    assert_eq!(result.output_dir.as_deref(), Some(expected_dir.as_path()));
    assert_eq!(
        rt.doc_calls.lock().unwrap()[0].1.as_deref(),
        Some(expected_dir.as_path())
    );
    assert_eq!(result.text_context.title, "Heading");
    assert_eq!(
        result.text_context.chunk_list,
        vec!["# Heading\n\nBody".to_string()]
    );
    let uri = result.asset_base_uri.unwrap();
    assert!(
        uri.starts_with("file:///") && uri.ends_with("/My%20Report/"),
        "{uri}"
    );

    // Non-asset documents use the parent directory and keep their title.
    let md = dir.path().join("notes.md");
    std::fs::write(&md, b"x").unwrap();
    let rt = Mock {
        doc: Some(DocOutput {
            markdown: "plain".into(),
            title: Some("Doc Title".into()),
        }),
        ..Mock::default()
    };
    let result = process_input_with(&request(&md), &NoHooks, &rt).unwrap();
    assert_eq!(result.output_dir, None);
    assert_eq!(result.text_context.title, "Doc Title");
    assert_eq!(result.asset_base_uri.unwrap(), directory_uri(dir.path()));
}

#[test]
fn directories_are_rejected() {
    let dir = TempDir::new(None, "aimd-dir-").unwrap();
    let sub = dir.path().join("folder.md");
    std::fs::create_dir(&sub).unwrap();
    let err = process_input_with(&request(&sub), &NoHooks, &Mock::default()).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedInput);
    assert_eq!(
        err.message,
        format!("Path is not a file: {}", sub.display())
    );
}

#[test]
fn url_results_carry_platform_and_source_uri() {
    let rt = Mock {
        url: Some(UrlOutput {
            markdown: "# Video\n\ntext".into(),
            title: "Video".into(),
            platform: "bilibili".into(),
            segments: None,
        }),
        ..Mock::default()
    };
    // test_use_cases.py::test_use_case_url_passes_precision_to_url_processor
    let mut req = ProcessInput::new("https://www.bilibili.com/video/BV1");
    req.model = Some("qwen3-asr-1.7b".into());
    req.precision = Some("8bit".into());
    let result = process_input_with(&req, &NoHooks, &rt).unwrap();
    assert_eq!(result.task_type, TaskType::Transcript);
    assert_eq!(result.platform.as_deref(), Some("bilibili"));
    assert_eq!(
        result.asset_base_uri.as_deref(),
        Some("https://www.bilibili.com/video/BV1")
    );
    assert_eq!(result.text_context.title, "Video");
    assert_eq!(result.markdown, "# Video\n\ntext");
    assert_eq!(
        rt.url_requests.lock().unwrap().as_slice(),
        &[(
            "https://www.bilibili.com/video/BV1".to_string(),
            Some("qwen3-asr-1.7b".to_string()),
            Some("8bit".to_string())
        )]
    );
    // A non-transcript task for a URL is rejected before the URL pipeline.
    let mut ocr_req = ProcessInput::new("https://youtu.be/x");
    ocr_req.task_type = Some(TaskType::Ocr);
    assert!(process_input_with(&ocr_req, &NoHooks, &rt).is_err());
    assert_eq!(rt.url_requests.lock().unwrap().len(), 1);

    struct Stub;
    impl Runtime for Stub {
        fn preflight_local(&self, _: LocalModality) -> Result<()> {
            Ok(())
        }
        fn load_asr(&self, _: &str) -> Result<Arc<dyn AsrModel>> {
            unreachable!()
        }
        fn load_ocr(&self, _: &str) -> Result<Arc<dyn OcrModel>> {
            unreachable!()
        }
    }
    let err = process_input_with(&req, &NoHooks, &Stub).unwrap_err();
    assert_eq!(err.kind, ErrorKind::BackendUnavailable);
}

#[test]
fn url_audio_fallback_transcribes_through_the_runtime() {
    let dir = TempDir::new(None, "aimd-urlasr-").unwrap();
    let wav = dir.path().join("audio_x.wav");
    write_wav(&wav);
    let model = ScriptedAsr::new(vec![Ok("transcribed url audio".into())]);
    let rt = Mock::default().with_asr(ASR_8BIT, model.clone());
    let request = TranscribeRequest {
        language: Some("en".into()),
        context: Some("Vocabulary: Linus\n\nThe following background".into()),
        model: Some("qwen3-asr-1.7b".into()),
        precision: Some("8bit".into()),
        ..TranscribeRequest::default()
    };
    let text = UrlAudioTranscriber(&rt)
        .transcribe(&wav, &request, &NoHooks)
        .unwrap();
    assert_eq!(text, "transcribed url audio");
    assert_eq!(rt.loads.lock().unwrap().as_slice(), &[ASR_8BIT.to_string()]);
    assert_eq!(
        model.calls.lock().unwrap()[0],
        (
            wav.clone(),
            Some("English".into()),
            Some("Vocabulary: Linus\n\nThe following background".into())
        )
    );
    // Unsupported download formats fail like local files do.
    let bad = dir.path().join("audio_x.part");
    std::fs::write(&bad, b"x").unwrap();
    let err = UrlAudioTranscriber(&rt)
        .transcribe(&bad, &request, &NoHooks)
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedInput);
}

#[test]
fn local_results_get_markitdown_normalization() {
    let dir = TempDir::new(None, "aimd-norm-").unwrap();
    let md = dir.path().join("notes.md");
    std::fs::write(&md, b"x").unwrap();
    let rt = Mock {
        doc: Some(DocOutput {
            markdown: "# T  \r\n\n\n\nbody\t\n".into(),
            title: None,
        }),
        ..Mock::default()
    };
    let result = process_input_with(&request(&md), &NoHooks, &rt).unwrap();
    assert_eq!(result.markdown, "# T\n\nbody\n");
    assert_eq!(result.text_context.title, "T");
}

// ---------------------------------------------------------------------------
// Local ASR
// ---------------------------------------------------------------------------

#[test]
fn local_transcript_uses_resolved_model_language_and_context() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let wav = dir.path().join("clip.wav");
    write_wav(&wav);
    let model = ScriptedAsr::new(vec![Ok("  hello world  ".into())]);
    let rt = Mock::default().with_asr(ASR_4BIT, model.clone());
    let mut req = request(&wav);
    req.language = Some("en".into());
    req.context = Some("Kubernetes".into());
    let (result, logs) = capture(|| process_input_with(&req, &NoHooks, &rt));
    let result = result.unwrap();
    assert_eq!(result.task_type, TaskType::Transcript);
    assert_eq!(result.markdown, "hello world");
    assert_eq!(result.segments, None);
    // Anchored local ASR uses the model's alignment units.
    let rt = Mock::default().with_asr(ASR_4BIT, Arc::new(AlignedAsr));
    let anchored = ProcessInput {
        anchors: true,
        ..ProcessInput::new(wav.to_string_lossy())
    };
    let aligned = process_input_with(&anchored, &NoHooks, &rt).unwrap();
    assert_eq!(
        aligned.markdown,
        "[00:00:01] Hello there.\n\n[00:00:05] Bye now."
    );
    let segments = aligned.segments.clone().unwrap();
    assert_eq!((segments[0].start, segments[0].end), (Some(1.0), Some(1.9)));
    // Segments alone keep the plain transcript.
    let segments_only = ProcessInput {
        include_segments: true,
        ..ProcessInput::new(wav.to_string_lossy())
    };
    let plain = process_input_with(&segments_only, &NoHooks, &rt).unwrap();
    assert_eq!(plain.markdown, "Hello there. Bye now.");
    assert_eq!(plain.segments, aligned.segments);
    assert_eq!(result.text_context.title, "clip");
    assert_eq!(result.asset_base_uri, None);
    let calls = model.calls.lock().unwrap();
    assert_eq!(
        calls[0],
        (
            wav.clone(),
            Some("English".into()),
            Some("Kubernetes".into())
        )
    );
    // No ffprobe/ffmpeg: duration unknown -> warn and run unsegmented.
    assert!(warnings(&logs)[0].starts_with("Could not determine audio/video duration:"));
    assert!(rt.runs().is_empty(), "native wav needs no ffmpeg");
}

#[test]
fn non_wav_audio_is_decoded_with_ffmpeg_to_16k_mono() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let mp3 = dir.path().join("talk.mp3");
    std::fs::write(&mp3, b"ID3").unwrap();
    let model = ScriptedAsr::new(vec![Ok("text".into())]);
    let rt = Mock::default()
        .tool("ffmpeg")
        .tool("ffprobe")
        .with_asr(ASR_4BIT, model.clone())
        .respond(|program, args| match program {
            "ffprobe" => ok_output("12.5\n"),
            "ffmpeg" => {
                std::fs::write(args.last().unwrap(), b"wav").unwrap();
                ok_output("")
            }
            _ => unreachable!(),
        });
    let (result, logs) = capture(|| process_input_with(&request(&mp3), &NoHooks, &rt));
    assert_eq!(result.unwrap().markdown, "text");
    let runs = rt.runs();
    assert_eq!(runs[0].0, "ffprobe");
    assert_eq!(runs[1].0, "ffmpeg");
    let args = &runs[1].1;
    assert_eq!(
        &args[..7],
        [
            "-y",
            "-i",
            &mp3.to_string_lossy(),
            "-ar",
            "16000",
            "-ac",
            "1"
        ]
    );
    let decoded = PathBuf::from(args.last().unwrap());
    assert_eq!(model.calls.lock().unwrap()[0].0, decoded);
    assert!(!decoded.exists(), "temporary wav is removed");
    assert!(
        logs.iter()
            .any(|(_, m)| m == "Audio/video duration is 12.50 seconds")
    );
}

#[test]
fn missing_ffmpeg_for_conversion_is_backend_unavailable() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let mp3 = dir.path().join("talk.mp3");
    std::fs::write(&mp3, b"ID3").unwrap();
    let rt = Mock::default().with_asr(ASR_4BIT, ScriptedAsr::new(vec![]));
    let (err, _) = capture(|| process_input_with(&request(&mp3), &NoHooks, &rt));
    let err = err.unwrap_err();
    assert_eq!(err.kind, ErrorKind::BackendUnavailable);
    assert!(
        err.message
            .starts_with("Cannot convert .mp3 to WAV: ffmpeg not found.")
    );
}

#[test]
fn repetition_loop_on_4bit_retries_with_8bit() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let wav = dir.path().join("loop.wav");
    write_wav(&wav);
    let looped = "好".repeat(200);
    let first = ScriptedAsr::new(vec![Ok(looped)]);
    let retry = ScriptedAsr::new(vec![Ok("clean text".into())]);
    let rt = Mock::default()
        .with_asr(ASR_4BIT, first.clone())
        .with_asr(ASR_8BIT, retry.clone());
    let (result, logs) = capture(|| process_input_with(&request(&wav), &NoHooks, &rt));
    assert_eq!(result.unwrap().markdown, "clean text");
    // No --language: both passes leave detection to the model.
    assert_eq!(first.calls.lock().unwrap()[0].1, None);
    assert_eq!(retry.calls.lock().unwrap()[0].1, None);
    assert_eq!(
        *rt.loads.lock().unwrap(),
        vec![ASR_4BIT.to_string(), ASR_8BIT.to_string()]
    );
    assert!(warnings(&logs).contains(&format!(
        "Repetition loop detected with 4-bit model ({ASR_4BIT}). Switching to 8-bit fallback \
         model: {ASR_8BIT}"
    )));

    // 8-bit models have no fallback: keep the text and warn.
    let rt = Mock::default().with_asr(ASR_8BIT, ScriptedAsr::new(vec![Ok("啊".repeat(200))]));
    let mut req = request(&wav);
    req.precision = Some("8bit".into());
    let (result, logs) = capture(|| process_input_with(&req, &NoHooks, &rt));
    assert_eq!(result.unwrap().markdown, "啊".repeat(200));
    assert!(warnings(&logs).contains(
        &"Repetition loop detected, but no 8-bit fallback model is available.".to_string()
    ));
}

#[test]
fn long_audio_is_segmented_with_progress_and_cancellation() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let wav = dir.path().join("long.wav");
    write_wav(&wav);
    let seg_dir = dir.path().join("tmp");
    let make_rt = |outputs: Vec<std::result::Result<String, AsrFailure>>| {
        Mock::default()
            .tool("ffmpeg")
            .tool("ffprobe")
            .with_asr(ASR_4BIT, ScriptedAsr::new(outputs))
            .respond(|program, args| match program {
                "ffprobe" => ok_output("1500.25"),
                "ffmpeg" => {
                    let template = args.last().unwrap();
                    for i in 0..3 {
                        let path = template.replace("%03d", &format!("{i:03}"));
                        write_wav(Path::new(&path));
                    }
                    ok_output("")
                }
                _ => unreachable!(),
            })
    };
    let mut req = request(&wav);
    req.temp_dir = Some(seg_dir.clone());
    std::fs::create_dir_all(&seg_dir).unwrap();

    let rt = make_rt(vec![
        Ok("one".into()),
        Err(AsrFailure::Empty),
        Ok("three".into()),
    ]);
    let hooks = RecHooks::default();
    let (result, _) = capture(|| process_input_with(&req, &hooks, &rt));
    assert_eq!(result.unwrap().markdown, "one three");
    let seg_args = &rt.runs()[1].1;
    assert_eq!(
        seg_args[3..13],
        [
            "-f",
            "segment",
            "-segment_time",
            "600.0",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1"
        ]
    );
    let events = hooks.events.lock().unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(
        events[1],
        (
            "transcribing".into(),
            Some(2),
            Some(3),
            Some("Transcribing segment 2 of 3".into())
        )
    );
    drop(events);
    assert_eq!(
        std::fs::read_dir(&seg_dir).unwrap().count(),
        0,
        "segments cleaned up"
    );

    let rt = make_rt(vec![
        Err(AsrFailure::Empty),
        Err(AsrFailure::Empty),
        Ok(" ".into()),
    ]);
    let (err, _) = capture(|| process_input_with(&req, &NoHooks, &rt));
    assert_eq!(
        err.unwrap_err().message,
        "ASR produced empty transcription for all segments"
    );

    // Anchors: 295 s segments; unaligned text is spread over its segment.
    let anchored = ProcessInput {
        anchors: true,
        ..req.clone()
    };
    let rt = make_rt(vec![
        Ok("one".into()),
        Err(AsrFailure::Empty),
        Ok("three".into()),
    ]);
    let (result, _) = capture(|| process_input_with(&anchored, &NoHooks, &rt));
    let result = result.unwrap();
    assert_eq!(result.markdown, "[00:00:00] one\n\n[00:09:50] three");
    assert_eq!(rt.runs()[1].1[6], "295.0");
    let segments = result.segments.unwrap();
    assert_eq!(
        (segments[1].start, segments[1].end),
        (Some(590.0), Some(885.0))
    );

    let rt = make_rt(vec![Ok("one".into()), Ok("two".into()), Ok("three".into())]);
    let hooks = RecHooks {
        cancel_after: Some(1),
        ..RecHooks::default()
    };
    let (err, _) = capture(|| process_input_with(&req, &hooks, &rt));
    let err = err.unwrap_err();
    assert_eq!(err.kind, ErrorKind::Cancelled);
    assert_eq!(err.message, "Transcription cancelled between segments");
    assert_eq!(
        std::fs::read_dir(&seg_dir).unwrap().count(),
        0,
        "segments cleaned up"
    );
}

#[test]
fn local_asr_gates_fail_fast() {
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let wav = dir.path().join("a.wav");
    write_wav(&wav);
    let run = |req: ProcessInput, rt: &Mock| capture(|| process_input_with(&req, &NoHooks, rt)).0;

    let mut req = request(&wav);
    req.precision = Some("bf16".into());
    assert_eq!(
        run(req, &Mock::default()).unwrap_err().kind,
        ErrorKind::BackendUnavailable
    );

    let mut req = request(&wav);
    req.model = Some("Qwen/Qwen3-ASR-1.7B-hf".into());
    assert_eq!(
        run(req, &Mock::default()).unwrap_err().kind,
        ErrorKind::BackendUnavailable
    );

    let mut req = request(&wav);
    req.language = Some("xx".into());
    assert_eq!(
        run(req, &Mock::default()).unwrap_err().kind,
        ErrorKind::UnsupportedInput
    );

    let mut req = request(&wav);
    req.precision = Some("fp8".into());
    assert_eq!(
        run(req, &Mock::default()).unwrap_err().kind,
        ErrorKind::ProcessingFailed
    );

    let rt = Mock {
        preflight: Some(AimdError::backend_unavailable("no metal")),
        ..Mock::default()
    };
    assert_eq!(run(request(&wav), &rt).unwrap_err().message, "no metal");
    assert!(rt.loads.lock().unwrap().is_empty());

    let rt = Mock::default().with_asr(
        ASR_4BIT,
        ScriptedAsr::new(vec![Err(AsrFailure::Other("boom".into()))]),
    );
    assert_eq!(
        run(request(&wav), &rt).unwrap_err().message,
        "Qwen3-ASR transcription failed: boom"
    );
}

#[test]
fn asr_input_validation_messages() {
    let rt = Mock::default();
    let err = asr::transcribe_file(
        &rt,
        Path::new("gone.wav"),
        &ProcessInput::default(),
        None,
        &NoHooks,
    )
    .unwrap_err();
    assert_eq!(err.message, "Audio/video file not found: gone.wav");
    let dir = TempDir::new(None, "aimd-asr-").unwrap();
    let txt = dir.path().join("a.txt");
    std::fs::write(&txt, b"x").unwrap();
    let err =
        asr::transcribe_file(&rt, &txt, &ProcessInput::default(), None, &NoHooks).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedInput);
    assert_eq!(
        err.message,
        "Unsupported file format: .txt. Supported formats: .aac, .avi, .flac, .flv, .m4a, .m4v, \
         .mkv, .mov, .mp3, .mp4, .mp4a, .ogg, .opus, .ts, .wav, .webm, .wma, .wmv"
    );
}

// ---------------------------------------------------------------------------
// Remote ASR (test_remote_backends)
// ---------------------------------------------------------------------------

#[test]
fn remote_asr_posts_multipart_and_parses_qwen_output() {
    let dir = TempDir::new(None, "aimd-rasr-").unwrap();
    let wav = dir.path().join("clip.wav");
    write_wav(&wav);
    let (base, server) = serve(vec![(
        200,
        r#"{"text":"language English<asr_text>remote words"}"#.into(),
    )]);
    let mut req = request(&wav);
    req.asr_base_url = Some(base.clone());
    req.asr_api_key = Some("secret".into());
    req.language = Some("en".into());
    req.context = Some("ctx".into());
    req.precision = Some("4bit".into());
    let rt = Mock::default();
    let (result, logs) = capture(|| process_input_with(&req, &NoHooks, &rt));
    assert_eq!(result.unwrap().markdown, "remote words");
    let recorded = server.join().unwrap();
    let r = &recorded[0];
    assert_eq!(r.request_line, "POST /v1/audio/transcriptions HTTP/1.1");
    assert_eq!(r.header("authorization"), Some("Bearer secret"));
    assert!(
        r.header("content-type")
            .unwrap()
            .starts_with("multipart/form-data; boundary=----aimd-")
    );
    let body = String::from_utf8_lossy(&r.body);
    assert!(body.contains("name=\"model\"\r\n\r\nQwen3-ASR-1.7B\r\n"));
    assert!(body.contains("name=\"language\"\r\n\r\nen\r\n"));
    assert!(body.contains("name=\"prompt\"\r\n\r\nctx\r\n"));
    assert!(body.contains("filename=\"clip.wav\"\r\nContent-Type: audio/x-wav\r\n\r\nRIFF"));
    let precision_warnings: Vec<_> = warnings(&logs)
        .into_iter()
        .filter(|w| w.starts_with("Ignoring ASR precision"))
        .collect();
    assert_eq!(precision_warnings.len(), 1);
    assert!(
        rt.loads.lock().unwrap().is_empty(),
        "remote skips local models"
    );
}

#[test]
fn remote_asr_error_mapping() {
    let dir = TempDir::new(None, "aimd-rasr-").unwrap();
    let wav = dir.path().join("clip.wav");
    write_wav(&wav);
    let cases = [
        (
            503,
            "down",
            ErrorKind::BackendUnavailable,
            "Remote ASR backend returned HTTP 503: down",
        ),
        (
            400,
            "bad",
            ErrorKind::ProcessingFailed,
            "Remote ASR request failed with HTTP 400: bad",
        ),
        (
            200,
            "not json",
            ErrorKind::ProcessingFailed,
            "Remote ASR backend returned an invalid JSON response",
        ),
    ];
    for (status, body, kind, message) in cases {
        let (base, server) = serve(vec![(status, body.into())]);
        let mut req = request(&wav);
        req.asr_base_url = Some(base);
        let (err, _) = capture(|| process_input_with(&req, &NoHooks, &Mock::default()));
        let err = err.unwrap_err();
        assert_eq!((err.kind, err.message.as_str()), (kind, message));
        server.join().unwrap();
    }
    // Empty text is an empty transcription -> empty markdown (the CLI then
    // rejects it when persisting).
    let (base, server) = serve(vec![(200, r#"{"text":"  "}"#.into())]);
    let mut req = request(&wav);
    req.asr_base_url = Some(base);
    let (result, _) = capture(|| process_input_with(&req, &NoHooks, &Mock::default()));
    assert_eq!(result.unwrap().markdown, "");
    server.join().unwrap();

    let mut req = request(&wav);
    let url = closed_port_url();
    req.asr_base_url = Some(url.clone());
    let (err, _) = capture(|| process_input_with(&req, &NoHooks, &Mock::default()));
    let err = err.unwrap_err();
    assert_eq!(err.kind, ErrorKind::BackendUnavailable);
    assert!(
        err.message
            .starts_with(&format!("Remote ASR backend is unavailable at {url}/v1: ")),
        "{}",
        err.message
    );
}

// ---------------------------------------------------------------------------
// OCR (test_ocr)
// ---------------------------------------------------------------------------

#[test]
fn image_ocr_returns_stripped_single_page() {
    let dir = TempDir::new(None, "aimd-ocr-").unwrap();
    let png = dir.path().join("scan.png");
    std::fs::write(&png, b"png").unwrap();
    let rt = Mock::default().with_ocr(OCR_4BIT, vec![Ok("  # Title\n\nBody  ".into())]);
    let hooks = RecHooks::default();
    let result = process_input_with(&request(&png), &hooks, &rt).unwrap();
    assert_eq!(result.task_type, TaskType::Ocr);
    assert_eq!(result.markdown, "# Title\n\nBody");
    assert_eq!(result.text_context.title, "scan");
    assert_eq!(result.asset_base_uri.unwrap(), directory_uri(dir.path()));
    assert_eq!(
        hooks.events.lock().unwrap()[0],
        (
            "ocr".into(),
            Some(1),
            Some(1),
            Some("Processed OCR page 1 of 1".into())
        )
    );
}

#[test]
fn pdf_ocr_renders_with_mutool_and_assembles_pages() {
    let dir = TempDir::new(None, "aimd-ocr-").unwrap();
    let pdf = dir.path().join("scan.pdf");
    std::fs::write(&pdf, b"%PDF").unwrap();
    let make_rt = |outputs| {
        Mock {
            pdf_ocr: true,
            pdf_pages: 4,
            ..Mock::default()
        }
        .tool("mutool")
        .with_ocr(OCR_4BIT, outputs)
        .respond(|program, args| {
            assert_eq!(program, "mutool");
            let template = &args[3];
            let range = args.last().unwrap();
            let (a, b) = range.split_once('-').unwrap();
            for page in a.parse::<u32>().unwrap()..=b.parse().unwrap() {
                std::fs::write(template.replace("%04d", &format!("{page:04}")), b"png").unwrap();
            }
            ok_output("")
        })
    };
    let rt = make_rt(vec![
        Ok("second".into()),
        Ok("".into()),
        Ok("fourth".into()),
    ]);
    let mut req = request(&pdf);
    req.start = Some(1);
    let hooks = RecHooks::default();
    let err = process_input_with(&req, &hooks, &rt).unwrap_err();
    // Empty page content from the model is a hard error (Python parity).
    assert_eq!(err.message, "Unlimited-OCR-2 produced empty content");

    let rt = make_rt(vec![
        Ok("second".into()),
        Ok("third".into()),
        Ok("fourth".into()),
    ]);
    let result = process_input_with(&req, &hooks, &rt).unwrap();
    assert_eq!(
        result.markdown,
        "## Page 2\n\nsecond\n\n## Page 3\n\nthird\n\n## Page 4\n\nfourth"
    );
    // One render per page; local models read PPM.
    let runs = rt.runs();
    assert_eq!(runs.len(), 3);
    for (run, page) in runs.iter().zip(2..) {
        let args = &run.1;
        assert_eq!(args[0..3], ["draw", "-q", "-o"]);
        assert!(args[3].ends_with("/page_%04d.ppm"));
        assert_eq!(
            args[4..],
            [
                "-F",
                "pnm",
                "-r",
                "144",
                "-c",
                "rgb",
                &pdf.to_string_lossy(),
                &format!("{page}-{page}")
            ]
        );
    }
    let args = &runs[0].1;
    let rendered_dir = Path::new(&args[3]).parent().unwrap().to_path_buf();
    assert!(!rendered_dir.exists(), "rendered pages are cleaned up");

    // A page that fails to render stops the job with mutool's error after
    // the pages before it were recognized; nothing is left on disk.
    let rt = make_rt(vec![Ok("second".into()), Ok("third".into())]).respond(|_, args| {
        let range = args.last().unwrap();
        if range == "3-3" {
            return Ok(CommandOutput {
                status: Some(1),
                stdout: Vec::new(),
                stderr: b"broken page".to_vec(),
            });
        }
        let page = range.split_once('-').unwrap().0.parse::<u32>().unwrap();
        std::fs::write(args[3].replace("%04d", &format!("{page:04}")), b"ppm").unwrap();
        ok_output("")
    });
    let err = process_input_with(&req, &NoHooks, &rt).unwrap_err();
    assert_eq!(
        err.message,
        format!("mutool draw failed for {}: broken page", pdf.display())
    );
    let runs = rt.runs();
    assert_eq!(runs.len(), 2, "rendering stops at the failed page");
    assert!(!Path::new(&runs[0].1[3]).parent().unwrap().exists());

    let mut req = request(&pdf);
    req.start = Some(2);
    req.end = Some(5);
    let err = process_input_with(&req, &NoHooks, &make_rt(vec![])).unwrap_err();
    assert_eq!(
        err.message,
        format!(
            "Invalid OCR page range 2-5 for {} with 4 pages.",
            pdf.display()
        )
    );
}

#[test]
fn pdf_ocr_cancellation_and_missing_mutool() {
    let dir = TempDir::new(None, "aimd-ocr-").unwrap();
    let pdf = dir.path().join("scan.pdf");
    std::fs::write(&pdf, b"%PDF").unwrap();
    let rt = Mock {
        pdf_pages: 2,
        ..Mock::default()
    }
    .with_ocr(OCR_4BIT, vec![]);
    let mut req = request(&pdf);
    req.task_type = Some(TaskType::Ocr);
    let err = process_input_with(&req, &NoHooks, &rt).unwrap_err();
    assert_eq!(err.kind, ErrorKind::BackendUnavailable);
    assert!(err.message.contains("brew install mupdf"));

    let hooks = RecHooks {
        cancel_after: Some(0),
        ..RecHooks::default()
    };
    let err = process_input_with(&req, &hooks, &rt).unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ErrorKind::Cancelled, "OCR cancelled before model inference")
    );

    let rt = Mock {
        pdf_pages: 2,
        ..Mock::default()
    }
    .tool("mutool")
    .with_ocr(OCR_4BIT, vec![Ok("a".into()), Ok("b".into())])
    .respond(|_, args| {
        for page in 1..=2 {
            std::fs::write(args[3].replace("%04d", &format!("{page:04}")), b"png").unwrap();
        }
        ok_output("")
    });
    let hooks = RecHooks {
        cancel_after: Some(1),
        ..RecHooks::default()
    };
    let err = process_input_with(&req, &hooks, &rt).unwrap_err();
    assert_eq!(err.message, "OCR cancelled between pages");
}

#[test]
fn pdf_ocr_multi_page_windows() {
    let dir = TempDir::new(None, "aimd-ocr-").unwrap();
    let pdf = dir.path().join("scan.pdf");
    std::fs::write(&pdf, b"%PDF").unwrap();
    let make_rt = || {
        Mock {
            pdf_ocr: true,
            pdf_pages: 5,
            ..Mock::default()
        }
        .tool("mutool")
        .with_ocr_window(
            OCR2_4BIT,
            2,
            ["one", "two", "", "four", "five"]
                .map(|t| Ok(t.to_string()))
                .into(),
        )
        .respond(|_, args| {
            let (a, b) = args.last().unwrap().split_once('-').unwrap();
            for page in a.parse::<u32>().unwrap()..=b.parse().unwrap() {
                std::fs::write(args[3].replace("%04d", &format!("{page:04}")), b"png").unwrap();
            }
            ok_output("")
        })
    };
    let req = request(&pdf);
    let rt = make_rt();
    let hooks = RecHooks::default();
    let result = process_input_with(&req, &hooks, &rt).unwrap();
    // A blank page inside a window is not an error; it renders empty.
    assert_eq!(
        result.markdown,
        "## Page 1\n\none\n\n## Page 2\n\ntwo\n\n## Page 3\n\n## Page 4\n\nfour\n\n## Page 5\n\nfive"
    );
    assert_eq!(*rt.ocr[OCR2_4BIT].requests.lock().unwrap(), [2, 2, 1]);
    assert!(rt.ocr[OCR2_4BIT].calls.lock().unwrap().is_empty());
    let messages: Vec<_> = hooks
        .events
        .lock()
        .unwrap()
        .iter()
        .filter_map(|e| e.3.clone())
        .filter(|m| m.starts_with("Processed OCR pages"))
        .collect();
    assert_eq!(
        messages,
        [
            "Processed OCR pages 1-2 of 5",
            "Processed OCR pages 3-4 of 5",
            "Processed OCR pages 5-5 of 5",
        ]
    );

    // Cancellation lands between windows, never inside one.
    let rt = make_rt();
    let n = RecHooks::default();
    process_input_with(&req, &n, &rt).unwrap();
    let first_window = n
        .events
        .lock()
        .unwrap()
        .iter()
        .position(|e| e.3.as_deref() == Some("Processed OCR pages 1-2 of 5"))
        .unwrap();
    let rt = make_rt();
    let hooks = RecHooks {
        cancel_after: Some(first_window + 1),
        ..RecHooks::default()
    };
    let err = process_input_with(&req, &hooks, &rt).unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ErrorKind::Cancelled, "OCR cancelled between page windows")
    );
    assert_eq!(*rt.ocr[OCR2_4BIT].requests.lock().unwrap(), [2]);
}

#[test]
fn ocr_page_range_and_model_validation() {
    let dir = TempDir::new(None, "aimd-ocr-").unwrap();
    let png = dir.path().join("a.png");
    std::fs::write(&png, b"png").unwrap();
    let rt = Mock::default();
    let check = |start: Option<i64>, end: Option<i64>, message: &str| {
        let mut req = request(&png);
        req.start = start;
        req.end = end;
        let err = process_input_with(&req, &NoHooks, &rt).unwrap_err();
        assert_eq!(err.message, message);
    };
    check(
        Some(-1),
        None,
        "OCR start page must be greater than or equal to 0.",
    );
    check(
        None,
        Some(-2),
        "OCR end page must be greater than or equal to 0.",
    );
    check(
        Some(3),
        Some(1),
        "OCR start page must be less than or equal to end page.",
    );

    let mut req = request(&png);
    req.model = Some("glm-ocr".into());
    let err = process_input_with(&req, &NoHooks, &rt).unwrap_err();
    assert_eq!(err.kind, ErrorKind::ProcessingFailed);
    assert!(
        err.message
            .starts_with("Unsupported MLX VLM OCR model 'glm-ocr'.")
    );
    let rt = Mock::default().with_ocr(OCR_4BIT, vec![Err(OcrFailure::Other("bad".into()))]);
    assert_eq!(
        process_input_with(&request(&png), &NoHooks, &rt)
            .unwrap_err()
            .message,
        "Unlimited-OCR-2 inference failed: bad"
    );
    // Located OCR: one anchor per layout block; lines are segments only.
    let seg = |text: &str, kind: &str, bbox: [f32; 4]| aimd_core::Segment {
        text: text.into(),
        kind: Some(kind.into()),
        bbox: Some(bbox),
        ..aimd_core::Segment::default()
    };
    let (text, segments) = ocr::locate_page(
        3,
        crate::LocatedOcr {
            markdown: "# T\n\nbody".into(),
            blocks: Some(vec![
                seg("# T", "title", [0.1, 0.1, 0.9, 0.2]),
                seg("body", "text", [0.1, 0.3, 0.9, 0.8]),
            ]),
            lines: Some(vec![seg("body", "line", [0.1, 0.3, 0.9, 0.35])]),
        },
    );
    assert_eq!(
        text,
        "<!-- aimd:loc page=3 bbox=0.100,0.100,0.900,0.200 kind=title -->\n# T\n\n\
         <!-- aimd:loc page=3 bbox=0.100,0.300,0.900,0.800 kind=text -->\nbody"
    );
    assert_eq!(segments.len(), 3);
    assert!(segments.iter().all(|s| s.page == Some(3)));
    let (text, segments) = ocr::locate_page(
        1,
        crate::LocatedOcr {
            markdown: "a b".into(),
            blocks: None,
            lines: Some(vec![seg("a", "line", [0.0, 0.0, 0.5, 0.1])]),
        },
    );
    assert!(text.ends_with("kind=page -->\na b"));
    assert_eq!(segments[0].kind.as_deref(), Some("line"));
    assert_eq!(
        ocr::pages_to_markdown(&[ocr::OcrPage {
            page_index: Some(0),
            text: " ".into()
        }])
        .unwrap_err()
        .message,
        "OCR returned empty content"
    );
}

#[test]
fn remote_ocr_posts_chat_completion_with_page_window() {
    let dir = TempDir::new(None, "aimd-rocr-").unwrap();
    let png = dir.path().join("a.jpg");
    std::fs::write(&png, b"\xff\xd8jpg").unwrap();
    let completion =
        |text: &str| serde_json::json!({"choices": [{"message": {"content": text}}]}).to_string();
    let (base, server) = serve(vec![(200, completion(" hello "))]);
    let mut req = request(&png);
    req.ocr_base_url = Some(format!("{base}/"));
    req.ocr_model = Some("served".into());
    req.precision = Some("8bit".into());
    let (result, logs) = capture(|| process_input_with(&req, &NoHooks, &Mock::default()));
    assert_eq!(result.unwrap().markdown, "hello");
    assert_eq!(
        warnings(&logs),
        vec![
            "Ignoring OCR precision for remote inference; server-side weights already define \
              their precision."
                .to_string()
        ]
    );
    let r = &server.join().unwrap()[0];
    assert_eq!(r.request_line, "POST /v1/chat/completions HTTP/1.1");
    assert_eq!(r.header("authorization"), Some("Bearer not-needed"));
    let payload: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
    assert_eq!(payload["model"], "served");
    assert_eq!(payload["window_size"], 128);
    assert_eq!(payload["ngram_size"], 35);
    assert_eq!(payload["skip_special_tokens"], false);
    assert_eq!(
        payload["messages"][0]["content"][0]["text"],
        "<image>\ndocument parsing."
    );
    assert_eq!(
        payload["messages"][0]["content"][1]["image_url"]["url"],
        format!(
            "data:image/jpeg;base64,{}",
            crate::http::base64_encode(b"\xff\xd8jpg")
        )
    );

    // PDF pages use the multi-page window; invalid completions are errors.
    let pdf = dir.path().join("s.pdf");
    std::fs::write(&pdf, b"%PDF").unwrap();
    let (base, server) = serve(vec![
        (200, completion("p1")),
        (200, r#"{"choices":[]}"#.into()),
    ]);
    let rt = Mock {
        pdf_ocr: true,
        pdf_pages: 2,
        ..Mock::default()
    }
    .tool("mutool")
    .respond(|_, args| {
        for page in 1..=2 {
            std::fs::write(args[3].replace("%04d", &format!("{page:04}")), b"png").unwrap();
        }
        ok_output("")
    });
    let mut req = request(&pdf);
    req.ocr_base_url = Some(base);
    let err = process_input_with(&req, &NoHooks, &rt).unwrap_err();
    assert_eq!(
        err.message,
        "Remote OCR backend returned an invalid chat completion"
    );
    // Remote servers get PNG pages.
    assert_eq!(rt.runs()[0].1[5], "png");
    let recorded = server.join().unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&recorded[0].body).unwrap();
    assert_eq!(payload["window_size"], 1024);
}
