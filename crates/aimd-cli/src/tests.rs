//! Ports of the `test_cli.py` scenarios over a fake runtime.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aimd_core::log::{Level, capture};
use aimd_core::{AimdError, ProcessHooks, Result};
use aimd_process::{AsrFailure, AsrModel, DocOutput, LocalModality, OcrFailure, OcrModel, Runtime};

use crate::app::{output_path_from_title, parse_dotenv, run};

struct Fake {
    transcript: String,
    doc: DocOutput,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            transcript: "hello".into(),
            doc: DocOutput {
                markdown: "# Doc Title\n\nbody".into(),
                title: None,
            },
        }
    }
}

struct Text(String);

impl AsrModel for Text {
    fn transcribe(
        &self,
        _: &Path,
        _: &str,
        _: Option<&str>,
    ) -> std::result::Result<String, AsrFailure> {
        if self.0.is_empty() {
            Err(AsrFailure::Empty)
        } else {
            Ok(self.0.clone())
        }
    }
}

impl OcrModel for Text {
    fn recognize(&self, _: &Path) -> std::result::Result<String, OcrFailure> {
        Ok(self.0.clone())
    }
}

impl Runtime for Fake {
    fn env(&self, _: &str) -> Option<String> {
        None
    }
    fn which(&self, _: &str) -> Option<PathBuf> {
        None
    }
    fn preflight_local(&self, _: LocalModality) -> Result<()> {
        Ok(())
    }
    fn load_asr(&self, _: &str) -> Result<Arc<dyn AsrModel>> {
        Ok(Arc::new(Text(self.transcript.clone())))
    }
    fn load_ocr(&self, _: &str) -> Result<Arc<dyn OcrModel>> {
        Ok(Arc::new(Text("# Scanned Page\n\ntext".into())))
    }
    fn convert_document(
        &self,
        _: &Path,
        _: Option<&Path>,
        _: Option<&Path>,
        _: &dyn ProcessHooks,
    ) -> Result<DocOutput> {
        Ok(self.doc.clone())
    }
    fn url_transcript(
        &self,
        _: &aimd_core::ProcessInput,
        _: &dyn ProcessHooks,
    ) -> Result<aimd_process::UrlOutput> {
        Err(AimdError::backend_unavailable("no url"))
    }
}

struct Outcome {
    code: i32,
    stdout: String,
    stderr: String,
    logs: Vec<(Level, String)>,
}

fn invoke(args: &[&str], rt: &Fake) -> Outcome {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let (code, logs) = capture(|| run(&args, &|_| None, rt, &mut out, &mut err));
    aimd_core::log::set_level(Level::Info);
    Outcome {
        code,
        stdout: String::from_utf8(out).unwrap(),
        stderr: String::from_utf8(err).unwrap(),
        logs,
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aimd-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// 0.1 s of 16 kHz mono s16 silence.
fn write_wav(path: &Path) {
    let data_len = 3200u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    for v in [1u16, 1] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [16000u32, 32000] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [2u16, 16] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    b.resize(b.len() + data_len as usize, 0);
    std::fs::write(path, b).unwrap();
}

#[test]
fn title_sanitization_matches_python() {
    let p = |t: &str| {
        output_path_from_title(t, "ocr", None)
            .to_string_lossy()
            .into_owned()
    };
    assert_eq!(p("My: Video / Part?1"), "My__Video___Part_1_ocr.md");
    assert_eq!(p("  spaced\t out  "), "spaced_out_ocr.md");
    assert_eq!(p("._hidden._"), "hidden_ocr.md");
    assert_eq!(p("***"), "output_ocr.md");
    assert_eq!(p(""), "output_ocr.md");
    let long = format!("{}._tail", "a".repeat(98));
    assert_eq!(p(&long), format!("{}_ocr.md", "a".repeat(98)));
    assert_eq!(
        output_path_from_title("t", "transcript", Some(Path::new("/x/y"))),
        PathBuf::from("/x/y/t_transcript.md")
    );
}

#[test]
fn transcript_writes_output_and_reports_success() {
    let dir = scratch("transcript");
    let wav = dir.join("clip.wav");
    write_wav(&wav);
    let out = dir.join("nested/out.md");
    let o = invoke(
        &[&wav.to_string_lossy(), "-o", &out.to_string_lossy()],
        &Fake::default(),
    );
    assert_eq!(o.code, 0, "{:?}", o.logs);
    assert_eq!(std::fs::read_to_string(&out).unwrap(), "hello");
    assert_eq!(
        o.stdout,
        format!(
            "Successfully transcribed\nOutput saved to {}\n",
            out.display()
        )
    );
    let infos: Vec<&str> = o
        .logs
        .iter()
        .filter(|(l, _)| *l == Level::Info)
        .map(|(_, m)| m.as_str())
        .collect();
    assert_eq!(infos[0], format!("Input: {}", wav.display()));
    assert!(infos.contains(&"Task: transcript"));
    assert!(infos.contains(&format!("Transcript saved to: {}", out.display()).as_str()));
}

#[test]
fn empty_transcript_exits_with_error() {
    // test_cli.py::test_cli_rejects_empty_transcript_output
    let dir = scratch("empty");
    let wav = dir.join("clip.wav");
    write_wav(&wav);
    let rt = Fake {
        transcript: String::new(),
        ..Fake::default()
    };
    let out = dir.join("out.md");
    let o = invoke(&[&wav.to_string_lossy(), "-o", &out.to_string_lossy()], &rt);
    assert_eq!(o.code, 1);
    assert!(
        o.logs
            .contains(&(Level::Error, "Transcription returned empty content".into()))
    );
    assert!(!out.exists());
}

#[test]
fn ocr_default_output_lands_next_to_input() {
    let dir = scratch("ocr");
    let png = dir.join("scan.png");
    std::fs::write(&png, b"png").unwrap();
    let o = invoke(&[&png.to_string_lossy()], &Fake::default());
    assert_eq!(o.code, 0, "{:?}", o.logs);
    let expected = dir.join("scan_ocr.md");
    assert_eq!(
        std::fs::read_to_string(&expected).unwrap(),
        "# Scanned Page\n\ntext"
    );
    assert_eq!(
        o.stdout,
        format!(
            "Successfully OCR processed\nOutput saved to {}\n",
            expected.display()
        )
    );
}

#[test]
fn asset_conversion_reports_directory_and_ignores_output() {
    let dir = scratch("assets");
    let epub = dir.join("book.epub");
    std::fs::write(&epub, b"zip").unwrap();
    let o = invoke(
        &[&epub.to_string_lossy(), "-o", "ignored.md"],
        &Fake::default(),
    );
    assert_eq!(o.code, 0);
    let out_dir = dir.join("book");
    assert_eq!(
        o.stdout,
        format!(
            "Successfully converted document with assets\nOutput saved to {}\n",
            out_dir.display()
        )
    );
    assert!(o.logs.contains(&(
        Level::Warning,
        "Ignoring --output for document asset conversions; output is a directory.".into()
    )));
    assert!(o.logs.contains(&(
        Level::Info,
        format!("Main file: {}", out_dir.join("book.md").display())
    )));
    assert!(!Path::new("ignored.md").exists());
}

#[test]
fn converted_documents_use_title_for_default_name() {
    let dir = scratch("convert");
    let md = dir.join("notes.md");
    std::fs::write(&md, b"x").unwrap();
    let o = invoke(
        &[&md.to_string_lossy(), "--task", " Convert "],
        &Fake::default(),
    );
    assert_eq!(o.code, 0, "{:?}", o.logs);
    let expected = dir.join("Doc_Title_converted.md");
    assert_eq!(
        std::fs::read_to_string(&expected).unwrap(),
        "# Doc Title\n\nbody"
    );
    assert!(o.stdout.starts_with("Successfully converted\n"));
}

#[test]
fn error_exits_and_messages() {
    let rt = Fake::default();
    let o = invoke(&["nofile.wav"], &rt);
    assert_eq!(o.code, 1);
    assert_eq!(
        o.logs.last().unwrap(),
        &(Level::Error, "Input file not found: nofile.wav".into())
    );

    let o = invoke(&["nofile"], &rt);
    assert_eq!(o.code, 1);
    assert!(
        o.logs
            .last()
            .unwrap()
            .1
            .starts_with("Unsupported input source.")
    );

    let o = invoke(&["x.wav", "--log-level", "verbose"], &rt);
    assert_eq!(o.code, 1);
    assert_eq!(
        o.stderr,
        "Error: Invalid log level 'verbose'. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL\n"
    );

    let o = invoke(&["x.wav", "--task", "summarize"], &rt);
    assert_eq!(o.code, 1);
    assert_eq!(
        o.stderr,
        "Error: Unsupported task. Supported tasks: transcript, convert, ocr.\n"
    );

    let o = invoke(&[], &rt);
    assert_eq!(o.code, 2);
    assert_eq!(
        o.stderr,
        "Usage: aimd [OPTIONS] {input_source}\nTry 'aimd --help' for help.\n\nError: Missing \
         argument 'input_source'.\n"
    );

    let o = invoke(&["--help"], &rt);
    assert_eq!(o.code, 0);
    assert!(o.stdout.starts_with("Usage: aimd [OPTIONS] {input_source}"));

    let dir = scratch("tempdir");
    let tmp = dir.join("t/nested");
    let o = invoke(&["nofile.wav", "--temp-dir", &tmp.to_string_lossy()], &rt);
    assert_eq!(o.code, 1);
    assert!(tmp.is_dir());
    assert!(o.logs.contains(&(
        Level::Info,
        format!("Using custom temp directory: {}", tmp.display())
    )));
}

#[test]
fn dotenv_parsing() {
    let pairs = parse_dotenv(
        "# comment\nAIMD_A=\"from file\"\nexport AIMD_B=v # note\nAIMD_C='q # kept'\nbad line\n",
    );
    assert_eq!(
        pairs,
        vec![
            ("AIMD_A".to_string(), "from file".to_string()),
            ("AIMD_B".to_string(), "v".to_string()),
            ("AIMD_C".to_string(), "q # kept".to_string()),
        ]
    );
}
