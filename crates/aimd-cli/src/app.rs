//! `aimd` command flow (ports `aimd.interfaces.cli.app.process`).

use std::io::Write;
use std::path::{Path, PathBuf};

use aimd_core::log::{LEVEL_NAMES, Level, set_level};
use aimd_core::output::{
    OutputFormat, PersistError, build_output_text, payload_text, persist_output, resolve_path,
    result_payload, subtitle_text,
};
use aimd_core::{NoHooks, ProcessInput, TaskType, log_debug, log_error, log_info, log_warn};
use aimd_process::Runtime;

use crate::args::{Cli, Parsed, USAGE, help_text, parse};

/// `_create_output_path_from_title`: `dir` is `None` for a bare relative
/// name (Python's `Path(".") / name` renders without `./`).
pub fn output_path_from_title(
    title: &str,
    suffix: &str,
    extension: &str,
    dir: Option<&Path>,
) -> PathBuf {
    let replaced: String = title
        .chars()
        .map(|c| if "<>:\"/\\|?*".contains(c) { '_' } else { c })
        .collect();
    let mut collapsed = String::new();
    let mut in_space = false;
    for c in aimd_core::pyfmt::strip(&replaced).chars() {
        if aimd_core::pyfmt::is_space(c) {
            if !in_space {
                collapsed.push('_');
            }
            in_space = true;
        } else {
            collapsed.push(c);
            in_space = false;
        }
    }
    let mut sanitized = collapsed.trim_matches(['.', '_']).to_string();
    if sanitized.chars().count() > 100 {
        sanitized = sanitized
            .chars()
            .take(100)
            .collect::<String>()
            .trim_end_matches(['.', '_'])
            .to_string();
    }
    if sanitized.is_empty() {
        sanitized = "output".into();
    }
    let name = format!("{sanitized}_{suffix}.{extension}");
    match dir {
        Some(d) => d.join(name),
        None => PathBuf::from(name),
    }
}

/// `Path(x).parent` as a join base (`None` for `.`).
fn parent_of(input: &str) -> Option<PathBuf> {
    match Path::new(input).parent() {
        Some(p) if !p.as_os_str().is_empty() && p != Path::new(".") => Some(p.to_path_buf()),
        _ => None,
    }
}

/// Python `str(Path(x))` for display (drops `./` prefixes and trailing `/`).
fn py_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    let parts: Vec<&str> = s
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    let joined = parts.join("/");
    match (s.starts_with('/'), joined.is_empty()) {
        (true, _) => format!("/{joined}"),
        (false, true) => ".".into(),
        (false, false) => joined,
    }
}

/// Run the CLI; returns the exit code.
pub fn run(
    args: &[String],
    env: &dyn Fn(&str) -> Option<String>,
    rt: &dyn Runtime,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    let cli = match parse(args, env) {
        Parsed::Run(cli) => cli,
        Parsed::Help => {
            let _ = write!(stdout, "{}", help_text());
            return 0;
        }
        Parsed::Usage {
            message,
            show_usage,
        } => {
            if show_usage {
                let _ = writeln!(stderr, "{USAGE}\nTry 'aimd --help' for help.\n");
            }
            let _ = writeln!(stderr, "Error: {message}");
            return 2;
        }
    };
    let Some(level) = Level::parse(&cli.log_level) else {
        let _ = writeln!(
            stderr,
            "Error: Invalid log level '{}'. Valid levels: {}",
            cli.log_level,
            LEVEL_NAMES.join(", ")
        );
        return 1;
    };
    set_level(level);
    if level == Level::Debug {
        log_debug!("Logging level configured to: {}", cli.log_level);
    }
    let requested = match cli.task.as_deref() {
        None => None,
        Some(task) => match TaskType::parse(&aimd_core::pyfmt::strip(task).to_lowercase()) {
            Some(t) => Some(t),
            None => {
                let _ = writeln!(
                    stderr,
                    "Error: Unsupported task. Supported tasks: transcript, convert, ocr."
                );
                return 1;
            }
        },
    };
    let format = match cli.format.as_deref().map(OutputFormat::parse) {
        None => OutputFormat::Markdown,
        Some(Some(format)) => format,
        Some(None) => {
            let _ = writeln!(
                stderr,
                "Error: Unsupported format. Supported formats: {}.",
                OutputFormat::NAMES
            );
            return 1;
        }
    };
    if format.is_subtitle() && cli.raw_transcript {
        let _ = writeln!(
            stderr,
            "Error: --format {} needs timed segments; drop --raw-transcript.",
            format.extension()
        );
        return 1;
    }
    log_info!("Input: {}", cli.input_source);
    match process(&cli, requested, format, rt, stdout) {
        Ok(()) => 0,
        Err(message) => {
            log_error!("{message}");
            1
        }
    }
}

fn process(
    cli: &Cli,
    requested: Option<TaskType>,
    format: OutputFormat,
    rt: &dyn Runtime,
    stdout: &mut dyn Write,
) -> Result<(), String> {
    let task_name = match requested {
        Some(TaskType::Transcript) => "Transcription",
        Some(TaskType::Convert) => "Conversion",
        Some(TaskType::Ocr) => "OCR",
        None => "Processing",
    };
    let generic = |e: &dyn std::fmt::Display| format!("{task_name} failed: {e}");
    let temp_dir = cli.temp_dir.as_ref().map(PathBuf::from);
    if let Some(dir) = &temp_dir {
        std::fs::create_dir_all(dir).map_err(|e| generic(&e))?;
        log_info!("Using custom temp directory: {}", py_path(dir));
    }
    let request = ProcessInput {
        input_source: cli.input_source.clone(),
        task_type: requested,
        model: cli.model.clone(),
        language: cli.language.clone(),
        start: cli.start,
        end: cli.end,
        save_original: cli.save_original.as_ref().map(PathBuf::from),
        cookies: cli.cookies.as_ref().map(PathBuf::from),
        cookies_from_browser: cli.cookies_from_browser.clone(),
        temp_dir,
        raw_transcript: cli.raw_transcript,
        precision: cli.precision.clone(),
        context: cli.context.clone(),
        metadata_context: !cli.no_context,
        anchors: cli.anchors && !cli.no_anchors,
        include_segments: format != OutputFormat::Markdown,
        asr_base_url: cli.asr_base_url.clone(),
        asr_model: cli.asr_model.clone(),
        asr_api_key: cli.asr_api_key.clone(),
        ocr_base_url: cli.ocr_base_url.clone(),
        ocr_model: cli.ocr_model.clone(),
        ocr_api_key: cli.ocr_api_key.clone(),
    };
    let result = aimd_process::process_input_with(&request, &NoHooks, rt).map_err(|e| e.message)?;
    log_info!("Task: {}", result.task_type.as_str());
    if let Some(platform) = &result.platform {
        log_info!("Platform: {platform}");
    }
    if result.task_type == TaskType::Convert
        && let Some(dir) = &result.output_dir
    {
        let stem = Path::new(&cli.input_source)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if cli.output.is_some() {
            log_warn!("Ignoring --output for document asset conversions; output is a directory.");
        }
        log_info!("Document converted with assets to: {}", py_path(dir));
        log_info!("Main file: {}", py_path(&dir.join(format!("{stem}.md"))));
        log_info!("Images extracted to: {}", py_path(&dir.join("images")));
        let _ = writeln!(stdout, "Successfully converted document with assets");
        let _ = writeln!(stdout, "Output saved to {}", py_path(dir));
        return Ok(());
    }
    let output = match &cli.output {
        Some(path) => PathBuf::from(path),
        None => {
            let suffix = match result.task_type {
                TaskType::Transcript => "transcript",
                TaskType::Convert => "converted",
                TaskType::Ocr => "ocr",
            };
            let dir = if result.task_type == TaskType::Transcript {
                Some(std::env::current_dir().map_err(|e| generic(&e))?)
            } else {
                parent_of(&cli.input_source)
            };
            output_path_from_title(
                &result.text_context.title,
                suffix,
                format.extension(),
                dir.as_deref(),
            )
        }
    };
    let shown = py_path(&output);
    log_info!("Output: {shown}");
    let text = match format {
        OutputFormat::Markdown => None,
        OutputFormat::Json => {
            build_output_text(result.task_type, &result.markdown).map_err(|e| e.message)?;
            let file = resolve_path(&output).to_string_lossy().into_owned();
            Some(payload_text(&result_payload(&result, Some(file), None, true)) + "\n")
        }
        OutputFormat::Srt | OutputFormat::Vtt => {
            Some(subtitle_text(&result, format).map_err(|e| e.message)?)
        }
    };
    persist_output(
        &output,
        result.task_type,
        text.as_deref().unwrap_or(&result.markdown),
    )
    .map_err(|e| match e {
        PersistError::Aimd(e) => e.message,
        PersistError::Io(e) => generic(&e),
    })?;
    let (log_line, echo) = match result.task_type {
        TaskType::Transcript => ("Transcript saved to", "Successfully transcribed"),
        TaskType::Ocr => ("OCR output saved to", "Successfully OCR processed"),
        TaskType::Convert => ("Converted file saved to", "Successfully converted"),
    };
    log_info!("{log_line}: {shown}");
    let _ = writeln!(stdout, "{echo}");
    let _ = writeln!(stdout, "Output saved to {shown}");
    Ok(())
}

/// `KEY=VALUE` pairs of a `.env` file (comments, `export`, quotes).
pub fn parse_dotenv(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let mut value = value.trim();
        if value.len() >= 2
            && ((value.starts_with('"') && value.ends_with('"'))
                || (value.starts_with('\'') && value.ends_with('\'')))
        {
            value = &value[1..value.len() - 1];
        } else if let Some(pos) = value.find(" #") {
            value = value[..pos].trim_end();
        }
        if !key.is_empty() {
            pairs.push((key.to_string(), value.to_string()));
        }
    }
    pairs
}

/// `load_dotenv()`: `./.env` values that never override the existing
/// environment. Call before any thread is spawned.
pub fn load_dotenv(path: &Path) {
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    for (key, value) in parse_dotenv(&text) {
        if std::env::var_os(&key).is_none() {
            // SAFETY: called once at startup before any thread is spawned.
            unsafe { std::env::set_var(key, value) };
        }
    }
}
