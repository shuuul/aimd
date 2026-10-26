//! Hand-written parser for the frozen Typer flag surface.

use aimd_core::output::{CONTEXT_HELP_TEXT, MODEL_HELP_TEXT, PRECISION_HELP_TEXT};

/// Parsed command line (env-var fallbacks applied by [`parse`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cli {
    pub input_source: String,
    pub task: Option<String>,
    pub output: Option<String>,
    pub model: Option<String>,
    pub precision: Option<String>,
    pub asr_base_url: Option<String>,
    pub asr_model: Option<String>,
    pub asr_api_key: Option<String>,
    pub ocr_base_url: Option<String>,
    pub ocr_model: Option<String>,
    pub ocr_api_key: Option<String>,
    pub context: Option<String>,
    pub no_context: bool,
    pub language: Option<String>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub save_original: Option<String>,
    pub cookies: Option<String>,
    pub cookies_from_browser: Option<String>,
    pub log_level: String,
    pub raw_transcript: bool,
    pub temp_dir: Option<String>,
}

/// Parse result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Parsed {
    Run(Box<Cli>),
    Help,
    /// Usage error (exit 2): message and whether to print the usage lines.
    Usage {
        message: String,
        show_usage: bool,
    },
}

pub const USAGE: &str = "Usage: aimd [OPTIONS] {input_source}";

#[derive(Clone, Copy)]
enum Kind {
    Str,
    Int,
    Flag,
}

/// (long, short, field id, kind, env var)
type OptionSpec = (
    &'static str,
    Option<&'static str>,
    &'static str,
    Kind,
    Option<&'static str>,
);

const OPTIONS: &[OptionSpec] = &[
    ("--task", None, "task", Kind::Str, None),
    ("--output", Some("-o"), "output", Kind::Str, None),
    ("--model", Some("-m"), "model", Kind::Str, None),
    ("--precision", None, "precision", Kind::Str, None),
    (
        "--asr-base-url",
        None,
        "asr_base_url",
        Kind::Str,
        Some("AIMD_ASR_BASE_URL"),
    ),
    (
        "--asr-model",
        None,
        "asr_model",
        Kind::Str,
        Some("AIMD_ASR_MODEL"),
    ),
    (
        "--asr-api-key",
        None,
        "asr_api_key",
        Kind::Str,
        Some("AIMD_ASR_API_KEY"),
    ),
    (
        "--ocr-base-url",
        None,
        "ocr_base_url",
        Kind::Str,
        Some("AIMD_OCR_BASE_URL"),
    ),
    (
        "--ocr-model",
        None,
        "ocr_model",
        Kind::Str,
        Some("AIMD_OCR_MODEL"),
    ),
    (
        "--ocr-api-key",
        None,
        "ocr_api_key",
        Kind::Str,
        Some("AIMD_OCR_API_KEY"),
    ),
    ("--context", None, "context", Kind::Str, None),
    ("--no-context", None, "no_context", Kind::Flag, None),
    ("--language", Some("-l"), "language", Kind::Str, None),
    ("--start", None, "start", Kind::Int, None),
    ("--end", None, "end", Kind::Int, None),
    (
        "--save-original",
        Some("-s"),
        "save_original",
        Kind::Str,
        None,
    ),
    ("--cookies", Some("-c"), "cookies", Kind::Str, None),
    (
        "--cookies-from-browser",
        None,
        "cookies_from_browser",
        Kind::Str,
        None,
    ),
    ("--log-level", None, "log_level", Kind::Str, None),
    ("--raw-transcript", None, "raw_transcript", Kind::Flag, None),
    (
        "--temp-dir",
        None,
        "temp_dir",
        Kind::Str,
        Some("AIMD_TEMP_DIR"),
    ),
];

fn usage(message: impl Into<String>) -> Parsed {
    Parsed::Usage {
        message: message.into(),
        show_usage: true,
    }
}

fn set(cli: &mut Cli, field: &str, value: String) {
    let slot = match field {
        "task" => &mut cli.task,
        "output" => &mut cli.output,
        "model" => &mut cli.model,
        "precision" => &mut cli.precision,
        "asr_base_url" => &mut cli.asr_base_url,
        "asr_model" => &mut cli.asr_model,
        "asr_api_key" => &mut cli.asr_api_key,
        "ocr_base_url" => &mut cli.ocr_base_url,
        "ocr_model" => &mut cli.ocr_model,
        "ocr_api_key" => &mut cli.ocr_api_key,
        "context" => &mut cli.context,
        "language" => &mut cli.language,
        "save_original" => &mut cli.save_original,
        "cookies" => &mut cli.cookies,
        "cookies_from_browser" => &mut cli.cookies_from_browser,
        "temp_dir" => &mut cli.temp_dir,
        "log_level" => {
            cli.log_level = value;
            return;
        }
        _ => unreachable!("unknown field {field}"),
    };
    *slot = Some(value);
}

/// Click `INT`: optional sign, digits, surrounding whitespace, `_` allowed
/// between digits (Python `int()`).
fn parse_int(value: &str) -> Option<i64> {
    let t = aimd_core::pyfmt::strip(value);
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    digits
        .replace('_', "")
        .parse::<i64>()
        .ok()
        .map(|v| v * sign)
}

/// Parse `args` (without the program name); `env` supplies the Typer
/// env-var fallbacks (empty values count as unset).
pub fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Parsed {
    let mut cli = Cli {
        log_level: "INFO".into(),
        ..Cli::default()
    };
    let mut seen: Vec<&str> = Vec::new();
    let mut positionals: Vec<String> = Vec::new();
    let mut only_positional = false;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if only_positional || arg == "-" || !arg.starts_with('-') {
            positionals.push(arg.clone());
            continue;
        }
        if arg == "--" {
            only_positional = true;
            continue;
        }
        if arg == "--help" {
            return Parsed::Help;
        }
        // Split `--opt=value` and `-ovalue`.
        let (name, inline): (String, Option<String>) = if arg.starts_with("--") {
            match arg.split_once('=') {
                Some((n, v)) => (n.to_string(), Some(v.to_string())),
                None => (arg.clone(), None),
            }
        } else {
            let mut chars = arg.chars();
            chars.next();
            let short: String = format!("-{}", chars.next().unwrap_or_default());
            let rest: String = chars.collect();
            (short, (!rest.is_empty()).then_some(rest))
        };
        let Some(&(long, _, field, kind, _)) = OPTIONS
            .iter()
            .find(|(l, s, ..)| *l == name || s.is_some_and(|s| s == name))
        else {
            return usage(format!("No such option: {name}"));
        };
        match kind {
            Kind::Flag => {
                if inline.is_some() {
                    return usage(format!("Option '{long}' does not take a value."));
                }
                match field {
                    "no_context" => cli.no_context = true,
                    _ => cli.raw_transcript = true,
                }
            }
            Kind::Str | Kind::Int => {
                let value = match inline {
                    Some(v) => v,
                    None if i < args.len() => {
                        i += 1;
                        args[i - 1].clone()
                    }
                    None => {
                        return Parsed::Usage {
                            message: format!("Option '{name}' requires an argument."),
                            show_usage: false,
                        };
                    }
                };
                if let Kind::Int = kind {
                    let Some(n) = parse_int(&value) else {
                        return usage(format!(
                            "Invalid value for '{name}': '{value}' is not a valid int."
                        ));
                    };
                    if field == "start" {
                        cli.start = Some(n);
                    } else {
                        cli.end = Some(n);
                    }
                } else {
                    set(&mut cli, field, value);
                }
            }
        }
        seen.push(field);
    }
    for &(_, _, field, _, envvar) in OPTIONS {
        if let Some(var) = envvar
            && !seen.contains(&field)
            && let Some(value) = env(var).filter(|v| !v.is_empty())
        {
            set(&mut cli, field, value);
        }
    }
    let mut positionals = positionals.into_iter();
    let Some(input) = positionals.next() else {
        return usage("Missing argument 'input_source'.");
    };
    let extra: Vec<String> = positionals.collect();
    if !extra.is_empty() {
        return usage(format!(
            "Got unexpected extra argument(s) ({})",
            extra.join(" ")
        ));
    }
    cli.input_source = input;
    Parsed::Run(Box::new(cli))
}

/// Plain-text `--help`.
pub fn help_text() -> String {
    let rows: [(&str, &str); 22] = [
        (
            "--task TEXT",
            "Optional explicit task: transcript, convert, or ocr. Defaults to auto-routing.",
        ),
        (
            "-o, --output PATH",
            "Output file path. If not specified, auto-generated from input",
        ),
        ("-m, --model TEXT", MODEL_HELP_TEXT),
        ("--precision TEXT", PRECISION_HELP_TEXT),
        (
            "--asr-base-url TEXT",
            "OpenAI-compatible ASR base URL; bare server URLs add /v1. [env var: AIMD_ASR_BASE_URL]",
        ),
        (
            "--asr-model TEXT",
            "Model ID served by the remote ASR endpoint. [env var: AIMD_ASR_MODEL]",
        ),
        (
            "--asr-api-key TEXT",
            "Bearer token for the remote ASR endpoint. [env var: AIMD_ASR_API_KEY]",
        ),
        (
            "--ocr-base-url TEXT",
            "OpenAI-compatible OCR base URL; bare server URLs add /v1. [env var: AIMD_OCR_BASE_URL]",
        ),
        (
            "--ocr-model TEXT",
            "Model ID served by the remote OCR endpoint. [env var: AIMD_OCR_MODEL]",
        ),
        (
            "--ocr-api-key TEXT",
            "Bearer token for the remote OCR endpoint. [env var: AIMD_OCR_API_KEY]",
        ),
        ("--context TEXT", CONTEXT_HELP_TEXT),
        (
            "--no-context",
            "Disable automatic ASR context built from URL metadata (title/description/tags).",
        ),
        (
            "-l, --language TEXT",
            "ASR language (e.g., zh, en, ja, yue); omit to auto-detect. OCR ignores it.",
        ),
        (
            "--start INTEGER",
            "0-based inclusive start page for OCR PDF inputs.",
        ),
        (
            "--end INTEGER",
            "0-based inclusive end page for OCR PDF inputs.",
        ),
        (
            "-s, --save-original PATH",
            "Save original downloaded audio/video file to a path or directory.",
        ),
        (
            "-c, --cookies PATH",
            "Path to Netscape-format cookies file for URL extraction.",
        ),
        (
            "--cookies-from-browser TEXT",
            "Browser cookie source for URL extraction.",
        ),
        (
            "--log-level TEXT",
            "Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL [default: INFO]",
        ),
        (
            "--raw-transcript",
            "Preserve original subtitle formatting (SRT/VTT timestamps). By default, subtitles are simplified to plain text.",
        ),
        (
            "--temp-dir PATH",
            "Custom temporary directory for intermediate files. Overrides AIMD_TEMP_DIR env var. Useful for sandboxed environments. [env var: AIMD_TEMP_DIR]",
        ),
        ("--help", "Show this message and exit."),
    ];
    let mut out = format!(
        "{USAGE}\n\n  Process audio/video/url/doc inputs to markdown.\n\nArguments:\n  input_source  Audio \
         file, video file, video URL, document, image, or scanned PDF to process [required]\n\nOptions:\n"
    );
    for (flag, help) in rows {
        out.push_str(&format!("  {flag}\n"));
        let mut line = String::from("      ");
        for word in help.split_whitespace() {
            if line.len() + word.len() > 78 && !line.trim().is_empty() {
                out.push_str(line.trim_end());
                out.push('\n');
                line = String::from("      ");
            }
            line.push_str(word);
            line.push(' ');
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn run(v: &[&str]) -> Cli {
        match parse(&args(v), &no_env) {
            Parsed::Run(cli) => *cli,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parses_every_flag_form() {
        let cli = run(&[
            "in.pdf",
            "--task",
            "ocr",
            "-o",
            "out.md",
            "-mqwen3-asr-0.6b",
            "--precision=8bit",
            "--start",
            "-1",
            "--end",
            " 3 ",
            "--no-context",
            "--raw-transcript",
            "-l",
            "en",
            "--context",
            "ctx",
            "-s",
            "keep/",
            "-c",
            "c.txt",
            "--cookies-from-browser",
            "chrome",
            "--log-level",
            "debug",
            "--asr-base-url",
            "http://a",
            "--ocr-api-key",
            "k",
        ]);
        assert_eq!(cli.input_source, "in.pdf");
        assert_eq!(cli.task.as_deref(), Some("ocr"));
        assert_eq!(cli.output.as_deref(), Some("out.md"));
        assert_eq!(cli.model.as_deref(), Some("qwen3-asr-0.6b"));
        assert_eq!(cli.precision.as_deref(), Some("8bit"));
        assert_eq!((cli.start, cli.end), (Some(-1), Some(3)));
        assert!(cli.no_context && cli.raw_transcript);
        assert_eq!(cli.language.as_deref(), Some("en"));
        assert_eq!(cli.save_original.as_deref(), Some("keep/"));
        assert_eq!(cli.cookies.as_deref(), Some("c.txt"));
        assert_eq!(cli.log_level, "debug");
        assert_eq!(cli.asr_base_url.as_deref(), Some("http://a"));
        assert_eq!(cli.ocr_api_key.as_deref(), Some("k"));
        assert_eq!(run(&["--", "-dash.wav"]).input_source, "-dash.wav");
    }

    #[test]
    fn env_fallbacks_apply_only_when_flag_absent() {
        let env = |k: &str| match k {
            "AIMD_ASR_BASE_URL" => Some("http://env".to_string()),
            "AIMD_TEMP_DIR" => Some("/tmp/x".to_string()),
            "AIMD_OCR_MODEL" => Some(String::new()),
            _ => None,
        };
        let Parsed::Run(cli) = parse(&args(&["a.wav", "--asr-base-url", "http://flag"]), &env)
        else {
            panic!()
        };
        assert_eq!(cli.asr_base_url.as_deref(), Some("http://flag"));
        assert_eq!(cli.temp_dir.as_deref(), Some("/tmp/x"));
        assert_eq!(cli.ocr_model, None);
    }

    #[test]
    fn usage_errors_match_click() {
        // test_cli.py usage scenarios (messages verbatim from Click/Typer).
        let err = |v: &[&str]| match parse(&args(v), &no_env) {
            Parsed::Usage {
                message,
                show_usage,
            } => (message, show_usage),
            other => panic!("{other:?}"),
        };
        assert_eq!(err(&[]), ("Missing argument 'input_source'.".into(), true));
        assert_eq!(
            err(&["--start", "abc", "x"]),
            (
                "Invalid value for '--start': 'abc' is not a valid int.".into(),
                true
            )
        );
        assert_eq!(
            err(&["--model"]),
            ("Option '--model' requires an argument.".into(), false)
        );
        assert_eq!(
            err(&["--bogus", "x"]),
            ("No such option: --bogus".into(), true)
        );
        assert_eq!(
            err(&["a", "b"]),
            ("Got unexpected extra argument(s) (b)".into(), true)
        );
        assert_eq!(parse(&args(&["x", "--help"]), &no_env), Parsed::Help);
        assert!(help_text().contains("--cookies-from-browser TEXT"));
    }
}
