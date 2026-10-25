//! Hermetic URL pipeline tests (ports `tests/test_url.py` and the URL parts
//! of `tests/test_asr_context.py`).
//!
//! yt-dlp is replaced by a POSIX `sh` script that logs its argv and emulates
//! metadata dumps, subtitle downloads (`--load-info-json`) and audio
//! downloads from files the test writes into a scratch directory. ASR is a
//! recording mock. No network, no sleeps.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde_json::{Value, json};

use aimd_core::{AimdError, ErrorKind, ProcessHooks, ProcessInput, Result};
use aimd_url::{
    AudioTranscriber, CookieSource, ReadableBackend, TranscribeRequest, UrlConfig, YtDlp,
    build_cookie_sources, convert_url_with, download_audio, extract_subtitles, extract_video_info,
    parse_cookies_from_browser, try_download_with_format,
};

const FAKE_YT_DLP: &str = r#"#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"
printf '%s\037' "$@" >> "$DIR/calls"
printf '\n' >> "$DIR/calls"
prev=""; OUT=""; SUBLANGS=""; SUBFMT=""; FMT=""; CODEC=""; COOKIE="none"; MODE="audio"
for a in "$@"; do
  case "$prev" in
    -o) OUT="$a" ;;
    --sub-langs) SUBLANGS="$a" ;;
    --sub-format) SUBFMT="$a" ;;
    -f) FMT="$a" ;;
    --audio-format) CODEC="$a" ;;
    --cookies) COOKIE="file" ;;
    --cookies-from-browser) COOKIE="$a" ;;
  esac
  case "$a" in
    --list-impersonate-targets) MODE="impersonate" ;;
    --dump-single-json) MODE="dump" ;;
    --load-info-json) MODE="subs" ;;
  esac
  prev="$a"
done
if [ -f "$DIR/hook.sh" ]; then . "$DIR/hook.sh"; fi
case "$MODE" in
  impersonate)
    if [ -f "$DIR/impersonate" ]; then cat "$DIR/impersonate"; fi
    exit 0 ;;
  dump)
    if [ -f "$DIR/info.json" ]; then cat "$DIR/info.json"; exit 0; fi
    echo "ERROR: no info configured" >&2; exit 1 ;;
  subs)
    LANG_ID=$(printf '%s' "$SUBLANGS" | tr -d '\\')
    BASE=$(printf '%s' "$OUT" | sed 's/\.%(ext)s$//')
    SRC="$DIR/subs/$LANG_ID.$SUBFMT"
    if [ -f "$SRC" ]; then cp "$SRC" "$BASE.$LANG_ID.$SUBFMT"; exit 0; fi
    echo "ERROR: Unable to download video subtitles for '$LANG_ID': HTTP Error 429" >&2
    exit 1 ;;
  audio)
    BASE=$(printf '%s' "$OUT" | sed 's/\.%(ext)s$//')
    EXT="${CODEC:-webm}"
    if [ -f "$DIR/audio" ]; then cp "$DIR/audio" "$BASE.$EXT"; exit 0; fi
    echo "ERROR: Requested format is not available" >&2
    exit 1 ;;
esac
"#;

/// Scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("aimd-url-test-{label}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir.canonicalize().unwrap())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Fake {
    scratch: Scratch,
}

impl Fake {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        std::fs::write(scratch.0.join("yt-dlp"), FAKE_YT_DLP).unwrap();
        std::fs::create_dir_all(scratch.0.join("subs")).unwrap();
        Fake { scratch }
    }

    fn dir(&self) -> &Path {
        &self.scratch.0
    }

    fn ytdlp(&self) -> YtDlp {
        YtDlp {
            command: vec!["sh".into(), self.dir().join("yt-dlp").into_os_string()],
        }
    }

    fn config(&self) -> UrlConfig {
        UrlConfig {
            ytdlp: self.ytdlp(),
            readable_backend: ReadableBackend::DomSmoothie,
            curl: OsString::from("curl"),
        }
    }

    fn hook(&self, body: &str) -> &Self {
        std::fs::write(self.dir().join("hook.sh"), body).unwrap();
        self
    }

    fn info(&self, info: &Value) -> &Self {
        std::fs::write(self.dir().join("info.json"), info.to_string()).unwrap();
        self
    }

    fn sub(&self, lang: &str, ext: &str, body: &str) -> &Self {
        std::fs::write(self.dir().join("subs").join(format!("{lang}.{ext}")), body).unwrap();
        self
    }

    fn audio(&self) -> &Self {
        std::fs::write(self.dir().join("audio"), "fake audio bytes").unwrap();
        self
    }

    fn impersonate(&self, listing: &str) -> &Self {
        std::fs::write(self.dir().join("impersonate"), listing).unwrap();
        self
    }

    fn calls(&self) -> Vec<Vec<String>> {
        let Ok(text) = std::fs::read_to_string(self.dir().join("calls")) else {
            return Vec::new();
        };
        text.lines()
            .map(|line| {
                line.split('\x1f')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .collect()
    }

    fn calls_with(&self, flag: &str) -> Vec<Vec<String>> {
        self.calls()
            .into_iter()
            .filter(|c| c.iter().any(|a| a == flag))
            .collect()
    }

    fn dump_calls(&self) -> Vec<Vec<String>> {
        self.calls_with("--dump-single-json")
    }

    fn sub_calls(&self) -> Vec<Vec<String>> {
        self.calls_with("--load-info-json")
    }

    fn audio_calls(&self) -> Vec<Vec<String>> {
        self.calls_with("-f")
    }
}

fn after<'a>(call: &'a [String], flag: &str) -> Option<&'a str> {
    call.iter()
        .position(|a| a == flag)
        .and_then(|i| call.get(i + 1))
        .map(String::as_str)
}

fn has(call: &[String], flag: &str) -> bool {
    call.iter().any(|a| a == flag)
}

fn has_seq(call: &[String], seq: &[&str]) -> bool {
    call.windows(seq.len())
        .any(|w| w.iter().zip(seq).all(|(a, b)| a == b))
}

/// `none`, `file`, or the `--cookies-from-browser` value.
fn cookie_of(call: &[String]) -> String {
    if has(call, "--cookies") {
        return "file".into();
    }
    after(call, "--cookies-from-browser")
        .unwrap_or("none")
        .to_string()
}

fn sub_attempts(fake: &Fake) -> Vec<(String, String)> {
    fake.sub_calls()
        .iter()
        .map(|c| {
            (
                after(c, "--sub-langs").unwrap().replace('\\', ""),
                after(c, "--sub-format").unwrap().to_string(),
            )
        })
        .collect()
}

#[derive(Default)]
struct Hooks {
    cancelled: AtomicBool,
    stages: Mutex<Vec<String>>,
}

impl ProcessHooks for Hooks {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn progress(&self, stage: &str, _: Option<u64>, _: Option<u64>, _: Option<&str>) {
        self.stages.lock().unwrap().push(stage.to_string());
    }
}

impl Hooks {
    fn stages(&self) -> Vec<String> {
        self.stages.lock().unwrap().clone()
    }
}

struct MockAsr {
    result: Result<String>,
    seen: Mutex<Vec<(PathBuf, String, TranscribeRequest)>>,
}

impl MockAsr {
    fn ok(text: &str) -> Self {
        MockAsr {
            result: Ok(text.to_string()),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn err(err: AimdError) -> Self {
        MockAsr {
            result: Err(err),
            seen: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<TranscribeRequest> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|(_, _, r)| r.clone())
            .collect()
    }

    fn single(&self) -> TranscribeRequest {
        let reqs = self.requests();
        assert_eq!(reqs.len(), 1, "expected exactly one transcription");
        reqs.into_iter().next().unwrap()
    }
}

impl AudioTranscriber for MockAsr {
    fn transcribe(
        &self,
        audio_path: &Path,
        request: &TranscribeRequest,
        _hooks: &dyn ProcessHooks,
    ) -> Result<String> {
        let body = std::fs::read_to_string(audio_path).unwrap_or_default();
        self.seen
            .lock()
            .unwrap()
            .push((audio_path.to_path_buf(), body, request.clone()));
        self.result.clone()
    }
}

fn input(url: &str) -> ProcessInput {
    ProcessInput {
        input_source: url.to_string(),
        ..ProcessInput::default()
    }
}

fn input_in(url: &str, fake: &Fake) -> ProcessInput {
    let tmp = fake.dir().join("tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    ProcessInput {
        input_source: url.to_string(),
        temp_dir: Some(tmp),
        ..ProcessInput::default()
    }
}

const VTT: &str = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsubtitle text\n";

fn expect_err<T: std::fmt::Debug>(result: Result<T>, kind: ErrorKind, needle: &str) -> AimdError {
    let err = result.expect_err("expected an error");
    assert_eq!(err.kind, kind, "{err:?}");
    assert!(
        err.message.contains(needle),
        "{:?} lacks {needle:?}",
        err.message
    );
    err
}

// --- get_text_from_url ---------------------------------------------------

#[test]
fn extracts_info_once_and_reports_unknown_platform() {
    let fake = Fake::new("once");
    fake.info(&json!({
        "title": "Example",
        "webpage_url": "https://example.com/video",
        "subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/en.vtt"}]},
    }))
    .sub("en", "vtt", VTT);
    let hooks = Hooks::default();
    let asr = MockAsr::ok("unused");
    let out = convert_url_with(
        &input_in("https://example.com/video", &fake),
        &fake.config(),
        &asr,
        &hooks,
    )
    .unwrap();

    assert_eq!(out.title, "Example");
    assert_eq!(out.platform, "unknown");
    assert!(
        out.markdown.contains("**Platform:** unknown"),
        "{}",
        out.markdown
    );
    assert!(out.markdown.contains("subtitle text"), "{}", out.markdown);
    assert!(
        !out.markdown.contains("-->"),
        "subtitles are stripped by default"
    );
    assert_eq!(fake.dump_calls().len(), 1);
    assert!(asr.requests().is_empty());
    assert_eq!(hooks.stages(), ["extracting"]);
    // Unknown platforms get no YouTube-only options.
    assert!(!has(&fake.dump_calls()[0], "--js-runtimes"));
    // Scratch dirs are cleaned up.
    assert_eq!(
        std::fs::read_dir(fake.dir().join("tmp")).unwrap().count(),
        0
    );
}

#[test]
fn youtube_output_and_yt_dlp_options() {
    let fake = Fake::new("youtube");
    fake.impersonate("[info] Available impersonate targets\nChrome-131:Macos-14   curl_cffi\n")
        .info(&json!({
            "title": "Example",
            "channel": "Demo Channel",
            "webpage_url": "https://www.youtube.com/watch?v=test",
            "duration": 12,
            "upload_date": "20260307",
            "view_count": 34,
            "description": "desc",
            "subtitles": {"en": [{"ext": "vtt", "url": "https://example.com/en.vtt"}]},
        }))
        .sub("en", "vtt", VTT);
    let out = convert_url_with(
        &input_in("https://www.youtube.com/watch?v=test", &fake),
        &fake.config(),
        &MockAsr::ok("unused"),
        &Hooks::default(),
    )
    .unwrap();
    assert_eq!(out.platform, "youtube");
    assert!(out.markdown.contains("**Platform:** youtube"));
    assert!(out.markdown.contains("**Channel:** Demo Channel"));

    let dump = &fake.dump_calls()[0];
    for seq in [
        &["--ignore-config", "--no-warnings"][..],
        &[
            "--no-js-runtimes",
            "--js-runtimes",
            "deno",
            "--js-runtimes",
            "node",
        ],
        &["--impersonate", "chrome"],
        &["--skip-download"],
    ] {
        assert!(has_seq(dump, seq), "{seq:?} missing from {dump:?}");
    }
    assert_eq!(cookie_of(dump), "chrome:default");

    let sub = &fake.sub_calls()[0];
    for seq in [
        &["--socket-timeout", "40"][..],
        &["--retries", "2"],
        &["--retry-sleep", "http:linear=0.5:1.5:1"],
        &["--write-subs"],
        &["--sub-format", "vtt"],
        &["--ignore-no-formats-error"],
        &["--js-runtimes", "deno"],
    ] {
        assert!(has_seq(sub, seq), "{seq:?} missing from {sub:?}");
    }
    // The metadata cookie source is reused for subtitles.
    assert_eq!(cookie_of(sub), "chrome:default");
    // Impersonation probing is cached per yt-dlp command.
    assert_eq!(fake.calls_with("--list-impersonate-targets").len(), 1);
}

#[test]
fn impersonation_skipped_when_unavailable() {
    let fake = Fake::new("noimp");
    fake.impersonate("Chrome-131:Macos-14   curl_cffi (unavailable)\n");
    let args = fake.ytdlp().base_args("youtube");
    assert!(!args.iter().any(|a| a == "--impersonate"));
    assert!(args.iter().any(|a| a == "--no-js-runtimes"));
}

#[test]
fn cookie_probe_order_falls_back_to_no_cookie_and_is_reused() {
    let fake = Fake::new("probe");
    fake.hook(
        r#"if [ "$MODE" = dump ] && [ "$COOKIE" != none ]; then
  echo "ERROR: could not find $COOKIE cookies database in \"/nope\"" >&2; exit 1
fi
"#,
    )
    .info(&json!({
        "title": "Example",
        "automatic_captions": {"en-orig": [{"ext": "vtt", "url": "https://example.com/a"}]},
    }))
    .sub("en-orig", "vtt", VTT);
    let out = convert_url_with(
        &input_in("https://www.youtube.com/watch?v=x", &fake),
        &fake.config(),
        &MockAsr::ok("unused"),
        &Hooks::default(),
    )
    .unwrap();
    assert!(out.markdown.contains("subtitle text"));
    let order: Vec<String> = fake.dump_calls().iter().map(|c| cookie_of(c)).collect();
    assert_eq!(
        order,
        [
            "chrome:default",
            "chromium",
            "brave",
            "edge",
            "vivaldi",
            "opera",
            "firefox",
            "safari",
            "whale",
            "none"
        ]
    );
    let sub = &fake.sub_calls()[0];
    assert_eq!(cookie_of(sub), "none");
    assert!(has(sub, "--write-auto-subs"));
}

#[test]
fn subtitles_use_the_cookie_source_that_returned_metadata() {
    let fake = Fake::new("reuse");
    fake.hook(
        r#"if [ "$MODE" = dump ] && [ "$COOKIE" != firefox ]; then
  echo "ERROR: [youtube] x: Some transient failure" >&2; exit 1
fi
"#,
    )
    .info(&json!({"title": "Example", "subtitles": {"en": [{"ext": "vtt", "url": "u"}]}}))
    .sub("en", "vtt", VTT);
    convert_url_with(
        &input_in("https://www.youtube.com/watch?v=x", &fake),
        &fake.config(),
        &MockAsr::ok("unused"),
        &Hooks::default(),
    )
    .unwrap();
    assert_eq!(cookie_of(&fake.sub_calls()[0]), "firefox");
}

#[test]
fn build_cookie_sources_order_and_fallback() {
    let fake = Fake::new("cookies");
    let file = fake.dir().join("cookies.txt");
    std::fs::write(
        &file,
        "# Netscape HTTP Cookie File\n.bilibili.com\tTRUE\t/\tFALSE\t0\tSESSDATA\tx\n",
    )
    .unwrap();
    let names: Vec<String> = build_cookie_sources("bilibili", Some(&file), Some("chrome:default"))
        .unwrap()
        .iter()
        .map(CookieSource::name)
        .collect();
    assert_eq!(names, ["cookiefile", "cookiesfrombrowser:chrome:default"]);

    let names: Vec<String> = build_cookie_sources("youtube", None, None)
        .unwrap()
        .iter()
        .map(CookieSource::name)
        .collect();
    assert_eq!(
        names,
        [
            "cookiesfrombrowser:chrome:default",
            "cookiesfrombrowser:chromium",
            "cookiesfrombrowser:brave",
            "cookiesfrombrowser:edge",
            "cookiesfrombrowser:vivaldi",
            "cookiesfrombrowser:opera",
            "cookiesfrombrowser:firefox",
            "cookiesfrombrowser:safari",
            "cookiesfrombrowser:whale",
            "no-cookie",
        ]
    );
    let bili = build_cookie_sources("bilibili", None, None).unwrap();
    assert!(!bili.contains(&CookieSource::NoCookie));
}

#[test]
fn explicit_invalid_cookie_options_fail_fast() {
    let fake = Fake::new("badcookie");
    let url = "https://www.youtube.com/watch?v=x";
    let asr = MockAsr::ok("unused");

    let mut inp = input_in(url, &fake);
    inp.cookies_from_browser = Some("   ".into());
    expect_err(
        convert_url_with(&inp, &fake.config(), &asr, &Hooks::default()),
        ErrorKind::UnsupportedInput,
        "cookies_from_browser cannot be empty",
    );

    let mut inp = input_in(url, &fake);
    inp.cookies = Some(fake.dir().join("missing.txt"));
    expect_err(
        convert_url_with(&inp, &fake.config(), &asr, &Hooks::default()),
        ErrorKind::UnsupportedInput,
        "missing.txt",
    );

    let not_netscape = fake.dir().join("cookies.json");
    std::fs::write(&not_netscape, "{\"cookies\": []}").unwrap();
    let mut inp = input_in(url, &fake);
    inp.cookies = Some(not_netscape);
    expect_err(
        convert_url_with(&inp, &fake.config(), &asr, &Hooks::default()),
        ErrorKind::UnsupportedInput,
        "Netscape",
    );
    assert!(
        fake.dump_calls().is_empty(),
        "validated before running yt-dlp"
    );

    // A syntactically valid browser yt-dlp rejects: stop at the first error.
    fake.hook(
        r#"if [ "$MODE" = dump ]; then
  echo "yt-dlp: error: unsupported browser specified for cookies: \"$COOKIE\". Supported browsers are: brave, chrome" >&2; exit 2
fi
"#,
    );
    let mut inp = input_in(url, &fake);
    inp.cookies_from_browser = Some("netscape".into());
    expect_err(
        convert_url_with(&inp, &fake.config(), &asr, &Hooks::default()),
        ErrorKind::UnsupportedInput,
        "unsupported browser specified for cookies",
    );
    assert_eq!(fake.dump_calls().len(), 1, "no fallback to other sources");
}

#[test]
fn parse_cookies_from_browser_rejects_bad_specs() {
    assert!(parse_cookies_from_browser("chrome").is_ok());
    assert_eq!(
        parse_cookies_from_browser("   ").unwrap_err(),
        "cookies_from_browser cannot be empty"
    );
}

const BILI_412_HOOK: &str = r#"case "$COOKIE" in
  chrome:default) echo "ERROR: [BiliBili] BV1: HTTP Error 412: Precondition Failed" >&2; exit 1 ;;
  none) ;;
  *) echo "ERROR: could not find $COOKIE cookies database" >&2; exit 1 ;;
esac
"#;

#[test]
fn bilibili_412_metadata_surfaces_cookie_hint() {
    let fake = Fake::new("bili412");
    fake.hook(BILI_412_HOOK);
    expect_err(
        extract_video_info(
            &fake.ytdlp(),
            "https://www.bilibili.com/video/BV1iVoVBgERD/",
            "bilibili",
            None,
            None,
        ),
        ErrorKind::ProcessingFailed,
        "Authenticated cookies are required for this URL",
    );
    let order: Vec<String> = fake.dump_calls().iter().map(|c| cookie_of(c)).collect();
    assert!(
        !order.contains(&"none".to_string()),
        "bilibili never tries no-cookie"
    );
    assert_eq!(order.len(), 9);
}

#[test]
fn bilibili_412_download_surfaces_cookie_hint() {
    let fake = Fake::new("bili412dl");
    fake.hook(BILI_412_HOOK).audio();
    let dl = fake.dir().join("dl");
    std::fs::create_dir_all(&dl).unwrap();
    expect_err(
        try_download_with_format(
            &fake.ytdlp(),
            "https://www.bilibili.com/video/BV1iVoVBgERD/",
            &dl,
            "sample_audio",
            "bestaudio",
            None,
            "bilibili",
            None,
            None,
        ),
        ErrorKind::ProcessingFailed,
        "Authenticated cookies are required for this download",
    );
    // ... and the full pipeline preserves that domain error.
    fake.info(&json!({"title": "Example", "id": "BV1"}));
    fake.hook(&format!(
        "if [ \"$MODE\" != dump ]; then\n{BILI_412_HOOK}fi\n"
    ));
    expect_err(
        convert_url_with(
            &input_in("https://www.bilibili.com/video/BV1", &fake),
            &fake.config(),
            &MockAsr::ok("unused"),
            &Hooks::default(),
        ),
        ErrorKind::ProcessingFailed,
        "Authenticated cookies are required for this download",
    );
}

#[test]
fn bilibili_subtitles_use_inline_data() {
    let fake = Fake::new("bilisub");
    fake.info(&json!({
        "title": "B站视频",
        "subtitles": {"zh-CN": [{"ext": "srt", "data": "1\n00:00:00,500 --> 00:00:03,000\n大家好，欢迎收看本期视频。\n"}]},
    }));
    let out = convert_url_with(
        &input_in("https://www.bilibili.com/video/BV1", &fake),
        &fake.config(),
        &MockAsr::ok("unused"),
        &Hooks::default(),
    )
    .unwrap();
    assert_eq!(out.platform, "bilibili");
    assert!(
        out.markdown.contains("大家好，欢迎收看本期视频。"),
        "{}",
        out.markdown
    );
    assert!(fake.sub_calls().is_empty());
    assert!(!has(&fake.dump_calls()[0], "--js-runtimes"));
}

#[test]
fn unsupported_and_invalid_urls() {
    let fake = Fake::new("unsupported");
    fake.hook("echo \"ERROR: Unsupported URL: https://example.com/\" >&2; exit 1\n");
    expect_err(
        convert_url_with(
            &input("https://example.com/"),
            &fake.config(),
            &MockAsr::ok(""),
            &Hooks::default(),
        ),
        ErrorKind::UnsupportedInput,
        "Unsupported URL",
    );
    assert_eq!(
        fake.dump_calls().len(),
        1,
        "unsupported URL stops the probe"
    );
    expect_err(
        convert_url_with(
            &input("ftp://example.com/x"),
            &fake.config(),
            &MockAsr::ok(""),
            &Hooks::default(),
        ),
        ErrorKind::UnsupportedInput,
        "Invalid URL",
    );
}

#[test]
fn missing_yt_dlp_is_backend_unavailable_with_install_hint() {
    let config = UrlConfig {
        ytdlp: YtDlp {
            command: vec![OsString::from("/nonexistent/aimd-test/yt-dlp")],
        },
        readable_backend: ReadableBackend::DomSmoothie,
        curl: "curl".into(),
    };
    expect_err(
        convert_url_with(
            &input("https://example.com/v"),
            &config,
            &MockAsr::ok(""),
            &Hooks::default(),
        ),
        ErrorKind::BackendUnavailable,
        "brew install yt-dlp",
    );
}

#[test]
fn stale_extractor_errors_get_upgrade_hint() {
    let fake = Fake::new("stale");
    fake.hook(
        "echo \"ERROR: [youtube] x: Failed to extract any player response; please report this issue on https://github.com/yt-dlp/yt-dlp/issues\" >&2; exit 1\n",
    );
    let err = expect_err(
        convert_url_with(
            &input("https://www.youtube.com/watch?v=x"),
            &fake.config(),
            &MockAsr::ok(""),
            &Hooks::default(),
        ),
        ErrorKind::ProcessingFailed,
        "brew upgrade yt-dlp",
    );
    assert!(
        err.message
            .starts_with("Failed to extract video information: ERROR: [youtube]")
    );

    let fake = Fake::new("notstale");
    fake.hook("echo \"ERROR: [generic] x: HTTP Error 404: Not Found\" >&2; exit 1\n");
    let err = convert_url_with(
        &input("https://example.com/v"),
        &fake.config(),
        &MockAsr::ok(""),
        &Hooks::default(),
    )
    .unwrap_err();
    assert!(!err.message.contains("brew upgrade"), "{}", err.message);
}

// --- subtitle selection (extract_subtitles) ------------------------------

fn subs(fake: &Fake, info: &Value, language: Option<&str>) -> Option<String> {
    subs_with(fake, info, language, &CookieSource::NoCookie)
}

fn subs_with(
    fake: &Fake,
    info: &Value,
    language: Option<&str>,
    source: &CookieSource,
) -> Option<String> {
    let work = fake.dir().join(format!("work-{}", fake.sub_calls().len()));
    std::fs::create_dir_all(&work).unwrap();
    extract_subtitles(
        &fake.ytdlp(),
        info,
        &info.to_string(),
        "youtube",
        language,
        Some(source),
        &work,
    )
    .unwrap()
}

fn yt_track(lang: &str, ext: &str, tlang: Option<&str>) -> Value {
    let mut url =
        format!("https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk&kind=asr&lang={lang}");
    if let Some(t) = tlang {
        url.push_str(&format!("&tlang={t}"));
    }
    url.push_str(&format!("&fmt={ext}"));
    json!({"ext": ext, "url": url})
}

#[test]
fn default_prefers_orig_over_inferred_chinese_title() {
    let fake = Fake::new("origzh");
    fake.sub("en-orig", "srt", "original english caption").sub(
        "zh-Hans",
        "srt",
        "chinese translation",
    );
    let info = json!({
        "title": "【深度访谈】OpenAI CEO 谈 AGI 的未来",
        "description": "本期节目讨论人工智能。",
        "language": "en",
        "subtitles": {},
        "automatic_captions": {
            "zh-Hans": [yt_track("en", "srt", Some("zh-Hans"))],
            "en-orig": [yt_track("en", "srt", None)],
        },
    });
    assert_eq!(
        subs(&fake, &info, None).as_deref(),
        Some("original english caption")
    );
    assert_eq!(sub_attempts(&fake), [("en-orig".into(), "srt".into())]);
}

#[test]
fn explicit_zh_prefers_chinese_over_en_orig() {
    let fake = Fake::new("zh");
    fake.sub("en-orig", "srt", "original english caption").sub(
        "zh-Hans",
        "srt",
        "chinese translation",
    );
    let info = json!({
        "title": "Why AI Moats Still Matter (And How They've Changed)",
        "language": "en",
        "automatic_captions": {
            "zh-Hans": [yt_track("en", "srt", Some("zh-Hans"))],
            "en-orig": [yt_track("en", "srt", None)],
        },
    });
    assert_eq!(
        subs(&fake, &info, Some("zh")).as_deref(),
        Some("chinese translation")
    );
    assert_eq!(sub_attempts(&fake)[0].0, "zh-Hans");
}

#[test]
fn prefers_english_when_title_is_english() {
    let fake = Fake::new("entitle");
    fake.sub("en", "srt", "english subtitle")
        .sub("zh-Hans", "srt", "zh");
    let info = json!({
        "title": "Deep Dive with OpenAI CEO on the Future of AGI",
        "description": "A conversation about artificial intelligence.",
        "subtitles": {
            "zh-Hans": [{"ext": "srt", "url": "https://example.com/zh"}],
            "en": [{"ext": "srt", "url": "https://example.com/en"}],
        },
    });
    assert_eq!(
        subs(&fake, &info, None).as_deref(),
        Some("english subtitle")
    );
    let call = &fake.sub_calls()[0];
    assert!(has(call, "--write-subs") && !has(call, "--write-auto-subs"));
    assert_eq!(sub_attempts(&fake), [("en".into(), "srt".into())]);
}

#[test]
fn skips_chinese_translations_for_english_youtube_video() {
    let fake = Fake::new("skipzh");
    for lang in ["zh-Hans", "zh-Hant", "en"] {
        fake.sub(lang, "srt", "wrong track");
    }
    fake.sub("en-orig", "srt", "english source caption");
    let info = json!({
        "id": "fgzr3PhzIMk",
        "title": "Why AI Moats Still Matter (And How They've Changed)",
        "description": "a16z General Partners David Haber, Alex Rampell, and Erik Torenberg discuss why 19 out of 20 AI startups building the same thing will die.",
        "language": "en",
        "subtitles": {},
        "automatic_captions": {
            "zh-Hans": [yt_track("en", "srt", Some("zh-Hans"))],
            "zh-Hant": [yt_track("en", "srt", Some("zh-Hant"))],
            "en-orig": [yt_track("en", "srt", None)],
            "en": [json!({"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk&kind=asr&lang=en&fmt=srt&src=en"})],
        },
    });
    assert_eq!(
        subs(&fake, &info, None).as_deref(),
        Some("english source caption")
    );
    assert_eq!(sub_attempts(&fake), [("en-orig".into(), "srt".into())]);
}

#[test]
fn retries_next_language_when_download_fails() {
    let fake = Fake::new("nextlang");
    fake.sub("en", "srt", "english fallback caption"); // en-orig fails
    let info = json!({
        "title": "Why AI Moats Still Matter (And How They've Changed)",
        "language": "en",
        "automatic_captions": {
            "zh-Hans": [yt_track("en", "srt", Some("zh-Hans"))],
            "en-orig": [yt_track("en", "srt", None)],
            "en": [json!({"ext": "srt", "url": "https://www.youtube.com/api/timedtext?v=fgzr3PhzIMk&kind=asr&lang=en&fmt=srt&src=en"})],
        },
    });
    assert_eq!(
        subs(&fake, &info, None).as_deref(),
        Some("english fallback caption")
    );
    assert_eq!(
        sub_attempts(&fake),
        [
            ("en-orig".into(), "srt".into()),
            ("en".into(), "srt".into())
        ]
    );
    // Transient-failure retries are delegated to yt-dlp itself.
    assert!(
        fake.sub_calls()
            .iter()
            .all(|c| has_seq(c, &["--retries", "2"]))
    );
}

#[test]
fn prefers_json3_for_youtube_auto_captions() {
    let fake = Fake::new("json3");
    fake.sub(
        "en-orig",
        "json3",
        r#"{"wireMagic": "pb3", "events": [{"tStartMs": 80, "dDurationMs": 3200, "segs": [{"utf8": "you've discovered"}]}]}"#,
    );
    for ext in ["srt", "vtt", "ttml"] {
        fake.sub("en-orig", ext, "should not be used");
    }
    let info = json!({
        "id": "Ppyt3MptX5k",
        "title": "The Scientist Who Scans Fathers' Brains: Parenthood Shrinks Your Brain (And That's GOOD!)",
        "language": "en",
        "subtitles": {},
        "automatic_captions": {
            "zh-Hans": [yt_track("en", "srt", Some("zh-Hans"))],
            "en-orig": [
                yt_track("en", "json3", None),
                yt_track("en", "srt", None),
                yt_track("en", "vtt", None),
                yt_track("en", "ttml", None),
            ],
        },
    });
    let content = subs(&fake, &info, None).unwrap();
    assert!(
        content.contains("00:00:00,080 --> 00:00:03,280"),
        "{content}"
    );
    assert!(content.contains("you've discovered"));
    assert_eq!(sub_attempts(&fake), [("en-orig".into(), "json3".into())]);
}

#[test]
fn retries_next_format_when_ttml_download_fails() {
    let fake = Fake::new("nextfmt");
    fake.sub("en-orig", "vtt", "vtt caption");
    let info = json!({
        "title": "English talk",
        "language": "en",
        "automatic_captions": {"en-orig": [
            {"ext": "ttml", "url": "https://example.com/en?fmt=ttml"},
            {"ext": "vtt", "url": "https://example.com/en?fmt=vtt"},
            {"ext": "srt", "url": "https://example.com/en?fmt=srt"},
        ]},
    });
    assert_eq!(subs(&fake, &info, None).as_deref(), Some("vtt caption"));
    assert_eq!(
        sub_attempts(&fake),
        [
            ("en-orig".into(), "ttml".into()),
            ("en-orig".into(), "vtt".into())
        ]
    );
}

#[test]
fn skips_empty_json3_and_uses_ttml() {
    let fake = Fake::new("emptyjson3");
    fake.sub("en-orig", "json3", r#"{"wireMagic": "pb3", "events": []}"#)
        .sub("en-orig", "ttml", "ttml caption");
    let info = json!({
        "title": "English talk",
        "language": "en",
        "automatic_captions": {"en-orig": [
            {"ext": "json3", "url": "https://example.com/en?fmt=json3"},
            {"ext": "ttml", "url": "https://example.com/en?fmt=ttml"},
        ]},
    });
    assert_eq!(subs(&fake, &info, None).as_deref(), Some("ttml caption"));
    assert_eq!(
        sub_attempts(&fake),
        [
            ("en-orig".into(), "json3".into()),
            ("en-orig".into(), "ttml".into())
        ]
    );
}

#[test]
fn extract_subtitles_uses_given_cookie_source() {
    let fake = Fake::new("subcookie");
    fake.sub("en-orig", "srt", "subtitle text");
    let info = json!({"automatic_captions": {"en-orig": [{"ext": "srt", "url": "https://example.com/subs"}]}});
    let source = CookieSource::Browser {
        raw: "chrome:default".into(),
        spec: parse_cookies_from_browser("chrome:default").unwrap(),
        explicit: false,
    };
    assert_eq!(
        subs_with(&fake, &info, None, &source).as_deref(),
        Some("subtitle text")
    );
    assert_eq!(cookie_of(&fake.sub_calls()[0]), "chrome:default");
}

// --- audio download ------------------------------------------------------

#[test]
fn download_audio_prefers_audio_only_streams() {
    for (url, first) in [
        (
            "https://www.youtube.com/watch?v=test",
            "bestaudio[acodec^=opus][abr<=128]/bestaudio[abr<=128]/bestaudio",
        ),
        ("https://www.bilibili.com/video/BV1", "bestaudio"),
    ] {
        let fake = Fake::new("strategy");
        fake.audio();
        let dl = fake.dir().join("dl");
        std::fs::create_dir_all(&dl).unwrap();
        let path = download_audio(
            &fake.ytdlp(),
            &json!({"title": "Example", "id": "id1"}),
            url,
            &dl,
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(path, dl.join("Example.webm"));
        let calls = fake.audio_calls();
        assert_eq!(calls.len(), 1, "{url}: first strategy succeeds");
        assert_eq!(after(&calls[0], "-f"), Some(first));
        assert!(!has(&calls[0], "-x"));
        assert_eq!(calls[0].last().map(String::as_str), Some(url));
    }
}

#[test]
fn download_audio_walks_strategies_on_failure() {
    let fake = Fake::new("walk");
    fake.hook(
        "if [ \"$MODE\" = audio ] && [ -z \"$CODEC\" ]; then echo 'ERROR: nope' >&2; exit 1; fi\n",
    )
    .audio();
    let dl = fake.dir().join("dl");
    std::fs::create_dir_all(&dl).unwrap();
    let path = download_audio(
        &fake.ytdlp(),
        &json!({"id": "id1"}),
        "https://example.com/v",
        &dl,
        None,
        None,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(path, dl.join("audio_id1.m4a"));
    let formats: Vec<String> = fake
        .audio_calls()
        .iter()
        .map(|c| after(c, "-f").unwrap().to_string())
        .collect();
    // One attempt per cookie source for the failing strategy, then m4a.
    assert_eq!(formats.len(), 11);
    assert_eq!(formats.last().unwrap(), "bestaudio[ext=m4a]/bestaudio");
}

#[test]
fn postprocessor_only_when_codec_requested() {
    let fake = Fake::new("codec");
    fake.audio();
    let dl = fake.dir().join("dl");
    std::fs::create_dir_all(&dl).unwrap();
    let ytdlp = fake.ytdlp();
    let plain = try_download_with_format(
        &ytdlp,
        "https://example.com/video",
        &dl,
        "sample_audio",
        "bestaudio",
        None,
        "unknown",
        None,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(plain, dl.join("sample_audio.webm"));
    let call = fake.audio_calls().pop().unwrap();
    assert!(!has(&call, "-x") && !has(&call, "--audio-format"));
    assert!(has_seq(&call, &["--ignore-config", "--no-warnings"]));
    assert_eq!(after(&call, "-o"), dl.join("sample_audio.%(ext)s").to_str());

    let m4a = try_download_with_format(
        &ytdlp,
        "https://example.com/video",
        &dl,
        "sample_audio_2",
        "bestaudio[ext=m4a]/bestaudio",
        Some("m4a"),
        "unknown",
        None,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(m4a, dl.join("sample_audio_2.m4a"));
    let call = fake.audio_calls().pop().unwrap();
    assert!(has_seq(
        &call,
        &["-x", "--audio-format", "m4a", "--audio-quality", "192"]
    ));
}

#[test]
fn download_applies_explicit_cookies() {
    let fake = Fake::new("dlcookie");
    fake.audio();
    let dl = fake.dir().join("dl");
    std::fs::create_dir_all(&dl).unwrap();
    let file = fake.dir().join("cookies.txt");
    std::fs::write(
        &file,
        "# HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tv\n",
    )
    .unwrap();
    try_download_with_format(
        &fake.ytdlp(),
        "https://www.youtube.com/watch?v=test",
        &dl,
        "a",
        "bestaudio",
        None,
        "youtube",
        Some(&file),
        None,
    )
    .unwrap()
    .unwrap();
    let call = fake.audio_calls().pop().unwrap();
    assert_eq!(after(&call, "--cookies"), file.to_str());

    try_download_with_format(
        &fake.ytdlp(),
        "https://www.youtube.com/watch?v=test",
        &dl,
        "b",
        "bestaudio",
        None,
        "youtube",
        None,
        Some("chrome:default"),
    )
    .unwrap()
    .unwrap();
    let call = fake.audio_calls().pop().unwrap();
    assert_eq!(
        after(&call, "--cookies-from-browser"),
        Some("chrome:default")
    );
}

// --- audio fallback through the pipeline ---------------------------------

fn linus_fake(label: &str) -> Fake {
    let fake = Fake::new(label);
    fake.info(&json!({
        "id": "id1",
        "title": "Interview with Linus Torvalds",
        "description": "A conversation about the Linux kernel and Git.",
        "tags": ["linux", "git"],
        "webpage_url": "https://example.com/video",
    }))
    .audio();
    fake
}

fn run_audio(
    fake: &Fake,
    inp: &ProcessInput,
    asr: &MockAsr,
    hooks: &Hooks,
) -> Result<aimd_url::UrlOutput> {
    convert_url_with(inp, &fake.config(), asr, hooks)
}

#[test]
fn audio_fallback_injects_metadata_context_by_default() {
    let fake = linus_fake("ctxdefault");
    let asr = MockAsr::ok("transcribed audio text");
    let hooks = Hooks::default();
    let out = run_audio(
        &fake,
        &input_in("https://example.com/video", &fake),
        &asr,
        &hooks,
    )
    .unwrap();
    assert!(out.markdown.contains("transcribed audio text"));
    let context = asr.single().context.unwrap();
    assert!(
        context.contains("Linus Torvalds") && context.contains("Linux kernel"),
        "{context}"
    );
    assert_eq!(
        hooks.stages(),
        ["extracting", "downloading", "transcribing"]
    );
    // The transcriber saw the downloaded file; the scratch dir is gone after.
    let (path, body, _) = asr.seen.lock().unwrap()[0].clone();
    assert_eq!(body, "fake audio bytes");
    assert_eq!(
        path.file_name().unwrap(),
        "Interview with Linus Torvalds.webm"
    );
    assert!(!path.exists());
    assert_eq!(
        std::fs::read_dir(fake.dir().join("tmp")).unwrap().count(),
        0
    );
}

#[test]
fn audio_fallback_metadata_context_variants() {
    let cases: [(Option<&str>, bool, Option<&str>); 3] = [
        (None, false, None),
        (
            Some("Vocabulary: Linus, Git."),
            true,
            Some("Vocabulary: Linus, Git.\n\nThe following"),
        ),
        (
            Some("Vocabulary: Linus, Git."),
            false,
            Some("Vocabulary: Linus, Git."),
        ),
    ];
    for (context, metadata, want) in cases {
        let fake = linus_fake("ctx");
        let asr = MockAsr::ok("transcribed audio text");
        let mut inp = input_in("https://example.com/video", &fake);
        inp.context = context.map(str::to_string);
        inp.metadata_context = metadata;
        run_audio(&fake, &inp, &asr, &Hooks::default()).unwrap();
        let got = asr.single().context;
        match want {
            None => assert_eq!(got, None),
            Some(w) if metadata => {
                let got = got.unwrap();
                assert!(got.starts_with(w), "{got}");
                assert!(got.contains("Linus Torvalds"));
            }
            Some(w) => assert_eq!(got.as_deref(), Some(w)),
        }
    }
}

#[test]
fn audio_fallback_forwards_inferred_language_model_precision() {
    let fake = Fake::new("lang");
    fake.info(&json!({
        "title": "中文播客：聊聊创业与产品",
        "description": "今天和嘉宾讨论创业经验。",
        "webpage_url": "https://example.com/episode",
    }))
    .audio();
    let asr = MockAsr::ok("transcribed chinese audio");
    let mut inp = input_in("https://example.com/episode", &fake);
    inp.model = Some("qwen3-asr-1.7b".into());
    inp.precision = Some("8bit".into());
    inp.asr_base_url = Some("http://127.0.0.1:9/v1".into());
    inp.asr_model = Some("remote-asr".into());
    let out = run_audio(&fake, &inp, &asr, &Hooks::default()).unwrap();
    assert!(out.markdown.contains("transcribed chinese audio"));
    let req = asr.single();
    assert_eq!(req.language.as_deref(), Some("zh"));
    assert_eq!(req.model.as_deref(), Some("qwen3-asr-1.7b"));
    assert_eq!(req.precision.as_deref(), Some("8bit"));
    assert_eq!(req.temp_dir, inp.temp_dir);
    assert_eq!(req.asr_base_url.as_deref(), Some("http://127.0.0.1:9/v1"));
    assert_eq!(req.asr_model.as_deref(), Some("remote-asr"));

    // An explicit language wins over inference.
    let fake = linus_fake("explicitlang");
    let asr = MockAsr::ok("transcribed audio text");
    let mut inp = input_in("https://example.com/video", &fake);
    inp.language = Some("de".into());
    run_audio(&fake, &inp, &asr, &Hooks::default()).unwrap();
    assert_eq!(asr.single().language.as_deref(), Some("de"));
}

#[test]
fn transcriber_backend_unavailable_propagates() {
    let fake = linus_fake("asrdown");
    let asr = MockAsr::err(AimdError::backend_unavailable("no usable ASR backend"));
    expect_err(
        run_audio(
            &fake,
            &input_in("https://www.youtube.com/watch?v=test", &fake),
            &asr,
            &Hooks::default(),
        ),
        ErrorKind::BackendUnavailable,
        "no usable ASR backend",
    );
}

#[test]
fn no_subtitles_and_no_audio_text_is_processing_failure() {
    // Download fails everywhere.
    let fake = Fake::new("nothing");
    fake.info(&json!({"title": "Example"}));
    let asr = MockAsr::ok("unused");
    expect_err(
        run_audio(
            &fake,
            &input_in("https://example.com/v", &fake),
            &asr,
            &Hooks::default(),
        ),
        ErrorKind::ProcessingFailed,
        "Failed to extract subtitles or transcribe audio for URL",
    );
    assert!(asr.requests().is_empty());

    // Transcript of <= 10 characters counts as no text.
    let fake = linus_fake("short");
    let asr = MockAsr::ok("too short");
    expect_err(
        run_audio(
            &fake,
            &input_in("https://example.com/v", &fake),
            &asr,
            &Hooks::default(),
        ),
        ErrorKind::ProcessingFailed,
        "Failed to extract subtitles or transcribe audio for URL",
    );

    // Whitespace-only subtitles fall through to audio.
    let fake = linus_fake("blanksub");
    let mut info: Value =
        serde_json::from_str(&std::fs::read_to_string(fake.dir().join("info.json")).unwrap())
            .unwrap();
    info["subtitles"] = json!({"en": [{"ext": "vtt", "url": "u"}]});
    fake.info(&info).sub("en", "vtt", "  \n ");
    let asr = MockAsr::ok("transcribed audio text");
    let out = run_audio(
        &fake,
        &input_in("https://example.com/v", &fake),
        &asr,
        &Hooks::default(),
    )
    .unwrap();
    assert!(out.markdown.contains("transcribed audio text"));
}

#[test]
fn raw_transcript_keeps_subtitle_timestamps() {
    let fake = Fake::new("raw");
    fake.info(&json!({"title": "Example", "subtitles": {"en": [{"ext": "vtt", "url": "u"}]}}))
        .sub("en", "vtt", VTT);
    let mut inp = input_in("https://example.com/v", &fake);
    inp.raw_transcript = true;
    let out = convert_url_with(&inp, &fake.config(), &MockAsr::ok(""), &Hooks::default()).unwrap();
    assert!(
        out.markdown.contains("00:00:01.000 --> 00:00:02.000"),
        "{}",
        out.markdown
    );
    assert!(out.markdown.contains("WEBVTT"));
}

#[test]
fn cancellation_checked_after_subtitle_lookup() {
    let fake = linus_fake("cancel");
    let hooks = Hooks::default();
    hooks.cancelled.store(true, Ordering::SeqCst);
    let asr = MockAsr::ok("transcribed audio text");
    expect_err(
        run_audio(
            &fake,
            &input_in("https://example.com/v", &fake),
            &asr,
            &hooks,
        ),
        ErrorKind::Cancelled,
        "URL processing cancelled after subtitle lookup",
    );
    assert!(fake.audio_calls().is_empty());
    assert!(asr.requests().is_empty());
}

#[test]
fn save_original_keeps_downloaded_audio() {
    let fake = linus_fake("save");
    let asr = MockAsr::ok("transcribed audio text");
    let target = fake.dir().join("keep").join("episode.m4a");
    let mut inp = input_in("https://example.com/v", &fake);
    inp.save_original = Some(target.clone());
    run_audio(&fake, &inp, &asr, &Hooks::default()).unwrap();
    // Stem from the requested path, extension from the download.
    assert!(fake.dir().join("keep").join("episode.webm").is_file());

    let fake = linus_fake("savedir");
    let asr = MockAsr::ok("transcribed audio text");
    let dir = fake.dir().join("originals");
    let mut inp = input_in("https://example.com/v", &fake);
    inp.save_original = Some(dir.clone());
    run_audio(&fake, &inp, &asr, &Hooks::default()).unwrap();
    assert!(dir.join("Interview with Linus Torvalds.webm").is_file());
}
