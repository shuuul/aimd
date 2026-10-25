//! Tier-3 E2E (`AIMD_E2E=1`, Metal GPU, HF cache): one run per model through
//! `process_input` with the native runtime (pipelines load once per
//! process via the runtime's cache).

use std::path::{Path, PathBuf};

use aimd_core::{NoHooks, ProcessInput, TaskType};

fn e2e_enabled() -> bool {
    if std::env::var("AIMD_E2E").as_deref() != Ok("1") {
        eprintln!("skipping: tier-3 E2E (set AIMD_E2E=1)");
        return false;
    }
    if aimd_mlx::device_is_cpu().unwrap_or(true) {
        eprintln!("skipping: E2E needs the MLX GPU device");
        return false;
    }
    true
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aimd-process-e2e-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn e2e_unlimited_ocr_png() {
    if !e2e_enabled() {
        return;
    }
    let png = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../aimd-engine/tests/golden/ocr_e2e/page_text.png");
    let result = aimd_process::process_input(&ProcessInput::new(png.to_string_lossy()), &NoHooks)
        .expect("OCR succeeds");
    assert_eq!(result.task_type, TaskType::Ocr);
    assert!(
        result.markdown.contains("Quarterly Report"),
        "{}",
        result.markdown
    );
    assert_eq!(result.text_context.title, "page_text");
}

#[test]
fn e2e_qwen3_asr_wav() {
    if !e2e_enabled() {
        return;
    }
    let dir = scratch("asr");
    let wav = dir.join("clip.wav");
    let status = std::process::Command::new("say")
        .args(["-v", "Samantha", "--data-format=LEI16@16000", "-o"])
        .arg(&wav)
        .arg("The quick brown fox jumps over the lazy dog.")
        .status();
    if !status.is_ok_and(|s| s.success()) {
        eprintln!("skipping: `say` unavailable");
        return;
    }
    let mut request = ProcessInput::new(wav.to_string_lossy());
    request.language = Some("en".into());
    let result = aimd_process::process_input(&request, &NoHooks).expect("ASR succeeds");
    let lower = result.markdown.to_lowercase();
    assert!(lower.contains("quick brown fox"), "{}", result.markdown);
    std::fs::remove_dir_all(dir).unwrap();
}
