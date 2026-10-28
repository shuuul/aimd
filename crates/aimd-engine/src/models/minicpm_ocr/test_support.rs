//! Shared MiniCPM-OCR test fixtures.
//!
//! One load per checkpoint per test binary: every checkpoint-backed test
//! goes through [`shared_pipeline`], a lazy `OnceLock<Option<_>>` per
//! precision (bf16 and load-time 4-bit) over the cached official
//! `openbmb/MiniCPM-OCR-1.0` snapshot. `None` (skip) when the snapshot is
//! absent or the MLX device is the CPU. Golden readers for
//! `tests/golden/minicpm_ocr/` live here too.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::pipeline::MiniCpmOcrPipeline;

pub(crate) const DENSE_REPO: &str = "models--openbmb--MiniCPM-OCR-1.0";
pub(crate) const DSPARK_REPO: &str = "models--openbmb--MiniCPM-OCR-1.0-DSpark";

fn hf_cache() -> PathBuf {
    PathBuf::from(
        std::env::var("AIMD_TEST_HF_CACHE")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
    )
}

/// The first cached snapshot of `repo` holding `config.json`,
/// `model.safetensors` and `tokenizer.json` (no device check).
pub(crate) fn snapshot(repo: &str) -> Option<PathBuf> {
    snapshot_with(
        repo,
        &["config.json", "model.safetensors", "tokenizer.json"],
    )
}

fn snapshot_with(repo: &str, required: &[&str]) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(hf_cache().join(repo).join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The dense snapshot when the checkpoint tier can run (GPU device).
pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    if aimd_mlx::device_is_cpu().unwrap_or(true) {
        eprintln!("skipping: checkpoint tier needs the GPU device (MLX CPU device selected)");
        return None;
    }
    let dir = snapshot(DENSE_REPO);
    if dir.is_none() {
        eprintln!("skipping: {DENSE_REPO} not in HF cache");
    }
    dir
}

/// `AIMD_E2E=1` gates the full-generation tier.
pub(crate) fn e2e_enabled() -> bool {
    std::env::var("AIMD_E2E").as_deref() == Ok("1")
}

/// Manifest key of a precision.
pub(crate) fn prec_key(bits: Option<u32>) -> &'static str {
    match bits {
        None => "bf16",
        Some(4) => "4bit",
        Some(_) => panic!("no goldens for this precision"),
    }
}

/// Loads the pipeline and, when its snapshot is cached, the DSpark
/// drafter (requests choose with `OcrRequest::speculative`).
fn load(bits: Option<u32>) -> Option<MiniCpmOcrPipeline> {
    let dir = checkpoint_dir()?;
    let mut p = MiniCpmOcrPipeline::load(&dir, bits).unwrap_or_else(|e| {
        panic!("MiniCPM-OCR checkpoint present but failed to load ({bits:?}): {e}")
    });
    match snapshot_with(DSPARK_REPO, &["config.json", "model.safetensors"]) {
        Some(d) => p
            .attach_drafter(&d)
            .unwrap_or_else(|e| panic!("DSpark checkpoint present but failed to load: {e}")),
        None => eprintln!("note: {DSPARK_REPO} not in HF cache; no drafter"),
    }
    Some(p)
}

/// The process-wide pipeline for `bits` (`None` = bf16, `Some(4)`).
pub(crate) fn shared_pipeline(bits: Option<u32>) -> Option<&'static MiniCpmOcrPipeline> {
    static BF16: OnceLock<Option<MiniCpmOcrPipeline>> = OnceLock::new();
    static Q4: OnceLock<Option<MiniCpmOcrPipeline>> = OnceLock::new();
    let cell = match bits {
        None => &BF16,
        Some(4) => &Q4,
        Some(_) => panic!("only bf16 and 4-bit have shared fixtures"),
    };
    cell.get_or_init(|| load(bits)).as_ref()
}

pub(crate) fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/minicpm_ocr")
}

pub(crate) fn read_bytes(name: &str) -> Vec<u8> {
    std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

pub(crate) fn json(name: &str) -> serde_json::Value {
    serde_json::from_slice(&read_bytes(name)).unwrap()
}

pub(crate) fn manifest() -> serde_json::Value {
    json("manifest.json")
}

pub(crate) fn preprocess_golden() -> serde_json::Value {
    json("preprocess.json")
}

pub(crate) fn u32s(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

pub(crate) fn usizes(v: &serde_json::Value) -> Vec<usize> {
    u32s(v).into_iter().map(|x| x as usize).collect()
}

pub(crate) fn bf16_le(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|p| half::bf16::from_bits(u16::from_le_bytes(*p)).to_f32())
        .collect()
}

pub(crate) fn f32_le(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_le_bytes(*p))
        .collect()
}

/// RGB8 pixels of a golden fixture image (`images/<name>.png`).
pub(crate) fn fixture_image(name: &str) -> (Vec<u8>, usize, usize) {
    let img = image::load_from_memory(&read_bytes(&format!("images/{name}.png"))).unwrap();
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    (rgb.into_raw(), w, h)
}
