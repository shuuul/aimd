//! Shared Qwen3-ForcedAligner test fixtures.
//!
//! One real-checkpoint load per test binary: checkpoint-backed tests get the
//! pipeline from [`shared_pipeline`], a lazy `OnceLock<Option<_>>` over the
//! cached `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` snapshot (`None` =
//! cache absent or MLX CPU device -> skip). Readers for
//! `tests/golden/aligner/` (written by `scripts/export_aligner_reference.py`)
//! live here too.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::pipeline::ForcedAlignerPipeline;

const REPO: &str = "models--mlx-community--Qwen3-ForcedAligner-0.6B-8bit";

/// Committed checkpoint-tier clips under `tests/golden/aligner/cases/`.
pub(crate) const CASES: &[&str] = &["en01", "zh08"];

fn hf_cache() -> PathBuf {
    PathBuf::from(
        std::env::var("AIMD_TEST_HF_CACHE")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
    )
}

/// The cached checkpoint snapshot (any device).
pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    let required = ["config.json", "model.safetensors", "vocab.json"];
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(hf_cache().join(REPO).join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// `AIMD_E2E=1` gates the end-to-end tier.
pub(crate) fn e2e_enabled() -> bool {
    std::env::var("AIMD_E2E").as_deref() == Ok("1")
}

/// The process-wide aligner, loaded once (checkpoint tier: GPU device only).
pub(crate) fn shared_pipeline() -> Option<&'static ForcedAlignerPipeline> {
    static PIPELINE: OnceLock<Option<ForcedAlignerPipeline>> = OnceLock::new();
    let p = PIPELINE
        .get_or_init(|| {
            if aimd_mlx::device_is_cpu().unwrap_or(true) {
                eprintln!("skipping: checkpoint tier needs the GPU device (MLX CPU selected)");
                return None;
            }
            let dir = checkpoint_dir()?;
            match ForcedAlignerPipeline::load(&dir) {
                Ok(p) => Some(p),
                Err(e) => panic!("aligner checkpoint present but failed to load: {e}"),
            }
        })
        .as_ref();
    if p.is_none() {
        eprintln!("skipping: Qwen3-ForcedAligner-0.6B-8bit not available");
    }
    p
}

/// `tests/golden/aligner/`.
pub(crate) fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/aligner")
}

pub(crate) fn read_json(path: &Path) -> serde_json::Value {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

pub(crate) fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Raw little-endian bf16 widened to f32 (exact).
pub(crate) fn read_bf16(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_le_bytes(*b).to_f32())
        .collect()
}
