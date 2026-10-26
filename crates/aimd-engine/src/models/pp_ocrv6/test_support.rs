//! Shared PP-OCRv6 test fixtures.
//!
//! One real-checkpoint load per test binary: checkpoint-backed tests get the
//! det + rec pipeline from [`shared_pipeline`], a lazy `OnceLock<Option<_>>`
//! over the cached `PaddlePaddle/PP-OCRv6_medium_{det,rec}_safetensors`
//! snapshots (`None` = cache absent or MLX CPU device -> skip). Readers for
//! `tests/golden/ppocrv6/` (written by `scripts/export_ppocrv6_reference.py`)
//! live here too.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::image_ops::BgrImage;
use super::pipeline::PpOcrV6Pipeline;

const DET_REPO: &str = "models--PaddlePaddle--PP-OCRv6_medium_det_safetensors";
const REC_REPO: &str = "models--PaddlePaddle--PP-OCRv6_medium_rec_safetensors";

fn hf_cache() -> PathBuf {
    PathBuf::from(
        std::env::var("AIMD_TEST_HF_CACHE")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
    )
}

fn snapshot(repo: &str) -> Option<PathBuf> {
    let required = ["config.json", "model.safetensors"];
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(hf_cache().join(repo).join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The cached det and rec snapshots, if both exist and the MLX device is the
/// GPU (the checkpoint tier is skipped on the CPU device).
pub(crate) fn checkpoint_dirs() -> Option<(PathBuf, PathBuf)> {
    if aimd_mlx::device_is_cpu().unwrap_or(true) {
        eprintln!("skipping: checkpoint tier needs the Metal device (MLX CPU device selected)");
        return None;
    }
    Some((snapshot(DET_REPO)?, snapshot(REC_REPO)?))
}

/// `AIMD_E2E=1` gates the end-to-end tier.
pub(crate) fn e2e_enabled() -> bool {
    std::env::var("AIMD_E2E").as_deref() == Ok("1")
}

/// The process-wide det + rec pipeline, loaded once.
pub(crate) fn shared_pipeline() -> Option<&'static PpOcrV6Pipeline> {
    static PIPELINE: OnceLock<Option<PpOcrV6Pipeline>> = OnceLock::new();
    let p = PIPELINE
        .get_or_init(|| {
            let (det, rec) = checkpoint_dirs()?;
            match PpOcrV6Pipeline::load(&det, &rec) {
                Ok(p) => Some(p),
                Err(e) => panic!("PP-OCRv6 checkpoints present but failed to load: {e}"),
            }
        })
        .as_ref();
    if p.is_none() {
        eprintln!("skipping: PP-OCRv6 medium det/rec not available");
    }
    p
}

/// `tests/golden/ppocrv6/`.
pub(crate) fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ppocrv6")
}

pub(crate) fn read_bytes(name: &str) -> Vec<u8> {
    std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

pub(crate) fn read_f32(name: &str) -> Vec<f32> {
    read_bytes(name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_le_bytes(*p))
        .collect()
}

pub(crate) fn manifest() -> serde_json::Value {
    serde_json::from_slice(&read_bytes("manifest.json")).unwrap()
}

/// A golden PNG decoded to BGR (lossless, so these are exactly the pixels
/// mlx4ocr saw).
pub(crate) fn golden_image(name: &str) -> BgrImage {
    BgrImage::decode(&read_bytes(name)).unwrap()
}

pub(crate) fn usize_of(v: &serde_json::Value) -> usize {
    v.as_u64().unwrap() as usize
}

pub(crate) fn quad_of(v: &serde_json::Value) -> [[f32; 2]; 4] {
    let pts = v.as_array().unwrap();
    std::array::from_fn(|i| {
        let p = pts[i].as_array().unwrap();
        [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32]
    })
}
