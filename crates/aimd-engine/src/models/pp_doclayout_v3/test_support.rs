//! Shared PP-DocLayoutV3 test fixtures.
//!
//! One real-checkpoint load per test binary: checkpoint-backed tests get the
//! detector from [`shared_pipeline`], a lazy `OnceLock<Option<_>>` over the
//! cached `PaddlePaddle/PP-DocLayoutV3_safetensors` snapshot (`None` = cache
//! absent or MLX CPU device -> skip). Readers for
//! `tests/golden/pp_doclayout_v3/` (written by
//! `scripts/export_pp_doclayout_v3_reference.py`) live here too.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::pipeline::PpDocLayoutV3;

const REPO: &str = "models--PaddlePaddle--PP-DocLayoutV3_safetensors";

fn hf_cache() -> PathBuf {
    PathBuf::from(
        std::env::var("AIMD_TEST_HF_CACHE")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
    )
}

fn snapshot() -> Option<PathBuf> {
    let required = ["config.json", "model.safetensors"];
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(hf_cache().join(REPO).join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The process-wide detector, loaded once (`None` on the MLX CPU device or
/// without the cached snapshot).
pub(crate) fn shared_pipeline() -> Option<&'static PpDocLayoutV3> {
    static PIPELINE: OnceLock<Option<PpDocLayoutV3>> = OnceLock::new();
    let p = PIPELINE
        .get_or_init(|| {
            if aimd_mlx::device_is_cpu().unwrap_or(true) {
                return None;
            }
            let dir = snapshot()?;
            match PpDocLayoutV3::load(&dir) {
                Ok(p) => Some(p),
                Err(e) => panic!("PP-DocLayoutV3 checkpoint present but failed to load: {e}"),
            }
        })
        .as_ref();
    if p.is_none() {
        eprintln!("skipping: PP-DocLayoutV3 needs the cached checkpoint and the Metal device");
    }
    p
}

/// `tests/golden/pp_doclayout_v3/`.
pub(crate) fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/pp_doclayout_v3")
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

/// A golden PNG decoded to RGB8 (lossless).
pub(crate) fn rgb_image(name: &str) -> image::RgbImage {
    image::load_from_memory(&read_bytes(name))
        .unwrap()
        .to_rgb8()
}

/// The golden pages (`manifest.pages[*]`).
pub(crate) fn pages() -> Vec<serde_json::Value> {
    manifest()["pages"].as_array().unwrap().clone()
}
