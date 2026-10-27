//! Shared Unlimited-OCR test fixtures.
//!
//! One real-checkpoint load per test binary: every checkpoint-backed test
//! (vision goldens, decoder goldens, pipeline E2E) goes through
//! [`shared_pipeline`], a lazy `OnceLock<Option<_>>` that loads
//! `mlx-community/Unlimited-OCR-4bit` from the HF cache once (or skips when
//! the snapshot is absent). Golden readers for `tests/golden/ocr_vision/`
//! and `tests/golden/ocr_decoder/` live here too.
//!
//! # Parity bounds
//!
//! The MLX port runs the same bf16 graph as mlx-vlm, so the residual is
//! kernel-order / bf16 rounding-boundary noise. It scales with the tensor
//! (its RMS / max magnitude), not with each element, so bounds are
//! tensor-level ([`Parity`]): relative RMS error and the worst element as
//! a fraction of max |ref|. Wiring bugs show up as O(1) relative RMS.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use aimd_mlx::{MlxArray, MlxDtype};

use super::pipeline::UnlimitedOcrPipeline;

const OCR_REPO_4BIT: &str = "models--mlx-community--Unlimited-OCR-4bit";
const OCR2_REPO_4BIT: &str = "models--mlx-community--Unlimited-OCR-2-4bit";

fn hf_cache() -> PathBuf {
    PathBuf::from(
        std::env::var("AIMD_TEST_HF_CACHE")
            .unwrap_or_else(|_| format!("{}/.cache/huggingface/hub", env!("HOME"))),
    )
}

/// The cached `Unlimited-OCR-4bit` snapshot holding config, weights and
/// tokenizer, if any. `None` on the MLX CPU device (`AIMD_MLX_DEVICE=cpu` or
/// no GPU): the checkpoint tier is far too slow there, so only hermetic tests
/// run (TEST RULES budget).
pub(crate) fn checkpoint_dir() -> Option<PathBuf> {
    snapshot_dir(OCR_REPO_4BIT)
}

/// The cached `Unlimited-OCR-2-4bit` snapshot (same rules as
/// [`checkpoint_dir`]).
pub(crate) fn checkpoint2_dir() -> Option<PathBuf> {
    snapshot_dir(OCR2_REPO_4BIT)
}

fn snapshot_dir(repo: &str) -> Option<PathBuf> {
    if aimd_mlx::device_is_cpu().unwrap_or(true) {
        eprintln!("skipping: checkpoint tier needs the Metal device (MLX CPU device selected)");
        return None;
    }
    let required = ["config.json", "model.safetensors", "tokenizer.json"];
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(hf_cache().join(repo).join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|dir| required.iter().all(|f| dir.join(f).is_file()))
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// `AIMD_E2E=1` gates the full-generation tier.
pub(crate) fn e2e_enabled() -> bool {
    std::env::var("AIMD_E2E").as_deref() == Ok("1")
}

/// The process-wide pipeline (vision + decoder + tokenizer), loaded once.
/// `None` (with a skip note) when the checkpoint is not cached.
pub(crate) fn shared_pipeline() -> Option<&'static UnlimitedOcrPipeline> {
    static PIPELINE: OnceLock<Option<UnlimitedOcrPipeline>> = OnceLock::new();
    let p = PIPELINE
        .get_or_init(|| {
            let Some(dir) = checkpoint_dir() else {
                eprintln!("skipping: {OCR_REPO_4BIT} not in HF cache");
                return None;
            };
            match UnlimitedOcrPipeline::load(&dir) {
                Ok(p) => Some(p),
                Err(e) => panic!("Unlimited-OCR checkpoint present but failed to load: {e}"),
            }
        })
        .as_ref();
    if p.is_none() {
        eprintln!("skipping: Unlimited-OCR-4bit not available");
    }
    p
}

/// The process-wide Unlimited-OCR-2 pipeline, loaded once (`None` when
/// the checkpoint is not cached).
pub(crate) fn shared_pipeline2() -> Option<&'static UnlimitedOcrPipeline> {
    static PIPELINE: OnceLock<Option<UnlimitedOcrPipeline>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| {
            let Some(dir) = checkpoint2_dir() else {
                eprintln!("skipping: {OCR2_REPO_4BIT} not in HF cache");
                return None;
            };
            match UnlimitedOcrPipeline::load(&dir) {
                Ok(p) => Some(p),
                Err(e) => panic!("Unlimited-OCR-2 checkpoint present but failed to load: {e}"),
            }
        })
        .as_ref()
}

/// Bounds for one compared tensor (see the module docs).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Parity {
    pub(crate) rel_rms: f64,
    pub(crate) max_frac: f64,
}

/// Compares sampled golden rows `want` (`rows.len() x cols`, in `rows`
/// order) against the full `got` (`[_, cols]`), printing the error and
/// recording a failure in `fails` when `bound` is exceeded.
pub(crate) fn compare_rows(
    tag: &str,
    got: &[f32],
    cols: usize,
    rows: &[usize],
    want: &[f32],
    bound: Parity,
    fails: &mut Vec<String>,
) {
    assert_eq!(want.len(), rows.len() * cols, "{tag}: golden size");
    let mut max_abs = 0f64;
    let mut worst = (0usize, 0usize);
    let (mut se, mut sr, mut mag) = (0f64, 0f64, 0f64);
    for (ri, &r) in rows.iter().enumerate() {
        assert!((r + 1) * cols <= got.len(), "{tag}: row {r} out of range");
        for c in 0..cols {
            let g = got[r * cols + c] as f64;
            let w = want[ri * cols + c] as f64;
            let d = (g - w).abs();
            if d.is_nan() || d > max_abs {
                max_abs = if d.is_nan() { f64::INFINITY } else { d };
                worst = (ri, c);
            }
            se += d * d;
            sr += w * w;
            mag = mag.max(w.abs());
        }
    }
    let rel_rms = (se / sr.max(1e-30)).sqrt();
    let max_frac = max_abs / mag.max(1e-30);
    let (ri, c) = worst;
    eprintln!(
        "{tag}: rel rms {rel_rms:.3e}, max abs {max_abs:.3e} = {:.2}% of max |ref| {mag:.3e} \
         (row {}, col {c}), {} rows",
        100.0 * max_frac,
        rows.get(ri).copied().unwrap_or(0),
        rows.len()
    );
    if !(rel_rms <= bound.rel_rms && max_frac <= bound.max_frac) {
        fails.push(format!(
            "{tag}: rel rms {rel_rms:.3e} (bound {:.3e}), max frac {max_frac:.3e} (bound {:.3e})",
            bound.rel_rms, bound.max_frac
        ));
    }
}

/// Host f32 copy of an MLX array viewed as `[_, last_dim]`.
pub(crate) fn mlx_rows_f32(a: &MlxArray) -> (Vec<f32>, usize) {
    let shape = a.shape().unwrap();
    let cols = *shape.last().unwrap();
    (a.astype(MlxDtype::F32).unwrap().to_f32_vec().unwrap(), cols)
}

fn read_bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

fn f32_le(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_le_bytes(*p))
        .collect()
}

fn bf16_le(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|p| half::bf16::from_bits(u16::from_le_bytes(*p)).to_f32())
        .collect()
}

fn json(dir: &Path, name: &str) -> serde_json::Value {
    serde_json::from_slice(&read_bytes(dir, name)).unwrap()
}

/// `tests/golden/ocr_vision/` (mlx-vlm vision tower stage dumps).
pub(crate) mod ocr_vision_golden {
    use super::*;

    pub(crate) fn dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ocr_vision")
    }

    pub(crate) fn manifest() -> serde_json::Value {
        json(&dir(), "manifest.json")
    }

    pub(crate) fn dump_entry<'a>(m: &'a serde_json::Value, stage: &str) -> &'a serde_json::Value {
        m["dumps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["stage"].as_str() == Some(stage))
            .unwrap_or_else(|| panic!("dump entry {stage} missing from manifest"))
    }

    pub(crate) fn dump_rows(e: &serde_json::Value) -> Vec<usize> {
        e["rows"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .unwrap_or_default()
    }

    pub(crate) fn read_f32(name: &str) -> Vec<f32> {
        f32_le(&read_bytes(&dir(), name))
    }

    pub(crate) fn read_bf16_f32(name: &str) -> Vec<f32> {
        bf16_le(&read_bytes(&dir(), name))
    }

    /// RGB8 pixels of `images/<name>`.
    pub(crate) fn source_image(name: &str) -> (Vec<u8>, usize, usize) {
        let img = image::load_from_memory(&read_bytes(&dir(), &format!("images/{name}"))).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        (rgb.into_raw(), w, h)
    }
}

/// `tests/golden/ocr_decoder/` (mlx-vlm decoder logits / forced streams /
/// ring cache dumps).
pub(crate) mod ocr_decoder_golden {
    use super::*;

    pub(crate) fn dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ocr_decoder")
    }

    pub(crate) fn manifest() -> serde_json::Value {
        json(&dir(), "manifest.json")
    }

    pub(crate) fn u32s(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    pub(crate) fn usizes(v: &serde_json::Value) -> Vec<usize> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    }

    pub(crate) fn read_f32(name: &str) -> Vec<f32> {
        f32_le(&read_bytes(&dir(), name))
    }

    pub(crate) fn read_bf16_f32(name: &str) -> Vec<f32> {
        bf16_le(&read_bytes(&dir(), name))
    }

    /// The image features the exporter fed the decoder: the vision golden
    /// `small_final_features.bf16u16` with its f16 round-trip snaps
    /// (`image_features_f16_snaps`), as a bf16 `[n_image, hidden]` array.
    pub(crate) fn image_features(m: &serde_json::Value) -> MlxArray {
        let (n, d) = (
            m["n_image"].as_u64().unwrap() as usize,
            m["hidden"].as_u64().unwrap() as usize,
        );
        let raw = ocr_vision_golden::read_bf16_f32("small_final_features.bf16u16");
        assert_eq!(raw.len(), n * d, "decoder golden feature geometry");
        let mut snaps = 0u64;
        let snapped: Vec<f32> = raw
            .iter()
            .map(|&v| {
                let w = half::f16::from_f32(v).to_f32();
                if w != v {
                    snaps += 1;
                    w
                } else {
                    v
                }
            })
            .collect();
        assert_eq!(
            snaps,
            m["image_features_f16_snaps"].as_u64().unwrap(),
            "f16 snap count"
        );
        MlxArray::from_f32(&[n, d], &snapped)
            .unwrap()
            .astype(MlxDtype::Bf16)
            .unwrap()
    }
}

pub(crate) mod ocr2_decoder_golden {
    use super::*;

    pub(crate) fn dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/ocr2_decoder")
    }

    pub(crate) fn manifest() -> serde_json::Value {
        json(&dir(), "manifest.json")
    }

    pub(crate) fn read_bf16_f32(name: &str) -> Vec<f32> {
        bf16_le(&read_bytes(&dir(), name))
    }

    /// The per-page image features in page order, concatenated as a bf16
    /// `[pages * per_page, hidden]` array.
    pub(crate) fn image_features(m: &serde_json::Value) -> MlxArray {
        let n = m["per_page_image_tokens"].as_u64().unwrap() as usize;
        let d = m["hidden"].as_u64().unwrap() as usize;
        let pages = m["pages"].as_array().unwrap().len();
        let mut all = Vec::with_capacity(pages * n * d);
        for p in 0..pages {
            let raw = read_bf16_f32(&format!("page{p}_features.bf16"));
            assert_eq!(raw.len(), n * d, "page {p} feature geometry");
            all.extend(raw);
        }
        MlxArray::from_f32(&[pages * n, d], &all)
            .unwrap()
            .astype(MlxDtype::Bf16)
            .unwrap()
    }
}
