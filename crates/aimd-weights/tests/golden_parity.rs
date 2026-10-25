//! Golden-fixture parity tests for the MLX affine dequantizer (WS-02).
//!
//! The fixtures under `tests/golden/quant/` are exported bit-exactly by
//! `scripts/export_quant_reference.py` (raw `mx.dequantize` output). Every
//! case must dequantize to identical 16-bit patterns here; any difference is
//! a packing bug, not a tolerance problem.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use aimd_weights::{QuantizedTensor, ScaleDtype, dequantize_affine};
use serde::Deserialize;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/quant")
        .canonicalize()
        .expect("tests/golden/quant fixtures must be present")
}

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    bits: u32,
    group_size: u32,
    mode: String,
    shape: Vec<usize>,
    payload_dtype: String,
    scale_dtype: String,
    dequant_dtype: String,
    files: HashMap<String, String>,
}

fn parse_scale_dtype(name: &str) -> ScaleDtype {
    match name {
        "F16" => ScaleDtype::F16,
        "BF16" => ScaleDtype::Bf16,
        other => panic!("unexpected scale dtype {other}"),
    }
}

fn read_case_bytes(case: &Case, key: &str) -> Vec<u8> {
    let rel = case
        .files
        .get(key)
        .unwrap_or_else(|| panic!("case {} missing {key} file entry", case.id));
    fs::read(fixture_dir().join(rel)).unwrap_or_else(|e| panic!("case {}: {key}: {e}", case.id))
}

fn load_u16_le(bytes: &[u8]) -> Vec<u16> {
    let (chunks, rest) = bytes.as_chunks::<2>();
    assert!(rest.is_empty(), "u16 byte length must be even");
    chunks
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// Every fixture case must dequantize bit-exactly against the exported
/// `mx.dequantize` bytes.
#[test]
fn golden_manifest_parity() {
    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(fixture_dir().join("manifest.json")).expect("reading manifest.json"),
    )
    .expect("parsing manifest.json");
    assert!(manifest.cases.len() >= 20, "expected full fixture set");
    let checkpoint_cases = manifest
        .cases
        .iter()
        .filter(|c| c.id.starts_with("checkpoint-"))
        .count();
    assert!(
        checkpoint_cases >= 3,
        "expected the 3 committed checkpoint slices, got {checkpoint_cases}"
    );

    for case in &manifest.cases {
        assert_eq!(case.mode, "affine", "case {}", case.id);
        assert_eq!(case.group_size, 64, "case {}", case.id);
        assert!(case.bits == 4 || case.bits == 8, "case {} bits", case.id);
        assert_eq!(case.payload_dtype, "U32", "case {}", case.id);
        assert_eq!(case.scale_dtype, case.dequant_dtype, "case {}", case.id);

        let quant = QuantizedTensor {
            bits: case.bits,
            group_size: case.group_size,
            shape: case.shape.clone(),
            scale_dtype: parse_scale_dtype(&case.scale_dtype),
            payload: read_case_bytes(case, "payload"),
            scales: load_u16_le(&read_case_bytes(case, "scales")),
            biases: load_u16_le(&read_case_bytes(case, "biases")),
        };
        let dequantized = dequantize_affine(&quant)
            .unwrap_or_else(|e| panic!("case {}: dequantize failed: {e}", case.id));
        let expected = load_u16_le(&read_case_bytes(case, "dequant"));

        assert_eq!(dequantized.shape, case.shape, "case {}", case.id);
        assert_eq!(
            dequantized.dtype,
            parse_scale_dtype(&case.dequant_dtype),
            "case {}",
            case.id
        );
        assert_eq!(
            dequantized.data.len(),
            expected.len(),
            "case {}: element count",
            case.id
        );
        let mismatches: Vec<usize> = dequantized
            .data
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(_, (got, want))| got != want)
            .map(|(i, _)| i)
            .collect();
        assert!(
            mismatches.is_empty(),
            "case {}: {} bit-exact mismatches, first at {} (got {:#06x}, want {:#06x})",
            case.id,
            mismatches.len(),
            mismatches[0],
            dequantized.data[mismatches[0]],
            expected[mismatches[0]]
        );
    }
}

/// Integration test against a real HF checkpoint; auto-skips when the HF
/// cache is absent. Override the snapshot directory with
/// `AIMD_TEST_CHECKPOINT_DIR`.
#[test]
fn checkpoint_snapshot_integration() {
    let Some(dir) = locate_checkpoint() else {
        eprintln!("skipping: no cached mlx-community Qwen3-ASR-1.7B-4bit snapshot");
        return;
    };

    let store = aimd_weights::load_checkpoint(&dir).expect("checkpoint loads");
    let quant_config = store
        .quantization()
        .expect("checkpoint config.json quantization block");
    assert_eq!(quant_config.mode.as_deref(), Some("affine"));
    assert_eq!(quant_config.bits, 4);
    assert_eq!(quant_config.group_size, 64);

    let bases = store.quantized_bases();
    assert!(bases.len() > 100, "expected many quantized tensors");
    assert!(bases.iter().all(|b| !b.ends_with(".weight")));

    // Assemble and dequantize the first few tensors; verify layout sanity.
    for base in bases.iter().take(4) {
        let tensor = store
            .quantized_tensor(base)
            .unwrap_or_else(|e| panic!("{base}: {e}"));
        assert_eq!(tensor.bits, 4);
        assert_eq!(tensor.group_size, 64);
        assert_eq!(tensor.scale_dtype, ScaleDtype::Bf16);
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        assert_eq!(cols % 64, 0);
        assert_eq!(tensor.payload.len(), rows * cols / 2);
        assert_eq!(tensor.scales.len(), rows * (cols / 64));
        let deq = tensor
            .dequantize()
            .unwrap_or_else(|e| panic!("{base}: {e}"));
        assert_eq!(deq.shape, vec![rows, cols]);
        assert!(
            deq.data
                .iter()
                .all(|bits| ScaleDtype::Bf16.to_f32(*bits).is_finite()),
            "{base}: dequantized values must be finite"
        );
    }
}

fn locate_checkpoint() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("AIMD_TEST_CHECKPOINT_DIR") {
        return PathBuf::from(dir).canonicalize().ok();
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3-ASR-1.7B-4bit/snapshots");
    let mut snaps: Vec<PathBuf> = fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").is_file())
        .collect();
    snaps.sort();
    snaps.into_iter().next()
}
