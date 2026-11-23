//! Golden-fixture parity tests for the MLX affine dequantizer (WS-02).
//!
//! The fixtures under `tests/golden/quant/` are exported bit-exactly by
//! `scripts/export_quant_reference.py` (raw `mx.dequantize` output). Every
//! case must dequantize to identical 16-bit patterns here; any difference is
//! a packing bug, not a tolerance problem.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[path = "common/mlx_quant.rs"]
mod mlx_quant;

use mlx_quant::{QuantError, QuantizedTensor, ScaleDtype, dequantize_affine};
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

mod tests {
    use super::*;

    fn q(bits: u32, group_size: u32, shape: [usize; 2]) -> QuantizedTensor {
        QuantizedTensor {
            bits,
            group_size,
            shape: shape.to_vec(),
            scale_dtype: ScaleDtype::F16,
            payload: vec![0; shape[0] * shape[1] * bits as usize / 8],
            scales: vec![0; shape[0] * (shape[1] / group_size as usize)],
            biases: vec![0; shape[0] * (shape[1] / group_size as usize)],
        }
    }

    #[test]
    fn rejects_unsupported_bits() {
        let mut bad = q(6, 64, [1, 64]);
        bad.bits = 6;
        assert!(matches!(
            dequantize_affine(&bad),
            Err(QuantError::UnsupportedQuant(_))
        ));
    }

    #[test]
    fn rejects_bad_grouping() {
        let mut bad = q(4, 64, [1, 96]);
        bad.shape = vec![1, 96];
        assert!(matches!(
            dequantize_affine(&bad),
            Err(QuantError::InvalidTensor(_))
        ));
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let mut bad = q(4, 64, [2, 128]);
        bad.payload.truncate(100);
        assert!(matches!(
            dequantize_affine(&bad),
            Err(QuantError::InvalidTensor(_))
        ));
        let mut bad = q(4, 64, [2, 128]);
        bad.scales.truncate(2);
        assert!(matches!(
            dequantize_affine(&bad),
            Err(QuantError::InvalidTensor(_))
        ));
    }

    #[test]
    fn rejects_non_matrix() {
        let mut bad = q(4, 64, [2, 64]);
        bad.shape = vec![2, 32, 2];
        assert!(matches!(
            dequantize_affine(&bad),
            Err(QuantError::InvalidTensor(_))
        ));
    }

    /// Hand-computed dequantization: one row, two groups, f16 scales.
    /// Group 0: scale f16 0x3C00 (= 1.0), bias 0x0000 (= 0.0); group 1:
    /// scale f16 0x3800 (= 0.5), bias f16 0xC000 (= -2.0). 4-bit elements
    /// packed two per byte, low nibble first.
    #[test]
    fn dequantizes_known_values() {
        let mut quant = q(4, 4, [1, 8]);
        // elements: g0: 0,1,2,3 -> bytes [0x10, 0x32]; g1: 15,0,1,2 -> [0x0F, 0x21]
        quant.payload = vec![0x10, 0x32, 0x0F, 0x21];
        quant.scales = vec![0x3C00, 0x3800];
        quant.biases = vec![0x0000, 0xC000];
        let out = dequantize_affine(&quant).unwrap();
        assert_eq!(out.shape, vec![1, 8]);
        assert_eq!(out.dtype, ScaleDtype::F16);
        // f16 bits: 0.0, 1.0, 2.0, 3.0, then 0.5*15-2 = 5.5, -2.0, -1.5, -1.0
        assert_eq!(
            out.data,
            vec![
                0x0000, 0x3C00, 0x4000, 0x4200, 0x4580, 0xC000, 0xBE00, 0xBC00
            ]
        );
    }
}
