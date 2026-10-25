//! Parity + error-path tests for the `aimd-mlx` C-ABI shim (WS-11a).
//!
//! Parity baseline for the quantized path: [`aimd_weights::dequantize_affine`]
//! (bit-exact vs `mx.dequantize`, guarded by `aimd-weights/tests/golden/quant`) cast to
//! f16, followed by a naive f32-accumulate matmul, for the golden quant
//! fixtures (the committed checkpoint slices + synthetic f16/bf16,
//! 4-bit/8-bit cases). Dense f16 matmul and the model stages are covered by
//! the model-level goldens in `aimd-engine` (ASR tower/decoder, OCR
//! vision/decoder), so they are not duplicated here.
//!
//! Tolerances are fp16/bf16-level (MLX is not bit-exact vs the reference on
//! either device). Measured values are printed with `eprintln!` and asserted
//! with headroom; see the constants next to each assertion.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use aimd_mlx::{MlxArray, MlxDtype, MlxError};
use aimd_weights::{QuantizedTensor, ScaleDtype, dequantize_affine};
use half::{bf16, f16};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Deterministic random data (no rand dependency in dev-deps).
// ---------------------------------------------------------------------------

/// xorshift64* — deterministic across runs and platforms.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    fn f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        self.f32_unit() * 2.0 - 1.0
    }

    fn f16_vec(&mut self, n: usize) -> Vec<f16> {
        (0..n).map(|_| f16::from_f32(self.signed())).collect()
    }

    fn bf16_vec(&mut self, n: usize) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(self.signed())).collect()
    }
}

// ---------------------------------------------------------------------------
// Golden quant fixture loading (same layout as aimd-weights' golden_parity).
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aimd-weights/tests/golden/quant")
        .canonicalize()
        .expect("aimd-weights/tests/golden/quant fixtures must be present")
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
    files: HashMap<String, String>,
}

impl Case {
    fn scale_dtype(&self) -> ScaleDtype {
        match self.scale_dtype.as_str() {
            "F16" => ScaleDtype::F16,
            "BF16" => ScaleDtype::Bf16,
            other => panic!("case {}: unexpected scale dtype {other}", self.id),
        }
    }

    fn read_bytes(&self, key: &str) -> Vec<u8> {
        let rel = self
            .files
            .get(key)
            .unwrap_or_else(|| panic!("case {} missing {key} file entry", self.id));
        fs::read(fixture_dir().join(rel)).unwrap_or_else(|e| panic!("case {}: {key}: {e}", self.id))
    }

    fn load_quantized(&self) -> QuantizedTensor {
        let scale_bytes = self.read_bytes("scales");
        let (chunks, rest) = scale_bytes.as_chunks::<2>();
        assert!(rest.is_empty(), "case {}: scales byte length", self.id);
        let scales: Vec<u16> = chunks.iter().map(|p| u16::from_le_bytes(*p)).collect();
        let bias_bytes = self.read_bytes("biases");
        let (chunks, rest) = bias_bytes.as_chunks::<2>();
        assert!(rest.is_empty(), "case {}: biases byte length", self.id);
        let biases: Vec<u16> = chunks.iter().map(|p| u16::from_le_bytes(*p)).collect();
        QuantizedTensor {
            bits: self.bits,
            group_size: self.group_size,
            shape: self.shape.clone(),
            scale_dtype: self.scale_dtype(),
            payload: self.read_bytes("payload"),
            scales,
            biases,
        }
    }
}

fn load_manifest() -> Manifest {
    serde_json::from_str(
        &fs::read_to_string(fixture_dir().join("manifest.json")).expect("reading manifest.json"),
    )
    .expect("parsing manifest.json")
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn scale_dtype_to_mlx(dtype: ScaleDtype) -> MlxDtype {
    match dtype {
        ScaleDtype::F16 => MlxDtype::F16,
        ScaleDtype::Bf16 => MlxDtype::Bf16,
    }
}

/// Widens an output array of any float dtype to `f32`.
fn out_f32(array: &MlxArray) -> Result<Vec<f32>, MlxError> {
    match array.dtype()? {
        MlxDtype::F16 => Ok(array.to_f16_vec()?.iter().map(|v| v.to_f32()).collect()),
        MlxDtype::Bf16 => Ok(array.to_bf16_vec()?.iter().map(|v| v.to_f32()).collect()),
        MlxDtype::F32 => array.to_f32_vec(),
        other => Err(MlxError::Op(format!("unexpected output dtype {other:?}"))),
    }
}

fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "output element count");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |a, v| a.max(v.abs()))
}

/// Casts dequantized bits (f16 or bf16 patterns) to f16 — the f16
/// reference-compute convention. bf16 -> f16 is exact for weight-range values
/// (wider mantissa; only the exponent range shrinks).
fn dequant_to_f16(dequant: &aimd_weights::DequantizedTensor) -> Vec<f16> {
    dequant
        .data
        .iter()
        .map(|&bits| match dequant.dtype {
            ScaleDtype::F16 => f16::from_bits(bits),
            ScaleDtype::Bf16 => f16::from_f32(ScaleDtype::Bf16.to_f32(bits)),
        })
        .collect()
}

/// Naive `x[m, k] @ wt[n, k]^T` with f32 accumulation, f16-rounded output —
/// the reference semantics for the quantized-matmul parity checks.
fn naive_matmul_f16_nt(x: &[f16], w: &[f16], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), n * k);
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let acc: f32 = (0..k)
                .map(|c| x[i * k + c].to_f32() * w[j * k + c].to_f32())
                .sum();
            out.push(f16::from_f32(acc).to_f32());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

#[test]
fn mlx_lifecycle_f16_roundtrip_and_metadata() {
    let data: Vec<f16> = [1.0f32, -0.5, 65504.0, 2.0]
        .iter()
        .map(|&v| f16::from_f32(v))
        .chain([f16::from_bits(0x7C01)]) // NaN pattern must survive bitwise
        .collect();
    let array = MlxArray::from_f16(&[5], &data).expect("from_f16");
    assert_eq!(array.dtype().unwrap(), MlxDtype::F16);
    assert_eq!(array.shape().unwrap(), vec![5]);
    assert_eq!(array.ndim().unwrap(), 1);
    assert_eq!(array.dim(0).unwrap(), 5);
    assert_eq!(array.size().unwrap(), 5);
    array.eval().expect("eval");

    let back = array.to_f16_vec().expect("to_f16_vec");
    assert_eq!(back.len(), data.len());
    for (got, want) in back.iter().zip(&data) {
        assert_eq!(got.to_bits(), want.to_bits(), "f16 bits must roundtrip");
    }
}

#[test]
fn mlx_lifecycle_bitexact_all_dtypes() {
    // Every wire dtype must survive import/export bit-exactly, including NaN
    // payloads, -0.0, and arbitrary integer words.
    let f32_data = [1.5f32, -0.0, f32::from_bits(0x7F80_0001)];
    let a = MlxArray::from_f32(&[3], &f32_data).unwrap();
    for (got, want) in a.to_f32_vec().unwrap().iter().zip(&f32_data) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    let bf16_data = [
        bf16::from_f32(1.25),
        bf16::from_bits(0xFF80),
        bf16::from_bits(0x0001),
    ];
    let a = MlxArray::from_bf16(&[3], &bf16_data).unwrap();
    for (got, want) in a.to_bf16_vec().unwrap().iter().zip(&bf16_data) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    let u32_data = [0xDEAD_BEEFu32, 0, 0xFFFF_FFFF];
    let a = MlxArray::from_u32(&[3], &u32_data).unwrap();
    assert_eq!(a.to_u32_vec().unwrap(), u32_data);

    let u8_data = [0xFFu8, 0x00, 0x7F];
    let a = MlxArray::from_u8(&[3], &u8_data).unwrap();
    // Readback through the wrong dtype must be rejected.
    let err = a.to_f16_bytes().unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
}

// ---------------------------------------------------------------------------
// Quantized matmul parity against the golden fixtures.
// ---------------------------------------------------------------------------

/// All committed fixture cases: checkpoint-realistic quantized_matmul parity.
///
/// Activations use the scale dtype (bf16 for the bf16-checkpoint cases, f16
/// for the f16 synthetic ones) — exactly how the MLX runtime feeds quantized
/// layers, so mlx keeps the promoted compute dtype at the scale dtype and its
/// internal dequant matches `dequantize_affine`'s rounding. The reference is
/// `dequantize_affine` (bit-exact vs mx.dequantize) cast to f16 + a naive
/// f32-accumulate matmul.
#[test]
fn mlx_quantized_matmul_parity_fixtures() {
    let manifest = load_manifest();
    assert!(manifest.cases.len() >= 25, "expected the full fixture set");
    let checkpoint_cases = manifest
        .cases
        .iter()
        .filter(|c| c.id.starts_with("checkpoint-"))
        .count();
    assert!(
        checkpoint_cases >= 3,
        "expected the 3 committed checkpoint slices, got {checkpoint_cases}"
    );

    // MLX's CPU `quantized_matmul` loses precision with bf16 inputs, and the
    // error grows with K (vs a float64 reference: 1-2% of max|out| at K=128,
    // 6.5-17% at K=4096; f16, dense matmul and f32 inputs stay <= 0.2%;
    // Metal bf16 stays < 0.8%), consistent with bf16 rather than f32
    // accumulation. Reproduced in Python mlx 0.32.2 and brew mlx 0.32.4. On the
    // CPU device the bf16 cases therefore run with f32 activations, scales and
    // biases: the packed-weight layout under test is unchanged, and the bound
    // stays the same as on Metal.
    let promote_bf16 = aimd_mlx::device_is_cpu().unwrap();
    let rel_tol = 0.008;
    let mut worst: (&str, f32, f32, f32) = ("", 0.0, 0.0, 0.0);
    for case in &manifest.cases {
        assert_eq!(case.mode, "affine", "case {}", case.id);
        assert_eq!(case.payload_dtype, "U32", "case {}", case.id);
        let [rows, cols] = [case.shape[0], case.shape[1]];
        let bits = case.bits;
        let group_size = case.group_size;
        let groups = cols / group_size as usize;

        let quant = case.load_quantized();
        let dequant = dequantize_affine(&quant)
            .unwrap_or_else(|e| panic!("case {}: dequantize failed: {e}", case.id));
        let w_f16 = dequant_to_f16(&dequant);

        // Random activations in the scale dtype: [m, cols].
        let m = 2;
        let mut rng = Rng::new(0x00C0FFEE ^ case.id.len() as u64);
        let (x_mlx, x_f16) = match case.scale_dtype() {
            ScaleDtype::Bf16 => {
                let x = rng.bf16_vec(m * cols);
                let f16_x: Vec<f16> = x.iter().map(|v| f16::from_f32(v.to_f32())).collect();
                (MlxArray::from_bf16(&[m, cols], &x).unwrap(), f16_x)
            }
            ScaleDtype::F16 => {
                let x = rng.f16_vec(m * cols);
                (MlxArray::from_f16(&[m, cols], &x).unwrap(), x.clone())
            }
        };

        // MLX arrays for the packed weight triple.
        let packed_cols = cols * bits as usize / 32;
        let (w, scales, biases) = MlxArray::quantized_weight(
            &quant.payload,
            &[rows, packed_cols],
            &quant.scales,
            &quant.biases,
            &[rows, groups],
            scale_dtype_to_mlx(case.scale_dtype()),
        )
        .expect("weight triple");

        let promote = promote_bf16 && case.scale_dtype() == ScaleDtype::Bf16;
        let (x_mlx, scales, biases, out_dtype) = if promote {
            (
                x_mlx.astype(MlxDtype::F32).unwrap(),
                scales.astype(MlxDtype::F32).unwrap(),
                biases.astype(MlxDtype::F32).unwrap(),
                MlxDtype::F32,
            )
        } else {
            (
                x_mlx,
                scales,
                biases,
                scale_dtype_to_mlx(case.scale_dtype()),
            )
        };

        let out = x_mlx
            .quantized_matmul(
                &w,
                &scales,
                Some(&biases),
                true,
                Some(group_size as i32),
                Some(bits as i32),
                None,
            )
            .unwrap_or_else(|e| panic!("case {}: quantized_matmul failed: {e}", case.id));
        assert_eq!(out.dtype().unwrap(), out_dtype, "case {}", case.id);
        assert_eq!(out.shape().unwrap(), vec![m, rows], "case {}", case.id);
        let mlx_out = out_f32(&out).expect("readback");

        // Reference: x @ deq(W).T with f32 accumulation.
        let ref_out = naive_matmul_f16_nt(&x_f16, &w_f16, m, cols, rows);

        let diff = max_abs_diff(&mlx_out, &ref_out);
        let scale = max_abs(&ref_out);
        // fp16-tolerance regime: ~1 bf16 ulp (2^-9) of the output magnitude
        // dominates; assert with 4x headroom.
        let bound = rel_tol * scale + 1e-4;
        if diff > worst.1 {
            worst = (&case.id, diff, scale, bound);
        }
        assert!(
            diff <= bound,
            "case {}: quantized parity exceeded tolerance: {diff:.3e} > {bound:.3e} (max|out| {scale:.3})",
            case.id
        );
    }
    eprintln!(
        "mlx_quantized_matmul_parity_fixtures: {} cases, worst: {} \
         max_abs_diff={:.3e}, max|out|={:.3}, bound={:.3e}",
        manifest.cases.len(),
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
}

/// Mixed activation/scale dtypes: f16 activations with checkpoint bf16
/// scales/biases are accepted natively (mlx promotes f16 x bf16 -> f32), so
/// no fp16 cast workaround is needed. Output is f32. Tolerance is looser than
/// the same-dtype path because mlx's internal f32 dequant skips the bf16
/// rounding the fixture dequantizer applies (~1% relative divergence at
/// k=2048).
#[test]
fn mlx_quantized_matmul_f16_x_bf16_scales_promotes() {
    let manifest = load_manifest();
    let case = manifest
        .cases
        .iter()
        .find(|c| c.id == "checkpoint-asr-4bit-model-layers-0-self_attn-k_proj-r8")
        .expect("checkpoint 4-bit fixture");
    let [rows, cols] = [case.shape[0], case.shape[1]];
    let quant = case.load_quantized();

    let m = 2;
    let mut rng = Rng::new(0xBEEF);
    let x = rng.f16_vec(m * cols);
    let x_mlx = MlxArray::from_f16(&[m, cols], &x).unwrap();
    let (w, scales, biases) = MlxArray::quantized_weight(
        &quant.payload,
        &[rows, cols * case.bits as usize / 32],
        &quant.scales,
        &quant.biases,
        &[rows, cols / case.group_size as usize],
        MlxDtype::Bf16,
    )
    .unwrap();

    let out = x_mlx
        .quantized_matmul(
            &w,
            &scales,
            Some(&biases),
            true,
            Some(case.group_size as i32),
            Some(case.bits as i32),
            None,
        )
        .expect("bf16 scales must be accepted");
    assert_eq!(out.dtype().unwrap(), MlxDtype::F32);
    assert_eq!(out.shape().unwrap(), vec![m, rows]);
    let mlx_out = out_f32(&out).expect("readback");

    let w_f16 = dequant_to_f16(&dequantize_affine(&quant).unwrap());
    let ref_out = naive_matmul_f16_nt(&x, &w_f16, m, cols, rows);

    let diff = max_abs_diff(&mlx_out, &ref_out);
    let scale = max_abs(&ref_out);
    eprintln!(
        "mlx_quantized_matmul_f16_x_bf16_scales_promotes: max_abs_diff={diff:.3e}, \
         max|out|={scale:.3}"
    );
    // f32 compute on the MLX side vs bf16-rounded weights on the reference
    // side: ~2^-9 relative per weight, so bound ~2% of |out|.
    let bound = 0.02 * scale + 1e-3;
    assert!(
        diff <= bound,
        "promoted quantized parity: {diff:.3e} > {bound:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Shape / elementwise ops.
// ---------------------------------------------------------------------------

#[test]
fn mlx_transpose_and_reshape_row_major() {
    // [2,3,4] values 0..24, row-major.
    let data: Vec<u32> = (0..24).collect();
    let a = MlxArray::from_u32(&[2, 3, 4], &data).unwrap();

    // transpose with axes [1,0,2]: out[j, i, k] == a[i, j, k].
    let t = a.transpose(&[1, 0, 2]).unwrap();
    assert_eq!(t.shape().unwrap(), vec![3, 2, 4]);
    let out = t.to_u32_vec().unwrap();
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..4usize {
                let want = data[i * 12 + j * 4 + k];
                assert_eq!(
                    out[j * 8 + i * 4 + k],
                    want,
                    "axes [1,0,2] at ({i},{j},{k})"
                );
            }
        }
    }

    // Reverse transpose (empty axes): out[k, j, i] == a[i, j, k].
    let rt = a.transpose(&[]).unwrap();
    assert_eq!(rt.shape().unwrap(), vec![4, 3, 2]);
    let out = rt.to_u32_vec().unwrap();
    for i in 0..2usize {
        for j in 0..3usize {
            for k in 0..4usize {
                assert_eq!(out[k * 6 + j * 2 + i], data[i * 12 + j * 4 + k]);
            }
        }
    }

    // Reshape keeps row-major order.
    let r = a.reshape(&[2, 12]).unwrap();
    assert_eq!(r.shape().unwrap(), vec![2, 12]);
    let flat = r.reshape(&[24]).unwrap().to_u32_vec().unwrap();
    assert_eq!(flat, data);
}

#[test]
fn mlx_elementwise_f32_reference() {
    let a_data = [1.0f32, 4.0, 9.0, 16.0];
    let b_data = [2.0f32, 2.0, 3.0, 4.0];
    let a = MlxArray::from_f32(&[4], &a_data).unwrap();
    let b = MlxArray::from_f32(&[4], &b_data).unwrap();

    for (got, want) in a
        .add(&b)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip([3.0, 6.0, 12.0, 20.0])
    {
        assert!((got - want).abs() <= 1e-6, "add: {got} vs {want}");
    }
    for (got, want) in a
        .multiply(&b)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip([2.0, 8.0, 27.0, 64.0])
    {
        assert!((got - want).abs() <= 1e-6, "multiply: {got} vs {want}");
    }
    for (got, want) in a
        .divide(&b)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip([0.5, 2.0, 3.0, 4.0])
    {
        assert!((got - want).abs() <= 1e-6, "divide: {got} vs {want}");
    }
    for (got, want) in a
        .sqrt()
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip([1.0, 2.0, 3.0, 4.0])
    {
        assert!((got - want).abs() <= 1e-6, "sqrt: {got} vs {want}");
    }
    for (got, want) in a
        .power(&b)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip([1.0, 16.0, 729.0, 65536.0])
    {
        assert!((got - want).abs() <= 1e-3, "power: {got} vs {want}");
    }
}

#[test]
fn mlx_softmax_matches_f32_reference() {
    let rows = [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
    let a = MlxArray::from_f32(&[2, 3], &rows).unwrap();
    let out = a.softmax(-1).unwrap();
    assert_eq!(out.shape().unwrap(), vec![2, 3]);
    let got = out.to_f32_vec().unwrap();
    for row in 0..2 {
        let slice = &got[row * 3..(row + 1) * 3];
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-6, "softmax row sum {sum}");
    }
    // Hand reference for row 0: exp([1,2,3]) / sum.
    let exps: Vec<f64> = [1.0f64, 2.0, 3.0].iter().map(|v| v.exp()).collect();
    let total: f64 = exps.iter().sum();
    for (k, e) in exps.iter().enumerate() {
        assert!((got[k] - (*e / total) as f32).abs() <= 1e-6);
    }
    // Uniform row stays uniform.
    for v in &got[3..6] {
        assert!((v - 1.0 / 3.0).abs() <= 1e-6);
    }
}

#[test]
fn mlx_silu_matches_f32_reference() {
    let x = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let a = MlxArray::from_f32(&[5], &x).unwrap();
    let got = a.silu().unwrap().to_f32_vec().unwrap();
    for (g, &v) in got.iter().zip(&x) {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        let want = v * sigmoid;
        assert!((g - want).abs() <= 1e-6, "silu({v}): {g} vs {want}");
    }
    // Compiled swiglu == unfused silu(gate) * up, bit-for-bit in bf16
    // (the decoders' dtype), across shapes (shapeless compile) and on a
    // transposed (strided) gate.
    for (rows, cols) in [(1usize, 6usize), (3, 4)] {
        let n = rows * cols;
        let g: Vec<f32> = (0..n).map(|i| i as f32 * 0.37 - 1.1).collect();
        let u: Vec<f32> = (0..n).map(|i| 0.5 - i as f32 * 0.21).collect();
        let bf = |v: &[f32], shape: &[usize]| {
            MlxArray::from_f32(shape, v)
                .unwrap()
                .astype(MlxDtype::Bf16)
                .unwrap()
        };
        let (gate, up) = (bf(&g, &[rows, cols]), bf(&u, &[rows, cols]));
        let unfused = gate.silu().unwrap().multiply(&up).unwrap();
        assert_eq!(
            gate.swiglu(&up).unwrap().to_bf16_vec().unwrap(),
            unfused.to_bf16_vec().unwrap()
        );
        let gate_t = bf(&g, &[cols, rows]).transpose(&[1, 0]).unwrap();
        let unfused_t = gate_t.silu().unwrap().multiply(&up).unwrap();
        assert_eq!(
            gate_t.swiglu(&up).unwrap().to_bf16_vec().unwrap(),
            unfused_t.to_bf16_vec().unwrap()
        );
    }
}

#[test]
fn mlx_rms_norm_matches_f32_reference() {
    // [2,4] with a [4] weight; reference: x / sqrt(mean(x^2) + eps) * w.
    let x = [
        1.0f32, -2.0, 3.0, -4.0, //
        0.5, 0.25, -0.75, 1.5,
    ];
    let w = [1.0f32, 2.0, 3.0, 4.0];
    let eps = 1e-5f32;
    let a = MlxArray::from_f32(&[2, 4], &x).unwrap();
    let weight = MlxArray::from_f32(&[4], &w).unwrap();

    let got = a
        .rms_norm(Some(&weight), eps)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let mut want = vec![0.0f32; 8];
    for row in 0..2 {
        let row_x = &x[row * 4..(row + 1) * 4];
        let ms: f32 = row_x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inv = 1.0 / (ms + eps).sqrt();
        for k in 0..4 {
            want[row * 4 + k] = row_x[k] * inv * w[k];
        }
    }
    for (g, wnt) in got.iter().zip(&want) {
        assert!((g - wnt).abs() <= 1e-5, "rms_norm: {g} vs {wnt}");
    }

    // Weight-less variant divides only.
    let got = a.rms_norm(None, eps).unwrap().to_f32_vec().unwrap();
    for row in 0..2 {
        let row_x = &x[row * 4..(row + 1) * 4];
        let ms: f32 = row_x.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inv = 1.0 / (ms + eps).sqrt();
        for k in 0..4 {
            assert!((got[row * 4 + k] - row_x[k] * inv).abs() <= 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// Error paths (shape/dtype validation, typed error propagation).
// ---------------------------------------------------------------------------

#[test]
fn mlx_array_new_length_mismatch_is_an_error() {
    // 2 f16 elements declared, 3 bytes supplied: the safe wrapper must reject
    // before any FFI call.
    let err = MlxArray::from_bytes(&[0u8; 3], MlxDtype::F16, &[2]).unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
    assert!(last_error_is_nonempty(err));
}

#[test]
fn mlx_to_f16_bytes_requires_f16() {
    let a = MlxArray::from_f32(&[2], &[1.0, 2.0]).unwrap();
    let err = a.to_f16_bytes().unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
    // The last-error slot must carry the shim's diagnostic.
    assert!(last_error_is_nonempty(err));
}

fn last_error_is_nonempty(err: MlxError) -> bool {
    matches!(&err, MlxError::Op(msg) if !msg.is_empty())
}

#[test]
fn mlx_matmul_inner_dim_mismatch_is_an_error() {
    let a = MlxArray::from_f16(&[64, 128], &vec![f16::ONE; 64 * 128]).unwrap();
    let b = MlxArray::from_f16(&[64, 96], &vec![f16::ONE; 64 * 96]).unwrap();
    let err = a.matmul(&b, None).unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
}

#[test]
fn mlx_reshape_element_count_mismatch_is_an_error() {
    let a = MlxArray::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let err = a.reshape(&[7]).unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
}

#[test]
fn mlx_quantized_matmul_dim_mismatch_reports_mlx_tag() {
    let manifest = load_manifest();
    let case = manifest
        .cases
        .iter()
        .find(|c| c.id == "checkpoint-asr-4bit-model-layers-0-self_attn-k_proj-r8")
        .expect("checkpoint 4-bit fixture");
    let [rows, cols] = [case.shape[0], case.shape[1]];
    let quant = case.load_quantized();
    let x = MlxArray::from_bf16(&[2, 100], &vec![bf16::ONE; 200]).unwrap();
    let (w, scales, biases) = MlxArray::quantized_weight(
        &quant.payload,
        &[rows, cols * case.bits as usize / 32],
        &quant.scales,
        &quant.biases,
        &[rows, cols / case.group_size as usize],
        MlxDtype::Bf16,
    )
    .unwrap();
    let err = x
        .quantized_matmul(
            &w,
            &scales,
            Some(&biases),
            true,
            Some(case.group_size as i32),
            Some(case.bits as i32),
            None,
        )
        .unwrap_err();
    match &err {
        MlxError::Op(msg) => {
            assert!(
                msg.contains("quantized_matmul"),
                "error must carry the mlx op tag: {msg}"
            );
        }
        other => panic!("expected MlxError::Op, got {other}"),
    }
}

#[test]
fn mlx_quantized_matmul_rejects_unsupported_bits() {
    let a = MlxArray::from_f16(&[1, 64], &[f16::ONE; 64]).unwrap();
    let w = MlxArray::from_u32(&[1, 8], &[0u32; 8]).unwrap();
    let s = MlxArray::from_f16(&[1, 1], &[f16::ONE]).unwrap();
    let b = MlxArray::from_f16(&[1, 1], &[f16::ZERO]).unwrap();
    // bits=5 has no affine packing in MLX (and 8 payload words would not be
    // consistent with it either).
    let err = a
        .quantized_matmul(&w, &s, Some(&b), true, Some(64), Some(5), None)
        .unwrap_err();
    assert!(matches!(err, MlxError::Op(_)), "{err}");
}

// ---------------------------------------------------------------------------
// Device selection.
// ---------------------------------------------------------------------------

/// `AIMD_MLX_DEVICE=cpu` (hermetic CI on virtualized macOS runners) must put
/// the shim on the MLX CPU device; otherwise it is whatever the host offers.
#[test]
fn mlx_device_follows_env_override() {
    let cpu = aimd_mlx::device_is_cpu().expect("device query");
    eprintln!("mlx device: {}", if cpu { "cpu" } else { "gpu" });
    match std::env::var("AIMD_MLX_DEVICE").as_deref() {
        Ok("cpu") => assert!(cpu, "AIMD_MLX_DEVICE=cpu must select the CPU device"),
        Ok("gpu") => assert!(!cpu, "AIMD_MLX_DEVICE=gpu must select the GPU"),
        _ => {}
    }
}

#[test]
fn mlx_moe_swiglu_4bit_matches_gather_qmm_graph() {
    // Custom Metal kernels need the GPU; the CPU device (hermetic CI) keeps
    // the gather_qmm path.
    if aimd_mlx::device_is_cpu().expect("device query") {
        return;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let bf = |v: f32| bf16::from_f32(v).to_bits();
    // (hidden, intermediate, experts, top-k): a small case and the
    // Unlimited-OCR decode shape (1280 / 896, top-6) with fewer experts.
    for (d, inter, experts, k) in [(256usize, 128usize, 8usize, 3usize), (1280, 896, 10, 6)] {
        let gs = 64usize;
        let mut quant = |rows: usize, cols: usize| {
            let r = experts * rows;
            let (words, groups) = (cols / 8, cols / gs);
            let payload: Vec<u8> = (0..r * words * 4).map(|_| next() as u8).collect();
            let scales: Vec<u16> = (0..r * groups)
                .map(|_| bf(0.001 + (next() % 1000) as f32 * 4e-6))
                .collect();
            let biases: Vec<u16> = (0..r * groups)
                .map(|_| bf(-0.03 + (next() % 1000) as f32 * 3e-5))
                .collect();
            MlxArray::quantized_weight(
                &payload,
                &[experts, rows, words],
                &scales,
                &biases,
                &[experts, rows, groups],
                MlxDtype::Bf16,
            )
            .unwrap()
        };
        let gate = quant(inter, d);
        let up = quant(inter, d);
        let down = quant(d, inter);
        let xv: Vec<bf16> = (0..d)
            .map(|_| bf16::from_f32((next() % 2000) as f32 / 1000.0 - 1.0))
            .collect();
        let picks: Vec<u32> = (0..k as u32)
            .map(|i| (i * 7 + 2) % experts as u32)
            .collect();
        let sv: Vec<bf16> = (0..k)
            .map(|i| bf16::from_f32(0.3 - i as f32 * 0.04))
            .collect();
        let x = MlxArray::from_bf16(&[d], &xv).unwrap();
        let inds = MlxArray::from_u32(&[k], &picks).unwrap();
        let scores = MlxArray::from_bf16(&[k], &sv).unwrap();

        let got = x
            .moe_swiglu_4bit(
                &inds,
                &scores,
                [&gate.0, &gate.1, &gate.2],
                [&up.0, &up.1, &up.2],
                [&down.0, &down.1, &down.2],
                gs as i32,
            )
            .unwrap();
        assert_eq!(got.shape().unwrap(), vec![d]);

        // Reference: the decoder's unfused SwitchGLU graph.
        let idx = inds.reshape(&[1, 1, k]).unwrap();
        let gq = |x: &MlxArray, q: &(MlxArray, MlxArray, MlxArray)| {
            x.gather_qmm(
                &q.0,
                &q.1,
                Some(&q.2),
                None,
                Some(&idx),
                true,
                Some(gs as i32),
                Some(4),
                false,
            )
            .unwrap()
        };
        let x5 = x.reshape(&[1, 1, 1, 1, d]).unwrap();
        let h = gq(&x5, &gate).swiglu(&gq(&x5, &up)).unwrap();
        let want = gq(&h, &down)
            .reshape(&[1, 1, k, d])
            .unwrap()
            .multiply(&scores.reshape(&[1, 1, k, 1]).unwrap())
            .unwrap()
            .sum_axis(-2, false)
            .unwrap()
            .reshape(&[d])
            .unwrap();
        // Same accumulation order and rounding as mlx: bit-identical.
        assert_eq!(
            got.to_bf16_vec().unwrap(),
            want.to_bf16_vec().unwrap(),
            "d={d} inter={inter}"
        );
    }
}

#[test]
fn mlx_moe_router_matches_mlx_ops() {
    if aimd_mlx::device_is_cpu().expect("device query") {
        return;
    }
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // The Unlimited-OCR gate (64 experts over 1280, top-6) and a smaller
    // one; `dup` repeats weight rows so equal logits exercise tie order.
    for (d, e, k, dup) in [
        (1280usize, 64usize, 6usize, false),
        (1280, 64, 6, true),
        (512, 16, 3, true),
    ] {
        for trial in 0..8 {
            let rows: Vec<Vec<bf16>> = (0..e)
                .map(|_| {
                    (0..d)
                        .map(|_| bf16::from_f32(((next() % 2001) as f32 - 1000.0) * 4e-5))
                        .collect()
                })
                .collect();
            let wv: Vec<bf16> = (0..e)
                .flat_map(|r| rows[if dup { r / 2 * 2 } else { r }].clone())
                .collect();
            let xv: Vec<bf16> = (0..d)
                .map(|_| bf16::from_f32((next() % 2000) as f32 / 1000.0 - 1.0))
                .collect();
            let w = MlxArray::from_bf16(&[e, d], &wv).unwrap();
            let x = MlxArray::from_bf16(&[d], &xv).unwrap();
            let (inds, scores) = x.moe_router(&w, k as i32).unwrap();

            let x3 = x.reshape(&[1, 1, d]).unwrap();
            let probs = x3
                .matmul(&w.transpose(&[1, 0]).unwrap(), None)
                .unwrap()
                .softmax_precise(-1, true)
                .unwrap();
            let want_inds = probs
                .argpartition((e - k) as i32, -1)
                .unwrap()
                .slice(&[0, 0, e - k], &[1, 1, e], &[1, 1, 1])
                .unwrap();
            let want_scores = probs.take_along_axis(&want_inds, -1).unwrap();
            assert_eq!(
                inds.to_u32_vec().unwrap(),
                want_inds.to_u32_vec().unwrap(),
                "d={d} e={e} dup={dup} trial {trial}"
            );
            assert_eq!(
                scores.to_bf16_vec().unwrap(),
                want_scores.to_bf16_vec().unwrap(),
                "d={d} e={e} dup={dup} trial {trial}"
            );
        }
    }
}
