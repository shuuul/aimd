//! Checkpoint weight loading and packed-tensor sanity on MLX.

#![cfg(feature = "mlx")]

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use aimd_engine::weights::{ScaleDtype, WeightStore, load_checkpoint};
use aimd_mlx::{MlxArray, MlxDtype};

/// Integration test against a real HF checkpoint; auto-skips when the HF
/// cache is absent. Override the snapshot directory with
/// `AIMD_TEST_CHECKPOINT_DIR`.
#[test]
fn checkpoint_snapshot_integration() {
    if aimd_mlx::device_is_cpu().unwrap_or(true) {
        eprintln!("skipping: cached checkpoint test requires an MLX GPU");
        return;
    }
    let Some(store) = shared_store() else {
        eprintln!("skipping: no cached mlx-community Qwen3-ASR-1.7B-4bit snapshot");
        return;
    };

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
        let (weight, scales, biases) = MlxArray::quantized_weight(
            &tensor.payload,
            &[rows, cols / 8],
            &tensor.scales,
            &tensor.biases,
            &[rows, cols / 64],
            MlxDtype::Bf16,
        )
        .unwrap_or_else(|e| panic!("{base}: {e}"));
        let deq = weight
            .dequantize(&scales, Some(&biases), Some(64), Some(4))
            .unwrap_or_else(|e| panic!("{base}: {e}"));
        assert_eq!(deq.shape().unwrap(), vec![rows, cols]);
        assert!(
            deq.to_bf16_vec().unwrap().iter().all(|v| v.is_finite()),
            "{base}: dequantized values must be finite"
        );
    }
}

fn shared_store() -> Option<&'static WeightStore> {
    static STORE: OnceLock<Option<WeightStore>> = OnceLock::new();
    STORE
        .get_or_init(|| {
            locate_checkpoint().map(|dir| load_checkpoint(&dir).expect("checkpoint loads"))
        })
        .as_ref()
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
