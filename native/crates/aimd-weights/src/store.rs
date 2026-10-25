//! Checkpoint directory loading with safetensors mmap (WS-02).
//!
//! A checkpoint is an HF snapshot directory: one or more `*.safetensors`
//! shards plus `config.json`. All shards are memory-mapped once; tensors are
//! exposed as zero-copy byte views into the mappings. Quantized tensors stay
//! in their raw packed form until
//! [`WeightStore::quantized_tensor`] copies the `{base}`, `{base}.scales`,
//! `{base}.biases` triple into an owned [`QuantizedTensor`] for
//! dequantization.
//!
//! The `config.json` `quantization` block (e.g.
//! `{"group_size": 64, "bits": 4, "mode": "affine"}`) is parsed with support
//! for per-tensor overrides (nested objects keyed by tensor name, as written
//! by mlx for mixed-precision models). All parse/shape problems surface as
//! typed [`WeightsError`] values; malformed input never panics.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, Metadata, TensorInfo};
use serde_json::Value;

use crate::WeightsError;
use crate::mlx_quant::{QuantizedTensor, ScaleDtype};

/// `quantization` block of `config.json`.
#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    /// Default bit width for quantized tensors (4 or 8 for aimd models).
    pub bits: u32,
    /// Default quantization group size (64 for aimd models).
    pub group_size: u32,
    /// Quantization mode; `None` when the config omits it. Anything other
    /// than `"affine"` is rejected by [`WeightStore::quantized_tensor`].
    pub mode: Option<String>,
    /// Per-tensor overrides keyed by tensor base name (MLX writes these for
    /// mixed-precision checkpoints).
    pub overrides: BTreeMap<String, PerTensorQuant>,
}

/// Per-tensor quantization override (`{"bits": 8, "group_size": 64}`).
#[derive(Clone, Copy, Debug)]
pub struct PerTensorQuant {
    pub bits: u32,
    pub group_size: u32,
}

/// One memory-mapped `*.safetensors` shard.
struct Shard {
    path: PathBuf,
    mmap: Mmap,
    /// Byte offset of the tensor data blob (8-byte length + JSON header).
    data_start: usize,
    #[allow(dead_code)]
    metadata: Metadata,
}

/// Resolved tensor location: shard index plus header info.
#[derive(Clone, Debug)]
struct TensorRef {
    shard: usize,
    info: TensorInfo,
}

/// Zero-copy view over all tensors of a checkpoint snapshot directory.
///
/// Tensor metadata comes from the safetensors headers; quantized tensors
/// keep their raw form until explicitly dequantized.
pub struct WeightStore {
    root: PathBuf,
    shards: Vec<Shard>,
    index: HashMap<String, TensorRef>,
    quantization: Option<QuantizationConfig>,
}

/// Load every `*.safetensors` file under `dir` (an HF snapshot directory).
///
/// Shards are loaded in sorted file-name order; a tensor name appearing in
/// more than one shard is an error. `config.json` is parsed for the
/// quantization block when present (a checkpoint without `config.json` can
/// still serve raw tensors).
pub fn load_checkpoint(dir: &Path) -> Result<WeightStore, WeightsError> {
    if !dir.is_dir() {
        return Err(WeightsError::MissingCheckpoint(dir.display().to_string()));
    }

    let mut shard_paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| WeightsError::Io(format!("reading {}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err(WeightsError::MissingCheckpoint(format!(
            "{} (no *.safetensors files)",
            dir.display()
        )));
    }

    let mut shards = Vec::with_capacity(shard_paths.len());
    for path in shard_paths {
        shards.push(open_shard(&path)?);
    }

    let mut index = HashMap::new();
    for (shard_id, shard) in shards.iter().enumerate() {
        for (name, info) in shard.metadata.tensors() {
            if index
                .insert(
                    name.clone(),
                    TensorRef {
                        shard: shard_id,
                        info: info.clone(),
                    },
                )
                .is_some()
            {
                return Err(WeightsError::Safetensors(format!(
                    "duplicate tensor {name:?} across shards of {}",
                    dir.display()
                )));
            }
        }
    }

    let quantization = parse_quantization_config(&dir.join("config.json"))?;

    Ok(WeightStore {
        root: dir.to_path_buf(),
        shards,
        index,
        quantization,
    })
}

fn open_shard(path: &Path) -> Result<Shard, WeightsError> {
    let file = fs::File::open(path)
        .map_err(|e| WeightsError::Io(format!("opening {}: {e}", path.display())))?;
    // Safety: the file is opened read-only and never mutated by this process.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| WeightsError::Io(format!("mapping {}: {e}", path.display())))?;
    let (header_len, metadata) = SafeTensors::read_metadata(&mmap[..])
        .map_err(|e| WeightsError::Safetensors(format!("{}: {e}", path.display())))?;
    Ok(Shard {
        path: path.to_path_buf(),
        data_start: 8 + header_len,
        mmap,
        metadata,
    })
}

/// Parse the optional `quantization` block of `config.json`.
///
/// Unknown keys are tolerated; nested object values that carry `bits` are
/// treated as per-tensor overrides.
fn parse_quantization_config(path: &Path) -> Result<Option<QuantizationConfig>, WeightsError> {
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(None); // no config.json (or unreadable) -> raw-tensor store only
    };
    let root: Value = serde_json::from_str(&text)
        .map_err(|e| WeightsError::Json(format!("{}: {e}", path.display())))?;
    let Some(block) = root.get("quantization") else {
        return Ok(None);
    };
    let obj = block.as_object().ok_or_else(|| {
        WeightsError::Json(format!("{}: quantization is not an object", path.display()))
    })?;

    let bits = obj.get("bits").and_then(Value::as_u64).ok_or_else(|| {
        WeightsError::Json(format!("{}: quantization.bits missing", path.display()))
    })?;
    let group_size = obj
        .get("group_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            WeightsError::Json(format!(
                "{}: quantization.group_size missing",
                path.display()
            ))
        })?;
    let bits = u32::try_from(bits).map_err(|_| {
        WeightsError::UnsupportedQuant(format!("quantization.bits {bits} out of range"))
    })?;
    let group_size = u32::try_from(group_size).map_err(|_| {
        WeightsError::UnsupportedQuant(format!("quantization.group_size {group_size} out of range"))
    })?;
    let mode = obj
        .get("mode")
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase);

    let mut overrides = BTreeMap::new();
    for (key, value) in obj {
        if matches!(key.as_str(), "bits" | "group_size" | "mode") {
            continue;
        }
        let Some(entry) = value.as_object() else {
            continue; // unknown scalar extension; ignore
        };
        let (Some(bits), Some(group_size)) = (
            entry.get("bits").and_then(Value::as_u64),
            entry.get("group_size").and_then(Value::as_u64),
        ) else {
            continue;
        };
        overrides.insert(
            key.clone(),
            PerTensorQuant {
                bits: u32::try_from(bits).unwrap_or(0),
                group_size: u32::try_from(group_size).unwrap_or(0),
            },
        );
    }

    Ok(Some(QuantizationConfig {
        bits,
        group_size,
        mode,
        overrides,
    }))
}

impl WeightStore {
    /// Snapshot directory on disk.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Parsed `config.json` quantization block, if the checkpoint has one.
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization.as_ref()
    }

    /// All tensor names across every shard, sorted for determinism.
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.index.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// safetensors header info for one tensor.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|r| &r.info)
    }

    /// Raw little-endian bytes of one tensor (zero-copy into the mmap).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightsError> {
        let tensor = self
            .index
            .get(name)
            .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))?;
        let shard = &self.shards[tensor.shard];
        let (start, end) = tensor.info.data_offsets;
        shard.mmap[..]
            .get(shard.data_start + start..shard.data_start + end)
            .ok_or_else(|| {
                WeightsError::InvalidTensor(format!(
                    "tensor {name:?} offsets {start}..{end} out of bounds in {}",
                    shard.path.display()
                ))
            })
    }

    /// Names of quantized module paths: every tensor `<m>.scales` for which
    /// the matching payload (`<m>.weight`, or `<m>` for raw tuples) and
    /// `<m>.biases` exist, sorted.
    pub fn quantized_bases(&self) -> Vec<String> {
        let mut bases: Vec<String> = self
            .index
            .keys()
            .filter_map(|name| name.strip_suffix(".scales"))
            .filter(|base| {
                self.index.contains_key(&format!("{base}.biases"))
                    && (self.index.contains_key(&format!("{base}.weight"))
                        || self.index.contains_key(*base))
            })
            .map(str::to_string)
            .collect();
        bases.sort();
        bases.dedup();
        bases
    }

    /// Assemble the MLX affine quantization triple of one module into an
    /// owned [`QuantizedTensor`].
    ///
    /// MLX NN saves a quantized module `<m>` as `<m>.weight` (packed U32
    /// payload), `<m>.scales` and `<m>.biases` (f16/bf16). `base` may be:
    /// the module path (`<m>`), the payload tensor name (`<m>.weight`), or a
    /// raw tuple name (`X` with `X.scales` / `X.biases` alongside). The
    /// payload must be `U32`; scales and biases must both be `F16` or both
    /// `BF16`.
    pub fn quantized_tensor(&self, base: &str) -> Result<QuantizedTensor, WeightsError> {
        let (payload_name, scales_name, biases_name) = self.resolve_triple(base)?;
        if !self.index.contains_key(&biases_name) {
            return Err(WeightsError::MissingTensor(biases_name));
        }

        let module = payload_name
            .strip_suffix(".weight")
            .unwrap_or(&payload_name);
        let (bits, group_size) = self.quant_params(&payload_name, module)?;
        let payload_info = &self.index[&payload_name].info;
        let scales_info = &self.index[&scales_name].info;
        let biases_info = &self.index[&biases_name].info;

        if payload_info.dtype != Dtype::U32 {
            return Err(WeightsError::UnsupportedQuant(format!(
                "payload {payload_name:?} dtype {:?}, expected U32",
                payload_info.dtype
            )));
        }
        let scale_dtype = scale_dtype_of(&scales_name, scales_info.dtype)?;
        let biases_dtype = scale_dtype_of(&biases_name, biases_info.dtype)?;
        if scale_dtype != biases_dtype {
            return Err(WeightsError::InvalidTensor(format!(
                "scales dtype {} != biases dtype {} for {payload_name:?}",
                scale_dtype.as_str(),
                biases_dtype.as_str()
            )));
        }
        let [payload_shape @ .., words] = payload_info.shape.as_slice() else {
            return Err(WeightsError::InvalidTensor(format!(
                "payload {payload_name:?} must be at least 1-D, got {:?}",
                payload_info.shape
            )));
        };
        let [scale_rows, groups] = scales_info.shape.as_slice() else {
            return Err(WeightsError::InvalidTensor(format!(
                "scales {scales_name:?} must be 2-D, got {:?}",
                scales_info.shape
            )));
        };
        if scales_info.shape != biases_info.shape {
            return Err(WeightsError::InvalidTensor(format!(
                "scales shape {:?} != biases shape {:?} for {payload_name:?}",
                scales_info.shape, biases_info.shape
            )));
        }
        // MLX packs the last axis; all leading axes are rows.
        let rows: usize = payload_shape.iter().product();
        if rows != *scale_rows || scale_rows == &0usize {
            return Err(WeightsError::InvalidTensor(format!(
                "payload rows {rows} != scales rows {scale_rows} for {payload_name:?}"
            )));
        }
        let bits_usize = bits as usize;
        let group_size_usize = group_size as usize;
        if group_size_usize == 0 || groups == &0usize {
            return Err(WeightsError::InvalidTensor(format!(
                "scales groups {groups} invalid for {payload_name:?}"
            )));
        }
        let cols = groups * group_size_usize;
        let expected_words = cols
            .checked_mul(bits_usize)
            .and_then(|n| n.checked_div(32))
            .ok_or_else(|| WeightsError::InvalidTensor("shape overflows usize".to_string()))?;
        if words != &expected_words {
            return Err(WeightsError::InvalidTensor(format!(
                "payload words {words} != expected {expected_words} for {payload_name:?} \
                 (cols {cols}, bits {bits})"
            )));
        }

        let payload = self.tensor_bytes(&payload_name)?.to_vec();
        let scales_raw = self.tensor_bytes(&scales_name)?.to_vec();
        let biases_raw = self.tensor_bytes(&biases_name)?.to_vec();
        if scales_raw.len() % 2 != 0 || biases_raw.len() % 2 != 0 {
            return Err(WeightsError::InvalidTensor(format!(
                "scales/biases byte length must be even for {payload_name:?}"
            )));
        }
        let scales = u16_vec_le(&scales_raw);
        let biases = u16_vec_le(&biases_raw);
        if scales.len() != rows * groups {
            return Err(WeightsError::InvalidTensor(format!(
                "scales element count {} != rows*groups {} for {payload_name:?}",
                scales.len(),
                rows * groups
            )));
        }

        Ok(QuantizedTensor {
            bits,
            group_size,
            shape: {
                let mut shape = payload_shape.to_vec();
                shape.push(cols);
                shape
            },
            scale_dtype,
            payload,
            scales,
            biases,
        })
    }

    /// Resolve the (payload, scales, biases) tensor names for a module path,
    /// payload name, or raw tuple name.
    ///
    /// Resolution order:
    /// 1. `{base}.scales` exists: module path (`<m>`: payload `<m>.weight`,
    ///    scales `<m>.scales`) or raw tuple (`X`: payload `X`, scales
    ///    `X.scales`).
    /// 2. `base` ends in `.weight` and `{module}.scales` exists (module =
    ///    base without the suffix): module path given as payload name.
    /// 3. Otherwise: missing tensor.
    fn resolve_triple(&self, base: &str) -> Result<(String, String, String), WeightsError> {
        let scales_of = |b: &str| format!("{b}.scales");
        if self.index.contains_key(&scales_of(base)) {
            let module_payload = format!("{base}.weight");
            let payload = if self.index.contains_key(&module_payload) {
                module_payload
            } else {
                base.to_string()
            };
            return Ok((payload, scales_of(base), format!("{base}.biases")));
        }
        if let Some(module) = base.strip_suffix(".weight")
            && self.index.contains_key(&scales_of(module))
        {
            return Ok((
                base.to_string(),
                scales_of(module),
                format!("{module}.biases"),
            ));
        }
        Err(WeightsError::MissingTensor(base.to_string()))
    }

    /// Quantization parameters for one tensor from the parsed config.
    ///
    /// Per-tensor overrides are keyed by the payload tensor name (`<m>.weight`)
    /// or the module path (`<m>`).
    fn quant_params(&self, payload_name: &str, module: &str) -> Result<(u32, u32), WeightsError> {
        let Some(config) = self.quantization.as_ref() else {
            return Err(WeightsError::UnsupportedQuant(format!(
                "checkpoint {} has no config.json quantization block",
                self.root.display()
            )));
        };
        if let Some(mode) = config.mode.as_deref()
            && mode != "affine"
        {
            return Err(WeightsError::UnsupportedQuant(format!(
                "quantization mode {mode:?}, expected \"affine\""
            )));
        }
        match config
            .overrides
            .get(payload_name)
            .or_else(|| config.overrides.get(module))
        {
            Some(override_q) => Ok((override_q.bits, override_q.group_size)),
            None => Ok((config.bits, config.group_size)),
        }
    }
}

fn scale_dtype_of(name: &str, dtype: Dtype) -> Result<ScaleDtype, WeightsError> {
    match dtype {
        Dtype::F16 => Ok(ScaleDtype::F16),
        Dtype::BF16 => Ok(ScaleDtype::Bf16),
        other => Err(WeightsError::UnsupportedQuant(format!(
            "{name:?} dtype {other:?}, expected F16 or BF16"
        ))),
    }
}

fn u16_vec_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    /// Build a minimal valid safetensors file: 8-byte LE header length +
    /// JSON header + packed data blob.
    fn serialize_fake(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_json = serde_json::to_string(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn write(path: &Path, bytes: &[u8]) {
        let mut file = fs::File::create(path).unwrap();
        file.write_all(bytes).unwrap();
    }

    /// One quantized tensor: logical (2, 8), 4-bit, group 4.
    /// Payload (2, 1) u32 (one word per row); scales/biases (2, 2) f16.
    fn sample_checkpoint(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        let payload: Vec<u8> = vec![0x10, 0x32, 0x0F, 0x21, 0x00, 0x00, 0x00, 0x00];
        let scales: Vec<u8> = [0x3C00u16, 0x3800, 0x4000, 0x4000] // r0: 1.0, 0.5; r1: 2.0, 2.0
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let biases: Vec<u8> = vec![0u8; 8];
        let file = serialize_fake(&[
            ("model.blk0.weight", "U32", &[2, 1], payload),
            ("model.blk0.scales", "F16", &[2, 2], scales),
            ("model.blk0.biases", "F16", &[2, 2], biases),
            ("model.norm.weight", "F16", &[8], vec![0u8; 16]),
        ]);
        write(&dir.join("model-00001.safetensors"), &file);
        write(
            &dir.join("config.json"),
            br#"{"quantization": {"group_size": 4, "bits": 4, "mode": "affine"}}"#,
        );
    }

    #[test]
    fn loads_multi_shard_checkpoint() {
        let dir = std::env::temp_dir().join("aimd-weights-test-multi-shard");
        let _ = fs::remove_dir_all(&dir);
        sample_checkpoint(&dir);
        // second shard with a second, plain tensor
        let file = serialize_fake(&[("other.tensor", "U32", &[1], vec![1, 2, 3, 4])]);
        write(&dir.join("model-00002.safetensors"), &file);

        let store = load_checkpoint(&dir).unwrap();
        assert_eq!(store.root(), dir);
        assert_eq!(
            store.tensor_names(),
            vec![
                "model.blk0.biases",
                "model.blk0.scales",
                "model.blk0.weight",
                "model.norm.weight",
                "other.tensor"
            ]
        );
        assert_eq!(store.tensor_bytes("other.tensor").unwrap(), &[1, 2, 3, 4]);
        assert!(store.tensor_bytes("nope").is_err());

        let quant = store.quantization().unwrap();
        assert_eq!((quant.bits, quant.group_size), (4, 4));
        assert_eq!(quant.mode.as_deref(), Some("affine"));

        let bases = store.quantized_bases();
        assert_eq!(bases, vec!["model.blk0"]);

        // module path, and the payload name, both resolve to the same triple
        let tensor = store.quantized_tensor("model.blk0").unwrap();
        assert_eq!(tensor.shape, vec![2, 8]);
        let tensor = store.quantized_tensor("model.blk0.weight").unwrap();
        assert_eq!(tensor.bits, 4);
        assert_eq!(tensor.group_size, 4);
        assert_eq!(tensor.shape, vec![2, 8]);
        assert_eq!(tensor.scale_dtype, ScaleDtype::F16);
        let deq = tensor.dequantize().unwrap();
        assert_eq!(deq.shape, vec![2, 8]);
        // row 0 group 0: elements 0..4 = 0..3 * 1.0
        assert_eq!(&deq.data[0..4], &[0x0000, 0x3C00, 0x4000, 0x4200]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_missing_dir_and_shards() {
        assert!(matches!(
            load_checkpoint(Path::new("/definitely/not/here-aimd")),
            Err(WeightsError::MissingCheckpoint(_))
        ));
        let dir = std::env::temp_dir().join("aimd-weights-test-empty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(matches!(
            load_checkpoint(&dir),
            Err(WeightsError::MissingCheckpoint(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_duplicate_tensor_across_shards() {
        let dir = std::env::temp_dir().join("aimd-weights-test-dup");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = serialize_fake(&[("t", "U32", &[1], vec![0, 0, 0, 0])]);
        write(&dir.join("a.safetensors"), &file);
        write(&dir.join("b.safetensors"), &file);
        assert!(matches!(
            load_checkpoint(&dir),
            Err(WeightsError::Safetensors(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_malformed_safetensors() {
        let dir = std::env::temp_dir().join("aimd-weights-test-malformed");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write(&dir.join("bad.safetensors"), b"nonsense");
        assert!(matches!(
            load_checkpoint(&dir),
            Err(WeightsError::Safetensors(_))
        ));
        // truncated header: length promises more bytes than the file has
        write(&dir.join("bad.safetensors"), &0x0100u64.to_le_bytes());
        assert!(matches!(
            load_checkpoint(&dir),
            Err(WeightsError::Safetensors(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_unsupported_mode_and_dtypes() {
        let dir = std::env::temp_dir().join("aimd-weights-test-mode");
        let _ = fs::remove_dir_all(&dir);
        sample_checkpoint(&dir);
        write(
            &dir.join("config.json"),
            br#"{"quantization": {"group_size": 4, "bits": 4, "mode": "mxfp4"}}"#,
        );
        let store = load_checkpoint(&dir).unwrap();
        assert!(matches!(
            store.quantized_tensor("model.blk0"),
            Err(WeightsError::UnsupportedQuant(_))
        ));
        // f32 scales are rejected
        let payload: Vec<u8> = vec![0x10, 0x32, 0x0F, 0x21, 0x00, 0x00, 0x00, 0x00];
        let file = serialize_fake(&[
            ("bad.weight", "U32", &[2, 1], payload),
            ("bad.weight.scales", "F32", &[2, 2], vec![0u8; 16]),
            ("bad.weight.biases", "F32", &[2, 2], vec![0u8; 16]),
        ]);
        write(&dir.join("model-00001.safetensors"), &file);
        let store = load_checkpoint(&dir).unwrap();
        assert!(matches!(
            store.quantized_tensor("bad.weight"),
            Err(WeightsError::UnsupportedQuant(_))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn supports_per_tensor_overrides() {
        let dir = std::env::temp_dir().join("aimd-weights-test-override");
        let _ = fs::remove_dir_all(&dir);
        sample_checkpoint(&dir);
        write(
            &dir.join("config.json"),
            br#"{"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                "model.blk0.weight": {"group_size": 4, "bits": 4}}}"#,
        );
        let store = load_checkpoint(&dir).unwrap();
        let tensor = store.quantized_tensor("model.blk0").unwrap();
        assert_eq!((tensor.bits, tensor.group_size), (4, 4));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Raw tuples saved without a `.weight` payload (payload `X` + `X.scales`
    /// + `X.biases`) also assemble.
    #[test]
    fn assembles_raw_tuples() {
        let dir = std::env::temp_dir().join("aimd-weights-test-raw-tuple");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let payload: Vec<u8> = vec![0x10, 0x32, 0x0F, 0x21, 0x00, 0x00, 0x00, 0x00];
        let scales: Vec<u8> = [0x3C00u16, 0x3800, 0x4000, 0x4000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file = serialize_fake(&[
            ("block.a", "U32", &[2, 1], payload),
            ("block.a.scales", "F16", &[2, 2], scales),
            ("block.a.biases", "F16", &[2, 2], vec![0u8; 8]),
        ]);
        write(&dir.join("model.safetensors"), &file);
        write(
            &dir.join("config.json"),
            br#"{"quantization": {"group_size": 4, "bits": 4, "mode": "affine"}}"#,
        );
        let store = load_checkpoint(&dir).unwrap();
        assert_eq!(store.quantized_bases(), vec!["block.a"]);
        let tensor = store.quantized_tensor("block.a").unwrap();
        assert_eq!(tensor.shape, vec![2, 8]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_config_still_serves_raw_tensors() {
        let dir = std::env::temp_dir().join("aimd-weights-test-noconfig");
        let _ = fs::remove_dir_all(&dir);
        sample_checkpoint(&dir);
        fs::remove_file(dir.join("config.json")).unwrap();
        let store = load_checkpoint(&dir).unwrap();
        assert!(store.quantization().is_none());
        assert!(matches!(
            store.quantized_tensor("model.blk0.weight"),
            Err(WeightsError::UnsupportedQuant(_))
        ));
        assert!(store.tensor_bytes("model.norm.weight").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }
}
