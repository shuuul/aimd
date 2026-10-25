//! Model precision parsing (ports `aimd.core.precision`).

use std::fmt;

use crate::error::{AimdError, Result};
use crate::pyfmt::repr;

/// Canonical precision values in display order.
pub const SUPPORTED_PRECISIONS: [&str; 4] = ["4bit", "6bit", "8bit", "bf16"];
/// Affine-quantized precisions.
pub const QUANTIZED_PRECISIONS: [&str; 3] = ["4bit", "6bit", "8bit"];

/// A normalized precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Bit4,
    Bit6,
    Bit8,
    Bf16,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Bit4 => "4bit",
            Precision::Bit6 => "6bit",
            Precision::Bit8 => "8bit",
            Precision::Bf16 => "bf16",
        }
    }

    /// Canonical spelling (`4bit`, ...) to a value; no normalization.
    pub fn from_canonical(value: &str) -> Option<Precision> {
        match value {
            "4bit" => Some(Precision::Bit4),
            "6bit" => Some(Precision::Bit6),
            "8bit" => Some(Precision::Bit8),
            "bf16" => Some(Precision::Bf16),
            _ => None,
        }
    }

    pub fn is_quantized(self) -> bool {
        self != Precision::Bf16
    }

    /// Quantization bit width (`None` for bf16).
    pub fn bits(self) -> Option<u32> {
        match self {
            Precision::Bit4 => Some(4),
            Precision::Bit6 => Some(6),
            Precision::Bit8 => Some(8),
            Precision::Bf16 => None,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Normalize a user-supplied precision (`4-bit`, `BF16`, ` 8 bit`) or
/// reject it with the Python `ProcessingFailedError` message.
pub fn normalize_precision(precision: Option<&str>) -> Result<Option<Precision>> {
    let Some(raw) = precision else {
        return Ok(None);
    };
    let normalized = raw
        .trim_matches(crate::pyfmt::is_space)
        .to_lowercase()
        .replace(['-', ' '], "");
    Precision::from_canonical(&normalized)
        .map(Some)
        .ok_or_else(|| {
            AimdError::failed(format!(
                "Unsupported precision {}. Supported precisions: {}.",
                repr(raw),
                SUPPORTED_PRECISIONS.join(", ")
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn normalizes_dash_space_and_case_variants() {
        assert_eq!(normalize_precision(None).unwrap(), None);
        assert_eq!(
            normalize_precision(Some("4-bit")).unwrap(),
            Some(Precision::Bit4)
        );
        assert_eq!(
            normalize_precision(Some(" 6 BIT ")).unwrap(),
            Some(Precision::Bit6)
        );
        assert_eq!(
            normalize_precision(Some("BF16")).unwrap(),
            Some(Precision::Bf16)
        );
        assert_eq!(
            normalize_precision(Some("8bit")).unwrap(),
            Some(Precision::Bit8)
        );
    }

    #[test]
    fn rejects_unknown_precision_with_python_message() {
        let err = normalize_precision(Some("fp8")).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ProcessingFailed);
        assert_eq!(
            err.message,
            "Unsupported precision 'fp8'. Supported precisions: 4bit, 6bit, 8bit, bf16."
        );
    }
}
