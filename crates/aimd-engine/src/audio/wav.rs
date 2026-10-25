//! Hand-written RIFF/WAVE parser for the pipeline wav contract.
//!
//! The outer pipeline ffmpeg-normalizes audio to 16 kHz mono wav; this
//! parser accepts exactly that contract - little-endian PCM s16le or IEEE
//! f32le, mono, 16 kHz (plain or `WAVE_FORMAT_EXTENSIBLE`) - and rejects
//! everything else with typed [`AudioError`]s instead of guessing.
//!
//! s16 samples are scaled by `1/32768` (the same mapping ffmpeg uses), so
//! they cover `[-1, 1)` exactly. f32 samples are passed through unclamped;
//! normalized pipeline audio keeps them within `[-1, 1]`.

use std::path::{Path, PathBuf};

use super::error::AudioError;

/// Decoded wav contents.
#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    /// Sample rate in Hz (always 16000 for accepted files).
    pub sample_rate: u32,
    /// Channel count (always 1 for accepted files).
    pub channels: u16,
    /// Interleaved samples, normalized to `[-1, 1]` (mono: one per frame).
    pub samples: Vec<f32>,
}

/// RIFF chunk ids handled specially; everything else is skipped.
const FMT: &[u8; 4] = b"fmt ";
const DATA: &[u8; 4] = b"data";

/// `WAVE_FORMAT_EXTENSIBLE`: the real format lives in the SubFormat GUID.
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// Standard KSDATAFORMAT GUID tail shared by PCM / IEEE float subformats.
const SUBFORMAT_TAIL: [u8; 14] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const SUBFORMAT_PCM: u16 = 0x0001;
const SUBFORMAT_IEEE_FLOAT: u16 = 0x0003;

/// Load and decode a wav file.
pub fn load_wav(path: &Path) -> Result<WavAudio, AudioError> {
    let bytes = std::fs::read(path).map_err(|source| AudioError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_wav(&bytes, path)
}

#[derive(Debug, Clone, Copy)]
struct FmtInfo {
    format_code: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

#[derive(Debug, Clone, Copy)]
struct DataChunk {
    offset: usize,
    size: usize,
}

/// Parse an in-memory RIFF/WAVE payload.
pub(crate) fn parse_wav(bytes: &[u8], path: &Path) -> Result<WavAudio, AudioError> {
    let len = bytes.len() as u64;

    if bytes.len() < 12 {
        return Err(AudioError::Truncated {
            path: path.to_path_buf(),
            chunk: "RIFF header".into(),
            needed: 12,
            available: len,
        });
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(AudioError::NotRiff {
            path: path.to_path_buf(),
        });
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(AudioError::NotWave {
            path: path.to_path_buf(),
        });
    }

    let mut fmt: Option<FmtInfo> = None;
    let mut data: Option<DataChunk> = None;

    let mut pos: usize = 12;
    while pos < bytes.len() {
        if bytes.len() - pos < 8 {
            return Err(AudioError::Truncated {
                path: path.to_path_buf(),
                chunk: "chunk header".into(),
                needed: pos as u64 + 8,
                available: len,
            });
        }
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as u64;
        let body_start = pos + 8;
        let body_end = body_start as u64 + size;
        if body_end > len {
            let id_str = String::from_utf8_lossy(id).into_owned();
            return Err(AudioError::Truncated {
                path: path.to_path_buf(),
                chunk: id_str,
                needed: body_end,
                available: len,
            });
        }
        let body = &bytes[body_start..body_end as usize];
        if id == FMT {
            fmt = Some(parse_fmt(body, path)?);
        } else if id == DATA {
            data = Some(DataChunk {
                offset: body_start,
                size: body.len(),
            });
        }
        // Chunk bodies are word-aligned: odd sizes carry one pad byte.
        pos = body_end as usize + (size as usize & 1);
    }

    let fmt = fmt.ok_or_else(|| AudioError::MissingFmtChunk {
        path: path.to_path_buf(),
    })?;
    let data = data.ok_or_else(|| AudioError::MissingDataChunk {
        path: path.to_path_buf(),
    })?;

    validate_fmt(&fmt, path)?;
    if data.size % fmt.block_align as usize != 0 {
        return Err(AudioError::PartialDataFrame {
            path: path.to_path_buf(),
            size: data.size as u32,
            block_align: fmt.block_align,
        });
    }

    let payload = &bytes[data.offset..data.offset + data.size];
    let samples = match (fmt.format_code, fmt.bits_per_sample) {
        (SUBFORMAT_PCM, 16) => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
            .collect(),
        (SUBFORMAT_IEEE_FLOAT, 32) => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        _ => unreachable!("validate_fmt rejects other format/bit combinations"),
    };

    Ok(WavAudio {
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        samples,
    })
}

/// Parse the `fmt ` chunk body, resolving `WAVE_FORMAT_EXTENSIBLE`.
fn parse_fmt(body: &[u8], path: &Path) -> Result<FmtInfo, AudioError> {
    let truncated = || AudioError::Truncated {
        path: path.to_path_buf(),
        chunk: "fmt ".into(),
        needed: 16,
        available: body.len() as u64,
    };
    if body.len() < 16 {
        return Err(truncated());
    }
    let format_code = u16::from_le_bytes([body[0], body[1]]);
    let channels = u16::from_le_bytes([body[2], body[3]]);
    let sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    let block_align = u16::from_le_bytes([body[12], body[13]]);
    let bits_per_sample = u16::from_le_bytes([body[14], body[15]]);

    if format_code == WAVE_FORMAT_EXTENSIBLE {
        // Extension: u16 cb_size, u16 valid_bits, u32 channel_mask,
        // 16-byte SubFormat GUID (LE u16 code + standard tail).
        if body.len() < 40 || u16::from_le_bytes([body[16], body[17]]) < 22 {
            return Err(AudioError::InvalidFmtChunk {
                path: path.to_path_buf(),
                size: body.len() as u32,
            });
        }
        if body[26..40] != SUBFORMAT_TAIL {
            return Err(AudioError::UnsupportedFormat {
                path: path.to_path_buf(),
                code: format_code,
            });
        }
        let sub = u16::from_le_bytes([body[24], body[25]]);
        if sub != SUBFORMAT_PCM && sub != SUBFORMAT_IEEE_FLOAT {
            return Err(AudioError::UnsupportedFormat {
                path: path.to_path_buf(),
                code: sub,
            });
        }
        return Ok(FmtInfo {
            format_code: sub,
            channels,
            sample_rate,
            block_align,
            bits_per_sample,
        });
    }

    if format_code != SUBFORMAT_PCM && format_code != SUBFORMAT_IEEE_FLOAT {
        return Err(AudioError::UnsupportedFormat {
            path: path.to_path_buf(),
            code: format_code,
        });
    }
    Ok(FmtInfo {
        format_code,
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
    })
}

/// Enforce the pipeline contract: 16 kHz mono s16le or f32le.
fn validate_fmt(fmt: &FmtInfo, path: &Path) -> Result<(), AudioError> {
    let path_buf: PathBuf = path.to_path_buf();
    match (fmt.format_code, fmt.bits_per_sample) {
        (SUBFORMAT_PCM, 16) | (SUBFORMAT_IEEE_FLOAT, 32) => {}
        (code, bits) => {
            return Err(AudioError::UnsupportedBitsPerSample {
                path: path_buf,
                bits,
                code,
            });
        }
    }
    if fmt.channels != 1 {
        return Err(AudioError::UnsupportedChannels {
            path: path_buf,
            actual: fmt.channels,
        });
    }
    if fmt.sample_rate != 16000 {
        return Err(AudioError::UnsupportedSampleRate {
            path: path_buf,
            actual: fmt.sample_rate,
        });
    }
    if fmt.block_align != fmt.bits_per_sample / 8 * fmt.channels {
        return Err(AudioError::InvalidFmtChunk {
            path: path_buf,
            size: fmt.bits_per_sample as u32,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AudioError, WavAudio, load_wav, parse_wav};
    use std::io::Write;
    use std::path::PathBuf;

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut c = Vec::with_capacity(8 + body.len() + (body.len() & 1));
        c.extend_from_slice(id);
        c.extend_from_slice(&(body.len() as u32).to_le_bytes());
        c.extend_from_slice(body);
        if body.len() % 2 == 1 {
            c.push(0); // word-alignment pad byte
        }
        c
    }

    fn riff(form: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut v = vec![0u8; 8];
        v.extend_from_slice(form);
        for c in chunks {
            v.extend_from_slice(c);
        }
        let size = (v.len() - 8) as u32;
        v[4..8].copy_from_slice(&size.to_le_bytes());
        v[..4].copy_from_slice(b"RIFF");
        v
    }

    fn fmt_body(format: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let mut b = Vec::new();
        b.extend_from_slice(&format.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * block_align as u32).to_le_bytes());
        b.extend_from_slice(&block_align.to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b
    }

    fn extensible_fmt_body(subformat: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let mut b = fmt_body(0xFFFE, channels, rate, bits);
        b.extend_from_slice(&22u16.to_le_bytes()); // cb_size
        b.extend_from_slice(&bits.to_le_bytes()); // valid_bits
        b.extend_from_slice(&0u32.to_le_bytes()); // channel_mask
        b.extend_from_slice(&subformat.to_le_bytes()); // SubFormat GUID code
        b.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
        b
    }

    fn s16_data(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn f32_data(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn ok(bytes: &[u8]) -> WavAudio {
        parse_wav(bytes, std::path::Path::new("test.wav")).expect("parse succeeds")
    }

    fn err(bytes: &[u8]) -> AudioError {
        parse_wav(bytes, std::path::Path::new("test.wav")).expect_err("parse fails")
    }

    #[test]
    fn audio_parse_wav_s16le_mono_16k() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[0, 16384, -32768, 32767])),
            ],
        );
        let wav = ok(&bytes);
        assert_eq!(wav.sample_rate, 16000);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.samples, vec![0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn audio_parse_wav_f32le_mono_16k() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(3, 1, 16000, 32)),
                chunk(b"data", &f32_data(&[0.25, -0.75, 1.0, -1.0])),
            ],
        );
        let wav = ok(&bytes);
        assert_eq!(wav.samples, vec![0.25, -0.75, 1.0, -1.0]);
    }

    #[test]
    fn audio_parse_wav_accepts_extensible_formats() {
        let pcm = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &extensible_fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[4096])),
            ],
        );
        assert_eq!(ok(&pcm).samples, vec![0.125]);

        let float = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &extensible_fmt_body(3, 1, 16000, 32)),
                chunk(b"data", &f32_data(&[-0.5])),
            ],
        );
        assert_eq!(ok(&float).samples, vec![-0.5]);
    }

    #[test]
    fn audio_parse_wav_skips_unknown_chunks() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"JUNK", &[0xAB; 5]), // odd size exercises the pad byte
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"LIST", &[0u8; 3]),
                chunk(b"data", &s16_data(&[-16384])),
            ],
        );
        assert_eq!(ok(&bytes).samples, vec![-0.5]);
    }

    #[test]
    fn audio_parse_wav_accepts_empty_data() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &[]),
            ],
        );
        assert!(ok(&bytes).samples.is_empty());
    }

    #[test]
    fn audio_parse_wav_rejects_wrong_sample_rate() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 44100, 16)),
                chunk(b"data", &s16_data(&[0])),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedSampleRate { actual: 44100, .. }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_stereo() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 2, 16000, 16)),
                chunk(b"data", &s16_data(&[0, 0])),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedChannels { actual: 2, .. }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_unknown_format() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(6, 1, 16000, 8)),
                chunk(b"data", &[0u8]),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedFormat { code: 6, .. }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_bad_bits_per_sample() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 8)),
                chunk(b"data", &[0u8]),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedBitsPerSample {
                bits: 8,
                code: 1,
                ..
            }
        ));
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(3, 1, 16000, 64)),
                chunk(b"data", &f32_data(&[0.0, 0.0])),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedBitsPerSample {
                bits: 64,
                code: 3,
                ..
            }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_extensible_unknown_subformat() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &extensible_fmt_body(0x0303, 1, 16000, 32)),
                chunk(b"data", &f32_data(&[0.0])),
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::UnsupportedFormat { code: 0x0303, .. }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_truncated_data_chunk() {
        let mut bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[1, 2, 3])),
            ],
        );
        bytes.truncate(bytes.len() - 3); // cut into the declared data payload
        assert!(matches!(err(&bytes), AudioError::Truncated { chunk, .. } if chunk == "data"));
    }

    #[test]
    fn audio_parse_wav_rejects_partial_frame() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &[0x01, 0x02, 0x03]), // 3 bytes, block_align 2
            ],
        );
        assert!(matches!(
            err(&bytes),
            AudioError::PartialDataFrame {
                size: 3,
                block_align: 2,
                ..
            }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_missing_chunks() {
        let fmt_only = riff(b"WAVE", &[chunk(b"fmt ", &fmt_body(1, 1, 16000, 16))]);
        assert!(matches!(
            err(&fmt_only),
            AudioError::MissingDataChunk { .. }
        ));

        let data_only = riff(b"WAVE", &[chunk(b"data", &s16_data(&[0]))]);
        assert!(matches!(
            err(&data_only),
            AudioError::MissingFmtChunk { .. }
        ));
    }

    #[test]
    fn audio_parse_wav_rejects_bad_container() {
        let bytes = riff(b"AVI ", &[chunk(b"fmt ", &fmt_body(1, 1, 16000, 16))]);
        assert!(matches!(err(&bytes), AudioError::NotWave { .. }));

        let mut bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[0])),
            ],
        );
        bytes[0..4].copy_from_slice(b"RIFX");
        assert!(matches!(err(&bytes), AudioError::NotRiff { .. }));

        let truncated = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[0])),
            ],
        )[..6]
            .to_vec();
        assert!(
            matches!(err(&truncated), AudioError::Truncated { chunk, .. } if chunk == "RIFF header")
        );
    }

    #[test]
    fn audio_load_wav_file_roundtrip() {
        let bytes = riff(
            b"WAVE",
            &[
                chunk(b"fmt ", &fmt_body(1, 1, 16000, 16)),
                chunk(b"data", &s16_data(&[-32768, 32767])),
            ],
        );
        let path: PathBuf =
            std::env::temp_dir().join(format!("aimd_audio_test_{}_s16.wav", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let wav = load_wav(&path).unwrap();
        assert_eq!(wav.samples, vec![-1.0, 32767.0 / 32768.0]);
        std::fs::remove_file(&path).unwrap();

        let missing = load_wav(std::path::Path::new("/nonexistent/aimd_audio_test.wav"));
        assert!(matches!(missing, Err(AudioError::Io { .. })));
    }
}
