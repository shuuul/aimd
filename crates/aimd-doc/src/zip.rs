//! Minimal ZIP reader: central directory + stored/deflate entries.
//! Hand-written per the WS-15 dependency policy (inflate via `miniz_oxide`).

use std::path::{Component, Path, PathBuf};

use miniz_oxide::inflate::decompress_to_vec_with_limit;

struct Entry {
    name: String,
    method: u16,
    flags: u16,
    crc: u32,
    csize: usize,
    usize_: usize,
    local_offset: usize,
}

pub(crate) struct ZipArchive {
    data: Vec<u8>,
    entries: Vec<Entry>,
}

fn u16_at(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(o..o + 2)?.try_into().ok()?))
}
fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = !0u32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

impl ZipArchive {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        Self::parse(data)
    }

    pub(crate) fn parse(data: Vec<u8>) -> Result<Self, String> {
        const BAD: &str = "File is not a zip file";
        if data.len() < 22 {
            return Err(BAD.into());
        }
        let lowest = data.len().saturating_sub(22 + 65535);
        let mut eocd = None;
        let mut i = data.len() - 22;
        loop {
            if u32_at(&data, i) == Some(0x0605_4b50) {
                eocd = Some(i);
                break;
            }
            if i == lowest {
                break;
            }
            i -= 1;
        }
        let eocd = eocd.ok_or(BAD)?;
        let total = u16_at(&data, eocd + 10).ok_or(BAD)? as usize;
        let cd_offset = u32_at(&data, eocd + 16).ok_or(BAD)? as usize;
        if total == 0xFFFF || cd_offset == 0xFFFF_FFFF {
            return Err("ZIP64 archives are not supported".into());
        }
        let mut entries = Vec::with_capacity(total);
        let mut p = cd_offset;
        for _ in 0..total {
            if u32_at(&data, p) != Some(0x0201_4b50) {
                return Err("Bad magic number for central directory".into());
            }
            let trunc = "Truncated central directory";
            let flags = u16_at(&data, p + 8).ok_or(trunc)?;
            let method = u16_at(&data, p + 10).ok_or(trunc)?;
            let crc = u32_at(&data, p + 16).ok_or(trunc)?;
            let csize = u32_at(&data, p + 20).ok_or(trunc)? as usize;
            let usize_ = u32_at(&data, p + 24).ok_or(trunc)? as usize;
            let nlen = u16_at(&data, p + 28).ok_or(trunc)? as usize;
            let elen = u16_at(&data, p + 30).ok_or(trunc)? as usize;
            let clen = u16_at(&data, p + 32).ok_or(trunc)? as usize;
            let local_offset = u32_at(&data, p + 42).ok_or(trunc)? as usize;
            let name = data.get(p + 46..p + 46 + nlen).ok_or(trunc)?;
            entries.push(Entry {
                name: String::from_utf8_lossy(name).into_owned(),
                method,
                flags,
                crc,
                csize,
                usize_,
                local_offset,
            });
            p += 46 + nlen + elen + clen;
        }
        Ok(Self { data, entries })
    }

    fn read_entry(&self, e: &Entry) -> Result<Vec<u8>, String> {
        if e.flags & 1 != 0 {
            return Err(format!("File {} is encrypted", e.name));
        }
        let lo = e.local_offset;
        if u32_at(&self.data, lo) != Some(0x0403_4b50) {
            return Err(format!("Bad magic number for file header of {}", e.name));
        }
        let nlen = u16_at(&self.data, lo + 26).ok_or("Truncated local header")? as usize;
        let elen = u16_at(&self.data, lo + 28).ok_or("Truncated local header")? as usize;
        let start = lo + 30 + nlen + elen;
        let raw = self
            .data
            .get(start..start + e.csize)
            .ok_or_else(|| format!("Truncated data for {}", e.name))?;
        let out = match e.method {
            0 => raw.to_vec(),
            8 => decompress_to_vec_with_limit(raw, e.usize_)
                .map_err(|err| format!("Error decompressing {}: {err:?}", e.name))?,
            m => return Err(format!("compression type {m} is not supported")),
        };
        if out.len() != e.usize_ || crc32(&out) != e.crc {
            return Err(format!("Bad CRC-32 for file {}", e.name));
        }
        Ok(out)
    }

    /// `ZipFile.extractall`: names are sanitized (no absolute / `..` parts).
    pub(crate) fn extract_all(&self, dest: &Path) -> Result<(), String> {
        for e in &self.entries {
            let mut rel = PathBuf::new();
            for comp in Path::new(e.name.replace('\\', "/").as_str()).components() {
                if let Component::Normal(c) = comp {
                    rel.push(c);
                }
            }
            if rel.as_os_str().is_empty() {
                continue;
            }
            let target = dest.join(&rel);
            if e.name.ends_with('/') {
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                continue;
            }
            let bytes = self.read_entry(e)?;
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&target, bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
