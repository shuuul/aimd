//! Small hand-written helpers (per the WS-15 dependency policy): UUIDs,
//! UTC timestamps in the pydantic JSON shape, constant-time comparison,
//! and the `Path.expanduser().resolve(strict=False)` port.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Serialize, Serializer};

// ---------------------------------------------------------------------------
// UUID
// ---------------------------------------------------------------------------

fn random_bytes() -> [u8; 16] {
    let mut out = [0u8; 16];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut out))
        .is_ok()
    {
        return out;
    }
    // Fallback: std's randomly seeded SipHash over a counter and the clock.
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let state = RandomState::new();
    for (i, chunk) in out.chunks_mut(8).enumerate() {
        let mut h = state.build_hasher();
        h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        h.write_u128(now_micros() as u128);
        h.write_usize(i);
        chunk.copy_from_slice(&h.finish().to_le_bytes());
    }
    out
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Random (version 4) UUID, canonical lowercase hyphenated form.
pub fn uuid4() -> String {
    let mut bytes = random_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format_uuid(&bytes)
}

/// Python `str(UUID(value))`: accepts `urn:uuid:` prefixes, braces,
/// hyphens anywhere, and either case; returns the canonical form.
pub fn parse_uuid(value: &str) -> Option<String> {
    let hex = value.replace("urn:", "").replace("uuid:", "");
    let hex = hex.trim_matches(|c| c == '{' || c == '}').replace('-', "");
    if hex.len() != 32 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(format_uuid(&bytes))
}

// ---------------------------------------------------------------------------
// Timestamps
// ---------------------------------------------------------------------------

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// UTC instant with microsecond precision, serialized like pydantic's
/// `datetime` (`2026-08-15T12:00:00.123456Z`; no fraction when zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(pub i64);

impl Timestamp {
    pub fn now() -> Self {
        Timestamp(now_micros())
    }
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.0.div_euclid(1_000_000);
        let micros = self.0.rem_euclid(1_000_000);
        let days = secs.div_euclid(86_400);
        let sod = secs.rem_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        write!(
            f,
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}",
            sod / 3600,
            (sod / 60) % 60,
            sod % 60
        )?;
        if micros != 0 {
            write!(f, ".{micros:06}")?;
        }
        f.write_str("Z")
    }
}

impl Serialize for Timestamp {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

/// `hmac.compare_digest` for byte strings.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for (i, &x) in a.iter().enumerate() {
        let y = b.get(i).copied().unwrap_or(!x);
        diff |= (x ^ y) as usize;
    }
    diff == 0
}

/// `Path.expanduser()` for `~` and `~/...` (other forms are kept).
pub fn expand_user(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if (text == "~" || text.starts_with("~/"))
        && let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty())
    {
        let rest = text.strip_prefix('~').unwrap_or("").trim_start_matches('/');
        let home = PathBuf::from(home);
        return if rest.is_empty() {
            home
        } else {
            home.join(rest)
        };
    }
    path.to_path_buf()
}

/// `os.path.realpath(path, strict=False)`: absolute, `..` folded after
/// resolving every existing symlink component; missing tails are kept.
pub fn resolve_lenient(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut pending: VecDeque<OsString> = VecDeque::new();
    push_components(&mut pending, &absolute, false);
    let mut out = PathBuf::from("/");
    let mut links = 0;
    while let Some(part) = pending.pop_front() {
        if part == ".." {
            out.pop();
            continue;
        }
        let candidate = out.join(&part);
        match std::fs::symlink_metadata(&candidate) {
            Ok(meta) if meta.file_type().is_symlink() && links < 40 => {
                links += 1;
                match std::fs::read_link(&candidate) {
                    Ok(target) => {
                        if target.is_absolute() {
                            out = PathBuf::from("/");
                        }
                        push_components(&mut pending, &target, true);
                    }
                    Err(_) => out = candidate,
                }
            }
            _ => out = candidate,
        }
    }
    out
}

fn push_components(queue: &mut VecDeque<OsString>, path: &Path, front: bool) {
    let parts: Vec<OsString> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(p) => Some(p.to_os_string()),
            Component::ParentDir => Some(OsString::from("..")),
            _ => None,
        })
        .collect();
    if front {
        for part in parts.into_iter().rev() {
            queue.push_front(part);
        }
    } else {
        queue.extend(parts);
    }
}

/// URL scheme as `urllib.parse.urlparse` reports it (lowercased), or
/// `None` when the value has no valid `scheme:` prefix.
pub fn url_scheme(value: &str) -> Option<String> {
    let colon = value.find(':')?;
    let scheme = &value[..colon];
    let valid = scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then(|| scheme.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_roundtrip_and_python_forms() {
        let id = uuid4();
        assert_eq!(id.len(), 36);
        assert_eq!(&id[14..15], "4");
        assert_eq!(parse_uuid(&id).as_deref(), Some(id.as_str()));
        assert_eq!(
            parse_uuid("{6F1C2D38-9A27-4C49-8A0C-54D4F1C1E0AA}").as_deref(),
            Some("6f1c2d38-9a27-4c49-8a0c-54d4f1c1e0aa")
        );
        assert_eq!(
            parse_uuid("urn:uuid:6f1c2d389a274c498a0c54d4f1c1e0aa").as_deref(),
            Some("6f1c2d38-9a27-4c49-8a0c-54d4f1c1e0aa")
        );
        assert_eq!(parse_uuid("nope"), None);
        assert_eq!(parse_uuid("../../etc/passwd"), None);
    }

    #[test]
    fn timestamps_match_pydantic() {
        assert_eq!(Timestamp(0).to_string(), "1970-01-01T00:00:00Z");
        // 2026-08-15T12:00:00.123456Z
        assert_eq!(
            Timestamp(1_786_795_200_123_456).to_string(),
            "2026-08-15T12:00:00.123456Z"
        );
        assert_eq!(
            Timestamp(951_782_400_000_001).to_string(),
            "2000-02-29T00:00:00.000001Z"
        );
    }

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"Bearer x", b"Bearer x"));
        assert!(!constant_time_eq(b"Bearer x", b"Bearer y"));
        assert!(!constant_time_eq(b"Bearer", b"Bearer x"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    #[test]
    fn url_scheme_like_urlparse() {
        assert_eq!(url_scheme("HTTPS://x").as_deref(), Some("https"));
        assert_eq!(url_scheme("file:///etc").as_deref(), Some("file"));
        assert_eq!(url_scheme("/tmp/a:b"), None);
    }
}
