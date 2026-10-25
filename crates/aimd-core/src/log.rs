//! Minimal stderr logger shared by every aimd crate (replaces `logly`).
//!
//! Format mirrors the Python CLI sink:
//! `YYYY-MM-DD HH:MM:SS | LEVEL    | module:function - message` (local
//! time; the Rust port prints the module path in place of `module:function`).
//! Lines are colorized (bold, red for errors) only when stderr is a TTY.
//!
//! Tests capture records on the current thread with [`capture`] instead of
//! asserting on stderr.

use std::cell::RefCell;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicU8, Ordering};

/// Log severity (Python level names).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    Critical = 50,
}

impl Level {
    pub fn as_str(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warning => "WARNING",
            Level::Error => "ERROR",
            Level::Critical => "CRITICAL",
        }
    }

    /// Parse a level name case-insensitively (`DEBUG`, `info`, ...).
    pub fn parse(name: &str) -> Option<Level> {
        match name.to_ascii_uppercase().as_str() {
            "DEBUG" => Some(Level::Debug),
            "INFO" => Some(Level::Info),
            "WARNING" => Some(Level::Warning),
            "ERROR" => Some(Level::Error),
            "CRITICAL" => Some(Level::Critical),
            _ => None,
        }
    }

    fn from_u8(value: u8) -> Level {
        match value {
            0..=10 => Level::Debug,
            11..=20 => Level::Info,
            21..=30 => Level::Warning,
            31..=40 => Level::Error,
            _ => Level::Critical,
        }
    }
}

/// Valid level names in display order (`--log-level` help/validation).
pub const LEVEL_NAMES: [&str; 5] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];

static MIN_LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

/// Set the global minimum level (default `INFO`).
pub fn set_level(level: Level) {
    MIN_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Current global minimum level.
pub fn level() -> Level {
    Level::from_u8(MIN_LEVEL.load(Ordering::Relaxed))
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<(Level, String)>>> = const { RefCell::new(None) };
}

/// Run `f` and return every record logged on this thread while it ran
/// (all levels, regardless of the global minimum). Captured records are
/// not written to stderr.
pub fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<(Level, String)>) {
    let previous = CAPTURE.with(|c| c.borrow_mut().replace(Vec::new()));
    let out = f();
    let records = CAPTURE.with(|c| {
        let mut slot = c.borrow_mut();
        let records = slot.take().unwrap_or_default();
        *slot = previous;
        records
    });
    (out, records)
}

/// Emit one record. Prefer the `log_*!` macros.
pub fn emit(level: Level, module: &str, message: &str) {
    let captured = CAPTURE.with(|c| {
        if let Some(records) = c.borrow_mut().as_mut() {
            records.push((level, message.to_string()));
            true
        } else {
            false
        }
    });
    if captured || level < self::level() {
        return;
    }
    let line = format!(
        "{} | {:<8} | {} - {}",
        timestamp(),
        level.as_str(),
        module,
        message
    );
    let stderr = std::io::stderr();
    let color = stderr.is_terminal();
    let mut handle = stderr.lock();
    let _ = if color {
        let code = match level {
            Level::Debug => "\x1b[34;1m",
            Level::Info => "\x1b[1m",
            Level::Warning => "\x1b[33;1m",
            Level::Error | Level::Critical => "\x1b[1;31m",
        };
        writeln!(handle, "{code}{line}\x1b[0m")
    } else {
        writeln!(handle, "{line}")
    };
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => { $crate::log::emit($crate::log::Level::Debug, module_path!(), &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => { $crate::log::emit($crate::log::Level::Info, module_path!(), &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => { $crate::log::emit($crate::log::Level::Warning, module_path!(), &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => { $crate::log::emit($crate::log::Level::Error, module_path!(), &format!($($arg)*)) };
}

// ---------------------------------------------------------------------------
// Local timestamp without a date/time dependency
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod local_time {
    use std::os::raw::{c_char, c_int, c_long};

    #[repr(C)]
    pub struct Tm {
        pub tm_sec: c_int,
        pub tm_min: c_int,
        pub tm_hour: c_int,
        pub tm_mday: c_int,
        pub tm_mon: c_int,
        pub tm_year: c_int,
        pub tm_wday: c_int,
        pub tm_yday: c_int,
        pub tm_isdst: c_int,
        pub tm_gmtoff: c_long,
        pub tm_zone: *const c_char,
    }

    unsafe extern "C" {
        fn localtime_r(time: *const i64, result: *mut Tm) -> *mut Tm;
    }

    /// `(year, month, day, hour, minute, second)` in local time.
    pub fn now() -> Option<(i32, u32, u32, u32, u32, u32)> {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs() as i64;
        let mut tm = std::mem::MaybeUninit::<Tm>::zeroed();
        // SAFETY: `localtime_r` writes a full `struct tm` into `tm`; the
        // layout above matches the glibc/BSD/macOS definition (both carry
        // `tm_gmtoff` + `tm_zone` after the nine int fields).
        let ptr = unsafe { localtime_r(&secs, tm.as_mut_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: initialized by the successful call above.
        let tm = unsafe { tm.assume_init() };
        Some((
            tm.tm_year + 1900,
            (tm.tm_mon + 1) as u32,
            tm.tm_mday as u32,
            tm.tm_hour as u32,
            tm.tm_min as u32,
            tm.tm_sec as u32,
        ))
    }
}

#[cfg(not(unix))]
mod local_time {
    pub fn now() -> Option<(i32, u32, u32, u32, u32, u32)> {
        None
    }
}

fn timestamp() -> String {
    match local_time::now() {
        Some((y, mo, d, h, mi, s)) => format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}"),
        None => "0000-00-00 00:00:00".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_collects_records_on_this_thread() {
        let ((), records) = capture(|| {
            crate::log_warn!("first {}", 1);
            crate::log_debug!("second");
        });
        assert_eq!(
            records,
            vec![
                (Level::Warning, "first 1".to_string()),
                (Level::Debug, "second".to_string())
            ]
        );
    }

    #[test]
    fn level_names_parse_case_insensitively() {
        assert_eq!(Level::parse("debug"), Some(Level::Debug));
        assert_eq!(Level::parse("WARNING"), Some(Level::Warning));
        assert_eq!(Level::parse("verbose"), None);
    }

    #[test]
    fn timestamp_has_python_shape() {
        let ts = timestamp();
        assert_eq!(ts.len(), 19);
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[13..14], ":");
    }
}
