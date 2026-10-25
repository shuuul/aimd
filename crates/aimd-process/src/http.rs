//! Small blocking HTTP helpers over `ureq` (remote backends, HF downloads),
//! plus hand-written base64, multipart, and Python-`mimetypes` lookups.

use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

/// HTTP-level failure.
#[derive(Debug)]
pub enum HttpError {
    /// Non-2xx status with the decoded body.
    Status(u16, String),
    /// Connection, TLS, timeout, or I/O failure.
    Transport(String),
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

fn finish(
    result: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<Vec<u8>, HttpError> {
    let mut response = result.map_err(|e| HttpError::Transport(e.to_string()))?;
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .with_config()
        .limit(u64::MAX)
        .read_to_vec()
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    if (200..300).contains(&status) {
        Ok(body)
    } else {
        Err(HttpError::Status(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        ))
    }
}

/// POST `body` and return the 2xx response body.
pub fn post(
    url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    timeout: Duration,
) -> Result<Vec<u8>, HttpError> {
    let mut request = agent(timeout).post(url);
    for (k, v) in headers {
        request = request.header(*k, v.as_str());
    }
    finish(request.send(body))
}

/// GET and return the 2xx response body.
pub fn get(url: &str, headers: &[(&str, String)], timeout: Duration) -> Result<Vec<u8>, HttpError> {
    let mut request = agent(timeout).get(url);
    for (k, v) in headers {
        request = request.header(*k, v.as_str());
    }
    finish(request.call())
}

/// GET streaming into `dest`.
pub fn download(
    url: &str,
    headers: &[(&str, String)],
    dest: &Path,
    timeout: Duration,
) -> Result<u64, HttpError> {
    let mut request = agent(timeout).get(url);
    for (k, v) in headers {
        request = request.header(*k, v.as_str());
    }
    let mut response = request
        .call()
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        let body = response.body_mut().read_to_string().unwrap_or_default();
        return Err(HttpError::Status(status, body));
    }
    let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();
    let mut file = std::fs::File::create(dest).map_err(|e| HttpError::Transport(e.to_string()))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| HttpError::Transport(e.to_string()))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| HttpError::Transport(e.to_string()))?;
        total += n as u64;
    }
    file.flush()
        .map_err(|e| HttpError::Transport(e.to_string()))?;
    Ok(total)
}

/// Standard base64 with padding.
pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Python `mimetypes.guess_type` for the extensions aimd uploads.
pub fn guess_mime(path: &Path) -> Option<&'static str> {
    Some(match aimd_core::ext::suffix_lower(path).as_str() {
        ".wav" => "audio/x-wav",
        ".mp3" => "audio/mpeg",
        ".flac" => "audio/x-flac",
        ".m4a" => "audio/mp4a-latm",
        ".aac" => "audio/x-aac",
        ".ogg" | ".opus" => "audio/ogg",
        ".webm" => "video/webm",
        ".mp4" => "video/mp4",
        ".png" => "image/png",
        ".jpg" | ".jpeg" => "image/jpeg",
        ".webp" => "image/webp",
        ".tif" | ".tiff" => "image/tiff",
        _ => return None,
    })
}

/// `multipart/form-data` body (Python `_multipart_body` layout).
pub fn multipart_body(
    fields: &[(&str, &str)],
    file_name: &str,
    file_mime: &str,
    file_bytes: &[u8],
) -> (Vec<u8>, String) {
    let boundary = format!(
        "----aimd-{}",
        crate::tempdir::unique_name("").replace('-', "")
    );
    let mut body = Vec::with_capacity(file_bytes.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {file_mime}\r\n\r\n").as_bytes());
    body.extend_from_slice(file_bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (body, format!("multipart/form-data; boundary={boundary}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_python() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(&[0xff, 0xfe, 0x00, 0x01]), "//4AAQ==");
    }

    #[test]
    fn mime_table() {
        assert_eq!(guess_mime(Path::new("a.WAV")), Some("audio/x-wav"));
        assert_eq!(guess_mime(Path::new("a.jpeg")), Some("image/jpeg"));
        assert_eq!(guess_mime(Path::new("a.mkv")), None);
    }

    #[test]
    fn multipart_layout() {
        let (body, ct) = multipart_body(&[("model", "m")], "a.wav", "audio/x-wav", b"RIFF");
        let boundary = ct.strip_prefix("multipart/form-data; boundary=").unwrap();
        let text = String::from_utf8(body).unwrap();
        assert_eq!(
            text,
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n\
                 --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"a.wav\"\r\nContent-Type: audio/x-wav\r\n\r\nRIFF\r\n--{boundary}--\r\n"
            )
        );
    }
}
