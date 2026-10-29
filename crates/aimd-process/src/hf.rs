//! Hugging Face hub cache lookup and HTTPS snapshot download (the layout
//! `huggingface_hub` uses: `models--{org}--{name}/{refs,snapshots}`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use aimd_core::{AimdError, Result, log_info};
use serde_json::Value;

use crate::http::{self, HttpError};

/// Hub cache directory: `HF_HUB_CACHE` > `HF_HOME/hub` >
/// `XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub`.
pub fn hub_cache_dir(env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    let set = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(dir) = set("HF_HUB_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(home) = set("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    if let Some(xdg) = set("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("huggingface").join("hub");
    }
    PathBuf::from(set("HOME").unwrap_or_else(|| ".".into()))
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// `models--org--name`.
pub fn repo_folder(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

/// Check the files consumed by the native loaders, without loading weights.
/// HF's shared cache may contain only a config or a subset of the shards.
fn complete_snapshot(snapshot: &Path) -> bool {
    let json = |name: &str| -> Option<Value> {
        serde_json::from_slice(&std::fs::read(snapshot.join(name)).ok()?).ok()
    };
    let Some(config) = json("config.json") else {
        return false;
    };
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("");
    let preprocessing = if model_type.starts_with("pp_ocrv6_") {
        if model_type.ends_with("_rec") {
            snapshot.join("preprocessor_config.json").is_file()
        } else {
            model_type.ends_with("_det")
        }
    } else {
        snapshot.join("tokenizer.json").is_file()
            || ["vocab.json", "merges.txt", "tokenizer_config.json"]
                .iter()
                .all(|name| snapshot.join(name).is_file())
    };
    if !preprocessing {
        return false;
    }
    if snapshot.join("model.safetensors.index.json").exists() {
        return json("model.safetensors.index.json")
            .and_then(|index| index.get("weight_map").and_then(Value::as_object).cloned())
            .is_some_and(|map| {
                !map.is_empty()
                    && map.values().all(|name| {
                        name.as_str()
                            .is_some_and(|name| snapshot.join(name).is_file())
                    })
            });
    }
    snapshot.join("model.safetensors").is_file()
}

/// A complete cached snapshot, preferring `refs/main`.
pub fn cached_snapshot(cache: &Path, repo_id: &str) -> Option<PathBuf> {
    let root = cache.join(repo_folder(repo_id));
    if let Ok(sha) = std::fs::read_to_string(root.join("refs").join("main")) {
        let snap = root.join("snapshots").join(sha.trim());
        if complete_snapshot(&snap) {
            return Some(snap);
        }
    }
    let mut snaps: Vec<(std::time::SystemTime, PathBuf)> =
        std::fs::read_dir(root.join("snapshots"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| complete_snapshot(p))
            .map(|p| {
                let mtime = std::fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                (mtime, p)
            })
            .collect();
    snaps.sort();
    snaps.pop().map(|(_, p)| p)
}

fn is_truthy(value: Option<String>) -> bool {
    value.is_some_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

fn http_err(repo_id: &str, e: HttpError) -> AimdError {
    match e {
        HttpError::Status(code, body) => AimdError::backend_unavailable(format!(
            "Unable to download model {repo_id} from Hugging Face (HTTP {code}): {}",
            body.trim()
        )),
        HttpError::Transport(e) => AimdError::backend_unavailable(format!(
            "Unable to download model {repo_id} from Hugging Face: {e}"
        )),
    }
}

/// Files fetched for a snapshot (everything but dotfiles and docs).
fn wanted(name: &str) -> bool {
    !name.starts_with('.') && !name.ends_with(".md") && !name.contains("/.")
}

/// Download `repo_id`@main into the cache and return the snapshot path.
pub fn download_snapshot(
    cache: &Path,
    repo_id: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<PathBuf> {
    let endpoint = env("HF_ENDPOINT")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into());
    let endpoint = endpoint.trim_end_matches('/');
    let mut headers = Vec::new();
    if let Some(token) = env("HF_TOKEN").filter(|v| !v.is_empty()) {
        headers.push(("Authorization", format!("Bearer {token}")));
    }
    let api = format!("{endpoint}/api/models/{repo_id}/revision/main");
    let info =
        http::get(&api, &headers, Duration::from_secs(60)).map_err(|e| http_err(repo_id, e))?;
    let info: Value = serde_json::from_slice(&info).map_err(|e| {
        AimdError::backend_unavailable(format!(
            "Invalid Hugging Face API response for {repo_id}: {e}"
        ))
    })?;
    let sha = info
        .get("sha")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            AimdError::backend_unavailable(format!(
                "Hugging Face API response for {repo_id} has no sha"
            ))
        })?
        .to_string();
    let mut files: Vec<String> = info
        .get("siblings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| s.get("rfilename").and_then(Value::as_str))
        .filter(|n| wanted(n))
        .map(str::to_string)
        .collect();
    // Configuration lands last, but cache hits also check weights and tokenizer:
    // other HF clients may have downloaded config.json before anything else.
    files.sort_by_key(|n| n == "config.json");
    let root = cache.join(repo_folder(repo_id));
    let snapshot = root.join("snapshots").join(&sha);
    let io = |e: std::io::Error| {
        AimdError::backend_unavailable(format!("Cannot write model cache {}: {e}", root.display()))
    };
    std::fs::create_dir_all(&snapshot).map_err(io)?;
    log_info!(
        "Downloading {repo_id} ({} files) to {}",
        files.len(),
        cache.display()
    );
    for name in &files {
        let dest = snapshot.join(name);
        if dest.is_file() {
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        let mut partial = dest.clone().into_os_string();
        partial.push(".incomplete");
        let partial = PathBuf::from(partial);
        let url = format!("{endpoint}/{repo_id}/resolve/{sha}/{name}");
        log_info!("Downloading {name}");
        http::download(&url, &headers, &partial, Duration::from_secs(3600)).map_err(|e| {
            let _ = std::fs::remove_file(&partial);
            http_err(repo_id, e)
        })?;
        std::fs::rename(&partial, &dest).map_err(io)?;
    }
    if !complete_snapshot(&snapshot) {
        return Err(AimdError::backend_unavailable(format!(
            "Downloaded model {repo_id} has an incomplete snapshot at {}: \
             required configuration, tokenizer or weight files are missing.",
            snapshot.display()
        )));
    }
    std::fs::create_dir_all(root.join("refs")).map_err(io)?;
    std::fs::write(root.join("refs").join("main"), &sha).map_err(io)?;
    Ok(snapshot)
}

/// Cached snapshot, else a download (refused when `HF_HUB_OFFLINE` is set).
pub fn resolve_checkpoint(repo_id: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<PathBuf> {
    let cache = hub_cache_dir(env);
    if let Some(snapshot) = cached_snapshot(&cache, repo_id) {
        return Ok(snapshot);
    }
    if is_truthy(env("HF_HUB_OFFLINE")) {
        return Err(AimdError::backend_unavailable(format!(
            "Model {repo_id} is missing or incomplete in the Hugging Face cache ({}) and HF_HUB_OFFLINE is set.",
            cache.display()
        )));
    }
    download_snapshot(&cache, repo_id, env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: Vec<(&'static str, String)>) -> impl Fn(&str) -> Option<String> {
        move |k| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn cache_dir_precedence() {
        let env = env_of(vec![("HF_HOME", "/h".into()), ("HOME", "/u".into())]);
        assert_eq!(hub_cache_dir(&env), PathBuf::from("/h/hub"));
        let env = env_of(vec![
            ("HF_HUB_CACHE", "/c".into()),
            ("HF_HOME", "/h".into()),
        ]);
        assert_eq!(hub_cache_dir(&env), PathBuf::from("/c"));
        let env = env_of(vec![("HOME", "/u".into())]);
        assert_eq!(
            hub_cache_dir(&env),
            PathBuf::from("/u/.cache/huggingface/hub")
        );
    }

    #[test]
    fn cached_snapshot_and_offline_refusal() {
        let dir = crate::tempdir::TempDir::new(None, "aimd-hf-test-").unwrap();
        let cache = dir.path().to_path_buf();
        let root = cache.join("models--mlx-community--Tiny-4bit");
        let snap = root.join("snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::create_dir_all(root.join("refs")).unwrap();
        std::fs::write(root.join("refs/main"), "abc\n").unwrap();
        // Incomplete (no config.json) -> not cached.
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        let env = env_of(vec![
            ("HF_HUB_CACHE", cache.to_string_lossy().into_owned()),
            ("HF_HUB_OFFLINE", "1".into()),
        ]);
        let err = resolve_checkpoint("mlx-community/Tiny-4bit", &env).unwrap_err();
        assert!(
            err.message.contains("HF_HUB_OFFLINE is set"),
            "{}",
            err.message
        );
        std::fs::write(snap.join("config.json"), "{}").unwrap();
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        std::fs::write(snap.join("model.safetensors"), "weights").unwrap();
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        std::fs::write(snap.join("vocab.json"), "{}").unwrap();
        std::fs::write(snap.join("merges.txt"), "#version: 0.2\n").unwrap();
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        std::fs::write(snap.join("tokenizer_config.json"), "{}").unwrap();
        assert_eq!(
            resolve_checkpoint("mlx-community/Tiny-4bit", &env).unwrap(),
            snap
        );
        // An index is authoritative: one existing shard is not sufficient.
        std::fs::remove_file(snap.join("model.safetensors")).unwrap();
        std::fs::write(
            snap.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(snap.join("one.safetensors"), "weights").unwrap();
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        let err = resolve_checkpoint("mlx-community/Tiny-4bit", &env).unwrap_err();
        assert!(err.message.contains("incomplete"), "{}", err.message);
        std::fs::write(snap.join("two.safetensors"), "weights").unwrap();
        assert_eq!(
            cached_snapshot(&cache, "mlx-community/Tiny-4bit"),
            Some(snap.clone())
        );
        // PP-OCRv6 has no BPE tokenizer; recognition needs its character table.
        std::fs::write(
            snap.join("config.json"),
            r#"{"model_type":"pp_ocrv6_small_rec"}"#,
        )
        .unwrap();
        assert_eq!(cached_snapshot(&cache, "mlx-community/Tiny-4bit"), None);
        std::fs::write(snap.join("preprocessor_config.json"), "{}").unwrap();
        assert_eq!(
            cached_snapshot(&cache, "mlx-community/Tiny-4bit"),
            Some(snap.clone())
        );
        std::fs::remove_file(snap.join("preprocessor_config.json")).unwrap();
        std::fs::write(
            snap.join("config.json"),
            r#"{"model_type":"pp_ocrv6_medium_det"}"#,
        )
        .unwrap();
        assert_eq!(
            cached_snapshot(&cache, "mlx-community/Tiny-4bit"),
            Some(snap)
        );
    }
}
