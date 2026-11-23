//! Repository maintenance tasks, run as `cargo xtask <task>`.

mod check_modules;
mod check_specs;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage: cargo xtask <check-modules|check-specs> [--root <dir>]";

/// Walk up from `start` to the nearest directory containing `specs/`.
fn discover_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join("specs").is_dir())
        .map(Path::to_path_buf)
}

fn repo_root() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = discover_root(&cwd)
    {
        return root;
    }
    // Fall back to the workspace root (crates/xtask/../..).
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    discover_root(manifest).unwrap_or_else(|| manifest.to_path_buf())
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some(task @ ("check-modules" | "check-specs")) => {
            let mut root = None;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--root" => match args.next() {
                        Some(dir) => root = Some(PathBuf::from(dir)),
                        None => {
                            eprintln!("--root requires a directory\n{USAGE}");
                            return ExitCode::from(2);
                        }
                    },
                    _ => {
                        eprintln!("unexpected argument {arg:?}\n{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            let root = root.unwrap_or_else(repo_root);
            let (code, out) = if task == "check-modules" {
                check_modules::run(&root)
            } else {
                check_specs::report(&check_specs::check_tree(&root))
            };
            print!("{out}");
            ExitCode::from(code)
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
