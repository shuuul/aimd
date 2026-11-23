//! Check the top-level module boundaries defined in spec 010.

#[cfg(test)]
mod tests;
mod tokens;

use self::tokens::{Token, tokenize};
use std::fs;
use std::io;
use std::path::Path;

const CONTRACT: &[&str] = &[
    "model", "error", "hooks", "output", "log", "text", "anchors",
];
const LIBRARY_RULES: &[(&[&str], &[&str])] = &[
    (&["text"], &["util"]),
    (CONTRACT, &[]),
    (&["sys", "util"], &[]),
    (&["doc", "url"], &["sys", "util", "ext"]),
];
const INTERFACE_CRATES: &[(&str, &str)] = &[("aimd-cli", "aimd_cli"), ("aimd-mcp", "aimd_mcp")];
const INTERNAL_CRATES: &[&str] = &[
    "aimd_core",
    "aimd_doc",
    "aimd_engine",
    "aimd_mlx",
    "aimd_process",
    "aimd_url",
    "aimd_weights",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Import<'a> {
    Unqualified,
    Crate(Token<'a>),
    Module(Token<'a>),
    External(Token<'a>),
}

/// Visit the leaves of a Rust use tree, retaining just its crate and first module.
fn collect_imports<'a>(
    tokens: &mut &[Token<'a>],
    mut import: Import<'a>,
    out: &mut Vec<Import<'a>>,
) {
    take(tokens, "::");
    loop {
        if take(tokens, "{") {
            if take(tokens, "}") {
                record_import(out, import);
                return;
            }
            while !tokens.is_empty() && !take(tokens, "}") {
                collect_imports(tokens, import, out);
                if !take(tokens, ",") {
                    take(tokens, "}");
                    break;
                }
            }
            return;
        }
        let Some((token, rest)) = tokens.split_first() else {
            return;
        };
        if token.text != "*" && !token.is_name() {
            return;
        }
        *tokens = rest;
        import = match import {
            Import::Unqualified if token.text == "crate" => Import::Crate(*token),
            Import::Unqualified => Import::External(*token),
            Import::Crate(_) => Import::Module(*token),
            _ => import,
        };
        if !take(tokens, "::") {
            if take(tokens, "as")
                && let Some((_, rest)) = tokens.split_first()
            {
                *tokens = rest;
            }
            record_import(out, import);
            return;
        }
    }
}

fn record_import<'a>(out: &mut Vec<Import<'a>>, import: Import<'a>) {
    if import != Import::Unqualified && out.last() != Some(&import) {
        out.push(import);
    }
}

fn take(tokens: &mut &[Token<'_>], text: &str) -> bool {
    if let Some((token, rest)) = tokens.split_first()
        && token.text == text
    {
        *tokens = rest;
        return true;
    }
    false
}

#[derive(Clone, Copy)]
enum Scope<'a> {
    Library(&'a str),
    Interface(&'a str),
}

impl Scope<'_> {
    fn violation(self, import: Import<'_>) -> Option<(usize, String)> {
        match (self, import) {
            (Self::Library(source), Import::Module(target) | Import::Crate(target)) => {
                let (_, extra) = LIBRARY_RULES
                    .iter()
                    .find(|(sources, _)| sources.contains(&source))?;
                let name = target.name();
                if name == source || CONTRACT.contains(&name) || extra.contains(&name) {
                    return None;
                }
                let path = match import {
                    Import::Crate(_) => "crate".to_string(),
                    _ => format!("crate::{name}"),
                };
                Some((
                    target.line,
                    format!("module `{source}` cannot import `{path}`"),
                ))
            }
            // Rust checks item visibility through `aimd`. Keep the interfaces from
            // bypassing it by importing the engine, shim or another product crate.
            (Self::Interface(own_crate), Import::External(target)) => {
                let name = target.name();
                if INTERNAL_CRATES.contains(&name)
                    || INTERFACE_CRATES
                        .iter()
                        .any(|(_, krate)| *krate == name && name != own_crate)
                {
                    return Some((
                        target.line,
                        format!(
                            "interface `{own_crate}` must import through `aimd`, found `{name}`"
                        ),
                    ));
                }
                None
            }
            _ => None,
        }
    }
}

fn check_source(source: &str, scope: Scope<'_>, path: &Path, diagnostics: &mut Vec<String>) {
    let tokens = tokenize(source);
    let mut depth = 0;
    let mut inline_module = None;
    for (index, token) in tokens.iter().enumerate() {
        match token.text {
            "{" => {
                if depth == 0
                    && matches!(scope, Scope::Library("lib"))
                    && index >= 2
                    && tokens[index - 2].text == "mod"
                    && tokens[index - 1].is_name()
                {
                    inline_module = Some(tokens[index - 1].name());
                }
                depth += 1;
            }
            "}" => {
                depth -= 1;
                if depth == 0 {
                    inline_module = None;
                }
            }
            "use" => {
                let scope = inline_module.map_or(scope, Scope::Library);
                let mut imports = Vec::new();
                collect_imports(&mut &tokens[index + 1..], Import::Unqualified, &mut imports);
                for (line, message) in imports
                    .into_iter()
                    .filter_map(|import| scope.violation(import))
                {
                    diagnostics.push(format!("{}:{line}: {message}", path.display()));
                }
            }
            _ => {}
        }
    }
}

fn visit_sources(dir: &Path, check: &mut impl FnMut(&Path)) -> io::Result<()> {
    let mut entries = fs::read_dir(dir)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_unstable_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            visit_sources(&path, check)?;
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            check(&path);
        }
    }
    Ok(())
}

fn check_sources(root: &Path, dir: &Path, interface: Option<&str>, diagnostics: &mut Vec<String>) {
    let result = visit_sources(dir, &mut |path| {
        let relative = path.strip_prefix(dir).unwrap();
        let top = Path::new(relative.components().next().unwrap().as_os_str());
        let module = if top == relative {
            top.file_stem().unwrap()
        } else {
            top.as_os_str()
        };
        let scope = interface.map_or_else(
            || Scope::Library(module.to_str().unwrap_or("")),
            Scope::Interface,
        );
        let display = path.strip_prefix(root).unwrap_or(path);
        match fs::read_to_string(path) {
            Ok(source) => check_source(&source, scope, display, diagnostics),
            Err(error) => diagnostics.push(format!("{}: {error}", display.display())),
        }
    });
    if let Err(error) = result {
        diagnostics.push(format!(
            "{}: {error}",
            dir.strip_prefix(root).unwrap_or(dir).display()
        ));
    }
}

pub fn run(root: &Path) -> (u8, String) {
    let library = root.join("crates/aimd");
    match library.try_exists() {
        Ok(false) => return (0, "Module check skipped (crates/aimd is absent).\n".into()),
        Err(error) => return (1, format!("crates/aimd: {error}\n")),
        Ok(true) => {}
    }
    let mut diagnostics = Vec::new();
    check_sources(root, &library.join("src"), None, &mut diagnostics);
    for (dir, krate) in INTERFACE_CRATES {
        let src = root.join("crates").join(dir).join("src");
        match src.try_exists() {
            Ok(true) => check_sources(root, &src, Some(krate), &mut diagnostics),
            Ok(false) => {}
            Err(error) => diagnostics.push(format!("crates/{dir}/src: {error}")),
        }
    }
    if diagnostics.is_empty() {
        (0, "Module check passed.\n".into())
    } else {
        (1, diagnostics.join("\n") + "\n")
    }
}
