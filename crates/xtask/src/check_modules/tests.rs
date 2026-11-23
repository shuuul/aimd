use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn violations(scope: Scope<'_>, source: &str) -> Vec<String> {
    let mut diagnostics = Vec::new();
    check_source(source, scope, Path::new("sample.rs"), &mut diagnostics);
    diagnostics
}

#[test]
fn checks_every_library_boundary_and_allows_the_own_module() {
    assert!(
        violations(
            Scope::Library("doc"),
            "use crate::{doc::format, model::Task, sys::System, util::pytext, ext::is_doc};"
        )
        .is_empty()
    );
    assert!(violations(Scope::Library("url"), "use crate::url::parse;").is_empty());
    assert!(
        violations(
            Scope::Library("util"),
            "use crate::{util::tempdir, hooks::NoHooks};"
        )
        .is_empty()
    );
    assert!(violations(Scope::Library("runtime"), "use crate::{doc, url, *};").is_empty());
    assert_eq!(
        violations(
            Scope::Library("hooks"),
            "use crate::{error::Error, sys, ext};"
        ),
        [
            "sample.rs:1: module `hooks` cannot import `crate::sys`",
            "sample.rs:1: module `hooks` cannot import `crate::ext`"
        ]
    );
    for (source, target) in [
        ("doc", "url"),
        ("url", "doc"),
        ("util", "sys"),
        ("sys", "util"),
    ] {
        assert_eq!(
            violations(Scope::Library(source), &format!("use crate::{target};")),
            [format!(
                "sample.rs:1: module `{source}` cannot import `crate::{target}`"
            )]
        );
    }
}

#[test]
fn finds_nested_groups_and_siblings_after_aliases() {
    let source = "\
pub(crate) use {
    std::{self as std_alias, io},
    crate::{
        model::{A as B, C},
        {runtime::{self as rt, Runtime}, r#url::*},
        self as aimd_root,
        *,
    },
};";
    assert_eq!(
        violations(Scope::Library("doc"), source),
        [
            "sample.rs:5: module `doc` cannot import `crate::runtime`",
            "sample.rs:5: module `doc` cannot import `crate::url`",
            "sample.rs:6: module `doc` cannot import `crate::self`",
            "sample.rs:7: module `doc` cannot import `crate::*`",
        ]
    );
}

#[test]
fn ignores_comments_and_literals_and_preserves_line_numbers() {
    let source = r####"
        // use crate::runtime;
        /* outer /* nested */ use crate::runtime; */
        const TEXT: &str = "escaped \" use crate::runtime;";
        const RAW: &str = r##"quotes "# use crate::runtime;
        use crate::runtime;"##;
        const BYTES: &[u8] = br"use crate::runtime;";
        const C: &CStr = c"use crate::runtime;";
        const CHAR: char = '\'';
        const UNICODE: char = '🦀';
        const BRACE: char = '\u{7d}';
        fn example<'a>(value: &'a str) {
            let _: &'a str = value;
            use /* separator */ crate :: ocr;
            let _ = crate::runtime::direct_reference;
        }
    "####;
    assert_eq!(
        violations(Scope::Library("url"), source),
        ["sample.rs:14: module `url` cannot import `crate::ocr`"]
    );
}

#[test]
fn checks_root_inline_modules_with_their_own_scope() {
    let source = "\
use crate::runtime;
mod doc {
    mod nested { use crate::doc; use crate::url; }
}
use crate::url;";
    assert_eq!(
        violations(Scope::Library("lib"), source),
        ["sample.rs:3: module `doc` cannot import `crate::url`"]
    );
}

#[test]
fn interfaces_use_the_library_and_keep_their_own_targets_and_third_party_imports() {
    let source = "\
use {::aimd::{output, ProcessInput}, aimd_mcp::Server, rmcp::ServiceExt};
use {aimd_engine::{self as engine, tokenizer}, ::aimd_mlx as mlx, aimd_cli};";
    assert_eq!(
        violations(Scope::Interface("aimd_mcp"), source),
        [
            "sample.rs:2: interface `aimd_mcp` must import through `aimd`, found `aimd_engine`",
            "sample.rs:2: interface `aimd_mcp` must import through `aimd`, found `aimd_mlx`",
            "sample.rs:2: interface `aimd_mcp` must import through `aimd`, found `aimd_cli`",
        ]
    );
}

struct TempTree(PathBuf);

impl TempTree {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "xtask-check-modules-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, path: &str, content: &str) {
        let path = self.0.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn skips_before_consolidation_and_checks_nested_sources_once_aimd_exists() {
    let tree = TempTree::new();
    tree.write("crates/aimd-cli/src/main.rs", "use aimd_process::Runtime;");
    assert_eq!(run(&tree.0).0, 0);

    tree.write(
        "crates/aimd/src/doc/nested/mod.rs",
        "use crate::doc::convert;\nuse crate::runtime::Runtime;",
    );
    tree.write("crates/aimd/src/model.rs", "use crate::error::Error;");
    tree.write(
        "crates/aimd/src/doc/README.md",
        "use crate::runtime::Runtime;",
    );
    let (status, diagnostics) = run(&tree.0);
    assert_eq!(status, 1);
    assert_eq!(
        diagnostics.lines().collect::<Vec<_>>(),
        [
            "crates/aimd/src/doc/nested/mod.rs:2: module `doc` cannot import `crate::runtime`",
            "crates/aimd-cli/src/main.rs:1: interface `aimd_cli` must import through `aimd`, found `aimd_process`",
        ]
    );

    tree.write(
        "crates/aimd/src/doc/nested/mod.rs",
        "use crate::doc::convert;",
    );
    tree.write("crates/aimd-cli/src/main.rs", "use aimd::Runtime;");
    assert_eq!(run(&tree.0).0, 0);
}
