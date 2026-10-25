//! Regression tests for the spec validator; each builds a temporary
//! repository tree and asserts a specific diagnostic per failure path.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const HEADINGS: [&str; 9] = REQUIRED_HEADINGS;

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xtask-check-specs-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn write_file(root: &Path, rel: &str, content: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

struct Spec<'a> {
    num: u32,
    status: &'a str,
    created: &'a str,
    updated: &'a str,
    extra: &'a str,
}

fn spec(num: u32) -> Spec<'static> {
    Spec {
        num,
        status: "Draft",
        created: "2026-08-01",
        updated: "2026-08-10",
        extra: "",
    }
}

impl Spec<'_> {
    fn md(&self) -> String {
        let body = HEADINGS
            .iter()
            .map(|h| format!("## {h}\n\nBody."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut fm = format!(
            "---\nid: \"{:03}\"\ntitle: \"Example spec\"\nstatus: {}\ncreated: {}\nupdated: {}\ncoordinator: \"Coordinator\"\n",
            self.num, self.status, self.created, self.updated
        );
        if !self.extra.is_empty() {
            fm.push_str(self.extra.trim_end_matches('\n'));
            fm.push('\n');
        }
        format!("{fm}---\n\n{body}")
    }
}

fn readme(active_rows: &[String], archived_rows: &[String], template_link: bool) -> String {
    let mut lines: Vec<String> = vec!["# Project specs".into(), String::new()];
    if template_link {
        lines.push("Copy [000-template.md](000-template.md) to start a spec.".into());
    }
    lines.extend(
        [
            "",
            "## Active specs",
            "",
            "| Spec | Status | Outcome |",
            "| --- | --- | --- |",
        ]
        .map(String::from),
    );
    lines.extend(active_rows.iter().cloned());
    lines.extend(
        [
            "",
            "## Archived specs",
            "",
            "| Spec | Completed | Outcome |",
            "| --- | --- | --- |",
        ]
        .map(String::from),
    );
    lines.extend(archived_rows.iter().cloned());
    lines.push(String::new());
    lines.join("\n")
}

fn active_row(num: u32, status: &str) -> String {
    let target = format!("{num:03}-example.md");
    format!("| [{target}]({target}) | {status} | Done |")
}

fn archived_row(num: u32) -> String {
    let name = format!("{num:03}-completed.md");
    format!("| [{name}](archive/{name}) | 2026-08-20 | Done |")
}

fn write_valid_tree(root: &Path) {
    fs::create_dir_all(root.join("specs/archive")).unwrap();
    fs::copy(
        repo_root().join("specs/000-template.md"),
        root.join("specs/000-template.md"),
    )
    .unwrap();
    write_file(root, "specs/README.md", &readme(&[], &[], true));
}

fn write_active_spec(root: &Path, num: u32, content: &str) {
    write_file(root, &format!("specs/{num:03}-example.md"), content);
}

fn write_archived_spec(root: &Path, num: u32, content: &str) {
    write_file(
        root,
        &format!("specs/archive/{num:03}-completed.md"),
        content,
    );
}

fn run_check(root: &Path) -> (u8, String) {
    report(&check_tree(root))
}

fn assert_passes(root: &Path) {
    let (code, out) = run_check(root);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("Specs check passed."));
}

fn assert_fails_with(root: &Path, needles: &[&str]) {
    let (code, out) = run_check(root);
    assert_eq!(code, 1, "{out}");
    for needle in needles {
        assert!(out.contains(needle), "missing {needle:?} in:\n{out}");
    }
}

#[test]
fn repository_spec_tree_passes() {
    assert_passes(&repo_root());
}

#[test]
fn empty_index_and_template_pass() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    assert_passes(&tmp.0);
}

#[test]
fn indexed_active_and_archived_specs_pass() {
    let tmp = TempDir::new();
    let root = &tmp.0;
    write_valid_tree(root);
    write_active_spec(
        root,
        1,
        &Spec {
            status: "Active",
            ..spec(1)
        }
        .md(),
    );
    write_archived_spec(
        root,
        2,
        &Spec {
            status: "Completed",
            ..spec(2)
        }
        .md(),
    );
    write_file(
        root,
        "specs/README.md",
        &readme(&[active_row(1, "Active")], &[archived_row(2)], true),
    );
    assert_passes(root);
}

#[test]
fn invalid_filename_and_reserved_zero_fail() {
    let tmp = TempDir::new();
    let root = &tmp.0;
    write_valid_tree(root);
    write_file(root, "specs/001-Uppercase.md", &spec(1).md());
    write_file(root, "specs/000-copy.md", &spec(0).md());
    assert_fails_with(
        root,
        &[
            "specs/001-Uppercase.md: filename does not match NNN-kebab-case.md",
            "specs/000-copy.md: ID 000 is reserved for specs/000-template.md",
        ],
    );
}

#[test]
fn frontmatter_id_mismatch_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(2).md());
    assert_fails_with(
        &tmp.0,
        &["specs/001-example.md: frontmatter id '002' does not match filename ID 001"],
    );
}

#[test]
fn completed_status_in_active_directory_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(
        &tmp.0,
        1,
        &Spec {
            status: "Completed",
            ..spec(1)
        }
        .md(),
    );
    assert_fails_with(
        &tmp.0,
        &[
            "specs/001-example.md: status 'Completed' is not allowed in specs/ (expected Draft or Active)",
        ],
    );
}

#[test]
fn duplicate_ids_across_directories_fail() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    write_archived_spec(
        &tmp.0,
        1,
        &Spec {
            status: "Completed",
            ..spec(1)
        }
        .md(),
    );
    assert_fails_with(
        &tmp.0,
        &["spec ID 001 is used by both specs/001-example.md and specs/archive/001-completed.md"],
    );
}

#[test]
fn missing_and_duplicate_headings_fail() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md().replace("## Context\n", ""));
    write_active_spec(
        &tmp.0,
        2,
        &spec(2)
            .md()
            .replacen("## Decisions\n", "## Decisions\n\n## Decisions\n", 1),
    );
    assert_fails_with(
        &tmp.0,
        &[
            "specs/001-example.md: missing required heading '## Context'",
            "specs/002-example.md: duplicate required heading '## Decisions' (2 occurrences)",
        ],
    );
}

#[test]
fn impossible_and_reversed_dates_fail() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(
        &tmp.0,
        1,
        &Spec {
            created: "2026-13-01",
            ..spec(1)
        }
        .md(),
    );
    write_active_spec(
        &tmp.0,
        2,
        &Spec {
            created: "2026-08-10",
            updated: "2026-08-01",
            ..spec(2)
        }
        .md(),
    );
    assert_fails_with(
        &tmp.0,
        &[
            "specs/001-example.md: created '2026-13-01' is not a real YYYY-MM-DD date",
            "specs/002-example.md: updated (2026-08-01) is earlier than created (2026-08-10)",
        ],
    );
}

#[test]
fn structured_frontmatter_values_fail() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(
        &tmp.0,
        1,
        &Spec {
            extra: "tags: [a, b]\n",
            ..spec(1)
        }
        .md(),
    );
    write_active_spec(
        &tmp.0,
        2,
        &Spec {
            extra: "notes: |\n  blocked\n",
            ..spec(2)
        }
        .md(),
    );
    assert_fails_with(
        &tmp.0,
        &[
            "frontmatter key 'tags' uses structured YAML",
            "[a, b]",
            "frontmatter key 'notes' uses structured YAML",
        ],
    );
}

#[test]
fn missing_index_link_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    assert_fails_with(
        &tmp.0,
        &["specs/README.md: does not index specs/001-example.md in the Active specs table"],
    );
}

#[test]
fn missing_template_link_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_file(&tmp.0, "specs/README.md", &readme(&[], &[], false));
    assert_fails_with(
        &tmp.0,
        &["must link the template (000-template.md) exactly once"],
    );
}

#[test]
fn duplicate_index_link_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    let row = active_row(1, "Draft");
    write_file(
        &tmp.0,
        "specs/README.md",
        &readme(&[row.clone(), row], &[], true),
    );
    assert_fails_with(
        &tmp.0,
        &["links spec 001 ('001-example.md') more than once"],
    );
}

#[test]
fn wrong_section_index_link_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    write_file(
        &tmp.0,
        "specs/README.md",
        &readme(
            &[],
            &["| [001-example.md](001-example.md) | 2026-08-20 | Done |".to_string()],
            true,
        ),
    );
    assert_fails_with(
        &tmp.0,
        &["Archived specs table links '001-example.md'; expected 'archive/001-example.md'"],
    );
}

#[test]
fn broken_index_link_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_file(
        &tmp.0,
        "specs/README.md",
        &readme(
            &["| [001-missing.md](001-missing.md) | Draft | Done |".to_string()],
            &[],
            true,
        ),
    );
    assert_fails_with(
        &tmp.0,
        &["link '001-missing.md' does not resolve to a spec file"],
    );
}

#[test]
fn out_of_order_index_links_fail() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    write_active_spec(&tmp.0, 2, &spec(2).md());
    write_file(
        &tmp.0,
        "specs/README.md",
        &readme(&[active_row(2, "Draft"), active_row(1, "Draft")], &[], true),
    );
    assert_fails_with(
        &tmp.0,
        &["Active specs table is not in ascending ID order (002 before 001)"],
    );
}

#[test]
fn id_gap_fails() {
    let tmp = TempDir::new();
    write_valid_tree(&tmp.0);
    write_active_spec(&tmp.0, 1, &spec(1).md());
    write_active_spec(&tmp.0, 3, &spec(3).md());
    assert_fails_with(
        &tmp.0,
        &["spec IDs are not continuous from 001; missing: 002"],
    );
}

#[test]
fn python_compat_helpers() {
    assert_eq!(repr("it's"), "\"it's\"");
    assert_eq!(repr("a\\b'\""), "'a\\\\b\\'\"'");
    assert_eq!(parse_date("\"2026-08-01\""), Some((2026, 8, 1)));
    assert_eq!(parse_date("20260801"), Some((2026, 8, 1)));
    assert_eq!(parse_date("2026-W32-5"), Some((2026, 8, 7)));
    assert_eq!(parse_date("2026-02-29"), None);
    assert_eq!(parse_date("2028-02-29"), Some((2028, 2, 29)));
    assert_eq!(splitlines("a\r\nb\rc\n"), vec!["a", "b", "c"]);
    assert_eq!(md_links("[a](b) x [c](d)"), vec!["b", "d"]);
    assert_eq!(path_name("archive/001-x.md"), "001-x.md");
}
