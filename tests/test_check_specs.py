"""Regression tests for the tracked spec validator (scripts/check_specs.py).

Fixtures build a temporary repository tree and execute the real validator
against it via subprocess, asserting both the exit code and a specific
diagnostic for every failure path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts" / "check_specs.py"
TEMPLATE = REPO_ROOT / "specs" / "000-template.md"

HEADINGS = (
    "Context",
    "Goal and success criteria",
    "Scope and non-goals",
    "Decisions",
    "Workstreams",
    "Verification",
    "Documentation sync",
    "Progress and handoff",
    "Completion summary",
)


def run_check(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=root,
        capture_output=True,
        text=True,
    )


def write_file(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def spec_md(
    num: int,
    status: str = "Draft",
    created: str = "2026-08-01",
    updated: str = "2026-08-10",
    *,
    extra_frontmatter: str = "",
) -> str:
    body = "\n\n".join(f"## {heading}\n\nBody." for heading in HEADINGS)
    frontmatter = (
        "---\n"
        f'id: "{num:03d}"\n'
        'title: "Example spec"\n'
        f"status: {status}\n"
        f"created: {created}\n"
        f"updated: {updated}\n"
        'coordinator: "Coordinator"\n'
    )
    if extra_frontmatter:
        frontmatter += extra_frontmatter.rstrip("\n") + "\n"
    return frontmatter + "---\n\n" + body


def readme(
    *,
    active_rows: list[str] | None = None,
    archived_rows: list[str] | None = None,
    template_link: bool = True,
) -> str:
    lines = ["# Project specs", ""]
    if template_link:
        lines.append("Copy [000-template.md](000-template.md) to start a spec.")
    lines += [
        "",
        "## Active specs",
        "",
        "| Spec | Status | Outcome |",
        "| --- | --- | --- |",
    ]
    lines.extend(active_rows or [])
    lines += [
        "",
        "## Archived specs",
        "",
        "| Spec | Completed | Outcome |",
        "| --- | --- | --- |",
    ]
    lines.extend(archived_rows or [])
    lines.append("")
    return "\n".join(lines)


def active_row(num: int, target: str | None = None, status: str = "Draft") -> str:
    target = target or f"{num:03d}-example.md"
    return f"| [{target}]({target}) | {status} | Done |"


def archived_row(
    num: int, target: str | None = None, completed: str = "2026-08-20"
) -> str:
    target = target or f"archive/{num:03d}-completed.md"
    return f"| [{Path(target).name}]({target}) | {completed} | Done |"


def write_valid_tree(root: Path) -> None:
    (root / "specs" / "archive").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, root / "specs" / "000-template.md")
    write_file(root, "specs/README.md", readme())


def write_active_spec(root: Path, num: int, content: str) -> None:
    write_file(root, f"specs/{num:03d}-example.md", content)


def write_archived_spec(root: Path, num: int, content: str) -> None:
    write_file(root, f"specs/archive/{num:03d}-completed.md", content)


def test_repository_spec_tree_passes() -> None:
    result = run_check(REPO_ROOT)
    assert result.returncode == 0, result.stdout
    assert "Specs check passed." in result.stdout


def test_empty_index_and_template_pass(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    result = run_check(tmp_path)
    assert result.returncode == 0, result.stdout
    assert "Specs check passed." in result.stdout


def test_indexed_active_and_archived_specs_pass(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1, status="Active"))
    write_archived_spec(tmp_path, 2, spec_md(2, status="Completed"))
    write_file(
        tmp_path,
        "specs/README.md",
        readme(
            active_rows=[active_row(1, status="Active")],
            archived_rows=[archived_row(2)],
        ),
    )
    result = run_check(tmp_path)
    assert result.returncode == 0, result.stdout
    assert "Specs check passed." in result.stdout


def test_invalid_filename_and_reserved_zero_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_file(tmp_path, "specs/001-Uppercase.md", spec_md(1))
    write_file(tmp_path, "specs/000-copy.md", spec_md(0))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/001-Uppercase.md: filename does not match NNN-kebab-case.md"
        in result.stdout
    )
    assert (
        "specs/000-copy.md: ID 000 is reserved for specs/000-template.md"
        in result.stdout
    )


def test_frontmatter_id_mismatch_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(2))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/001-example.md: frontmatter id '002' does not match filename ID 001"
        in result.stdout
    )


def test_completed_status_in_active_directory_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1, status="Completed"))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/001-example.md: status 'Completed' is not allowed in specs/ "
        "(expected Draft or Active)" in result.stdout
    )


def test_duplicate_ids_across_directories_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    write_archived_spec(tmp_path, 1, spec_md(1, status="Completed"))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "spec ID 001 is used by both specs/001-example.md and specs/archive/001-completed.md"
        in result.stdout
    )


def test_missing_and_duplicate_headings_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    missing_context = spec_md(1).replace("## Context\n", "")
    duplicated_decisions = spec_md(2).replace(
        "## Decisions\n",
        "## Decisions\n\n## Decisions\n",
        1,
    )
    write_active_spec(tmp_path, 1, missing_context)
    write_active_spec(tmp_path, 2, duplicated_decisions)
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/001-example.md: missing required heading '## Context'" in result.stdout
    )
    assert (
        "specs/002-example.md: duplicate required heading '## Decisions' (2 occurrences)"
        in result.stdout
    )


def test_impossible_and_reversed_dates_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1, created="2026-13-01"))
    write_active_spec(
        tmp_path,
        2,
        spec_md(2, created="2026-08-10", updated="2026-08-01"),
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/001-example.md: created '2026-13-01' is not a real YYYY-MM-DD date"
        in result.stdout
    )
    assert (
        "specs/002-example.md: updated (2026-08-01) is earlier than created (2026-08-10)"
        in result.stdout
    )


def test_structured_frontmatter_values_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1, extra_frontmatter="tags: [a, b]\n"))
    write_active_spec(
        tmp_path, 2, spec_md(2, extra_frontmatter="notes: |\n  blocked\n")
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "frontmatter key 'tags' uses structured YAML" in result.stdout
    assert "[a, b]" in result.stdout
    assert "frontmatter key 'notes' uses structured YAML" in result.stdout


def test_missing_index_link_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "specs/README.md: does not index specs/001-example.md in the Active specs table"
        in result.stdout
    )


def test_missing_template_link_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_file(tmp_path, "specs/README.md", readme(template_link=False))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "must link the template (000-template.md) exactly once" in result.stdout


def test_duplicate_index_link_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    row = active_row(1)
    write_file(tmp_path, "specs/README.md", readme(active_rows=[row, row]))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "links spec 001 ('001-example.md') more than once" in result.stdout


def test_wrong_section_index_link_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    write_file(
        tmp_path,
        "specs/README.md",
        readme(
            archived_rows=["| [001-example.md](001-example.md) | 2026-08-20 | Done |"]
        ),
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "Archived specs table links '001-example.md'; expected 'archive/001-example.md'"
        in result.stdout
    )


def test_broken_index_link_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_file(
        tmp_path,
        "specs/README.md",
        readme(active_rows=["| [001-missing.md](001-missing.md) | Draft | Done |"]),
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "link '001-missing.md' does not resolve to a spec file" in result.stdout


def test_out_of_order_index_links_fail(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    write_active_spec(tmp_path, 2, spec_md(2))
    write_file(
        tmp_path,
        "specs/README.md",
        readme(active_rows=[active_row(2), active_row(1)]),
    )
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert (
        "Active specs table is not in ascending ID order (002 before 001)"
        in result.stdout
    )


def test_id_gap_fails(tmp_path: Path) -> None:
    write_valid_tree(tmp_path)
    write_active_spec(tmp_path, 1, spec_md(1))
    write_active_spec(tmp_path, 3, spec_md(3))
    result = run_check(tmp_path)
    assert result.returncode == 1
    assert "spec IDs are not continuous from 001; missing: 002" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
