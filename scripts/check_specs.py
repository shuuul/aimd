#!/usr/bin/env python3
"""Validate the tracked spec tree rooted at the current working directory.

Dependency-free (Python stdlib only); run from the repository root:

    uv run check:specs

or directly with `python scripts/check_specs.py`. Paths resolve from
`Path.cwd()` so the same script validates the real tree and pytest fixtures
that copy it into a temporary repository.

Exit code 0 with "Specs check passed." when every invariant holds; exit code
1 with "Specs check failed:" followed by one line per problem otherwise.
All problems are collected and printed together.

Frontmatter is deliberately restricted to flat, scalar `key: value` data, so
this validator parses it with an ad hoc parser that rejects structured YAML
(arrays, maps, block scalars, anchors, aliases, tags) instead of pulling in a
YAML dependency.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

REQUIRED_KEYS = ("id", "title", "status", "created", "updated", "coordinator")
REQUIRED_HEADINGS = (
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
ACTIVE_STATUSES = {"Draft", "Active"}
ARCHIVE_STATUS = "Completed"
FORMAL_ID_RE = re.compile(r"^(?P<num>\d{3})-(?P<slug>[a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)]+)\)")
SECTION_RE = re.compile(r"^##\s+(Active specs|Archived specs)\s*$")
SEPARATOR_RE = re.compile(r":?-+:?")

TEMPLATE_NAME = "000-template.md"
README_NAME = "README.md"


def parsing_headings(text: str) -> list[str]:
    """Level-2 headings outside fenced code blocks."""
    headings: list[str] = []
    fence = None  # (char, opener length)
    for line in text.splitlines():
        stripped = line.strip()
        opener = re.match(r"^(?P<mark>`{3,}|~{3,})", stripped)
        if opener:
            mark = opener.group("mark")
            if fence is None:
                fence = (mark[0], len(mark))
            elif stripped.startswith(fence[0]) and len(stripped) >= fence[1]:
                fence = None
            continue
        if fence is None:
            match = re.match(r"^##\s+(\S.*?)\s*$", stripped)
            if match:
                headings.append(match.group(1))
    return headings


def parse_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse flat `key: value` frontmatter; reject anything structured."""
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, ["frontmatter must start with a '---' delimiter line"]
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, ["frontmatter is missing the closing '---' delimiter line"]
    frontmatter: dict[str, str] = {}
    for i in range(1, end):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            errors.append(
                f"frontmatter line {i + 1} must be flat 'key: value', got comment: {stripped!r}"
            )
            continue
        if ":" not in raw:
            errors.append(
                f"frontmatter line {i + 1} is not flat 'key: value': {stripped!r}"
            )
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            errors.append(f"frontmatter line {i + 1} has an invalid key {key!r}")
            continue
        if not value:
            errors.append(f"frontmatter key {key!r} has an empty value")
            continue
        if value[:1] in "|>&*!{[":
            errors.append(
                f"frontmatter key {key!r} uses structured YAML ({value!r}); "
                "only flat scalar values are allowed"
            )
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key in frontmatter:
            errors.append(f"duplicate frontmatter key {key!r}")
        frontmatter[key] = value
    return frontmatter, errors


def parse_date(value: str | None) -> dt.date | None:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value.strip().strip("\"'"))
    except ValueError:
        return None


def check_spec_file(root: Path, path: Path, *, is_template: bool) -> list[str]:
    rel = path.relative_to(root).as_posix()
    errors: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{rel}: cannot read file: {exc}"]

    frontmatter, fm_errors = parse_frontmatter(text)
    errors.extend(f"{rel}: {msg}" for msg in fm_errors)

    if not is_template:
        match = FORMAL_ID_RE.match(path.name)
        if match is None:
            errors.append(f"{rel}: filename does not match NNN-kebab-case.md")
        else:
            file_id = int(match.group("num"))
            if file_id == 0:
                errors.append(f"{rel}: ID 000 is reserved for specs/000-template.md")
            else:
                try:
                    fm_id = int(frontmatter.get("id", "").strip("\"'"))
                except ValueError:
                    fm_id = -1
                if fm_id != file_id:
                    errors.append(
                        f"{rel}: frontmatter id {frontmatter.get('id')!r} "
                        f"does not match filename ID {file_id:03d}"
                    )

    for key in REQUIRED_KEYS:
        if key not in frontmatter:
            errors.append(f"{rel}: frontmatter is missing required key {key!r}")

    if not is_template:
        created = (
            parse_date(frontmatter["created"]) if "created" in frontmatter else None
        )
        updated = (
            parse_date(frontmatter["updated"]) if "updated" in frontmatter else None
        )
        if "created" in frontmatter and created is None:
            errors.append(
                f"{rel}: created {frontmatter['created']!r} is not a real YYYY-MM-DD date"
            )
        if "updated" in frontmatter and updated is None:
            errors.append(
                f"{rel}: updated {frontmatter['updated']!r} is not a real YYYY-MM-DD date"
            )
        if created is not None and updated is not None and updated < created:
            errors.append(
                f"{rel}: updated ({updated}) is earlier than created ({created})"
            )

    in_archive = path.parent.name == "archive"
    status = frontmatter.get("status")
    if in_archive:
        if status != ARCHIVE_STATUS:
            errors.append(
                f"{rel}: status {status!r} is not allowed in specs/archive/ "
                f"(expected {ARCHIVE_STATUS})"
            )
    elif status not in ACTIVE_STATUSES:
        errors.append(
            f"{rel}: status {status!r} is not allowed in specs/ (expected Draft or Active)"
        )

    counts: dict[str, int] = {}
    for heading in parsing_headings(text):
        counts[heading] = counts.get(heading, 0) + 1
    for heading in REQUIRED_HEADINGS:
        n = counts.get(heading, 0)
        if n == 0:
            errors.append(f"{rel}: missing required heading '## {heading}'")
        elif n > 1:
            errors.append(
                f"{rel}: duplicate required heading '## {heading}' ({n} occurrences)"
            )

    return errors


def check_readme(
    root: Path,
    readme: Path,
    active_index: dict[int, str],
    archived_index: dict[int, str],
) -> list[str]:
    rel = readme.relative_to(root).as_posix()
    errors: list[str] = []
    try:
        text = readme.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{rel}: cannot read file: {exc}"]

    template_links = [
        m.group("target")
        for line in text.splitlines()
        for m in MD_LINK_RE.finditer(line)
        if m.group("target") == TEMPLATE_NAME
    ]
    if len(template_links) != 1:
        errors.append(f"{rel}: must link the template ({TEMPLATE_NAME}) exactly once")

    lines = text.splitlines()
    h2 = [i for i, line in enumerate(lines) if re.match(r"^##\s+", line.strip())]
    sections: dict[str, tuple[int, int]] = {}
    for idx, i in enumerate(h2):
        match = SECTION_RE.match(lines[i].strip())
        if match is None:
            continue
        end = h2[idx + 1] if idx + 1 < len(h2) else len(lines)
        sections[match.group(1)] = (i + 1, end)

    for section, (start, end) in sections.items():
        expected = active_index if section == "Active specs" else archived_index
        linked: dict[int, str] = {}
        prev: int | None = None
        for lineno in range(start, end):
            stripped = lines[lineno].strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(SEPARATOR_RE.fullmatch(cell) for cell in cells):
                continue
            if cells and cells[0] == "Spec":
                continue
            link_matches = list(MD_LINK_RE.finditer(stripped))
            if not link_matches:
                errors.append(f"{rel}: {section} row has no spec link: {stripped!r}")
                continue
            if len(link_matches) > 1:
                errors.append(
                    f"{rel}: {section} row has multiple spec links: {stripped!r}"
                )
            target = link_matches[0].group("target")
            if target == TEMPLATE_NAME:
                errors.append(f"{rel}: {section} table must not link the template")
                continue
            match = FORMAL_ID_RE.match(Path(target).name)
            if match is None or int(match.group("num")) == 0:
                errors.append(
                    f"{rel}: {section} table links a non-spec file {target!r}"
                )
                continue
            link_id = int(match.group("num"))
            expected_target = f"{link_id:03d}-{match.group('slug')}.md"
            if section == "Archived specs":
                expected_target = f"archive/{expected_target}"
            if target != expected_target:
                errors.append(
                    f"{rel}: {section} table links {target!r}; "
                    f"expected {expected_target!r} (wrong section or path)"
                )
                continue
            if not (root / "specs" / target).is_file():
                errors.append(f"{rel}: link {target!r} does not resolve to a spec file")
                continue
            if link_id in linked:
                errors.append(
                    f"{rel}: {section} table links spec {link_id:03d} "
                    f"({target!r}) more than once"
                )
                continue
            linked[link_id] = target
            if prev is not None and link_id <= prev:
                errors.append(
                    f"{rel}: {section} table is not in ascending ID order "
                    f"({prev:03d} before {link_id:03d})"
                )
            prev = link_id

        for link_id, target in sorted(expected.items()):
            if link_id not in linked:
                errors.append(
                    f"{rel}: does not index specs/{target} in the {section} table"
                )
        for link_id in linked:
            if link_id not in expected:
                kind = "active" if section == "Active specs" else "archived"
                errors.append(
                    f"{rel}: {section} table indexes spec {link_id:03d}, "
                    f"which is not a {kind} spec"
                )

    return errors


def check_tree(root: Path) -> list[str]:
    errors: list[str] = []
    specs_dir = root / "specs"
    archive_dir = specs_dir / "archive"
    readme = specs_dir / "README.md"
    template_path = specs_dir / TEMPLATE_NAME

    if not specs_dir.is_dir():
        errors.append("specs/ directory is missing")
    if not archive_dir.is_dir():
        errors.append("specs/archive/ directory is missing")
    if not readme.is_file():
        errors.append("specs/README.md is missing")
    if not template_path.is_file():
        errors.append("specs/000-template.md is missing")
    if not specs_dir.is_dir():
        return errors

    active_index: dict[int, str] = {}
    archived_index: dict[int, str] = {}

    def add_index(
        index: dict[int, str], path: Path, label: str, into: list[str]
    ) -> None:
        match = FORMAL_ID_RE.match(path.name)
        if match is None:
            return
        num = int(match.group("num"))
        if num == 0:
            return
        if num in index:
            into.append(
                f"duplicate spec ID {num:03d} used by {index[num]!r} and "
                f"{path.name!r} in {label}"
            )
            return
        index[num] = path.name

    for path in sorted(specs_dir.glob("*.md")):
        if path.name == TEMPLATE_NAME:
            errors.extend(check_spec_file(root, path, is_template=True))
            continue
        if path.name == README_NAME:
            continue
        errors.extend(check_spec_file(root, path, is_template=False))
        add_index(active_index, path, "specs/", errors)
    for path in sorted(archive_dir.glob("*.md")):
        errors.extend(check_spec_file(root, path, is_template=False))
        add_index(archived_index, path, "specs/archive/", errors)

    all_ids = set(active_index) | set(archived_index)
    if all_ids:
        missing = [f"{i:03d}" for i in range(1, max(all_ids) + 1) if i not in all_ids]
        if missing:
            errors.append(
                f"spec IDs are not continuous from 001; missing: {', '.join(missing)}"
            )
        for dup in sorted(set(active_index) & set(archived_index)):
            errors.append(
                f"spec ID {dup:03d} is used by both specs/{active_index[dup]} "
                f"and specs/archive/{archived_index[dup]}"
            )

    errors.extend(check_readme(root, readme, active_index, archived_index))
    return errors


def main() -> int:
    problems = check_tree(Path.cwd())
    if problems:
        print("Specs check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Specs check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
