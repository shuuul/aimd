"""Readable HTML extraction through the Defuddle CLI.

Defuddle is distributed as a Node/TypeScript package. This wrapper keeps the
Python package boundary explicit and invokes ``npx defuddle`` when callers opt
into HTML extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess


@dataclass(slots=True, frozen=True)
class DefuddleResult:
    """Parsed HTML content and metadata returned by Defuddle."""

    content: str
    title: str | None = None
    author: str | None = None
    source: str | None = None


def extract_html_with_defuddle(
    source: str | Path,
    *,
    markdown: bool = True,
    npx_command: str = "npx",
) -> DefuddleResult:
    """Extract readable content from a URL, HTML file, or HTML piped source.

    Requires Node.js/npm at runtime. ``source`` is passed to
    ``npx defuddle parse`` as a URL or local file path.
    """
    command = [npx_command, "defuddle", "parse", str(source), "--json"]
    if markdown:
        command.append("--markdown")

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"defuddle failed: {stderr}")

    payload = json.loads(completed.stdout)
    content = payload.get("contentMarkdown") or payload.get("content") or ""
    return DefuddleResult(
        content=content,
        title=payload.get("title"),
        author=payload.get("author"),
        source=payload.get("source") or payload.get("url"),
    )
