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

from aimd.core.errors import BackendUnavailableError, ProcessingFailedError


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

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise BackendUnavailableError(
            f"npx is required for Defuddle HTML extraction ({npx_command} not found)"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise ProcessingFailedError(f"defuddle failed: {stderr}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProcessingFailedError(f"defuddle returned invalid JSON: {exc}") from exc

    content = payload.get("contentMarkdown") or payload.get("content") or ""
    return DefuddleResult(
        content=content,
        title=payload.get("title"),
        author=payload.get("author"),
        source=payload.get("source") or payload.get("url"),
    )
