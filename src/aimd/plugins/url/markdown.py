"""Transcript cleanup and Markdown formatting for URL processing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence

_TIMESTAMP_RE = re.compile(
    r"\d{1,2}:\d{2}:\d{2}[.,]\d{2,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{2,3}"
)
_SEQUENCE_RE = re.compile(r"^\d+$")
_VTT_HEADER_RE = re.compile(r"^WEBVTT", re.IGNORECASE)
_VTT_META_RE = re.compile(r"^(Kind:|Language:)", re.IGNORECASE)
_VTT_WORD_TIMESTAMP_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}\.\d{3}>")
_SUBTITLE_TAG_RE = re.compile(
    r"</?(?:b|c(?:\.[^>]*)?|font(?:\s+[^>]*)?|i|lang(?:\s+[^>]*)?|rt|ruby|u|v(?:\s+[^>]*)?)>",
    re.IGNORECASE,
)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SPEAKER_MARKERS = {">>", "<<"}
_SENTENCE_END_RE = re.compile(r"[.!?。！？…][\"'”’)\]]*$")
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "etc.",
    "inc.",
    "ltd.",
    "st.",
    "u.s.",
    "u.k.",
    "e.g.",
    "i.e.",
    "ph.d.",
    "phd.",
}
_NEXT_SENTENCE_RE = re.compile(r"^[\"'“‘(\[]?(?:[A-Z]|[\u3400-\u9fff])")


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _merge_rolling_caption_lines(lines: Iterable[str]) -> str:
    """Join caption cues, dropping YouTube ASR rolling-window overlap.

    Auto captions repeat a sliding window of words in successive cues. Matching
    the longest suffix/prefix overlap keeps the linear transcript stable across
    json3, srv1, SRT, VTT, and TTML.
    """
    tokens: list[str] = []
    for line in lines:
        current = line.split()
        if not current:
            continue
        if not tokens:
            tokens.extend(current)
            continue
        max_overlap = min(len(tokens), len(current))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if tokens[-size:] == current[:size]:
                overlap = size
                break
        tokens.extend(current[overlap:])
    return _format_linear_transcript(tokens)


def _split_speaker_marker(token: str) -> tuple[bool, str]:
    """Return whether a token starts a YouTube speaker turn, plus remainder."""
    if token in _SPEAKER_MARKERS:
        return True, ""
    for marker in _SPEAKER_MARKERS:
        if token.startswith(marker):
            return True, token[len(marker) :]
    return False, token


def _is_sentence_end(token: str, next_token: str | None) -> bool:
    """Return True when ``token`` should close a transcript line."""
    if token.lower() in _ABBREVIATIONS:
        return False
    if re.fullmatch(r"[A-Za-z]\.", token):
        return False
    if not _SENTENCE_END_RE.search(token):
        return False
    if next_token is None:
        return True
    speaker_change, remainder = _split_speaker_marker(next_token)
    if speaker_change:
        return True
    return bool(_NEXT_SENTENCE_RE.match(remainder or next_token))


def _format_linear_transcript(tokens: Sequence[str]) -> str:
    """Turn linearized caption tokens into one sentence per line.

    YouTube auto captions insert ``>>`` at speaker changes. Keep the marker at
    the start of that turn's first line.
    """
    lines: list[str] = []
    current: list[str] = []
    prefix: str | None = None

    def flush() -> None:
        nonlocal prefix
        if not current:
            return
        text = " ".join(current)
        if prefix:
            lines.append(f"{prefix} {text}")
            prefix = None
        else:
            lines.append(text)
        current.clear()

    pending = list(tokens)
    index = 0
    while index < len(pending):
        token = pending[index]
        speaker_change, remainder = _split_speaker_marker(token)
        if speaker_change:
            flush()
            prefix = ">>" if token.startswith(">>") else "<<"
            if remainder:
                pending[index] = remainder
                continue
            index += 1
            continue
        current.append(token)
        next_token = pending[index + 1] if index + 1 < len(pending) else None
        if _is_sentence_end(token, next_token):
            flush()
        index += 1
    flush()
    return "\n".join(lines)


def _ttml_cue_texts(text: str) -> list[str] | None:
    """Extract per-cue text from TTML ``<p>`` elements."""
    normalized = _BR_RE.sub(" ", text)
    try:
        root = ET.fromstring(normalized)
    except ET.ParseError:
        return None
    cues: list[str] = []
    for element in root.iter():
        if _local_tag(element.tag) != "p":
            continue
        body = " ".join("".join(element.itertext()).split())
        if body:
            cues.append(body)
    return cues


def _timed_cue_texts(text: str) -> list[str]:
    """Extract cue text from SRT/VTT bodies."""
    text_lines: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _VTT_HEADER_RE.match(stripped):
            continue
        if _VTT_META_RE.match(stripped):
            continue
        if _TIMESTAMP_RE.search(stripped):
            continue
        if _SEQUENCE_RE.match(stripped):
            continue
        if stripped.startswith("NOTE "):
            continue
        # YouTube enhanced VTT puts word-level timestamps and ``<c>`` tags in
        # cue text. Remove them before overlap matching; otherwise identical
        # rolling captions look different and are emitted more than once.
        cue_text = _VTT_WORD_TIMESTAMP_RE.sub("", stripped)
        cue_text = _SUBTITLE_TAG_RE.sub("", cue_text).strip()
        if cue_text:
            text_lines.append(cue_text)
    return text_lines


def strip_subtitle_formatting(text: str) -> str:
    """Strip subtitle formatting, returning a format-independent transcript.

    Handles SRT, WebVTT, and TTML. YouTube json3/srv1/srv3 tracks are converted
    to SRT before this runs. Rolling-window auto captions are linearized so the
    default Markdown body stays stable across formats, then split one sentence
    per line. YouTube ``>>`` speaker-change markers are kept at the start of
    the new turn. Returns the original text unchanged if no subtitle formatting
    is detected.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    head = stripped[:500]
    if "<tt " in head or 'xmlns="http://www.w3.org/ns/ttml"' in head:
        cues = _ttml_cue_texts(stripped)
        if cues:
            return _merge_rolling_caption_lines(cues)

    lines = stripped.splitlines()
    has_timestamps = any(_TIMESTAMP_RE.search(line) for line in lines[:30])
    if not has_timestamps:
        return text

    return _merge_rolling_caption_lines(_timed_cue_texts(stripped))


def format_content(
    info_dict: dict[str, object], content: str | None, platform: str | None = None
) -> str:
    """Format video metadata and extracted content into markdown."""
    title = info_dict.get("title", "Unknown Title")
    description = info_dict.get("description", "No description available")
    channel = info_dict.get("channel", info_dict.get("uploader", "Unknown"))
    duration = info_dict.get("duration", 0)
    view_count = info_dict.get("view_count", 0)
    upload_date = info_dict.get("upload_date", "")
    webpage_url = info_dict.get("webpage_url", "")

    formatted_content = f"""# {title}

**Channel:** {channel}
**Platform:** {platform or "unknown"}
**Duration:** {duration} seconds
**Upload Date:** {upload_date}
**View Count:** {view_count:,} views
**URL:** {webpage_url}

## Description

{description}

## Content

"""

    if content and content.strip():
        formatted_content += content
    else:
        formatted_content += "*No subtitles or transcription available for this video.*"

    return formatted_content
