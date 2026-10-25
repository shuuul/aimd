#!/usr/bin/env python
"""Export golden URL-plugin fixtures for the Rust ``aimd-url`` crate (WS-18).

Runs the Python reference implementation (``aimd.plugins.url``) over
deterministic inputs and writes into
``crates/aimd-url/tests/golden/subtitles/``:

- ``inputs/<name>.<ext>``  subtitle payloads. Synthetic cases are (re)written
  by this script; ``real_*`` files come from ``--fetch-youtube VIDEO_ID`` (a
  short public video, downloaded once through yt-dlp, every track format).
- ``cases.json``           per input: ``normalized`` =
  ``_normalize_subtitle_payload(input)``, ``stripped`` =
  ``strip_subtitle_formatting(normalized)``, ``stripped_raw`` =
  ``strip_subtitle_formatting(input)`` (``null`` where Python returns None).
- ``text.json``            pure-function parity: language preference and
  inference, ``build_metadata_context``, ``format_content``,
  ``parse_cookies_from_browser``, ``is_url``, ``detect_platform``,
  ``_format_linear_transcript``.

Usage::

    uv run python scripts/export_url_subtitle_reference.py [--out DIR] [--fetch-youtube jNQXAC9IVRw]

Deterministic: fixed synthetic inputs, no wall-clock, no randomness. Every
output file stays well under 1 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from aimd.plugins.url import detect_platform, is_url
from aimd.plugins.url.cookies import parse_cookies_from_browser
from aimd.plugins.url.markdown import (
    _format_linear_transcript,
    format_content,
    strip_subtitle_formatting,
)
from aimd.plugins.url.metadata import build_metadata_context
from aimd.plugins.url.subtitles import (
    _normalize_subtitle_payload,
    detect_content_language,
    get_preferred_languages,
    resolve_subtitle_language,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "crates" / "aimd-url" / "tests" / "golden" / "subtitles"
MAX_BYTES = 1_000_000

# Interview-style transcript used for the rolling auto-caption cases.
ROLLING_TEXT = (
    "you've discovered something about the brain I think is quite surprising. "
    ">> So we found that in fathers the brain is actually shrinking. Dr. Darby "
    "found that in the U.S. and the U.K., e.g. in Boston. >> Yeah, go for it. "
    '"Really?" she asked. (It was.) Mr. Smith said "No!" then left... '
    "Version 2.0 shipped in Jan. 2024 vs. the old one. A. B. Carter agreed."
)
ROLLING_ZH = (
    "大家好，欢迎来到节目。今天我们聊聊大脑的秘密！你知道吗？父亲的大脑会变小。"
    "这是真的…… 研究人员发现了这一点。"
)


def _ts(ms: int, sep: str) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def _rolling_windows(words: list[str], width: int, step: int) -> list[list[str]]:
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + width])
        if start + width >= len(words):
            break
    return windows


def _rolling_vtt(words: list[str]) -> str:
    """YouTube enhanced VTT: rolling windows with word timestamps + <c> tags."""
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    t = 80
    for window in _rolling_windows(words, 8, 4):
        lines.append(f"{_ts(t, '.')} --> {_ts(t + 2500, '.')} align:start position:0%")
        first, rest = window[0], window[1:]
        parts = [first]
        for i, word in enumerate(rest):
            parts.append(f"<{_ts(t + 100 * (i + 1), '.')}><c> {word}</c>")
        lines.append("".join(parts))
        lines.append("")
        t += 1300
    return "\n".join(lines)


def _rolling_srt(words: list[str], crlf: bool = False) -> str:
    blocks = []
    t = 80
    for index, window in enumerate(_rolling_windows(words, 7, 3), start=1):
        blocks.append(
            f"{index}\n{_ts(t, ',')} --> {_ts(t + 2000, ',')}\n{' '.join(window)}"
        )
        t += 1100
    text = "\n\n".join(blocks) + "\n"
    return text.replace("\n", "\r\n") if crlf else text


def _rolling_json3(words: list[str]) -> str:
    events: list[dict] = [{"tStartMs": 0, "dDurationMs": 0, "id": 1}]
    t = 80
    for window in _rolling_windows(words, 6, 3):
        segs = [{"utf8": window[0]}] + [
            {"utf8": f" {w}", "tOffsetMs": 120 * (i + 1)}
            for i, w in enumerate(window[1:])
        ]
        events.append({"tStartMs": t, "dDurationMs": 2400, "segs": segs})
        events.append({"tStartMs": t + 1200, "segs": [{"utf8": "\n"}]})
        t += 1200
    events.append({"tStartMs": t, "segs": [{"utf8": "tail without duration"}]})
    events.append({"tStartMs": t + 10.7, "dDurationMs": 0, "segs": [{"utf8": "x"}]})
    return json.dumps({"wireMagic": "pb3", "pens": [{}], "events": events}, indent=1)


def _rolling_srv1(words: list[str]) -> str:
    rows = ['<?xml version="1.0" encoding="utf-8" ?><transcript>']
    t = 0.08
    for window in _rolling_windows(words, 6, 3):
        text = " ".join(window).replace("&", "&amp;").replace("'", "&amp;#39;")
        rows.append(f'<text start="{t:.2f}" dur="2.4">{text}</text>')
        t += 1.2
    rows.append('<text start="bad" dur="1">skipped</text>')
    rows.append('<text start="99.5" dur="">empty dur</text>')
    rows.append("</transcript>")
    return "".join(rows)


def _rolling_srv3(words: list[str]) -> str:
    rows = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        '<timedtext format="3">',
        "<body>",
    ]
    t = 80
    for window in _rolling_windows(words, 6, 3):
        segs = "".join(
            f'<s t="{100 * i}" ac="0">{w} </s>' for i, w in enumerate(window)
        )
        rows.append(f'<p t="{t}" d="2400" w="1">{segs}</p>')
        rows.append(f'<p t="{t + 1200}" w="1" a="1">\n</p>')
        t += 1200
    rows += ["</body>", "</timedtext>"]
    return "\n".join(rows)


def _rolling_ttml(words: list[str], prefixed: bool = False) -> str:
    p = "tt:" if prefixed else ""
    ns = (
        'xmlns:tt="http://www.w3.org/ns/ttml"'
        if prefixed
        else 'xmlns="http://www.w3.org/ns/ttml"'
    )
    rows = [
        '<?xml version="1.0" encoding="utf-8" ?>',
        f'<{p}tt xml:lang="en" {ns} xmlns:tts="http://www.w3.org/ns/ttml#styling">',
        f"<{p}head><{p}styling/></{p}head>",
        f"<{p}body><{p}div>",
    ]
    t = 80
    for i, window in enumerate(_rolling_windows(words, 6, 3)):
        half = len(window) // 2
        body = " ".join(window[:half]) + "<br/>" + " ".join(window[half:])
        if i % 2:
            body = f'<{p}span tts:color="white">{body}</{p}span>'
        rows.append(
            f'<{p}p begin="{_ts(t, ".")}" end="{_ts(t + 2400, ".")}">{body.replace("&", "&amp;")}</{p}p>'
        )
        t += 1200
    rows += [f"</{p}div></{p}body>", f"</{p}tt>"]
    return "\n".join(rows)


def synthetic_inputs() -> dict[str, str]:
    words = ROLLING_TEXT.split()
    zh_words = list(ROLLING_ZH)
    return {
        "rolling.vtt": _rolling_vtt(words),
        "rolling.srt": _rolling_srt(words),
        "rolling_crlf.srt": _rolling_srt(words, crlf=True),
        "rolling.json3": _rolling_json3(words),
        "rolling.srv1": _rolling_srv1(words),
        "rolling.srv3": _rolling_srv3(words),
        "rolling.ttml": _rolling_ttml(words),
        "rolling_prefixed.ttml": _rolling_ttml(words, prefixed=True),
        "rolling_zh.vtt": _rolling_vtt(zh_words),
        "rolling_zh.json3": _rolling_json3(zh_words),
        "tags.srt": (
            "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i> <b>there</b>, "
            '<font color="#ffff00">friend</font>.\n\n'
            "2\n00:00:02,500 --> 00:00:04,000\n<v Bob>We meet again.</v> <u>Yes</u> "
            "<ruby>漢<rt>kan</rt></ruby> <c.yellow>tagged</c> <lang en>text</lang> "
            "<bx>kept</bx> <br> 3 < 4 > 2\n\n"
            "3\n00:00:04,000 --> 00:00:05,000\n42\n\n"
            "4\n0:00:05.00-->0:00:06.00\n１２３ fullwidth digits stay?\n"
        ),
        "notes.vtt": (
            "WEBVTT - with title\nKIND: captions\nlanguage: ja\n\n"
            "NOTE This is a comment\n\nNOTE\nmulti-line note body\n\n"
            "cue-1\n00:00.000 --> 00:01.000\nshort timestamps are not cue lines\n\n"
            "00:00:01.000 --> 00:00:02.000 line:90%\n- dash line one.\n- Dash line two.\n\n"
            "00:00:02.000 --> 00:00:03.000\n　ideographic　space\x1c sep\x1f end.\n"
        ),
        "bilibili.srt": (
            "1\n00:00:00,500 --> 00:00:03,000\n大家好，欢迎收看本期视频。\n\n"
            "2\n00:00:03,000 --> 00:00:06,200\n今天我们来聊一聊 Rust 和 Python！\n\n"
            "3\n00:00:06,200 --> 00:00:09,000\n你觉得呢？ OK. Let's go.\n"
        ),
        "speakers.srt": (
            "1\n00:00:00,000 --> 00:00:01,000\n>> Hi.\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\n>>>>double marker. <<Left marker? ok\n\n"
            "3\n00:00:02,000 --> 00:00:03,000\n>> >> Two in a row. etc. Ph.D. done\n"
        ),
        "empty_events.json3": '{"wireMagic": "pb3", "events": [{"tStartMs": 0}]}',
        "malformed.ttml": '<tt xmlns="http://www.w3.org/ns/ttml"><p>unclosed &nbsp;</tt>',
        "malformed_with_ts.ttml": (
            '<tt xmlns="http://www.w3.org/ns/ttml"><p>bad</tt>\n'
            "00:00:01.000 --> 00:00:02.000\nfallback cue text.\n"
        ),
        "plain.txt": "  Just a plain transcript.\nNo timestamps here.  \n",
        "whitespace.txt": " \n\t\n",
    }


def fetch_youtube(video_id: str, inputs: Path) -> None:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("json3", "srv1", "srv3", "ttml", "srt", "vtt"):
            opts = {
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "ignoreconfig": True,
                "skip_download": True,
                "writesubtitles": True,
                "subtitleslangs": ["en"],
                "subtitlesformat": fmt,
                "outtmpl": str(Path(tmp) / "sub.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            for path in Path(tmp).glob("sub.*"):
                target = inputs / f"real_{video_id}.{fmt}"
                target.write_bytes(path.read_bytes())
                path.unlink()


def subtitle_cases(inputs: Path) -> list[dict]:
    cases = []
    for path in sorted(inputs.iterdir()):
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8")
        normalized = _normalize_subtitle_payload(text)
        cases.append(
            {
                "name": path.name,
                "normalized": normalized,
                "stripped": (
                    strip_subtitle_formatting(normalized)
                    if normalized is not None
                    else None
                ),
                "stripped_raw": strip_subtitle_formatting(text),
            }
        )
    return cases


INFO_SAMPLES = [
    {},
    {"title": "Example", "webpage_url": "https://example.com/video"},
    {
        "title": "聊聊 PostgreSQL 与 Redis",
        "uploader": "SomeChannel",
        "channel": "Ignored Channel",
        "description": "本期嘉宾: 张三, 讨论 CUDA 优化。\n" + "细节" * 600,
        "tags": ["数据库", " PostgreSQL ", "", 7, "Redis"]
        + [f"t{i}" for i in range(30)],
        "chapters": [{"title": "开场"}, {"title": " "}, "x", {"title": "深入 CUDA"}]
        + [{"title": f"c{i}"} for i in range(40)],
        "duration": 123.456,
        "view_count": 1234567,
        "upload_date": "20260307",
        "webpage_url": "https://www.bilibili.com/video/BV1",
    },
    {
        "title": "Interview with Linus Torvalds",
        "uploader": "",
        "channel": "Linux Foundation",
        "description": "A conversation about the Linux kernel and Git.",
        "tags": ["linux", "git"],
        "duration": 19,
        "view_count": 446483224,
    },
    {"title": "  ", "tags": [], "chapters": None},
    {"title": "t", "description": "x" * 5000},
    {"title": "Float views", "view_count": 1234.5, "duration": 1e16},
]

LANGUAGE_AVAILABLE = [
    [],
    ["zh-Hans", "en-orig", "fr-orig", "en", "zh"],
    ["ab", "zh-Hans", "zh-Hant", "en-orig", "en", "fr"],
    ["zh-Hans", "zh-Hant", "en-orig", "en", "fr-orig"],
    ["ja", "en-US", "ai-zh", "zh-TW", "zh-orig", "en_GB", "original", "danmaku", "ko"],
    ["en", "en", "zh-CN", "de-orig", "Chinese", "English"],
]
LANGUAGE_PREFS = [
    None,
    "orig",
    "original",
    "en",
    "EN",
    "english",
    "zh",
    "zh-Hans",
    "chinese",
    "ja",
]

CONTENT_TEXTS = [
    ["如何用 AIMD 把 YouTube 视频转成 Markdown"],
    ["How to convert YouTube videos to Markdown with AIMD"],
    ["【深度访谈】OpenAI CEO 谈 AGI 的未来", "English transcript available"],
    ["Deep Dive with OpenAI CEO on the Future of AGI"],
    ["B站 UP主 推荐：Python 异步编程实战教程"],
    ["使用 ChatGPT 写代码"],
    ["WWDC 2024 Keynote Highlights"],
    ["", None],
    [None, None],
    ["!!"],
    ["中 with a long english sentence of many latin letters"],
    ["中 abc"],
    ["ab"],
    ["x" * 5000 + "中文"],
]

RESOLVE_CASES = [
    [None, "未使用的中文标题足够汉字", "未使用的中文描述足够汉字", "en"],
    ["en", "中文标题带有足够汉字", "这是中文描述", None],
    [None, "中文标题带有足够汉字", "English description only as filler text", None],
    [None, "Some English title", None, "zh-CN"],
    [None, "Some English title", None, "ja"],
    [None, None, None, "  "],
    ["", "English words here", None, None],
]

COOKIE_SPECS = [
    "chrome",
    "chrome:default",
    "chrome+gnomekeyring:default",
    "firefox:default::container-1",
    "firefox::c1",
    " Brave : Profile 1 ",
    "edge+",
    "   ",
    "+kwallet",
    ":profile",
]

URLS = [
    "https://youtube.com/watch?v=x",
    "HTTP://example.com",
    "https://youtu.be/abc",
    "https://www.bilibili.com/video/BV1",
    "https://xhslink.com/a",
    "https://www.xiaohongshu.com/explore/1",
    "https://www.xiaoyuzhoufm.com/episode/1",
    "ftp://example.com",
    "https://",
    "example.com/video",
    "http://[::1",
    "/local/file.mp3",
    "https://example.com/video",
]

LINEAR_TEXTS = [
    ROLLING_TEXT,
    ROLLING_ZH,
    ">> >> a. B. c? D! “Quoted” end. (Paren) [Bracket] ‘single’ ok.",
    "<<only markers <<",
    "",
]


def text_cases() -> dict:
    return {
        "preferred_languages": [
            {
                "language": pref,
                "available": avail,
                "result": get_preferred_languages(pref, avail),
            }
            for avail in LANGUAGE_AVAILABLE
            for pref in LANGUAGE_PREFS
        ],
        "detect_content_language": [
            {"texts": texts, "result": detect_content_language(*texts)}
            for texts in CONTENT_TEXTS
        ],
        "resolve_subtitle_language": [
            {
                "args": args,
                "result": resolve_subtitle_language(
                    args[0],
                    title=args[1],
                    description=args[2],
                    metadata_language=args[3],
                ),
            }
            for args in RESOLVE_CASES
        ],
        "metadata_context": [
            {"info": info, "result": build_metadata_context(info)}
            for info in INFO_SAMPLES
        ],
        "format_content": [
            {
                "info": info,
                "content": content,
                "platform": platform,
                "result": format_content(info, content, platform),
            }
            for info in INFO_SAMPLES
            for content, platform in (
                ("body text", "youtube"),
                (None, None),
                ("  ", "unknown"),
            )
        ],
        "cookies_from_browser": [
            {"spec": spec, **_cookie_result(spec)} for spec in COOKIE_SPECS
        ],
        "urls": [
            {"url": url, "is_url": is_url(url), "platform": detect_platform(url)}
            for url in URLS
        ],
        "linear_transcript": [
            {"text": text, "result": _format_linear_transcript(text.split())}
            for text in LINEAR_TEXTS
        ],
    }


def _cookie_result(spec: str) -> dict:
    try:
        browser, profile, keyring, container = parse_cookies_from_browser(spec)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "browser": browser,
        "profile": profile,
        "keyring": keyring,
        "container": container,
    }


def _write(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise SystemExit(f"{path} would be {len(data)} bytes (> {MAX_BYTES})")
    path.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fetch-youtube", metavar="VIDEO_ID")
    args = parser.parse_args()

    inputs = args.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, text in synthetic_inputs().items():
        _write(inputs / name, text)
    if args.fetch_youtube:
        fetch_youtube(args.fetch_youtube, inputs)

    cases = subtitle_cases(inputs)
    _write(
        args.out / "cases.json", json.dumps(cases, ensure_ascii=False, indent=1) + "\n"
    )
    _write(
        args.out / "text.json",
        json.dumps(text_cases(), ensure_ascii=False, indent=1) + "\n",
    )
    print(f"wrote {len(cases)} subtitle cases to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
