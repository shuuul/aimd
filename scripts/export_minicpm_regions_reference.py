#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "numpy==2.4.6",
#     "pillow==12.2.0",
#     "opencv-python-headless==4.13.0.92",
#     "pydantic>=2,<3",
#     "pyyaml>=6",
#     "requests>=2.28",
# ]
# ///
"""Export the MiniCPM-OCR region-pipeline goldens (spec 009 WS-05).

Reference: the official MiniCPM-OCR SDK (`minicpmvocr`, OpenBMB/MiniCPM-OCR
`sdk/`), imported by path from a local checkout (never vendored):

    git clone --depth 1 https://github.com/OpenBMB/MiniCPM-OCR /tmp/MiniCPM-OCR
    uv run scripts/export_minicpm_regions_reference.py --sdk /tmp/MiniCPM-OCR/sdk

(`--sdk` defaults to `$MINICPM_OCR_SDK`.) The SDK runs with its default
config: `ppdoclayout3_torch` crop chain (overlap filter -> rectangle crop ->
text-block merge), legacy prompt override mode, no external prompt map,
`skip_vlm_labels` derived from `use_chart_recognition = use_seal_recognition
= True`, `use_ocr_for_image_block = False` (only the three image labels skip
the VLM), no resize, no TTS.

Writes `crates/aimd-engine/tests/golden/minicpm_regions/`:

- `text_cases.json`: inputs -> SDK outputs for the string functions
  (`_truncate_repetitive_suffix`, `_normalize_crop_text`,
  `truncate_repetitive_content`, `postprocess_content`,
  `repair_display_formula`, `has_mixed_inline_formula_content`,
  `split_text_block`, `repair_latex_formula`, `otsl_pad_to_sqr_v2`,
  `convert_otsl_to_html`, `classify_engine_text`, `html.unescape`) plus
  `_normalize_and_dedupe_blocks` + `assemble_markdown` block lists. Crafted
  cases plus seeded random token soups.
- `crop_cases.json`: per page (the two PP-DocLayoutV3 golden pages and
  synthetic pages whose pixels are `(x * 7 + y * 13 + c * 31) % 251`), the
  layout regions and the SDK layout blocks: order, label, bbox, folded
  regions, crop size + FNV-1a 64 of the crop's RGB8 bytes, skip decision,
  prompt, budget, slice_nums, output format.
- `e2e_cases.json`: the SDK block executor (`_execute_block`) fed canned
  recogniser outputs per block, then `_normalize_and_dedupe_blocks` +
  `assemble_markdown`: the final markdown and per-block contents, plus the
  hash of every image the fake engine received.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "minicpm_regions"
LAYOUT_GOLDEN = REPO / "crates" / "aimd-engine" / "tests" / "golden" / "pp_doclayout_v3"
MAX_FILE = 500 * 1024


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def synthetic_page(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    chans = [((x * 7 + y * 13 + c * 31) % 251).astype(np.uint8) for c in range(3)]
    return Image.fromarray(np.stack(chans, axis=-1), "RGB")


# --- random token soups ------------------------------------------------------

LATEX_TOKENS = [
    "$",
    "$$",
    "\\(",
    "\\)",
    "\\[",
    "\\]",
    "\\left",
    "\\right",
    "\\left(",
    "\\right)",
    "\\left\\{",
    "\\right.",
    "\\leftarrow",
    "\\rightarrow",
    "{",
    "}",
    "\\{",
    "\\}",
    "(",
    ")",
    "[",
    "]",
    "|",
    "\\\\",
    "\\",
    "x",
    "y^2",
    "\\frac{a}{b}",
    " ",
    "  ",
    "\n",
    "\t",
    "a",
    "b",
    "\\begin{array}{cc}",
    "\\end{array}",
    "\\begin{cases}",
    "\\end{cases}",
    "\\begin{aligned}",
    "\\end{aligned}",
    "\\quad",
    "\\qquad",
    "\\,",
    "\\;",
    "\\ ",
    "~",
    "\\hspace{1em}",
    "&",
    "=",
    "\\mathring\\mathrm{A}",
    "\\langle",
    "\\rangle",
    "中",
    "文",
    "\\.",
    ".",
    "\\lceil",
]
TEXT_TOKENS = [
    "word",
    "Word",
    "the",
    "of",
    "中文",
    "测试",
    " ",
    "  ",
    "\n",
    "\n\n",
    "\n\n\n",
    "\t",
    "$",
    "$$",
    "\\(",
    "\\)",
    "\\[",
    "\\]",
    "x^2",
    "```",
    "```python",
    "<table>",
    "</table>",
    "<TABLE border=1>",
    "| a | b |",
    "|---|---|",
    "| 1 | 2 |",
    "\\begin{align}",
    "\\end{align}",
    "\\$",
    "a$b",
    "€",
    "é",
    "\u3000",
    "\xa0",
    "-",
    "1.",
    "#",
]
OTSL_TOKENS = [
    "<fcel>",
    "<fcel>",
    "<fcel>",
    "<ecel>",
    "<lcel>",
    "<ucel>",
    "<xcel>",
    "<nl>",
    "<nl>",
    "a",
    "b c",
    " ",
    "\n",
    "中",
    "&amp;",
    "&lt;x&gt;",
    "1",
    "2.5",
    '"q"',
    "'s'",
    "&#39;",
    "&copy",
    "&nbsp;",
    "<b>",
    "&",
]
REPEAT_UNITS = [
    "abc ",
    "the same words again and ",
    "数据 ",
    "x y z w v u ",
    "\\hline ",
    "0",
    "ab",
    "<td></td>",
    "lorem ipsum dolor sit amet consectetur ",
    "重复的内容",
]


def soup(rng: random.Random, tokens, lo: int, hi: int) -> str:
    return "".join(rng.choice(tokens) for _ in range(rng.randint(lo, hi)))


def repetitive(rng: random.Random) -> str:
    head = soup(rng, TEXT_TOKENS, 0, 12)
    unit = rng.choice(REPEAT_UNITS)
    if rng.random() < 0.5:
        unit = unit.upper() if rng.random() < 0.5 else unit
    body = unit * rng.randint(2, 40)
    tail = rng.choice(
        ["", " \\end{array} \\]", " $$", "</td></tr></table>", " tail", "\n"]
    )
    return head + body + tail


# --- text cases ---------------------------------------------------------------


def text_cases(sdk) -> dict:
    _text = sdk["text"]
    md = sdk["markdown"]
    otsl = sdk["otsl"]
    latex = sdk["latex"]
    utils = sdk["utils"]
    rng = random.Random(9005)
    labels = [
        "text",
        "display_formula",
        "inline_formula",
        "table",
        "chart",
        "doc_title",
        "paragraph_title",
        "seal",
        "header",
        "formula_number",
        "reference_content",
        "algorithm",
    ]

    cases: dict = {}

    # _truncate_repetitive_suffix
    trs = [
        "",
        "short text",
        " ".join(["w%d" % i for i in range(17)]),
        " ".join(["a b c d e f"] * 3),
        "intro " + " ".join(["a b c d e f"] * 3),
        "intro words here " + " ".join(["One two three four five six"] * 4) + " end",
        "x " * 10 + ("p q r s t u v " * 3) + "\\end{array} \\] $$ </td></tr></table>",
        "中 文 测 试 一 二 " * 5,
        "A B C D E F a b c d e f a b c d e f tail text",
    ]
    trs += [repetitive(rng) for _ in range(60)]
    trs += [soup(rng, TEXT_TOKENS, 10, 80) for _ in range(20)]
    cases["truncate_repetitive_suffix"] = [
        {"text": t, "out": _text._truncate_repetitive_suffix(t)} for t in trs
    ]

    # truncate_repetitive_content
    trc = [
        ("", 50),
        ("a" * 49, 50),
        ("ab" * 60, 50),
        ("x" * 200, 50),
        ("header " + "abcdefgh" * 20, 50),
        ("start " + "unit1234" * 6, 50),
        ("\n".join(["same line"] * 12), 50),
        ("\n".join(["same line"] * 9 + ["other"]), 50),
        ("\n".join(["l1", "l2"] * 6), 50),
        ("short", 5000),
        ("重复" * 40, 50),
        ("abcabcabcabcabcabcabcabcabcabcabcab", 10),
        ("q" + "rstuvwxyz" * 12, 50),
    ]
    for _ in range(60):
        s = repetitive(rng).strip()
        trc.append((s, rng.choice([50, 50, 10, 5000])))
    cases["truncate_repetitive_content"] = [
        {
            "content": c,
            "min_count": m,
            "out": md.truncate_repetitive_content(c, min_count=m),
        }
        for c, m in trc
    ]

    # repair_display_formula / _strip_spacing_tokens through it
    rdf = [
        "$$x$$",
        "$$ $$ $$y$$",
        "\\[\\quad\\]",
        "\\[ \\, \\; \\]",
        "\\[a\\\\\\]",
        "$$a\\$$",
        "$$a\\\\$$",
        "$$x$$ text $$",
        "$$",
        "$$$",
        "\\[ \\] and \\( \\)",
        "$$\\hspace{2em}~$$",
        "$$\\quadx$$",
        "$$\\negthinspace\\thinspace$$",
        "$$ {} $$",
    ]
    rdf += [soup(rng, LATEX_TOKENS, 1, 30) for _ in range(120)]
    cases["repair_display_formula"] = [
        {"content": c, "out": md.repair_display_formula(c)} for c in rdf
    ]

    # has_mixed_inline_formula_content
    hm = [
        "",
        "  ",
        "$x$",
        "$x$ and y",
        "\\(x\\)",
        "\\(x\\) text",
        "$$x$$",
        "a $b$ c",
        "\\$x$",
        "$x\\$ y$",
        "$x$$y$",
        "$a\\$$",
        "$\n$",
        "$x$\n",
        "\\(a\n\\)",
        "plain",
        "$x$ $y$",
        "$ $",
        "$\\$$",
        "x$y$",
    ]
    hm += [soup(rng, LATEX_TOKENS, 1, 20) for _ in range(150)]
    cases["has_mixed_inline_formula_content"] = [
        {"content": c, "out": md.has_mixed_inline_formula_content(c)} for c in hm
    ]

    # split_text_block
    stb = [
        "",
        "line one\nline two",
        "a\n\n\nb",
        "```\ncode\n  kept\n```\nafter",
        "<table><tr>\n<td>x</td>\n</tr></table>\nnext",
        "| a | b |\n|---|---|\n| 1 | 2 |\nnext para",
        "$$\nx = 1\n$$\ntext",
        "\\[\nx\n\\]\ntext",
        "\\begin{align}\na &= b\n\\end{align}\ntext",
        "inline $x\ncontinues$ here\nnext",
        "inline \\(x\nmore\\) end\nnext",
        "  indented  \n\tTabbed\t",
        "中文第一行\n中文第二行",
    ]
    stb += [soup(rng, TEXT_TOKENS + LATEX_TOKENS[:20], 1, 40) for _ in range(110)]
    cases["split_text_block"] = []
    for c in stb:
        for protect in (False, True):
            cases["split_text_block"].append(
                {
                    "content": c,
                    "protect_inline_math": protect,
                    "out": md.split_text_block(c, protect_inline_math=protect),
                }
            )

    # repair_latex_formula
    rlf = [
        "",
        "  ",
        "x}",
        "{x",
        "\\{x",
        "\\left( x \\right)",
        "\\left x \\right",
        "\\left( x",
        "{\\left( a } \\right)",
        "\\left( {a \\right) }",
        "\\leftarrow x",
        "\\begin{array}{cc} a",
        "a \\end{array}",
        "\\begin{array}{lr} a \\end{array} \\end{array}",
        "\\end{cases}",
        "x \\end{array} \\right. \\end{array} \\right.",
        "x \\] \\]",
        "x\\\\",
        "x \\\\\n",
        "\\mathring\\mathrm{A}",
        "\\left\\langle x \\right\\rangle",
        "\\left\\lbrace x \\right\\rbrace",
        "\\begin{align*} x",
        "\\left. \\right|",
    ]
    rlf += [soup(rng, LATEX_TOKENS, 1, 30) for _ in range(100)]
    cases["repair_latex_formula"] = []
    for c in rlf:
        for label in ("display_formula", "inline_formula", ""):
            cases["repair_latex_formula"].append(
                {
                    "s": c,
                    "label": label,
                    "out": latex.repair_latex_formula(c, label=label),
                }
            )

    # OTSL
    ot = [
        "",
        "plain text",
        "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>",
        "<fcel>Name<fcel>Value<nl><fcel>x<lcel><nl><fcel>y<fcel>2<nl>",
        "<fcel>h<lcel><lcel><nl><fcel>a<ecel><fcel>b<nl>",
        "<fcel>r<fcel>s<nl><ucel><fcel>t<nl><ucel><fcel>u<nl>",
        "<fcel>x<lcel><nl><xcel><xcel><nl><fcel>z<fcel>w<nl>",
        "<fcel>a<fcel>b<fcel>c<nl><fcel>d<nl><fcel>e<fcel>f<fcel>g<fcel>h<nl>",
        "<fcel>a &amp; b<fcel>&lt;tag&gt;<nl><fcel>\"q\"<fcel>'s'<nl>",
        "<ecel><ecel><nl><ecel><ecel><nl>",
        "<nl><nl>",
        "<fcel>no newline",
        "junk<fcel>a<fcel>b\n<nl>\n<fcel>c<fcel>d\n",
        "<fcel><fcel>b<nl><fcel>c<fcel><nl>",
        "<table><tr><td>x</td></tr></table>",
        "<fcel>中文<fcel>表格<nl><fcel>数据<lcel><nl>",
        "<fcel>a\n<fcel>b\n",
    ]
    ot += [soup(rng, OTSL_TOKENS, 1, 40) for _ in range(150)]
    cases["otsl_pad_to_sqr_v2"] = [
        {"s": s, "out": otsl.otsl_pad_to_sqr_v2(s)} for s in ot
    ]
    cases["convert_otsl_to_html"] = [
        {"s": s, "out": otsl.convert_otsl_to_html(s)} for s in ot
    ]

    # html.unescape on the supported entity subset
    un = [
        "&amp;",
        "&amp",
        "&ampx",
        "&lt;&gt;&quot;&#39;&#x27;&apos;",
        "&nbsp;&copy&reg;",
        "&#0;",
        "&#128;",
        "&#x110000;",
        "&#xD800;",
        "&#65;&#x41;&#X41",
        "&#1234567890;",
        "&eacute;&Eacute&szlig;",
        "&unknown;",
        "&;",
        "&#;",
        "& amp;",
        "&amp;amp;",
        "&ndash;&mdash;&hellip;&lsquo;&rsquo;&ldquo;&rdquo;",
        "&times;&divide;&deg;&plusmn;",
        "&notit;",
        "&notin;",
        "&#x1F600;",
        "&#9;",
        "&#x0D;",
        "&#x80;&#x9F;",
        "&#11;",
        "&le;&ge;&ne;&minus;&bull;&middot;",
        "&alpha;&beta;&pi;&Sigma;",
    ]
    # every HTML 4 name (with and without `;`), plus a numeric sweep
    import html.entities

    names = sorted(html.entities.name2codepoint) + [
        "apos",
        "AMP",
        "GT",
        "LT",
        "QUOT",
        "COPY",
        "REG",
    ]
    un += ["".join(f"&{n};" for n in names), "".join(f"&{n}x" for n in names)]
    un += [
        "".join(f"&#{n};" for n in range(lo, lo + 0x40)) for lo in range(0, 0x300, 0x40)
    ]
    un += [
        "".join(
            f"&#x{n:x};"
            for n in [0xFDCF, 0xFDD0, 0xFDEF, 0xFDF0, 0xFFFD, 0xFFFE, 0xFFFF]
        )
        + "".join(
            f"&#x{(p << 16) | 0xFFFE:x};&#x{(p << 16) | 0xFFFF:x};"
            for p in range(1, 17)
        )
        + "&#x10FFFF;&#x110000;&#xD7FF;&#xDFFF;&#xE000;"
    ]
    cases["html_unescape"] = [{"s": s, "out": html.unescape(s)} for s in un]

    # classify_engine_text
    ce = [
        "",
        "text",
        "<!-- note -->",
        "<!-- request failed: timeout -->",
        "  <!--\nTraceback (most recent call last)\n-->  ",
        "<!-- CUDA out of memory -->x",
        "<!-- EngineCore died -->",
        "prefix <!-- failed -->",
    ]
    cases["classify_engine_text"] = [
        {
            "raw": r,
            "text": utils.classify_engine_text(r)[0],
            "error": utils.classify_engine_text(r)[1],
        }
        for r in ce
    ]

    # postprocess_content
    pc = [
        ("$$x$$", "latex", "display_formula"),
        ("\\[ a+b \\]", "latex", "display_formula"),
        ("\\[a\\] \\[b\\]", "latex", "display_formula"),
        ("$x^2$", "latex", "inline_formula"),
        ("\\(x\\)", "latex", "inline_formula"),
        ("$", "latex", "inline_formula"),
        ("$$", "latex", "display_formula"),
        ("\\left( x", "latex", "display_formula"),
        ("\\left( x", "latex", "inline_formula"),
        ("```html\n<table><tr><td>a</td></tr>\n```", "html", "table"),
        ("<table><tr><td>a</td></tr><table>", "html", "table"),
        ("```<table>```", "html", "chart"),
        ("  plain  ", "text", "text"),
    ]
    for _ in range(80):
        label = rng.choice(["display_formula", "inline_formula"])
        pc.append((soup(rng, LATEX_TOKENS, 1, 30), "latex", label))
    for _ in range(20):
        pc.append((repetitive(rng), "text", "text"))
    for _ in range(20):
        pc.append(
            (
                "```html" * rng.randint(0, 1)
                + soup(rng, OTSL_TOKENS + ["<table>", "</table>", "```"], 1, 20),
                "html",
                rng.choice(["table", "chart"]),
            )
        )
    cases["postprocess_content"] = [
        {
            "content": c,
            "output_format": f,
            "label": lab,
            "out": md.postprocess_content(c, f, label=lab),
        }
        for c, f, lab in pc
    ]

    # _normalize_crop_text
    nct = []
    for _ in range(200):
        label = rng.choice(labels)
        if label in ("table", "chart"):
            t = soup(rng, OTSL_TOKENS, 1, 30)
        elif "formula" in label:
            t = soup(rng, LATEX_TOKENS, 1, 30)
        elif rng.random() < 0.3:
            t = repetitive(rng)
        else:
            t = soup(rng, TEXT_TOKENS + LATEX_TOKENS[:10], 1, 40)
        nct.append((t, label))
    nct += [
        ("see \\( a + b \\) and \\(c\\)", "text"),
        ("\\[x\\]  \\[y\\]", "display_formula"),
        ("a  \t b\n\n\n\nc", "text"),
        ("<fcel>a<fcel>b<nl>", "table"),
        ("<fcel>a<fcel>b<nl>", "text"),
    ]
    cases["normalize_crop_text"] = [
        {"text": t, "label": lab, "out": _text._normalize_crop_text(t, lab)}
        for t, lab in nct
    ]

    # block lists: _normalize_and_dedupe_blocks + assemble_markdown
    prompt_map = sdk["prompt_map"]
    all_labels = sdk["px_labels"]
    block_cases = []
    crafted = [
        [
            {"order": 0, "label": "doc_title", "content": "Title"},
            {"order": 1, "label": "header", "content": "running header"},
            {"order": 2, "label": "text", "content": "para one\npara two"},
            {"order": 3, "label": "image", "content": ""},
            {"order": 4, "label": "display_formula", "content": "E = mc^2"},
            {
                "order": 5,
                "label": "inline_formula",
                "content": "x^2",
                "inline_formula_mixed": False,
            },
            {
                "order": 6,
                "label": "inline_formula",
                "content": "where $x$ is\nreal",
                "inline_formula_mixed": True,
            },
            {
                "order": 7,
                "label": "table",
                "content": "<fcel>a<fcel>b<nl><fcel>c<lcel><nl>",
            },
            {"order": 8, "label": "chart", "content": ""},
            {"order": 9, "label": "seal", "content": "SEAL"},
            {"order": 10, "label": "figure_title", "content": "Fig. 1"},
            {"order": 11, "label": "number", "content": "3"},
            {"order": 12, "label": "paragraph_title", "content": ""},
            {"order": 13, "label": "formula_number", "content": "(1)"},
        ],
        [
            {"order": 2, "label": "text", "content": "dup"},
            {"order": 2, "label": "text", "content": "dup"},
            {"order": 1, "label": "text", "content": "first"},
            {"order": 3, "label": "chart", "content": "<fcel>q<nl>"},
            {"order": 4, "label": "header_image", "content": ""},
        ],
        [],
    ]
    for _ in range(40):
        blocks = []
        for order in range(rng.randint(1, 10)):
            label = rng.choice(all_labels)
            if label in ("table", "chart"):
                content = soup(rng, OTSL_TOKENS, 0, 20)
            elif "formula" in label:
                content = soup(rng, LATEX_TOKENS, 0, 15)
            else:
                content = soup(rng, TEXT_TOKENS, 0, 20)
            block = {
                "order": rng.randint(0, 12) if rng.random() < 0.2 else order,
                "label": label,
                "content": content,
            }
            if label == "inline_formula":
                block["inline_formula_mixed"] = rng.random() < 0.5
            blocks.append(block)
        crafted.append(blocks)
    for blocks in crafted:
        final = _text._normalize_and_dedupe_blocks(copy.deepcopy(blocks))
        block_cases.append(
            {
                "blocks": blocks,
                "final": [
                    {"order": b["order"], "label": b["label"], "content": b["content"]}
                    for b in final
                ],
                "markdown": md.assemble_markdown(final, prompt_map),
            }
        )
    cases["blocks"] = block_cases
    return cases


# --- crop cases -----------------------------------------------------------------

SYNTHETIC = [
    {
        "name": "merge_cross",
        "width": 900,
        "height": 800,
        "regions": [
            ("doc_title", [100, 30, 800, 80]),
            ("text", [100, 100, 400, 200]),
            ("text", [450, 110, 700, 190]),
            ("table", [100, 300, 700, 600]),
            ("text", [100, 650, 700, 700]),
            ("footer", [0, 760, 900, 800]),
        ],
    },
    {
        "name": "updown",
        "width": 700,
        "height": 600,
        "regions": [
            ("text", [100, 100, 500, 150]),
            ("image", [400, 120, 600, 300]),
            ("text", [100, 160, 450, 210]),
            ("text", [150, 215, 450, 260]),
            ("text", [120, 400, 400, 440]),
            ("chart", [420, 380, 690, 590]),
        ],
    },
    {
        "name": "tall_group",
        "width": 300,
        "height": 400,
        "regions": [
            ("text", [10, 10, 40, 200]),
            ("text", [45, 10, 75, 200]),
            ("vertical_text", [100, 10, 130, 390]),
            ("vertical_text", [100, 200, 130, 395]),
            ("seal", [200, 300, 290, 390]),
        ],
    },
    {
        "name": "filters",
        "width": 640,
        "height": 480,
        "regions": [
            ("header", [0, 0, 640, 30]),
            ("text", [10, 40, 15, 400]),
            ("text", [20, 40, 400, 44]),
            ("reference", [20, 50, 400, 100]),
            ("text", [20, 110, 400, 200]),
            ("inline_formula", [50, 120, 120, 140]),
            ("text", [25, 112, 395, 198]),
            ("inline_formula", [500, 120, 560, 140]),
            ("display_formula", [450, 110, 630, 200]),
            ("formula_number", [600, 210, 639, 240]),
            ("header_image", [0, 250, 100, 300]),
            ("footer_image", [540, 430, 640, 480]),
            ("abstract", [20, 260, 300, 330]),
            ("algorithm", [20, 340, 300, 420]),
            ("aside_text", [610, 300, 640, 420]),
            ("vision_footnote", [320, 340, 600, 360]),
            ("reference_content", [320, 370, 600, 400]),
            ("content", [320, 260, 600, 330]),
            ("footnote", [20, 430, 500, 460]),
            ("number", [620, 460, 640, 480]),
        ],
    },
    {
        "name": "align_chain",
        "width": 800,
        "height": 700,
        "regions": [
            ("text", [100, 100, 600, 140]),
            ("text", [300, 145, 600, 185]),
            ("text", [200, 190, 600, 230]),
            ("text", [250, 235, 400, 275]),
            ("table", [500, 150, 780, 300]),
            ("text", [100, 400, 300, 440]),
            ("text", [320, 405, 500, 435]),
            ("text", [520, 400, 700, 440]),
            ("paragraph_title", [100, 500, 300, 530]),
            ("figure_title", [100, 540, 300, 560]),
        ],
    },
]


def crop_page(sdk, name: str, image: Image.Image, regions: list) -> dict:
    det_cls = sdk["detector_cls"]
    detector = det_cls(copy.deepcopy(sdk["default_config"]["layout"]))
    detections = [
        {
            "index": i,
            "label": r["label"],
            "class_id": r["class_id"],
            "score": r["score"],
            "bbox": [float(v) for v in r["bbox"]],
            "polygon": [],
        }
        for i, r in enumerate(regions)
    ]
    blocks = detector.crop(image, detections)
    sink: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        layout, _raw = sdk["build_layout"](
            image=image,
            image_id=name,
            source_path=f"{name}.png",
            source_type="image",
            page_index=None,
            detections=detections,
            blocks=blocks,
            crops_root=Path(tmp),
            crop_sink=sink,
        )
    out_blocks = []
    for block in layout["blocks"]:
        crop = np.ascontiguousarray(sink[block["crop_filename"]], dtype=np.uint8)
        cfg = sdk["get_element_cfg"](sdk["prompt_map"], block["label"])
        out_blocks.append(
            {
                "order": block["order"],
                "label": block["label"],
                "bbox": block["bbox"],
                "regions": block["regions"],
                "crop_width": block["crop_width"],
                "crop_height": block["crop_height"],
                "crop_fnv": fnv1a64(crop.tobytes()),
                "skip_vlm": block["skip_vlm"],
                "recognised": cfg is not None and not block["skip_vlm"],
                "prompt": cfg["prompt"] if cfg else None,
                "max_new_tokens": cfg["max_new_tokens"] if cfg else None,
                "slice_nums": cfg["slice_nums"] if cfg else None,
                "output_format": cfg["output_format"] if cfg else None,
                "resize_min_side": cfg["resize_min_side"] if cfg else None,
            }
        )
    return {"layout": layout, "sink": sink, "blocks": out_blocks}


def page_inputs(sdk) -> list:
    pages = []
    manifest = json.loads((LAYOUT_GOLDEN / "manifest.json").read_text())
    for page in manifest["pages"]:
        image = Image.open(LAYOUT_GOLDEN / "images" / f"{page['name']}.png").convert(
            "RGB"
        )
        assert image.size == (page["width"], page["height"])
        pages.append(
            {
                "name": page["name"],
                "source": f"pp_doclayout_v3/images/{page['name']}.png",
                "width": page["width"],
                "height": page["height"],
                "image": image,
                "regions": page["regions"],
            }
        )
    px = sdk["px_labels"]
    for syn in SYNTHETIC:
        regions = [
            {
                "index": i,
                "label": label,
                "class_id": px.index(label),
                "score": 0.9,
                "bbox": bbox,
            }
            for i, (label, bbox) in enumerate(syn["regions"])
        ]
        pages.append(
            {
                "name": syn["name"],
                "source": "synthetic",
                "width": syn["width"],
                "height": syn["height"],
                "image": synthetic_page(syn["width"], syn["height"]),
                "regions": regions,
            }
        )
    return pages


# --- e2e ------------------------------------------------------------------------

CANNED = {
    "doc_title": ["MiniCPM-OCR Region Test"],
    "paragraph_title": ["1. Introduction", "2. Results \\(n=3\\)", "3. Notes"],
    "text": [
        "First line of text with \\( a+b \\) inline.\nSecond   line\t\twith  spaces.",
        "重复 " + "the loop repeats here " * 4 + "\\end{array}",
        "Plain paragraph.\n\n\n\nAfter gap.",
        "| a | b |\n|---|---|\n| 1 | 2 |\ntrailing",
        "Another paragraph with $x$ math.",
    ],
    "table": [
        "<fcel>Name<fcel>Value<fcel>Unit<nl><fcel>alpha<lcel><fcel>m<nl><fcel>&amp; beta<fcel>2.5<ucel><nl>",
        "```html\n<table><tr><td>x</td></tr>\n```",
    ],
    "display_formula": ["\\[ E = mc^2 \\left( 1 + v \\]", "$$\\int_0^1 f(x)\\,dx$$"],
    "inline_formula": ["$x^2$", "where \\(y\\) holds"],
    "formula_number": ["(1)"],
    "figure_title": ["Figure 1: A picture"],
    "number": ["12"],
    "header": ["Running header"],
    "footer": ["Page footer"],
    "chart": ["<fcel>Q1<fcel>10<nl><fcel>Q2<fcel>20<nl>"],
    "seal": ["OFFICIAL SEAL"],
    "vertical_text": ["竖排文字"],
    "abstract": ["Abstract body."],
    "algorithm": ["for i in range(3):\n    print(i)"],
    "aside_text": ["aside"],
    "vision_footnote": ["note"],
    "reference_content": ["[1] Ref."],
    "content": ["Contents ...... 1"],
    "footnote": ["1 footnote"],
}


class FakeEngine:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.queue = []

    def recognize(self, image, prompt, max_new_tokens, slice_nums):
        arr = np.ascontiguousarray(np.asarray(image.convert("RGB")), dtype=np.uint8)
        text = self.queue.pop(0)
        self.calls.append(
            {
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "fnv": fnv1a64(arr.tobytes()),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "slice_nums": slice_nums,
            }
        )
        return {"text": text, "output_token_count": 0}


def e2e_page(sdk, page: dict, cropped: dict) -> dict:
    counters: dict = {}
    raw_by_order = {}
    engine = FakeEngine()
    outputs = []
    for block in cropped["layout"]["blocks"]:
        label = block["label"]
        pool = CANNED.get(label, ["fallback text"])
        i = counters.get(label, 0)
        counters[label] = i + 1
        raw = pool[i % len(pool)]
        raw_by_order[block["order"]] = raw
        engine.queue = [raw]
        out = sdk["execute_block"](
            engine,
            sdk["prompt_map"],
            page["name"],
            Path("."),
            block,
            crop_array=cropped["sink"][block["crop_filename"]],
        )
        outputs.append(out["result_block"])
    final = sdk["text"]._normalize_and_dedupe_blocks(outputs)
    markdown = sdk["markdown"].assemble_markdown(final, sdk["prompt_map"])
    return {
        "name": page["name"],
        "raw": [{"order": o, "text": t} for o, t in raw_by_order.items()],
        "calls": engine.calls,
        "final": [
            {
                "order": b["order"],
                "label": b["label"],
                "content": b["content"],
                "inline_formula_mixed": bool(b.get("inline_formula_mixed", False)),
                "status": b["status"],
            }
            for b in final
        ],
        "markdown": markdown,
    }


# --- driver ---------------------------------------------------------------------


def load_sdk(sdk_dir: Path) -> dict:
    if not (sdk_dir / "minicpmvocr" / "__init__.py").is_file():
        raise SystemExit(
            f"--sdk must point at the SDK dir holding minicpmvocr/: {sdk_dir}"
        )
    sys.path.insert(0, str(sdk_dir))
    import minicpmvocr
    from minicpmvocr import _utils, constants
    from minicpmvocr.config import DEFAULT_CONFIG
    from minicpmvocr.layouts.ppdoclayout3_torch import PPDocLayoutV3TorchDetector
    from minicpmvocr.pipeline import _blocks, _layout_record, _text
    from minicpmvocr.postprocess import latex_repair, markdown
    from minicpmvocr.postprocess.otsl import converter
    from minicpmvocr.prompts import get_element_cfg, load_prompt_map

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    assert cfg["layout"]["skip_vlm_labels"] is None
    assert cfg["prompt"] == {"override_mode": "legacy", "prompt_map": ""}
    return {
        "version": getattr(minicpmvocr, "__version__", "?"),
        "default_config": cfg,
        "prompt_map": load_prompt_map(cfg),
        "px_labels": list(constants.PX_LABELS),
        "detector_cls": PPDocLayoutV3TorchDetector,
        "build_layout": _layout_record._build_layout_from_blocks,
        "execute_block": _blocks._execute_block,
        "get_element_cfg": get_element_cfg,
        "text": _text,
        "markdown": markdown,
        "otsl": converter,
        "latex": latex_repair,
        "utils": _utils,
    }


def write_json(path: Path, payload) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    size = len(data.encode("utf-8"))
    assert size <= MAX_FILE, f"{path.name} is {size} bytes (> {MAX_FILE})"
    path.write_text(data, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)} ({size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sdk", default=os.environ.get("MINICPM_OCR_SDK", ""))
    parser.add_argument(
        "--sdk-commit", default=os.environ.get("MINICPM_OCR_SDK_COMMIT", "")
    )
    args = parser.parse_args()
    if not args.sdk:
        raise SystemExit("pass --sdk <MiniCPM-OCR checkout>/sdk or set MINICPM_OCR_SDK")
    sdk = load_sdk(Path(args.sdk).resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"sdk_version": sdk["version"], "sdk_commit": args.sdk_commit}

    write_json(OUT / "text_cases.json", {"meta": meta, **text_cases(sdk)})

    crop_pages = []
    e2e_pages = []
    for page in page_inputs(sdk):
        cropped = crop_page(sdk, page["name"], page["image"], page["regions"])
        crop_pages.append(
            {
                "name": page["name"],
                "source": page["source"],
                "width": page["width"],
                "height": page["height"],
                "regions": [
                    {"label": r["label"], "class_id": r["class_id"], "bbox": r["bbox"]}
                    for r in page["regions"]
                ],
                "blocks": cropped["blocks"],
            }
        )
        e2e_pages.append(e2e_page(sdk, page, cropped))
    write_json(OUT / "crop_cases.json", {"meta": meta, "pages": crop_pages})
    write_json(OUT / "e2e_cases.json", {"meta": meta, "pages": e2e_pages})


if __name__ == "__main__":
    main()
