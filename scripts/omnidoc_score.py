# /// script
# requires-python = ">=3.11"
# dependencies = ["rapidfuzz"]
# ///
"""Score corpus_gate outputs on the OmniDocBench subset.

usage: uv run scripts/omnidoc_score.py CORPUS_DIR out-a.json [out-b.json ...]

Both sides drop formulas ($..$, $$..$$, \\(..\\), \\[..\\]), HTML tags,
comments and markup, keeping letters/digits/CJK.
- sim: 1 - character edit distance / longer length (order-sensitive,
  OmniDocBench-style text edit distance on the whole page).
- f1: token-bag F1 (English words, single CJK characters), order-free, so
  multi-column reading-order differences do not dominate.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

FORMULA = re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$[^$\n]+?\$", re.S)
TAG = re.compile(r"<!--.*?-->|<[^>]+>", re.S)
KEEP = re.compile(r"[0-9A-Za-z一-鿿㐀-䶿]+")
TOK = re.compile(r"[0-9A-Za-z]+|[一-鿿㐀-䶿]")


def norm(text: str) -> str:
    text = TAG.sub(" ", FORMULA.sub(" ", text))
    return "".join(KEEP.findall(text)).lower()


def tokens(text: str) -> Counter:
    text = TAG.sub(" ", FORMULA.sub(" ", text)).lower()
    return Counter(TOK.findall(text))


def f1(ref: Counter, hyp: Counter) -> float:
    hit = sum((ref & hyp).values())
    if not ref and not hyp:
        return 1.0
    p = hit / max(1, sum(hyp.values()))
    r = hit / max(1, sum(ref.values()))
    return 0.0 if hit == 0 else 2 * p * r / (p + r)


corpus = Path(sys.argv[1])
items = json.load(open(corpus / "manifest.json"))["ocr"]
rows = []
for path in sys.argv[2:]:
    native = json.load(open(path))
    ((key, docs),) = native.items()
    name = Path(path).stem.removeprefix("out-")
    groups = defaultdict(list)
    stops = 0
    for it in items:
        pages = docs[it["id"]]["pages"]
        hyp = "\n\n".join(p["markdown"] for p in pages)
        stops += all(p["finish"] == "stop" for p in pages)
        a, b = norm(it["reference"]), norm(hyp)
        sim = 1 - Levenshtein.distance(a, b) / max(1, len(a), len(b))
        score = (sim, f1(tokens(it["reference"]), tokens(hyp)))
        for g in ("all", it["source"], it["lang"]):
            groups[g].append(score)
    rows.append((name, groups, stops))

cols = ["all", "en", "zh"] + sorted({it["source"] for it in items})
width = max(len(c) for c in cols)
print(f"{len(items)} pages; cell = sim / f1 (higher is better)\n")
print(f"{'':{width}} " + " ".join(f"{n:>16}" for n, _, _ in rows))
for c in cols:
    cells = []
    for _, g, _ in rows:
        v = g[c]
        cells.append(
            f"{sum(s for s, _ in v) / len(v):.3f} / {sum(f for _, f in v) / len(v):.3f}"
        )
    print(
        f"{c + f' ({len(rows[0][1][c])})':{width + 5}}"
        + " ".join(f"{x:>16}" for x in cells)
    )
print(
    f"{'stop':{width + 5}}" + " ".join(f"{f'{s}/{len(items)}':>16}" for _, _, s in rows)
)
