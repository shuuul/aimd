# /// script
# requires-python = ">=3.11"
# ///
"""Build a corpus_gate-format subset of OmniDocBench.

usage: uv run scripts/omnidoc_build.py OMNIDOC_DIR OUT_DIR [PER_LANG]
Picks PER_LANG english + PER_LANG chinese pages per data source (seed 0),
skipping pages with masked regions. Reference = reading-order text of the
content blocks (headers/footers/page numbers/figures/isolated equations
excluded; tables as their HTML).
"""

import json
import random
import subprocess
import sys
from pathlib import Path

KEEP = {
    "text_block",
    "title",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "figure_footnote",
    "page_footnote",
    "equation_caption",
    "reference",
    "code_txt",
    "code_txt_caption",
    "list_group",
    "table",
}

src, out = Path(sys.argv[1]), Path(sys.argv[2])
per_lang = int(sys.argv[3]) if len(sys.argv) > 3 else 4
pages = json.load(open(src / "OmniDocBench.json"))
rng = random.Random(0)
by = {}
for p in pages:
    a = p["page_info"]["page_attribute"]
    if a["data_source"] == "historical_document":
        continue
    if any(d["category_type"].endswith("_mask") for d in p["layout_dets"]):
        continue
    lang = {"english": "en", "simplified_chinese": "zh"}.get(a["language"])
    if lang:
        by.setdefault((a["data_source"], lang), []).append(p)

chosen = []
for key in sorted(by):
    group = by[key]
    rng.shuffle(group)
    chosen += [(key, p) for p in group[:per_lang]]


def block_text(d):
    if d["category_type"] == "table":
        return d.get("html") or d.get("text") or ""
    return d.get("text") or ""


items = []
for (source, lang), p in chosen:
    dets = [
        d
        for d in p["layout_dets"]
        if d["category_type"] in KEEP
        and not d.get("ignore")
        and d.get("order") is not None
    ]
    dets.sort(key=lambda d: d["order"])
    ref = "\n\n".join(t for t in map(block_text, dets) if t.strip())
    name = p["page_info"]["image_path"]
    items.append(
        {
            "id": f"{source}-{lang}-{Path(name).stem[-8:]}",
            "pages": [f"images/{name}"],
            "reference": ref,
            "source": source,
            "lang": lang,
        }
    )

out.mkdir(parents=True, exist_ok=True)
need = [i["pages"][0] for i in items if not (out / i["pages"][0]).exists()]
if need:
    cmd = [
        "uvx",
        "hf",
        "download",
        "opendatalab/OmniDocBench",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(out),
    ]
    for n in need:
        cmd += ["--include", n]
    subprocess.run(cmd, check=True)
missing = [i["id"] for i in items if not (out / i["pages"][0]).exists()]
assert not missing, missing
json.dump(
    {"asr": [], "ocr": items},
    open(out / "manifest.json", "w"),
    ensure_ascii=False,
    indent=1,
)
print(len(items), "pages")
