#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Export golden cases for the Unlimited-OCR-2 raw-output -> Markdown converter.

Runs the upstream stdlib-only ``process_unlimitedOCR2.py`` ``convert`` over a set
of hand-written raw outputs and writes ``{name, input, expected}`` records to
``crates/aimd-engine/tests/golden/ocr2_markdown/cases.json``. The Rust port
(``unlimited_ocr::markdown2``) must reproduce ``expected`` byte for byte.

Downloads nothing: pass the path of the upstream script.

    uv run scripts/export_ocr2_markdown_reference.py /path/to/process_unlimitedOCR2.py
"""

import importlib.util
import json
import sys
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[1]
    / "crates/aimd-engine/tests/golden/ocr2_markdown/cases.json"
)

TABLE_A = (
    '<table border="1"><tr><th>Name</th><th>Qty</th></tr>'
    "<tr><td>apple</td><td>1</td></tr></table>"
)
TABLE_B_REPEAT = (
    "<table><tr><th>Ｎame</th><th>Qty</th></tr><tr><td>pear</td><td>2</td></tr></table>"
)
TABLE_C_REPEAT2 = (
    "<table><tr><th> name </th><th>QTY</th></tr>"
    "<tr><td>plum</td><td>3</td></tr></table>"
)

CASES = {
    "multi_page_blank": (
        "<PAGE><BLOCK>title<LEVEL1>[0, 0, 100, 50]<FORK>Doc Title\n"
        "<BLOCK>text[0, 60, 100, 200]<FORK>Page one text.\n"
        "<BLOCK>header[0, 0, 10, 10]<FORK>Running header"
        "<PAGE>"
        "<PAGE><BLOCK>text[0, 0, 10, 10]<FORK>Page three text.\n"
        "<BLOCK>page_number[0, 0, 10, 10]<FORK>3"
    ),
    "text_chain": (
        "<PAGE><BLOCK>text[0, 0, 10, 10]<FORK>Intro.\n"
        "<BLOCK>cross_text[0, 900, 999, 999]<FORK>The sentence starts here and "
        "<PAGE><BLOCK>append_text[0, 0, 999, 50]<FORK>continues on page two and "
        "<BLOCK>text[0, 60, 999, 100]<FORK>Separate paragraph."
        "<PAGE><BLOCK>append_text[0, 0, 999, 50]<FORK>ends on page three.\n"
        "<BLOCK>append_text[0, 60, 999, 90]<FORK> Tail."
    ),
    "table_chain_repeated_headers": (
        "<PAGE><BLOCK>cross_table[0, 500, 999, 999]<FORK>" + TABLE_A + "<PAGE>"
        "<BLOCK>append_table[0, 0, 999, 300]<FORK>" + TABLE_B_REPEAT + "<PAGE>"
        "<BLOCK>append_table[0, 0, 999, 300]<FORK>" + TABLE_C_REPEAT2 + "\n"
        "<BLOCK>text[0, 400, 999, 450]<FORK>After table."
    ),
    "table_fragment_without_rows": (
        "<PAGE><BLOCK>cross_table[0, 500, 999, 999]<FORK>" + TABLE_A + "<PAGE>"
        "<BLOCK>append_table[0, 0, 999, 300]<FORK>stray table text"
    ),
    "table_gap_pages": (
        "<PAGE><BLOCK>cross_table[0, 500, 999, 999]<FORK>" + TABLE_A + "<PAGE>"
        "<BLOCK>text[0, 0, 10, 10]<FORK>Gap text<PAGE>"
        "<BLOCK>append_table[0, 0, 999, 300]<FORK>" + TABLE_B_REPEAT + "<PAGE>"
        "<BLOCK>table[0, 0, 999, 300]<FORK>" + TABLE_A + "<PAGE>"
        "<BLOCK>append_table[0, 0, 999, 300]<FORK>" + TABLE_C_REPEAT2 + "<PAGE>"
        "<BLOCK>append_table[0, 0, 5, 5]<FORK>" + TABLE_B_REPEAT
    ),
    "append_without_chain": (
        "<PAGE><BLOCK>append_text[0, 0, 10, 10]<FORK>Orphan append.\n"
        "<BLOCK>append_table[0, 0, 10, 10]<FORK>" + TABLE_A
    ),
    "merge_columns": (
        "<PAGE><BLOCK>text[0, 0, 50, 50]<FORK>Left column <MERGE>\n"
        "<BLOCK>text[50, 0, 100, 50]<FORK>right column<MERGE>"
        "<BLOCK>title<LEVEL2>[0, 60, 100, 80]<FORK>Heading joined"
        "<BLOCK>image[0, 90, 10, 99]<FORK>"
        "<MERGE><BLOCK>text[0, 0, 5, 5]<FORK>After image"
        "<PAGE><MERGE><BLOCK>text[0, 0, 50, 50]<FORK>New page, no cross-page merge"
    ),
    "level_first_and_bbox_first": (
        "<PAGE><BLOCK>title<LEVEL2>[10, 10, 20, 20]<FORK>Level first\n"
        "<BLOCK>title[10, 30, 20, 40]<LEVEL3><FORK>Bbox first\n"
        "<BLOCK>title< LEVEL 4 }[1, 2, 3, 4, 5]<FORK>Spaced level, five numbers\n"
        "<BLOCK>title</LEVEL5><FORK>Closing level token, no bbox\n"
        "<BLOCK>Title<level2>[1,2,3,4]<FORK>Mixed case label"
    ),
    "missing_level": (
        "<PAGE><BLOCK>title[1, 2, 3, 4]<FORK>No level\n"
        "<BLOCK>title<FORK>Bare title\n"
        "<BLOCK>title<LEVEL0>[1, 2, 3, 4]<FORK>Level zero\n"
        "<BLOCK>title<LEVEL>[1, 2, 3, 4]<FORK>Empty level number"
    ),
    "image_chart_bbox": (
        "<PAGE><BLOCK>image[-5, 20, 1200, 400]<FORK>\n"
        "<BLOCK>chart[100, 100, 500, 500]<FORK>[image]\n"
        "<BLOCK>figure[100, 100, 500, 500]<FORK>Figure caption text\n"
        "<BLOCK>seal[1, 2, 3, 4, 5]<FORK>\n"
        "<BLOCK>image<FORK>no bbox image caption\n"
        "<BLOCK>image_caption[1, 2, 3, 4]<FORK>Caption below"
        "<PAGE><BLOCK>picture[7, 8, 9, 10]<FORK>"
    ),
    "noise_content": (
        "<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>[non-text]\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>[UNREADABLE]\n"
        "<BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK>[Image]\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>Kept [non-text] inline\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>"
    ),
    "bare_blocks": (
        "title<LEVEL1>[1, 2, 3, 4]<FORK>Bare Title\n"
        "text[1, 2, 3, 4]<FORK>bare body one<FORK>"
        "text[5, 6, 7, 8]<FORK>bare body two\n"
        "<|det|>text[9, 10, 11, 12]<FORK>det prefixed\n"
        "foo[1, 2, 3, 4]<FORK>unknown label stays text\n"
        "<FORK><BLOCK>text[1, 2, 3, 4]<FORK>fork before block\n"
        "  <FORK>\n<BLOCK>text[1, 2, 3, 4]<FORK>fork with blank line"
        "<PAGE>cross_table[1, 2, 3, 4, 5]<FORK>" + TABLE_A
    ),
    "stray_tokens": (
        "<PAGE>junk <|det|> before\n<BLOCK>text[1, 2, 3, 4]<FORK>Body <|ref|>with<|/ref|> "
        "stray <image> tokens <FORK> and <LEVEL3 and <MERGE> done\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>[10, 20, 30, 40]\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>a := b \\coloneqq c \\eqqcolon d\n"
        "<BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK><|det|>\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>trailing   \nspaces\t\n\n\n\n\nmany blanks\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>   [ 1, 2, 3, 4 ]   \n"
        "<PAGE 2><BLOCK>text[1, 2, 3, 4]<FORK>numbered page tag</PAGE>"
    ),
    "no_blocks_raw": "Plain text before <PAGE> more plain <MERGE> text\n<PAGE>\n<PAGE>third",
    "empty_and_blank": "",
    "whitespace_only": "  \n\t   ",
    "fenced_markdown": (
        "```markdown\n<PAGE><BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK>Fenced\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>Inside fence.\n```"
    ),
    "multiple_top_titles": (
        "<PAGE><BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK>First\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>x\n"
        "<BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK>Second\n"
        "<BLOCK>title<LEVEL3>[1, 2, 3, 4]<FORK>Deep\n"
        "<BLOCK>text[1, 2, 3, 4]<FORK>y\n"
        "<BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>Third"
        "<PAGE><BLOCK>title<LEVEL1>[1, 2, 3, 4]<FORK>Fourth"
    ),
    "single_top_title": (
        "<PAGE><BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>Only root\n"
        "<BLOCK>title<LEVEL3>[1, 2, 3, 4]<FORK>Child\n"
        "<BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>Other root"
    ),
    "adjacent_titles_merge": (
        "<PAGE><BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>Part one of title\n"
        "<BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>part   two of\u001ftitle\n"
        "<BLOCK>title<LEVEL3>[1, 2, 3, 4]<FORK>Different level\n"
        "<BLOCK>title<LEVEL3>[1, 2, 3, 4]<FORK>\n"
        "<BLOCK>title[1, 2, 3, 4]<FORK>no level a\n"
        "<BLOCK>title[1, 2, 3, 4]<FORK>no level b"
        "<PAGE><BLOCK>title<LEVEL3>[1, 2, 3, 4]<FORK>Other page"
    ),
    "bad_colspan": (
        '<PAGE><BLOCK>table[1, 2, 3, 4]<FORK><table><tr><td colspan="2.5">a</td>'
        '<td rowspan="abc">b</td><td colspan="">c</td><td COLSPAN="3">d</td>'
        '<td colspan="+2">e</td><td colspan=" 4 ">f</td><td rowspan="x7y9">g</td>'
        '<td colspan="1_0">h</td><td colspan="0x2">i</td></tr></table>'
    ),
    "fullwidth_comma_bbox": (
        "<PAGE><BLOCK>text[10，20，30，40]<FORK>Hello full-width\n"
        "<BLOCK>image[10 ， 20, 30，40，50]<FORK>\n"
        "<BLOCK>title<LEVEL2>[1，2，3，4]<FORK>Head"
    ),
    "equation_and_misc_labels": (
        "<PAGE><BLOCK>equation[1, 2, 3, 4]<FORK>$$a \\coloneqq b$$\n"
        "<BLOCK>image_footnote[1, 2, 3, 4]<FORK>foot\n"
        "<BLOCK>ref_text[1, 2, 3, 4]<FORK>[1] Ref\n"
        "<BLOCK>cross_equation[1, 2, 3, 4]<FORK>cross eq<PAGE>"
        "<BLOCK>append_equation[1, 2, 3, 4]<FORK> append eq"
    ),
    "real_output": (
        "<PAGE><BLOCK>title<LEVEL2>[42, 65, 855, 101]<FORK>Preparing Context for Language Models\n"
        "<BLOCK>text[40, 130, 954, 225]<FORK>Context engineering is the practice.\n"
        "<BLOCK>cross_text[40, 884, 947, 955]<FORK>Local inference keeps private material on the device and finish in seconds\n"
        "<PAGE><BLOCK>append_text[40, 37, 955, 84]<FORK>instead of minutes.\n"
        "<BLOCK>title<LEVEL2>[42, 110, 221, 136]<FORK>Conclusion"
    ),
}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} /path/to/process_unlimitedOCR2.py")
    spec = importlib.util.spec_from_file_location("process_unlimitedOCR2", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    records = [
        {"name": name, "input": raw, "expected": mod.convert(raw)}
        for name, raw in CASES.items()
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(records, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(records)} cases -> {OUT}")


if __name__ == "__main__":
    main()
