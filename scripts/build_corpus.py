# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build the labeled ASR/OCR golden corpus from the sources in ``scripts/corpus``.

Usage: uv run scripts/build_corpus.py [OUT_DIR]   (default: target/corpus)

- ASR: every ``asr.tsv`` sentence spoken by three macOS ``say`` voices per
  language, written as 16 kHz mono 16-bit WAV.
- OCR: every ``ocr/*.md`` document rendered by Pandoc (HTML) and headless
  Chrome (A4 PDF), then rasterized like production with
  ``mutool draw -r 144 -c rgb``. Long documents span several pages.

Writes ``OUT_DIR/manifest.json`` (labels are the source text) for
``cargo run --release -p aimd-engine --example corpus_gate`` and
``uv run scripts/corpus_baseline.py``. macOS only (``say``, Chrome). The
speech is synthetic, so absolute error rates are optimistic; the gate
compares the native engine against the mlx-audio / mlx-vlm baseline on
identical inputs.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "corpus"
VOICES = {
    "en": ["Samantha", "Daniel", "Reed (English (US))"],
    "zh": [
        "Tingting",
        "Eddy (Chinese (China mainland))",
        "Flo (Chinese (China mainland))",
    ],
}
LANGUAGE_NAMES = {"en": "English", "zh": "Chinese"}
CHROME = os.environ.get(
    "AIMD_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
CSS = """
body { font-family: Helvetica, "PingFang SC", sans-serif; font-size: 13pt;
       line-height: 1.5; margin: 0; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #333; padding: 4px 10px; }
"""


def run(*args: str | Path) -> None:
    subprocess.run([str(a) for a in args], check=True, capture_output=True)


def build_asr(out: Path) -> list[dict]:
    items = []
    audio = out / "asr"
    audio.mkdir(parents=True, exist_ok=True)
    with open(SOURCES / "asr.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for index, voice in enumerate(VOICES[row["language"]]):
                wav = audio / f"{row['id']}-v{index}.wav"
                if not wav.exists():
                    run(
                        "say",
                        "-v",
                        voice,
                        "--data-format=LEI16@16000",
                        "-o",
                        wav,
                        row["text"],
                    )
                items.append(
                    {
                        "id": wav.stem,
                        "language": row["language"],
                        "language_name": LANGUAGE_NAMES[row["language"]],
                        "voice": voice,
                        "audio": str(wav.relative_to(out)),
                        "reference": row["text"],
                    }
                )
    return items


def build_ocr(out: Path) -> list[dict]:
    items = []
    pages_dir = out / "ocr"
    pages_dir.mkdir(parents=True, exist_ok=True)
    css = pages_dir / "corpus.css"
    css.write_text(CSS, encoding="utf-8")
    for source in sorted((SOURCES / "ocr").glob("*.md")):
        html = pages_dir / f"{source.stem}.html"
        pdf = pages_dir / f"{source.stem}.pdf"
        run(
            "pandoc",
            source,
            "-s",
            "--metadata",
            f"pagetitle={source.stem}",
            "--css",
            css.name,
            "-o",
            html,
        )
        run(
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        )
        for old in pages_dir.glob(f"{source.stem}-p*.png"):
            old.unlink()
        run(
            "mutool",
            "draw",
            "-q",
            "-o",
            pages_dir / f"{source.stem}-p%d.png",
            "-F",
            "png",
            "-r",
            "144",
            "-c",
            "rgb",
            pdf,
        )
        pages = sorted(
            pages_dir.glob(f"{source.stem}-p*.png"),
            key=lambda p: int(p.stem.rsplit("-p", 1)[1]),
        )
        items.append(
            {
                "id": source.stem,
                "pages": [str(p.relative_to(out)) for p in pages],
                "reference": source.read_text(encoding="utf-8"),
            }
        )
    return items


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "target/corpus").resolve()
    for tool in ("say", "pandoc", "mutool"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found")
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME} (set AIMD_CHROME)")
    manifest = {"asr": build_asr(out), "ocr": build_ocr(out)}
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pages = sum(len(d["pages"]) for d in manifest["ocr"])
    print(
        f"{out}: {len(manifest['asr'])} clips, {len(manifest['ocr'])} documents / {pages} pages"
    )


if __name__ == "__main__":
    main()
