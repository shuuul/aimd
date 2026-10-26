#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-audio==0.6.1",
#     "numpy==2.5.4",
#     "transformers==5.20.2",
#     "soynlp==0.0.493",
# ]
# ///
"""Export Qwen3-ForcedAligner reference fixtures for the native port (spec 005 WS-02).

Reference: the mlx-audio ``ForcedAlignerModel``
(``mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner``) on the product
checkpoint ``mlx-community/Qwen3-ForcedAligner-0.6B-8bit``, read from the HF
cache offline. It runs the same MLX graph the Rust port runs (Qwen3-ASR audio
tower -> Qwen3 decoder, one causal forward, ``lm_head`` over 5000 time classes
of 80 ms, argmax at every ``<timestamp>`` slot).

Unit splitting follows the OFFICIAL ``qwen_asr.inference.qwen3_forced_aligner
.Qwen3ForceAlignProcessor`` (qwen-asr 0.0.6), which differs from mlx-audio's
copy in two places; the exporter patches the mlx-audio processor so both
sides follow the official behaviour:

- Chinese goes through ``tokenize_space_lang`` (whitespace split, keep
  letters/numbers/apostrophe, then one unit per CJK ideograph), not
  mlx-audio's ``tokenize_chinese_mixed``.
- Korean uses soynlp ``LTokenizer(scores=...)`` with the official
  ``korean_dict_jieba.dict`` word list (every word scores 1.0); mlx-audio uses
  an empty score table. The dict ships inside the qwen-asr wheel, which is
  fetched from PyPI by pinned URL + sha256 (qwen-asr itself pulls torch,
  gradio and vllm, so it is not installed).
- Japanese needs the nagisa BiLSTM tagger and is not ported (the Rust
  ``ALIGNER_LANGUAGES`` omits it).

Outputs:

1. Default mode writes ``crates/aimd-engine/tests/golden/aligner/``:

   - ``units.json``          unit-splitting cases (text, language, units)
   - ``fix_timestamp.json``  ``fix_timestamp`` cases (raw ms -> repaired ms)
   - ``cases/<name>/audio.wav``            16 kHz mono s16 input clip
   - ``cases/<name>/audio_features.f32``   tower output ``[pads, 1024]``
   - ``cases/<name>/ts_logits.bf16``       lm_head logits at the timestamp
                                           slots ``[n_ts, 5000]`` (raw bf16)
   - ``cases/<name>/meta.json``            text, language, units, pad count,
                                           prompt ids, argmax ids, raw ms, items
   - ``manifest.json``       versions, checkpoint, sha256s

   and ``crates/aimd-engine/src/models/qwen3_aligner/korean_dict.txt``
   (sorted unique dict words of >= 2 characters: ``LTokenizer`` only scores
   prefixes of length >= 2).

2. ``--corpus`` aligns every clip of the golden ASR corpus
   (``target/corpus``, see ``scripts/build_corpus.py``) with its reference
   text, plus 60 s / 300 s English and Chinese clips concatenated from it,
   and writes ``target/aligner/reference.json`` (items + reference latency)
   for ``cargo run --release -p aimd-engine --example aligner_parity``.

Usage::

    uv run scripts/export_aligner_reference.py
    uv run scripts/export_aligner_reference.py --corpus
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "crates" / "aimd-engine" / "tests" / "golden" / "aligner"
KOREAN_DICT_OUT = (
    REPO_ROOT
    / "crates"
    / "aimd-engine"
    / "src"
    / "models"
    / "qwen3_aligner"
    / "korean_dict.txt"
)
CHECKPOINT = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
QWEN_ASR_WHEEL = (
    "https://files.pythonhosted.org/packages/01/12/"
    "d3027a7e4dc2eea0b12a4bf8414a7109f055004e1771666e01d8859d3ca0/"
    "qwen_asr-0.0.6-py3-none-any.whl"
)
QWEN_ASR_WHEEL_SHA256 = (
    "b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3"
)
KOREAN_DICT_MEMBER = "qwen_asr/inference/assets/korean_dict_jieba.dict"

# Committed checkpoint-tier cases: short corpus clips (wav committed).
CASES = [
    (
        "en01",
        "asr/en01-v0.wav",
        "English",
        "The quick brown fox jumps over the lazy dog.",
    ),
    ("zh08", "asr/zh08-v0.wav", "Chinese", "我们的会议改到星期二下午三点。"),
]

UNIT_CASES = [
    ("English", "The quick brown fox jumps over the lazy dog."),
    ("English", "Don't stop: it's 3.14, not 22/7 -- right?!"),
    ("English", "  Context   engineering\tdecides\nwhat a model sees.  "),
    ("English", "GPU-accelerated inference 用MLX运行。"),
    ("English", "...!!! ---"),
    ("English", ""),
    ("Chinese", "今天天气很好，我们一起去公园散步吧。"),
    ("Chinese", "我用GPT-4写代码，效率提高了50%！"),
    ("Chinese", "hello,world 你好 AI 时代"),
    ("Chinese", "上下文工程（context engineering）很重要。"),
    ("Chinese", "𠀀𪜀㐀豈"),
    ("Cantonese", "我哋今日去飲茶，好唔好？"),
    ("German", "Größenwahn ist übermäßig — sagt Jürgen."),
    ("French", "L'homme qu'il a vu, c'était « là-bas »."),
    ("Spanish", "¿Dónde está la biblioteca? ¡Aquí!"),
    ("Italian", "Perché l'università è chiusa?"),
    ("Portuguese", "A ação começou às três horas."),
    ("Russian", "Съешь же ещё этих мягких французских булок, да выпей чаю."),
    ("Korean", "안녕하세요. 오늘 날씨가 정말 좋네요!"),
    ("Korean", "회의는 화요일 오후 세 시로 변경되었습니다."),
    ("Korean", "대한민국의 수도는 서울입니다"),
    ("Korean", "나 너 AI가 좋아"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def korean_scores() -> dict[str, float]:
    with urllib.request.urlopen(QWEN_ASR_WHEEL) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != QWEN_ASR_WHEEL_SHA256:
        sys.exit(f"qwen-asr wheel sha256 mismatch: {digest}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(KOREAN_DICT_MEMBER).decode("utf-8")
    scores = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        scores[line.split()[0]] = 1.0
    return scores


def load_reference(scores: dict[str, float]):
    from mlx_audio.stt.utils import load_model
    from soynlp.tokenizer import LTokenizer

    model = load_model(CHECKPOINT)
    proc = model.aligner_processor
    proc.ko_tokenizer = LTokenizer(scores=scores)

    def encode_timestamp(text: str, language: str):
        lang = language.lower()
        if lang == "japanese":
            raise ValueError("Japanese (nagisa) is not part of the native port")
        if lang == "korean":
            word_list = proc.tokenize_korean(text)
        else:
            word_list = proc.tokenize_space_lang(text)
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        return word_list, "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

    proc.encode_timestamp = encode_timestamp
    return model


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


def instrumented_align(model, audio: np.ndarray, text: str, language: str) -> dict:
    """``ForcedAlignerModel.generate`` step by step, keeping the stages."""
    proc = model.aligner_processor
    input_features, mask, pads = model._preprocess_audio(audio)
    word_list, prompt = proc.encode_timestamp(text, language)
    prompt = prompt.replace("<|audio_pad|>", "<|audio_pad|>" * pads)
    ids = model._tokenizer.encode(prompt, return_tensors="np", add_special_tokens=False)
    input_ids = mx.array(ids)
    features = model.get_audio_features(input_features, mask)
    logits = model(
        input_ids, input_features=input_features, feature_attention_mask=mask
    )
    mx.eval(logits, features)
    ts_id = model.config.timestamp_token_id
    ids_np = np.array(ids[0])
    positions = np.where(ids_np == ts_id)[0]
    ts_logits = logits[0][mx.array(positions)]
    argmax = np.array(mx.argmax(logits[0], axis=-1))[positions]
    raw_ms = argmax * model.config.timestamp_segment_time
    parsed = proc.parse_timestamp(word_list, raw_ms)
    items = [
        {
            "text": it["text"],
            "start": round(it["start_time"] / 1000.0, 3),
            "end": round(it["end_time"] / 1000.0, 3),
        }
        for it in parsed
    ]
    # The staged run must equal the public entry point.
    public = model.generate(audio, text, language=language)
    assert [(i.text, i.start_time, i.end_time) for i in public] == [
        (i["text"], i["start"], i["end"]) for i in items
    ], "instrumented forward diverged from ForcedAlignerModel.generate"
    return {
        "units": word_list,
        "audio_pad_count": int(pads),
        "prompt_ids": [int(x) for x in ids_np],
        "features": np.array(features.astype(mx.float32)).reshape(pads, -1),
        "ts_logits": ts_logits,
        "ts_argmax": [int(x) for x in argmax],
        "raw_ms": [float(x) for x in raw_ms],
        "fixed_ms": [int(x) for x in proc.fix_timestamp(raw_ms)],
        "items": items,
    }


def bf16_bytes(arr: mx.array) -> bytes:
    assert arr.dtype == mx.bfloat16, arr.dtype
    return bytes(memoryview(arr.view(mx.uint16)))


def fix_timestamp_cases(proc) -> list[dict]:
    rng = np.random.default_rng(5)
    cases = [
        [0, 80, 160, 240],
        [0, 80, 1600, 240, 320, 400],
        [400, 80, 160, 240],
        [0, 80, 160, 80, 0],
        [0, 800, 800, 800, 160, 240, 320, 400],
        [0, 80, 4000, 4000, 4000, 160, 240],
        [160, 160, 80, 80, 0, 0],
        [0, 3000, 3100, 3200, 400, 480, 560, 640],
        [80],
        [240, 0],
    ]
    for _ in range(6):
        base = np.sort(rng.integers(0, 300, size=40)) * 80
        noise = rng.random(40) < 0.15
        base[noise] = rng.integers(0, 300, size=int(noise.sum())) * 80
        cases.append(base.tolist())
    out = []
    for raw in cases:
        arr = np.array(raw, dtype=np.float64)
        out.append(
            {"raw_ms": [float(x) for x in raw], "fixed_ms": proc.fix_timestamp(arr)}
        )
    return out


def export_goldens(model, scores, corpus: Path) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    words = sorted({w for w in scores if len(w) >= 2})
    KOREAN_DICT_OUT.write_text("\n".join(words) + "\n", encoding="utf-8")

    proc = model.aligner_processor
    units = [
        {"language": lang, "text": text, "units": proc.encode_timestamp(text, lang)[0]}
        for lang, text in UNIT_CASES
    ]
    (GOLDEN / "units.json").write_text(
        json.dumps(units, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (GOLDEN / "fix_timestamp.json").write_text(
        json.dumps(fix_timestamp_cases(proc), indent=1) + "\n", encoding="utf-8"
    )

    shas = {}
    for name, rel, language, text in CASES:
        case = GOLDEN / "cases" / name
        case.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(corpus / rel, case / "audio.wav")
        audio = read_wav(case / "audio.wav")
        out = instrumented_align(model, audio, text, language)
        out["features"].astype("<f4").tofile(case / "audio_features.f32")
        (case / "ts_logits.bf16").write_bytes(bf16_bytes(out["ts_logits"]))
        meta = {
            "text": text,
            "language": language,
            "samples": int(audio.shape[0]),
            "units": out["units"],
            "audio_pad_count": out["audio_pad_count"],
            "prompt_ids": out["prompt_ids"],
            "classify_num": int(out["ts_logits"].shape[-1]),
            "ts_argmax": out["ts_argmax"],
            "raw_ms": out["raw_ms"],
            "fixed_ms": out["fixed_ms"],
            "items": out["items"],
        }
        (case / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        for f in sorted(case.iterdir()):
            shas[f"cases/{name}/{f.name}"] = sha256(f)
        print(f"{name}: {len(out['units'])} units, {out['audio_pad_count']} pads")
    for f in ["units.json", "fix_timestamp.json"]:
        shas[f] = sha256(GOLDEN / f)
    manifest = {
        "checkpoint": CHECKPOINT,
        "reference": "mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner.ForcedAlignerModel",
        "unit_splitting": "qwen-asr 0.0.6 Qwen3ForceAlignProcessor (Chinese via "
        "tokenize_space_lang, Korean LTokenizer with korean_dict_jieba.dict)",
        "versions": {
            "mlx": mx.__version__,
            "mlx-audio": "0.6.1",
            "numpy": np.__version__,
            "soynlp": "0.0.493",
            "qwen-asr wheel sha256": QWEN_ASR_WHEEL_SHA256,
        },
        "korean_dict_sha256": sha256(KOREAN_DICT_OUT),
        "sha256": shas,
    }
    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def concat_clips(entries, corpus: Path, seconds: float):
    """Concatenate corpus clips (0.3 s gaps) up to ``seconds``, pad with silence."""
    target = int(seconds * 16000)
    gap = np.zeros(int(0.3 * 16000), dtype=np.float32)
    parts, texts, total, i = [], [], 0, 0
    while True:
        e = entries[i % len(entries)]
        wav = read_wav(corpus / e["audio"])
        if total + wav.shape[0] + gap.shape[0] > target:
            break
        parts += [wav, gap]
        texts.append(e["reference"])
        total += wav.shape[0] + gap.shape[0]
        i += 1
    parts.append(np.zeros(target - total, dtype=np.float32))
    return np.concatenate(parts), texts


def export_corpus(model, corpus: Path) -> None:
    manifest = json.loads((corpus / "manifest.json").read_text())
    out_dir = REPO_ROOT / "target" / "aligner"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [
        {
            "id": e["id"],
            "audio": str(corpus / e["audio"]),
            "language": e["language_name"],
            "text": e["reference"],
        }
        for e in manifest["asr"]
    ]
    for lang, name in [("en", "English"), ("zh", "Chinese")]:
        entries = [e for e in manifest["asr"] if e["language"] == lang]
        for seconds in (60, 300):
            audio, texts = concat_clips(entries, corpus, seconds)
            path = out_dir / f"{lang}_{seconds}s.wav"
            write_wav(path, audio)
            sep = " " if lang == "en" else ""
            clips.append(
                {
                    "id": f"{lang}_{seconds}s",
                    "audio": str(path),
                    "language": name,
                    "text": sep.join(texts),
                }
            )
    # Warm-up so the first timing excludes graph compilation.
    model.generate(clips[0]["audio"], clips[0]["text"], language=clips[0]["language"])
    for clip in clips:
        audio = read_wav(Path(clip["audio"]))
        t0 = time.perf_counter()
        result = model.generate(audio, clip["text"], language=clip["language"])
        clip["latency_s"] = time.perf_counter() - t0
        clip["items"] = [
            {"text": i.text, "start": i.start_time, "end": i.end_time} for i in result
        ]
        print(f"{clip['id']}: {len(clip['items'])} units, {clip['latency_s']:.3f}s")
    out = out_dir / "reference.json"
    out.write_text(
        json.dumps(
            {"checkpoint": CHECKPOINT, "clips": clips}, ensure_ascii=False, indent=1
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--corpus", action="store_true", help="export target/aligner/reference.json"
    )
    ap.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "target" / "corpus")
    args = ap.parse_args()
    scores = korean_scores()
    model = load_reference(scores)
    if args.corpus:
        export_corpus(model, args.corpus_dir)
    else:
        export_goldens(model, scores, args.corpus_dir)


if __name__ == "__main__":
    main()
