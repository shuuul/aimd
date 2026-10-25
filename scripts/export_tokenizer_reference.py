#!/usr/bin/env python
"""Export golden tokenizer/chat-template fixtures for the native engine (WS-04).

Runs the reference Hugging Face tokenization paths over deterministic inputs and
writes one JSON case per template scenario into ``crates/aimd-engine/tests/golden/tokenizer/``:

- ``cases/<name>.json``  per case: ``{"model", "case", "checkpoint",
  "prompt_text", "prompt_ids", ...}`` plus the parameters the Rust builders
  take (``audio_pad_count`` for ASR, ``image_pad_count`` for OCR)
- ``manifest.json``      versions, template evidence, per-case index, sha256s

The fixtures are committed so the Rust ``aimd-engine`` tokenizer tests can run
in CI without multi-GB checkpoints; the live-checkpoint tests (encode/decode
roundtrips) skip when the HF cache is absent.

Reference paths reproduced (evidence):

ASR (``Qwen/Qwen3-ASR-1.7B-hf``, transformers ``Qwen3ASRProcessor``)
    The chat template ``chat_template.jinja`` always emits a system turn, then
    the user turn containing one audio placeholder, then either an assistant
    generation prompt or a continued assistant message::

        <|im_start|>system\n{system}<|im_end|>\n
        <|im_start|>user\n<|audio_start|>{audio pads}<|audio_end|><|im_end|>\n
        <|im_start|>assistant\n[{language {Name}<asr_text>}]

    - ``processor.apply_transcription_request(audio, language)`` (used by
      ``src/aimd/plugins/asr/models/transformers.py`` when no context is
      given) appends an assistant message pre-filled with
      ``language {Name}<asr_text>`` (empty prefill without a language) and
      renders with ``continue_final_message=True``.
    - With context, the same adapter builds
      ``[system text, user audio]`` + ``add_generation_prompt=True``, or
      ``[system text, user audio, assistant "language {Name}<asr_text>"]`` +
      ``continue_final_message=True``.
    - The single ``<|audio_pad|>`` in the rendered prompt is replaced by the
      processor with ``<|audio_pad|> * num_audio_tokens`` before tokenization;
      the count is audio-dependent (WS-05/WS-06 mel front-end) and recorded
      per case as ``audio_pad_count``.
    - Language names come from the fixed map in
      ``src/aimd/plugins/asr/models/transformers.py`` (``LANGUAGE_CODE_TO_NAME``).

OCR (``mlx-community/Unlimited-OCR-4bit``, the production mlx-vlm path
``mlx_vlm.models.unlimited_ocr.processing_unlimitedocr.UnlimitedOCRProcessor``)
    The processor chat template concatenates message contents (single user
    turn here), producing ``<image>document parsing.``-style prompts. The
    processor's ``tokenize_with_images`` then splits the formatted prompt on
    ``<image>``, encodes every text segment with ``add_special_tokens=False``
    and inserts ``num_image_tokens`` ``<image>`` ids (id 128815) per image,
    prefixed by the bos id 0::

        [0] + encode(seg_i, no specials) + [128815] * image_pad_count
            + encode(seg_last, no specials)

    The image-pad run is contiguous for single-image gundam mode (global view
    plus local crops); the count is recorded as ``image_pad_count``. Note the
    committed ``chat_template.jinja`` in the checkpoint adds a trailing space
    that the production mlx-vlm processor template does not; the fixtures
    lock the production behavior (no trailing space).

Both checkpoints load through transformers' ``tokenizers`` backend, i.e. the
reference Rust ``tokenizers`` crate implementation, so ids are expected to
match the native engine bit-for-bit. Every case carries an exporter-side
self-check that the recorded ``prompt_text``/segment encoding round-trips to
``prompt_ids`` before anything is written.

Usage::

    uv run python scripts/export_tokenizer_reference.py [--out crates/aimd-engine/tests/golden/tokenizer]

Deterministic: fixed synthetic inputs, no wall-clock, no randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "crates" / "aimd-engine" / "tests" / "golden" / "tokenizer"

ASR_CHECKPOINT = "Qwen/Qwen3-ASR-1.7B-hf"
OCR_CHECKPOINT = "mlx-community/Unlimited-OCR-4bit"

# Language names fixed by the Python adapter
# (src/aimd/plugins/asr/models/transformers.py::LANGUAGE_CODE_TO_NAME); the
# model itself was trained with these exact full names in the forced suffix
# "language <NAME><asr_text>".
LANGUAGE_CODE_TO_NAME = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "ms": "Malay",
    "yue": "Cantonese",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}

# Context samples: free-form biasing text injected as the system turn.
CTX_MIX = "高频词汇： bluesky、ZCode；话题：上下文工程"
CTX_CJK = (
    "会议记录：大家好，欢迎参加上下文工程研讨会。今天讨论分词器与聊天模板的逐位对齐。🎉"
)

# OCR user turns (text after the single <image> placeholder).
OCR_DOCUMENT_PARSING = "document parsing."
OCR_CJK = "识别这一页：你好，世界，上下文工程。"

AUDIO_TOKEN = "<|audio_pad|>"
IMAGE_TOKEN = "<image>"


# ---------------------------------------------------------------------------
# Deterministic synthetic audio (only the length affects the prompt ids; the
# content drives the mel front-end, not the tokenizer)
# ---------------------------------------------------------------------------


def gen_sine_1_5s(sr: int = 16000) -> tuple[np.ndarray, dict]:
    """1.5 s of 440 Hz sine: 150 mel frames -> 20 audio tokens."""
    n = int(sr * 1.5)
    t = np.arange(n, dtype=np.float64) / sr
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32), {
        "type": "sine",
        "freq_hz": 440.0,
        "duration_s": 1.5,
    }


def gen_am_burst_1_8s(sr: int = 16000) -> tuple[np.ndarray, dict]:
    """1.8 s AM burst: 180 mel frames -> 23 audio tokens."""
    freqs = [300.0, 700.0, 1220.0, 2600.0]
    amps = [0.45, 0.25, 0.2, 0.15]
    n = int(sr * 1.8)
    t = np.arange(n, dtype=np.float64) / sr
    carrier = sum(a * np.sin(2.0 * np.pi * f * t) for f, a in zip(freqs, amps))
    am = (0.5 * (1.0 + np.sin(2.0 * np.pi * 3.0 * t))) ** 2
    gate = np.clip(np.sin(2.0 * np.pi * 1.2 * t) * 4.0 + 0.5, 0.0, 1.0)
    wav = np.clip(0.6 * am * gate * carrier, -1.0, 1.0)
    return wav.astype(np.float32), {
        "type": "am_burst",
        "freqs_hz": freqs,
        "duration_s": 1.8,
    }


def gen_tone_0_61s(sr: int = 16000) -> tuple[np.ndarray, dict]:
    """0.61 s tone: 38 mel frames -> 5 audio tokens (partial final chunk)."""
    n = int(sr * 0.61)
    t = np.arange(n, dtype=np.float64) / sr
    return (0.5 * np.sin(2.0 * np.pi * 640.0 * t)).astype(np.float32), {
        "type": "sine",
        "freq_hz": 640.0,
        "duration_s": 0.61,
    }


def gen_tone_4_0s(sr: int = 16000) -> tuple[np.ndarray, dict]:
    """4.0 s tone: 400 mel frames -> 52 audio tokens (four full chunks)."""
    n = int(sr * 4.0)
    t = np.arange(n, dtype=np.float64) / sr
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32), {
        "type": "sine",
        "freq_hz": 440.0,
        "duration_s": 4.0,
    }


# (name, context, language, audio generator)
ASR_CASES = [
    ("asr_no_context_no_language", None, None, gen_sine_1_5s),
    ("asr_no_context_language_chinese", None, "Chinese", gen_sine_1_5s),
    ("asr_context_only", CTX_MIX, None, gen_am_burst_1_8s),
    ("asr_context_language_chinese", CTX_MIX, "Chinese", gen_am_burst_1_8s),
    ("asr_cjk_context_only", CTX_CJK, None, gen_tone_0_61s),
    ("asr_cjk_context_language_japanese", CTX_CJK, "Japanese", gen_tone_4_0s),
]

# (name, user turn text, image size)
OCR_CASES = [
    ("ocr_document_parsing", OCR_DOCUMENT_PARSING, (640, 640)),
    ("ocr_cjk_document_parsing", OCR_CJK, (640, 640)),
    ("ocr_document_parsing_multicrop", OCR_DOCUMENT_PARSING, (1024, 768)),
]


# ---------------------------------------------------------------------------
# ASR reference (transformers Qwen3ASRProcessor, mirrors
# src/aimd/plugins/asr/models/transformers.py::_build_transcription_inputs)
# ---------------------------------------------------------------------------


def asr_reference(
    processor, audio: np.ndarray, context: str | None, language: str | None
) -> tuple[str, list[int], int]:
    """Return (prompt_text, prompt_ids, audio_pad_count) for one ASR case.

    Reproduces the adapter behavior exactly: no context -> the
    ``apply_transcription_request`` convenience path; with context -> manual
    messages with a system turn.
    """
    tokenizer = processor.tokenizer
    audio_pad_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)

    if not context:
        # The adapter calls processor.apply_transcription_request directly.
        out = processor.apply_transcription_request(
            audio=audio, language=language, return_tensors=None
        )
        ids = [int(x) for x in out["input_ids"][0]]
        rendered = processor.apply_chat_template(
            [
                [
                    {"role": "user", "content": [{"type": "audio", "audio": audio}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"language {language}<asr_text>"
                                if language is not None
                                else "",
                            }
                        ],
                    },
                ]
            ],
            tokenize=False,
            continue_final_message=True,
        )[0]
    else:
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": context}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
        ]
        chat_kwargs: dict[str, object] = {"tokenize": True, "return_dict": True}
        if language is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"language {language}<asr_text>"}
                    ],
                }
            )
            chat_kwargs["continue_final_message"] = True
        else:
            chat_kwargs["add_generation_prompt"] = True
        out = processor.apply_chat_template([messages], **chat_kwargs)
        ids = [int(x) for x in out["input_ids"][0]]
        rendered_kwargs = {
            key: value
            for key, value in chat_kwargs.items()
            if key in ("continue_final_message", "add_generation_prompt")
        }
        rendered = processor.apply_chat_template(
            [messages], tokenize=False, **rendered_kwargs
        )[0]

    # The processor expands the single <|audio_pad|> into num_audio_tokens
    # copies before tokenizing; recover that count from the final ids.
    audio_pad_count = sum(1 for token_id in ids if token_id == audio_pad_id)
    if audio_pad_count < 1:
        raise RuntimeError("ASR case produced no audio tokens")
    if rendered.count(AUDIO_TOKEN) != 1:
        raise RuntimeError(f"expected exactly one {AUDIO_TOKEN} in the rendered prompt")

    prompt_text = rendered.replace(AUDIO_TOKEN, AUDIO_TOKEN * audio_pad_count, 1)

    # Self-check: the expanded prompt text must encode back to the ids the
    # processor produced (add_special_tokens=True adds nothing for this
    # tokenizer; the post-processor is a no-op).
    reencoded = [int(x) for x in tokenizer(prompt_text)["input_ids"]]
    if reencoded != ids:
        raise RuntimeError("ASR prompt_text does not re-encode to prompt_ids")
    return prompt_text, ids, audio_pad_count


# ---------------------------------------------------------------------------
# OCR reference (production mlx-vlm UnlimitedOCRProcessor)
# ---------------------------------------------------------------------------


def ocr_reference(
    processor, user_text: str, image_size: tuple[int, int]
) -> tuple[str, list[int], int]:
    """Return (prompt_text, prompt_ids, image_pad_count) for one OCR case."""
    image_token_id = processor.image_token_id
    bos_id = processor.bos_id

    messages = [{"role": "user", "content": f"{IMAGE_TOKEN}{user_text}"}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not rendered.endswith(user_text):
        raise RuntimeError(f"unexpected OCR rendering {rendered!r}")

    from PIL import Image

    image = Image.new("RGB", image_size, "white")
    out = processor(
        text=rendered,
        images=[image],
        cropping=True,
        base_size=1024,
        image_size=640,
    )
    ids = [int(x) for x in out["input_ids"][0]]

    # Self-check: reproduce the processor's tokenize_with_images structure for
    # the single-image case: bos + one contiguous image-token run + text.
    segments = rendered.split(IMAGE_TOKEN)
    if len(segments) != 2:
        raise RuntimeError("expected exactly one <image> placeholder")
    head = [
        int(x)
        for x in processor.tokenizer(segments[0], add_special_tokens=False)["input_ids"]
    ]
    tail = [
        int(x)
        for x in processor.tokenizer(segments[1], add_special_tokens=False)["input_ids"]
    ]
    image_pad_count = sum(1 for token_id in ids if token_id == image_token_id)
    image_positions = [
        index for index, token_id in enumerate(ids) if token_id == image_token_id
    ]
    run_start = 1 + len(head)  # bos + head text ids
    if image_positions != list(range(run_start, run_start + image_pad_count)):
        raise RuntimeError("OCR image tokens are not a single contiguous run")

    reconstructed = [bos_id] + head + [image_token_id] * image_pad_count + tail
    if reconstructed != ids:
        raise RuntimeError(
            "OCR structure reconstruction does not match processor output"
        )

    return rendered, ids, image_pad_count


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_checkpoint_dir(repo_id: str) -> Path:
    """Locate a cached HF snapshot without network access."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        # Cached snapshots can be "incomplete" (missing .gitattributes/README
        # while all model files are present); fall back to a direct cache scan.
        import os

        cache = Path(
            os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")
        )
        repo_dir = cache / ("models--" + repo_id.replace("/", "--"))
        snapshots = sorted(
            path
            for path in repo_dir.glob("snapshots/*")
            if (path / "tokenizer.json").is_file()
            and (path / "tokenizer_config.json").is_file()
        )
        if snapshots:
            return snapshots[-1]
        raise RuntimeError(
            f"checkpoint {repo_id!r} not found in the local HF cache; "
            "run with the checkpoints cached (see CHECKPOINTS in the mission)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import transformers

    out_dir = args.out
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict] = []
    total_bytes = 0

    # ------------------------------------------------------------------ ASR
    from transformers import AutoProcessor

    asr_processor = AutoProcessor.from_pretrained(ASR_CHECKPOINT)
    asr_tokenizer = asr_processor.tokenizer
    asr_special = {
        "eos_token": asr_tokenizer.eos_token,
        "eos_token_id": asr_tokenizer.eos_token_id,
        "pad_token": asr_tokenizer.pad_token,
        "pad_token_id": asr_tokenizer.pad_token_id,
        "im_start_id": asr_tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": asr_tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "audio_start_id": asr_tokenizer.convert_tokens_to_ids("<|audio_start|>"),
        "audio_pad_id": asr_tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
        "audio_end_id": asr_tokenizer.convert_tokens_to_ids("<|audio_end|>"),
        "asr_text_id": asr_tokenizer.convert_tokens_to_ids("<asr_text>"),
    }
    if asr_special["im_start_id"] is None or asr_special["audio_pad_id"] is None:
        print("FATAL: ASR tokenizer is missing template special tokens")
        return 1

    for name, context, language, generator in ASR_CASES:
        audio, recipe = generator()
        prompt_text, ids, audio_pad_count = asr_reference(
            asr_processor, audio, context, language
        )
        case = {
            "model": "asr",
            "case": name,
            "checkpoint": ASR_CHECKPOINT,
            "context": context,
            "language": language,
            "audio_pad_count": audio_pad_count,
            "audio_recipe": recipe,
            "prompt_text": prompt_text,
            "prompt_ids": ids,
        }
        case_path = cases_dir / f"{name}.json"
        case_path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
        manifest_cases.append(
            {
                "name": name,
                "model": "asr",
                "file": str(case_path.relative_to(out_dir)),
                "prompt_ids_len": len(ids),
                "audio_pad_count": audio_pad_count,
                "sha256": sha256_file(case_path),
            }
        )
        total_bytes += case_path.stat().st_size
        print(
            f"{name:36s} ids={len(ids):5d} audio_pads={audio_pad_count:3d} "
            f"language={language or '-'}"
        )

    # ------------------------------------------------------------------ OCR
    import mlx_vlm  # noqa: F401  (records the version in the manifest)
    from mlx_vlm.models.unlimited_ocr.processing_unlimitedocr import (
        UnlimitedOCRProcessor,
    )

    ocr_snapshot = resolve_checkpoint_dir(OCR_CHECKPOINT)
    ocr_processor = UnlimitedOCRProcessor.from_pretrained(str(ocr_snapshot))
    ocr_special = {
        "bos_token": ocr_processor.tokenizer.bos_token,
        "bos_id": ocr_processor.bos_id,
        "eos_token": ocr_processor.tokenizer.eos_token,
        "eos_id": ocr_processor.eos_id,
        "pad_id": ocr_processor.pad_id,
        "image_token": IMAGE_TOKEN,
        "image_token_id": ocr_processor.image_token_id,
    }

    for name, user_text, image_size in OCR_CASES:
        prompt_text, ids, image_pad_count = ocr_reference(
            ocr_processor, user_text, image_size
        )
        case = {
            "model": "ocr",
            "case": name,
            "checkpoint": OCR_CHECKPOINT,
            "user_text": user_text,
            "image_size": list(image_size),
            "image_pad_count": image_pad_count,
            "prompt_text": prompt_text,
            "prompt_ids": ids,
        }
        case_path = cases_dir / f"{name}.json"
        case_path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
        manifest_cases.append(
            {
                "name": name,
                "model": "ocr",
                "file": str(case_path.relative_to(out_dir)),
                "prompt_ids_len": len(ids),
                "image_pad_count": image_pad_count,
                "sha256": sha256_file(case_path),
            }
        )
        total_bytes += case_path.stat().st_size
        print(
            f"{name:36s} ids={len(ids):5d} image_pads={image_pad_count:4d} "
            f"image={image_size[0]}x{image_size[1]}"
        )

    # ------------------------------------------------- committed tokenizers
    # Copy the exact tokenizer.json files into the fixture directory so the
    # Rust golden tests can run builders + encoding hermetically in CI (no HF
    # cache required). The live-checkpoint tests cross-check these copies
    # against the cache snapshots.
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    asr_snapshot = resolve_checkpoint_dir(ASR_CHECKPOINT)
    ocr_snapshot = resolve_checkpoint_dir(OCR_CHECKPOINT)
    copies = [
        ("qwen3-asr-1.7b.tokenizer.json", asr_snapshot / "tokenizer.json"),
        ("unlimited-ocr.tokenizer.json", ocr_snapshot / "tokenizer.json"),
    ]
    tokenizer_copies_manifest = []
    for target_name, source in copies:
        target = ckpt_dir / target_name
        target.write_bytes(source.read_bytes())
        tokenizer_copies_manifest.append(
            {
                "file": str(target.relative_to(out_dir)),
                "checkpoint": source.parent.parent.parent.name.replace(
                    "models--", ""
                ).replace("--", "/"),
                "source": str(source),
                "sha256": sha256_file(target),
            }
        )
        total_bytes += target.stat().st_size
        print(f"copied {source} -> {target}")

    manifest = {
        "version": 1,
        "generator": "scripts/export_tokenizer_reference.py",
        "checkpoints": {
            "asr": ASR_CHECKPOINT,
            "ocr": OCR_CHECKPOINT,
        },
        "generated_with": {
            "transformers": transformers.__version__,
            "mlx_vlm": mlx_vlm.__version__,
            "numpy": np.__version__,
        },
        "language_map": LANGUAGE_CODE_TO_NAME,
        "special_tokens": {"asr": asr_special, "ocr": ocr_special},
        "committed_tokenizers": tokenizer_copies_manifest,
        "template_evidence": {
            "asr": (
                "chat_template.jinja of Qwen/Qwen3-ASR-1.7B-hf; system turn always "
                "emitted; user turn holds <|audio_start|><|audio_pad|>*n<|audio_end|>; "
                "assistant continuation is 'language {Name}<asr_text>' with "
                "continue_final_message=True (apply_transcription_request) or "
                "add_generation_prompt=True without a language; context goes into "
                "the system turn"
            ),
            "ocr": (
                "mlx_vlm.models.unlimited_ocr.processing_unlimitedocr."
                "UnlimitedOCRProcessor: template concatenates message contents "
                "(no trailing space for a single user turn), then "
                "tokenize_with_images encodes text segments with "
                "add_special_tokens=False and inserts a contiguous run of "
                "<image> ids (128815) after bos id 0"
            ),
        },
        "cases": manifest_cases,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    total_bytes += manifest_path.stat().st_size

    print(f"\nwrote {len(manifest_cases)} cases + manifest to {out_dir}")
    print(f"fixture bytes: {total_bytes} ({total_bytes / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
