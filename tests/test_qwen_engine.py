from aimd_media.qwen_engine import (
    _parse_qwen_output,
    _resolve_language,
)


def test_resolve_qwen_language_code() -> None:
    assert _resolve_language("zh") == "Chinese"
    assert _resolve_language("English") == "English"
    assert _resolve_language(None) is None


def test_parse_qwen_output_extracts_asr_text() -> None:
    assert (
        _parse_qwen_output("language English<asr_text>Hello world", None)
        == "Hello world"
    )


def test_parse_qwen_output_keeps_forced_language_text() -> None:
    assert _parse_qwen_output("Hello world", "English") == "Hello world"


def test_parse_qwen_output_falls_back_to_plain_text() -> None:
    assert _parse_qwen_output("Hello world", None) == "Hello world"
