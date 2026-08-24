"""Tests for opt-in OpenAI-compatible ASR and OCR backends."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.error import URLError

import pytest

from aimd.core.errors import BackendUnavailableError
from aimd.core.remote import RemoteBackendConfig, resolve_remote_backend
from aimd.plugins.asr._plugin import transcribe_file
from aimd.plugins.asr.models.remote import RemoteASRModel
from aimd.plugins.ocr.backends import RemoteOCRBackend, create_ocr_backend
from aimd.plugins.ocr.remote import RemoteOCRClient


class _RecordingHandler(BaseHTTPRequestHandler):
    records: list[dict[str, object]] = []
    statuses: dict[str, int] = {}

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        self.records.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )
        status = self.statuses.get(self.path, 200)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.end_headers()
        if self.path.endswith("/audio/transcriptions"):
            payload = {"text": "remote transcription"}
        else:
            payload = {"choices": [{"message": {"content": "# Remote OCR\n\nbody"}}]}
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@contextmanager
def _mock_server():
    handler = type("RecordingHandler", (_RecordingHandler,), {})
    handler.records = []
    handler.statuses = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", handler
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_remote_config_explicit_over_env_and_normalizes_root_url(monkeypatch) -> None:
    monkeypatch.setenv("AIMD_ASR_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("AIMD_ASR_MODEL", "env-model")
    monkeypatch.setenv("AIMD_ASR_API_KEY", "env-key")

    config = resolve_remote_backend(
        "asr",
        base_url="http://explicit.example/",
        model="explicit-model",
        api_key="explicit-key",
    )

    assert config == RemoteBackendConfig(
        base_url="http://explicit.example/v1",
        model="explicit-model",
        api_key="explicit-key",
    )


def test_remote_config_unset_keeps_local_backend(monkeypatch) -> None:
    for name in ("BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"AIMD_OCR_{name}", raising=False)
    assert resolve_remote_backend("ocr") is None


@pytest.mark.asyncio
async def test_remote_asr_posts_multipart_with_context_and_auth(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test-audio")
    with _mock_server() as (base_url, handler):
        model = RemoteASRModel(
            RemoteBackendConfig(base_url, "Qwen3-ASR-1.7B", "secret")
        )
        result = await model.transcribe(
            audio, language="zh", context="Vocabulary: AIMD"
        )

    assert result == "remote transcription"
    record = handler.records[0]
    assert record["path"] == "/v1/audio/transcriptions"
    assert record["headers"]["Authorization"] == "Bearer secret"
    body = record["body"]
    assert b'name="model"' in body and b"Qwen3-ASR-1.7B" in body
    assert b'name="language"' in body and b"zh" in body
    assert b'name="prompt"' in body and b"Vocabulary: AIMD" in body
    assert b'name="file"; filename="sample.wav"' in body


@pytest.mark.asyncio
async def test_remote_asr_maps_server_failure_to_backend_unavailable(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    with _mock_server() as (base_url, handler):
        handler.statuses["/v1/audio/transcriptions"] = 503
        model = RemoteASRModel(RemoteBackendConfig(base_url, "model", "key"))
        with pytest.raises(BackendUnavailableError, match="HTTP 503"):
            await model.transcribe(audio)


@pytest.mark.asyncio
async def test_remote_asr_maps_connection_failure_to_backend_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    model = RemoteASRModel(RemoteBackendConfig("http://asr.example/v1", "model", "key"))
    monkeypatch.setattr(
        "aimd.plugins.asr.models.remote.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )

    with pytest.raises(BackendUnavailableError, match="offline"):
        await model.transcribe(audio)


@pytest.mark.asyncio
async def test_remote_asr_selection_skips_local_preflight_and_warns_once(
    monkeypatch, tmp_path: Path
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-test-audio")
    warnings: list[str] = []
    with _mock_server() as (base_url, _handler):
        monkeypatch.setenv("AIMD_ASR_BASE_URL", base_url)
        monkeypatch.setattr(
            "aimd.plugins.asr._plugin._select_backend_for_model",
            lambda _model: pytest.fail("local backend selection must not run"),
        )
        monkeypatch.setattr(
            "aimd.plugins.asr.audio_utils.get_audio_duration", lambda _path: 0.0
        )
        monkeypatch.setattr("aimd.plugins.asr._plugin.logger.warning", warnings.append)
        result = await transcribe_file(audio, precision="4bit")

    assert result == "remote transcription"
    assert len(warnings) == 1
    assert "Ignoring ASR precision" in warnings[0]


def test_remote_ocr_posts_image_and_unlimited_extras(tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake-png")
    with _mock_server() as (base_url, handler):
        client = RemoteOCRClient(
            RemoteBackendConfig(base_url, "Unlimited-OCR", "secret")
        )
        result = client.recognize_image(image)

    assert result == "# Remote OCR\n\nbody"
    record = handler.records[0]
    assert record["path"] == "/v1/chat/completions"
    payload = json.loads(record["body"])
    assert payload["model"] == "Unlimited-OCR"
    assert payload["skip_special_tokens"] is False
    assert payload["ngram_size"] == 35
    assert payload["window_size"] == 128
    content = payload["messages"][0]["content"]
    assert content[0]["text"].startswith("<image>")
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_remote_ocr_selection_skips_platform_backend(monkeypatch) -> None:
    monkeypatch.setenv("AIMD_OCR_BASE_URL", "http://ocr.example/v1")
    monkeypatch.setattr(
        "aimd.plugins.ocr.backends.select_ocr_backend",
        lambda: pytest.fail("local backend selection must not run"),
    )
    assert isinstance(create_ocr_backend(), RemoteOCRBackend)


def test_remote_ocr_processes_rendered_pdf_pages(monkeypatch, tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    backend = RemoteOCRBackend(
        RemoteBackendConfig("http://ocr.example/v1", "Unlimited-OCR", "key")
    )
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        "aimd.plugins.ocr.backends._render_pdf_pages",
        lambda *_args, **_kwargs: ((0, first), (1, second)),
    )
    monkeypatch.setattr(
        "aimd.plugins.ocr.backends._cleanup_rendered_pages", lambda _pages: None
    )
    monkeypatch.setattr(
        backend.client,
        "recognize_image",
        lambda path, *, multi_page=False: calls.append((path, multi_page)) or path.stem,
    )

    result = backend.recognize(pdf)

    assert [page.text for page in result.pages] == ["first", "second"]
    assert calls == [(first, True), (second, True)]


def test_remote_ocr_precision_warns_once(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"image")
    warnings: list[str] = []
    with _mock_server() as (base_url, _handler):
        backend = RemoteOCRBackend(
            RemoteBackendConfig(base_url, "Unlimited-OCR", "key")
        )
        monkeypatch.setattr("aimd.plugins.ocr.backends.logger.warning", warnings.append)
        backend.recognize(image, precision="4bit")

    assert len(warnings) == 1
    assert "Ignoring OCR precision" in warnings[0]
