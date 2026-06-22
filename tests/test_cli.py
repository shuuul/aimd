from pathlib import Path

from typer.testing import CliRunner

from aimd.core.adapters.cli.app import app
from aimd.core.application.models import InputRoute, ProcessResult
from aimd.core.types import TextContext


runner = CliRunner()


class _FakeProcessUseCase:
    def __init__(self, result, task_type="transcript"):
        self._result = result
        self._task_type = task_type

    def ensure_supported_input(self, input_source):  # noqa: ARG002
        source_kind = "document_file" if self._task_type == "convert" else "audio_file"
        return InputRoute(source_kind=source_kind, task_type=self._task_type)

    async def execute(self, request):  # noqa: ARG002
        return self._result


def test_cli_transcript_auto_output(monkeypatch, tmp_path: Path) -> None:
    class _Container:
        process_input_use_case = _FakeProcessUseCase(
            ProcessResult(
                task_type="transcript",
                text_context=TextContext(title="Demo Title", chunk_list=["hello"]),
            ),
            task_type="transcript",
        )

    monkeypatch.setattr(
        "aimd.core.adapters.cli.app.build_container", lambda: _Container()
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["input.mp3"])
    assert result.exit_code == 0
    out = Path("Demo_Title_transcript.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hello"


def test_cli_does_not_expose_task_option() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--task" not in result.stdout


def test_cli_convert_epub_output_dir(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "book"

    class _Container:
        process_input_use_case = _FakeProcessUseCase(
            ProcessResult(
                task_type="convert",
                text_context=TextContext(title="book", chunk_list=["x"]),
                output_dir=output_dir,
            ),
            task_type="convert",
        )

    monkeypatch.setattr(
        "aimd.core.adapters.cli.app.build_container", lambda: _Container()
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [str(tmp_path / "aimd.book.epub")])
    assert result.exit_code == 0
    assert "Successfully converted book with images" in result.stdout
