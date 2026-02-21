from pathlib import Path

from typer.testing import CliRunner

from aimd.adapters.cli.app import app
from aimd.application.models import ProcessResult
from aimd.types import TextContext


runner = CliRunner()


class _FakeProcessUseCase:
    def __init__(self, result):
        self._result = result

    async def execute(self, request):  # noqa: ARG002
        return self._result


def test_cli_transcript_auto_output(monkeypatch) -> None:
    class _Container:
        process_input_use_case = _FakeProcessUseCase(
            ProcessResult(
                task_type="transcript",
                text_context=TextContext(title="Demo Title", chunk_list=["hello"]),
            )
        )

    monkeypatch.setattr("aimd.adapters.cli.app.build_container", lambda: _Container())
    monkeypatch.setattr(
        "aimd.adapters.cli.app.ensure_supported_input",
        lambda _src, _checker: "transcript",
    )

    with runner.isolated_filesystem():
        result = runner.invoke(app, ["input.mp3"])
        assert result.exit_code == 0
        out = Path("Demo_Title_transcript.md")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "hello"


def test_cli_convert_epub_output_dir(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "book"

    class _Container:
        process_input_use_case = _FakeProcessUseCase(
            ProcessResult(
                task_type="convert",
                text_context=TextContext(title="book", chunk_list=["x"]),
                output_dir=output_dir,
            )
        )

    monkeypatch.setattr("aimd.adapters.cli.app.build_container", lambda: _Container())
    monkeypatch.setattr(
        "aimd.adapters.cli.app.ensure_supported_input",
        lambda _src, _checker: "convert",
    )

    with runner.isolated_filesystem():
        result = runner.invoke(app, [str(tmp_path / "book.epub")])
        assert result.exit_code == 0
        assert "Successfully converted EPUB with images" in result.stdout
