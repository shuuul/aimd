from pathlib import Path
from importlib import import_module

from typer.testing import CliRunner

from aimd.core.models import InputRoute, ProcessResult, TextContext


cli_app = import_module("aimd.interfaces.cli.app")
app = cli_app.app
runner = CliRunner()


def test_cli_transcript_auto_output(monkeypatch, tmp_path: Path) -> None:
    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="Demo Title", chunk_list=["hello"]),
        )

    monkeypatch.setattr(
        cli_app,
        "ensure_supported_input",
        lambda _: InputRoute(source_kind="audio_file", task_type="transcript"),
    )
    monkeypatch.setattr(
        cli_app,
        "process_input",
        _fake_process_input,
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
    output_dir = tmp_path / "doc"

    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="convert",
            text_context=TextContext(title="doc", chunk_list=["x"]),
            output_dir=output_dir,
        )

    monkeypatch.setattr(
        cli_app,
        "ensure_supported_input",
        lambda _: InputRoute(source_kind="document_file", task_type="convert"),
    )
    monkeypatch.setattr(
        cli_app,
        "process_input",
        _fake_process_input,
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [str(tmp_path / "document.epub")])
    assert result.exit_code == 0
    assert "Successfully converted document with assets" in result.stdout
