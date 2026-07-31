from importlib import import_module
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from aimd.core.models import ProcessResult, TextContext

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
        "process_input",
        _fake_process_input,
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["input.mp3"])
    assert result.exit_code == 0
    out = Path("Demo_Title_transcript.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "hello"


def test_cli_transcript_output_persists_all_chunks(monkeypatch, tmp_path: Path) -> None:
    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(
                title="Demo Title",
                chunk_list=["# Title\n\n## Content", "subtitle body"],
            ),
        )

    monkeypatch.setattr(
        cli_app,
        "process_input",
        _fake_process_input,
    )

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["https://www.youtube.com/watch?v=test"])
    assert result.exit_code == 0
    out = Path("Demo_Title_transcript.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "# Title\n\n## Content\n\nsubtitle body"


def test_cli_exposes_task_option() -> None:
    command = get_command(app)
    task_option = next(param for param in command.params if param.name == "task")
    assert "--task" in task_option.opts


def test_cli_task_option_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def _fake_process_input(request):
        seen["task_type"] = request.task_type
        return ProcessResult(
            task_type="ocr",
            text_context=TextContext(title="Page", chunk_list=["ocr text"]),
        )

    monkeypatch.setattr(cli_app, "process_input", _fake_process_input)
    monkeypatch.chdir(tmp_path)

    image = tmp_path / "page.png"
    image.write_bytes(b"fake")
    result = runner.invoke(app, [str(image), "--task", "ocr", "-o", "out.md"])

    assert result.exit_code == 0
    assert seen["task_type"] == "ocr"
    assert Path("out.md").read_text(encoding="utf-8") == "ocr text"


def test_cli_rejects_invalid_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["input.mp3", "--task", "nope"])
    assert result.exit_code == 1
    assert "Unsupported task" in result.output


def test_cli_convert_epub_output_dir(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "doc"
    warnings: list[str] = []

    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="convert",
            text_context=TextContext(title="doc", chunk_list=["x"]),
            output_dir=output_dir,
        )

    monkeypatch.setattr(
        cli_app,
        "process_input",
        _fake_process_input,
    )
    monkeypatch.setattr(cli_app.logger, "warning", warnings.append)

    monkeypatch.chdir(tmp_path)
    ignored_output = tmp_path / "ignored.md"
    result = runner.invoke(
        app,
        [str(tmp_path / "document.epub"), "--output", str(ignored_output)],
    )
    assert result.exit_code == 0
    assert "Successfully converted document with assets" in result.stdout
    assert warnings == [
        "Ignoring --output for document asset conversions; output is a directory."
    ]
    assert not ignored_output.exists()


def test_cli_rejects_empty_transcript_output(monkeypatch, tmp_path: Path) -> None:
    async def _fake_process_input(request):  # noqa: ARG001
        return ProcessResult(
            task_type="transcript",
            text_context=TextContext(title="Empty", chunk_list=[]),
        )

    monkeypatch.setattr(cli_app, "process_input", _fake_process_input)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["input.mp3"])

    assert result.exit_code == 1
    assert not Path("Empty_transcript.md").exists()
