# src/aimd

Package root containing CLI entry point and core utilities.

## OVERVIEW

Core package with typer-based CLI, constants, and tool registry. Dispatches to `tool/` modules based on input type.

## STRUCTURE

```
src/aimd/
├── cli.py          # Typer app, main() entry point
├── const.py        # AUDIO_EXTENSIONS, EPUB_EXTENSIONS, TRANSCRIPTION_ENGINES
├── utils.py        # is_url(), is_supported_url(), create_output_path_from_title()
├── types.py        # TextContext Pydantic model
├── __init__.py     # Exports
└── tool/           # Processing modules (see tool/AGENTS.md)
```

## WHERE TO LOOK

| Task | File | Key Functions |
|------|------|---------------|
| CLI commands | `cli.py` | `main()`, `process()`, `_get_task_type()` |
| Task dispatch | `cli.py` | `_process_transcript()`, `_process_convert()` |
| Constants | `const.py` | Extensions, engines, locale lists |
| File/URL utils | `utils.py` | `sanitize_filename()`, `is_url()`, `is_supported_url()` |
| Data model | `types.py` | `TextContext` |

## CONVENTIONS

- **Import paths**: Use relative imports (`from .tool.audio import ...`)
- **CLI options**: All options have short (`-o`) and long (`--output`) flags
- **Engine naming**: Engines are lowercase (`auto`, `mlx`, `cuda`, `cpu`)
- **Locale format**: `zh_CN`, `en_US` (underscore, not hyphen)

## ANTI-PATTERNS

- **No engine validation**: `_get_task_type()` doesn't validate engine availability
- **Late failure**: Engine import errors happen at runtime, not CLI parse
- **No output validation**: Output path created without extension check

## NOTES

- `py.typed` signals PEP 561 type compliance
- `__init__.py` exports core utilities for `aimd.*` access
