# src/aimd

Package root containing CLI entry point and core utilities.

## OVERVIEW

Core package now exposes both:
- Typer CLI (`aimd`)
- FastAPI service (`aimd-api`)

Dispatch and validation flow is centralized in `service.py`, with engine preflight checks in `capabilities.py` and typed errors in `errors.py`.

## STRUCTURE

```
src/aimd/
├── cli.py          # Typer app, main() entry point
├── api.py          # FastAPI app and HTTP routes
├── service.py      # Shared orchestration for CLI/API
├── capabilities.py # Engine availability + auto-selection preflight
├── errors.py       # Domain exceptions (AimdError subclasses)
├── const.py        # AUDIO_EXTENSIONS, EPUB_EXTENSIONS, TRANSCRIPTION_ENGINES
├── utils.py        # is_url(), is_supported_url(), create_output_path_from_title()
├── types.py        # TextContext Pydantic model
├── __init__.py     # Exports
└── tool/           # Processing modules (see tool/AGENTS.md)
```

## WHERE TO LOOK

| Task | File | Key Functions |
|------|------|---------------|
| CLI commands | `cli.py` | `main()`, `process()` |
| API routes | `api.py` | `/healthz`, `/v1/engines`, `/v1/process` |
| Shared task dispatch | `service.py` | `ensure_supported_input()`, `process_transcript_input()`, `process_convert_input()` |
| Engine preflight | `capabilities.py` | `get_engine_capabilities()`, `resolve_engine_with_preflight()` |
| Error taxonomy | `errors.py` | `AimdError` subclasses with `status_code` |
| Constants | `const.py` | Extensions, engines, language codes |
| File/URL utils | `utils.py` | `sanitize_filename()`, `is_url()`, `is_supported_url()` |
| Data model | `types.py` | `TextContext` |

## CONVENTIONS

- **Import paths**: Use relative imports (`from .tool.audio import ...`)
- **Thin interface layers**: CLI/API should delegate logic to `service.py`
- **CLI options**: All options have short (`-o`) and long (`--output`) flags
- **Engine naming**: Engines are lowercase (`auto`, `mlx`, `cuda`, `cpu`)
- **Language codes**: Whisper-style short codes (`zh`, `en`, `ja`). Only yap converts internally to `zh_CN`/`en_US`.
- **Error handling**: Prefer raising typed errors from `errors.py` rather than raw `ValueError/RuntimeError`

## ANTI-PATTERNS

- **Bypassing service layer**: Duplicating task-routing logic in CLI/API introduces drift
- **Unstructured exceptions**: Raising generic exceptions in service/API path breaks status consistency
- **Cross-layer imports**: Keep `tool/*` focused on processing; avoid API/CLI specific concerns

## NOTES

- `py.typed` signals PEP 561 type compliance
- `__init__.py` exports core utilities for `aimd.*` access
- API script entrypoint is `aimd-api`
