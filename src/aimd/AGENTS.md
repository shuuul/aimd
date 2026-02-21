# src/aimd

Core package organized with ports/adapters architecture.

## STRUCTURE

- `application/` — use-cases, canonical request/response models, bootstrap wiring
- `infrastructure/` — concrete implementations (capabilities, transcription, URL, documents)
- `adapters/` — CLI, HTTP API, MCP interface adapters
- `cli.py`, `api.py`, `mcp.py` — runtime entrypoints

## CONVENTIONS

- Keep orchestration in `application/use_cases/*`.
- Keep IO/third-party integrations in `infrastructure/*`.
- Keep interface-specific request/response mapping in `adapters/*`.
