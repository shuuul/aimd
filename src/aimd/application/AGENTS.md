# src/aimd/application

Application layer: orchestration and canonical models.

## Responsibilities

- Define request/response models used across interfaces.
- Implement use-cases (`process_input`, `list_engines`).
- Wire dependencies explicitly in `bootstrap.py`.

## Rules

- No direct adapter imports.
- Keep business flow in use-cases; avoid embedding integration logic here.
