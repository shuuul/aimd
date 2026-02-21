# src/aimd/adapters

Interface adapters for CLI, HTTP API, and MCP.

## Responsibilities

- Parse interface-level input.
- Map payloads/options to `application.models.ProcessInput`.
- Map use-case output/errors to interface response format.

## Rules

- Do not implement core processing logic in adapters.
- Reuse shared output persistence helpers from `application/services`.
