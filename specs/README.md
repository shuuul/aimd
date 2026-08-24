# Project specs

Tracked execution records for long-running, multi-workstream, or
handoff-heavy work in this repository. A spec records intent, decisions,
workstreams, progress, and acceptance evidence. Stable behavior, interfaces,
boundaries, commands, and maintenance decisions must still end up in code,
tests, schemas, public docs, and the nearest layered `AGENTS.md`.

Copy [000-template.md](000-template.md) to start a spec.

## Active specs

| Spec | Status | Outcome |
| --- | --- | --- |

## Archived specs

| Spec | Completed | Outcome |
| --- | --- | --- |
| [001-remote-asr-ocr.md](archive/001-remote-asr-ocr.md) | 2026-08-24 | Opt-in OpenAI-compatible ASR/OCR HTTP backends for dspark vLLM |

## Lifecycle

Exactly three top-level statuses:

| Status | Location | Meaning |
| --- | --- | --- |
| `Draft` | `specs/` | Intent, scope, or decomposition is not decision-complete. |
| `Active` | `specs/` | The execution contract is ready and workstreams may proceed. |
| `Completed` | `specs/archive/` | Acceptance and durable documentation sync are complete. |

Blocking is represented at the workstream level, not as a fourth spec status:
record the evidence, the required decision, the owner, and the next action in
the workstream row.

## Numbering

Formal specs are `NNN-kebab-case.md` files in `specs/`; `000` is reserved for
the template. IDs must be unique across `specs/` and `specs/archive/` and
continuous from `001`; never reuse, renumber, or delete an ID to close a gap.
Determine the next ID by scanning both directories. Reserve the next ID and
add the Active index row before parallel work starts.

## Coordination

The coordinator owns frontmatter, scope, cross-workstream decisions, the index
row, and closeout. Every workstream has a stable ID (`WS-01`, `WS-02`, ...).
Workers claim a row before editing and append handoff entries under
`Progress and handoff` instead of rewriting another worker's history.

## Validation

Run the structural validator and its regression tests after any spec change,
and always before archiving:

```bash
uv run python scripts/check_specs.py  # exit 1 lists every problem
uv run pytest -q tests/test_check_specs.py
```

The validator enforces filenames, flat frontmatter, real dates, per-directory
statuses, required sections, and the index tables in this README. It is
deliberately dependency-free and runs in CI before tests.

## Documentation sync

A spec closes only when its lasting behavior has moved into durable
documentation: the nearest `AGENTS.md`, `docs/`, code comments, and
`CHANGELOG.md` as relevant. Record the sync targets in the spec's
`Documentation sync` section. If experiment commits back the spec, record
their final Git refs (annotated tags like `experiments/<topic>-v<N>` or
custom refs like `refs/experiments/<topic>-v<N>`) under `Documentation sync`
→ `### Experiment refs`.

## Closeout

Before moving a spec to `specs/archive/`:

1. Satisfy every success criterion or record the decision that removed it.
2. Move lasting behavior and maintenance knowledge into the owning docs and
   `AGENTS.md`.
3. Record final commands and evidence in the spec.
4. If experiment commits back the spec, record their final Git refs.
5. Complete the completion summary and set `status: Completed`.
6. Move the file (unchanged except closeout edits) and its index row to
   `specs/archive/` in the same change.
7. Run `uv run python scripts/check_specs.py` and the focused validator tests.
