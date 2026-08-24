---
id: "001"
title: "Remote ASR and OCR HTTP backends"
status: Draft
created: 2026-08-24
updated: 2026-08-24
coordinator: "David"
---

# 001 — Remote ASR and OCR HTTP backends

## Context

aimd today selects ASR and OCR **only from local runtimes**:

- Darwin/Apple Silicon: MLX (`mlx-audio` / `mlx-vlm`) loading `mlx-community/Qwen3-ASR-*` and `mlx-community/Unlimited-OCR-*` (or GLM-OCR) by kebab-case alias plus `--precision`.
- Linux/CUDA: Transformers adapters loading `Qwen/Qwen3-ASR-*-hf` and `baidu/Unlimited-OCR` in-process.

Evidence: `src/aimd/plugins/asr/capabilities.py` (`select_transcription_backend`), `src/aimd/plugins/ocr/backends.py` (`select_ocr_backend`), `src/aimd/plugins/asr/const.py` aliases (`qwen3-asr-1.7b`, `qwen3-asr-0.6b`), OCR models in `src/aimd/plugins/ocr/models/`. There is no `base_url` / OpenAI-compatible client path. Linux without CUDA (and any machine that should not load weights) cannot use the LAN services.

On 2026-08-24 those services are already running on dspark, separate from DFlash:

| Role | Endpoint | Served model id | Notes |
| --- | --- | --- | --- |
| ASR | `http://192.168.100.114:8000/v1` | `Qwen3-ASR-1.7B` | vLLM 0.27, NVFP4 weights, `/v1/audio/transcriptions` |
| OCR | `http://192.168.100.114:10000/v1` | `Unlimited-OCR` | vLLM 0.27, W4A16_NVFP4, chat completions + image |

Clients (Mac, metacube, dspark) should call these URLs instead of loading 4–7 GiB of weights locally. Existing local MLX/Transformers paths stay the default when no remote URL is configured.

## Goal and success criteria

A user can point aimd at those HTTP services with env or CLI, keep the same kebab-case model aliases, and get markdown out without CUDA/MLX weights on the client.

- [ ] With `AIMD_ASR_BASE_URL=http://192.168.100.114:8000/v1` (and optional `AIMD_ASR_MODEL=Qwen3-ASR-1.7B`), `aimd audio.wav --model qwen3-asr-1.7b` transcribes via `POST /v1/audio/transcriptions` and does not import mlx-audio or load a local checkpoint. Verify: unit test with a mocked HTTP server plus one live LAN smoke test.
- [ ] With `AIMD_OCR_BASE_URL=http://192.168.100.114:10000/v1` (and optional `AIMD_OCR_MODEL=Unlimited-OCR`), `aimd scan.png --model unlimited-ocr` (and scanned PDFs already routed to OCR) calls chat completions with an image payload and Unlimited-OCR prompt conventions. Verify: mocked HTTP plus one live image smoke test.
- [ ] When those env vars (or CLI equivalents) are **unset**, Darwin still uses MLX and CUDA Linux still uses Transformers. Verify: existing ASR/OCR tests stay green; no network calls in that path.
- [ ] `--precision` on a remote request is ignored with a single warning (weights already quantized server-side). Verify: log assertion in the remote unit tests.
- [ ] `ProcessInput.context` / metadata context is forwarded to Qwen3-ASR as the transcription prompt / extra body the vLLM endpoint accepts; unsupported fields are skipped with a warning, matching current local behavior. Verify: request-body fixture.
- [ ] Missing or unreachable remote URL raises `BackendUnavailableError` (not a generic HTTP traceback). Verify: mocked connection failure.
- [ ] CLI, HTTP API, and MCP all honor the same remote settings (env and explicit flags). Verify: one test per interface or a shared resolver test used by all three.
- [ ] `uv run python scripts/check_specs.py` and `uv run pytest -q tests/test_check_specs.py` pass after this spec lands.

## Scope and non-goals

In scope:

- New remote adapters behind the existing `ASRModel` protocol and OCR backend/result types.
- Config: environment variables and matching CLI flags; optional `api_key` defaulting to `not-needed`.
- Selection rule: if an ASR/OCR base URL is set, use the remote backend and skip local CUDA/MLX preflight for that modality.
- OpenAI-compatible wire format only (vLLM on dspark).
- Docs in `AGENTS.md` (model naming, examples) and a short note in `docs/` if a provider page exists.
- Tests with mocked HTTP; optional live tests gated so CI does not need dspark.

Not in scope:

- Serving, quantizing, or restarting the dspark vLLM processes.
- Changing DFlash (`qwen3.8-27b` on `:8888`).
- Merging ASR/OCR/LLM weights or running them in one engine.
- GLM-OCR remote, Whisper, or non-OpenAI vendor APIs.
- NVFP4 conversion or ModelScope downloads inside aimd.
- Auth beyond a bearer/api_key header.
- Streaming responses.

## Decisions

| Date | Decision | Rationale | Affected workstreams |
| --- | --- | --- | --- |
| 2026-08-24 | Remote is opt-in via base URL, not a new `--model` family | Keep kebab-case aliases (`qwen3-asr-1.7b`, `unlimited-ocr`). URL switches *where* they run. | WS-01, WS-04 |
| 2026-08-24 | ASR uses `/v1/audio/transcriptions`; OCR uses `/v1/chat/completions` with image parts | Matches the live dspark vLLM endpoints already verified. | WS-02, WS-03 |
| 2026-08-24 | Do not require CUDA/MLX when a remote URL is set | Mac and CPU-only boxes must be able to call dspark. | WS-01 |
| 2026-08-24 | Ignore `--precision` for remote with a warning | Server weights are already NVFP4; local 4bit/8bit/bf16 maps do not apply. | WS-01, WS-04 |
| 2026-08-24 | Unlimited-OCR remote must send the model card prompt prefix and n-gram extras | Official recipe: prompt starts with `<image>` (e.g. document parsing), `skip_special_tokens=false`, `ngram_size=35`, `window_size=128` (1024 for multi-page). | WS-03 |

## Workstreams

Use `Pending`, `Claimed`, `In progress`, `Blocked`, or `Done`.

| ID | Deliverable | Owner | Status | Dependencies | Verification |
| --- | --- | --- | --- | --- | --- |
| WS-01 | Config resolver: `AIMD_ASR_BASE_URL` / `AIMD_ASR_MODEL` / `AIMD_ASR_API_KEY` and OCR twins; CLI `--asr-base-url` / `--ocr-base-url`; skip local backend preflight when URL set | Unassigned | Pending | None | Unit tests for precedence CLI > env > unset; CUDA-less selection when URL set |
| WS-02 | `asr/models/remote.py` implementing `ASRModel`: multipart POST `{base}/audio/transcriptions`, model id from config (default `Qwen3-ASR-1.7B`), ffmpeg decode still local if the API needs wav/mp3 | Unassigned | Pending | WS-01 | Mocked transcription 200; 503 -> `BackendUnavailableError`; context field in body |
| WS-03 | `ocr` remote adapter: render PDF pages as today, POST chat completions to `{base}/chat/completions`, model default `Unlimited-OCR`, image as data URL, Unlimited-OCR extras | Unassigned | Pending | WS-01 | Mocked image 200; PDF page loop; extras present in JSON |
| WS-04 | Wire resolver through CLI, HTTP API, MCP, and MarkItDown plugin kwargs; warn on remote+precision | Unassigned | Pending | WS-01 | Shared resolver tests imported by CLI/API/MCP tests |
| WS-05 | Tests, `AGENTS.md` examples, README index already updated; live smoke optional `AIMD_LIVE_REMOTE=1` | Unassigned | Pending | WS-02, WS-03, WS-04 | `uv run pytest -q` (no live LAN in CI); spec validator |

## Verification

- `uv run python scripts/check_specs.py`
- `uv run pytest -q tests/test_check_specs.py`
- New unit tests with `httpx`/`respx` or stdlib `http.server` mocks for ASR and OCR remote adapters.
- Regression: existing MLX/Transformers tests unchanged when remote env is unset.
- Manual (optional, LAN): from metacube or Mac,

```bash
AIMD_ASR_BASE_URL=http://192.168.100.114:8000/v1 \
AIMD_ASR_MODEL=Qwen3-ASR-1.7B \
  uv run aimd path/to/short.wav --model qwen3-asr-1.7b

AIMD_OCR_BASE_URL=http://192.168.100.114:10000/v1 \
AIMD_OCR_MODEL=Unlimited-OCR \
  uv run aimd path/to/scan.png --model unlimited-ocr
```

- Live smoke must not run in CI unless explicitly opted in.

## Documentation sync

- Durable product/developer docs: `README.md` examples if they document `--model`; `docs/architecture.md` plugin diagram if it lists only mlx/transformers.
- Nearest local `AGENTS.md`: add remote env vars, keep kebab-case aliases, state that remote skips local precision maps.
- Parent/package guidance: none.
- Root guidance and roadmap: this spec until archived.

### Experiment refs

None yet.

## Progress and handoff

### 2026-08-24 — David — spec draft

- Changed: added this Draft spec and the Active index row in `specs/README.md`.
- Evidence: dspark serving `Qwen3-ASR-1.7B` on `:8000` and `Unlimited-OCR` on `:10000`; aimd has no HTTP backend today (`select_transcription_backend` / `select_ocr_backend` are platform-only).
- Remaining: implementation workstreams WS-01..WS-05 after the spec is marked Active.
- Blockers: none for the draft.
- Next action: review wire format against a real `/v1/audio/transcriptions` and OCR chat request if the first implementation PR starts; then set `status: Active`.

## Completion summary

Not complete. Implementation has not started.
