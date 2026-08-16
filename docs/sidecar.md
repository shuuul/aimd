# Local desktop sidecar contract

The AIMD HTTP API can run as a same-user, same-machine desktop sidecar. This boundary is
not a remote or multi-user service.

## Launch and readiness

The desktop launcher owns process supervision:

1. Reserve a random available TCP port on loopback, then pass that explicit port as
   `AIMD_HTTP_PORT` before spawning `aimd-api`.
2. Set `AIMD_HTTP_HOST=127.0.0.1` (or another loopback address). AIMD rejects wildcard,
   LAN, public, hostname, zero, and out-of-range bind settings.
3. Generate a cryptographically random per-process bearer secret and pass it only through
   `AIMD_API_TOKEN`. The renderer must never receive this secret.
4. Pass every source/destination directory granted by the user in the path-separated
   `AIMD_ALLOWED_ROOTS` value.
5. Poll authenticated `GET /readyz` until it returns `200 {"status":"ok"}`. `GET
   /healthz` is the liveness endpoint and uses the same authentication.

The parent keeps ownership of the child process. On application shutdown it sends the
normal process termination signal and waits for the sidecar's graceful shutdown. AIMD
requests cooperative cancellation for active jobs, waits up to five seconds, then
cancels remaining asyncio tasks; the parent may force-kill the process after its own
deadline. A sidecar exit or failed readiness deadline is a startup/runtime failure, not a
reason to launch a second unsupervised process.

## Authentication and transport

When `AIMD_API_TOKEN` is set, every endpoint—including health, readiness, SSE, and
OpenAPI—requires `Authorization: Bearer <token>`. OpenAPI advertises `bearerAuth`. Desktop
launches must always set a token. Token-less mode remains available for an explicitly
trusted operator-run API and must not be used by the desktop.

Only loopback HTTP is supported. The parent chooses and communicates the port; AIMD does
not publish a discovery file or expose a network listener. The renderer talks to a narrow
Electron preload API, never directly to the sidecar.

## Shared-filesystem policy

`AIMD_ALLOWED_ROOTS` is an allow-list, not a working-directory hint. AIMD expands and
canonicalizes every local `input_source`, `output_file`, `save_original`, and `cookies`
path before processing. Canonical paths outside all roots are rejected with HTTP 403,
including `..` traversal and symlinks that resolve outside a root. Only `http://` and
`https://` URL inputs are accepted; other URI schemes are rejected.

The desktop grants the narrow roots selected by the user. It must not grant `/`, a home
directory, or another broad root merely for convenience. This is a same-user trust
boundary, not protection against a malicious local process racing filesystem changes
after validation; Electron main remains responsible for user consent and root lifetime.

## Jobs, cancellation, and output lifetime

`POST /v1/jobs` returns status and SSE URLs. `GET /v1/jobs/{id}` returns the current
snapshot and terminal artifact/error. `GET /v1/jobs/{id}/events` emits monotonic
`JobEvent` records and resumes after the `Last-Event-ID` sequence. `DELETE /v1/jobs/{id}`
requests cancellation.

Cancellation is cooperative:

- `requested` means work is waiting for a safe checkpoint.
- `cancelled` means a processor stopped at such a checkpoint (for example between EPUB
  chapters or segmented ASR work).
- `completed_after_request` means an uninterruptible model/subprocess finished safely;
  the completed artifact is retained instead of discarded.

`artifact.markdown` is the exact editor document. `chunk_list` is never reconstructed for
saving. Asset-producing document conversions create a durable, caller-owned `output_dir`
beside the source. AIMD does not remove it at job cleanup or shutdown, and ignores a
separate `output_file` that would detach Markdown from relative assets. In-memory terminal
job metadata is bounded independently of those filesystem outputs.
