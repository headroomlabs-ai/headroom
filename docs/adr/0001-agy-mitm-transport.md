# ADR 0001 — Transport for compressing Google Antigravity CLI (`agy`) traffic

- Status: Accepted (design-review-gate PASSED — PM/Architect/Designer/Security/CTO all APPROVED, 2026-06-15)
- Date: 2026-06-15
- Epic: `headroom-30y` · Task: `headroom-30y.1`

## Context

Headroom wraps coding agents by pointing them at its local proxy via a base-URL
environment variable (Claude Code → `ANTHROPIC_BASE_URL`, Codex → `config.toml`,
etc.) and compressing the JSON bodies that flow through.

`agy` (Google Antigravity CLI) cannot be wrapped this way. Verified empirically:

- `agy` is a stripped **Go** binary (not Node), config dir `~/.gemini/antigravity-cli/`.
- It exposes **no base-URL override**: `CODE_ASSIST_ENDPOINT`, `GOOGLE_GEMINI_BASE_URL`,
  `GOOGLE_CLOUD_CODE_ENDPOINT` are absent from the binary and are **ignored at runtime**
  (live test: `agy --print` returned correct output with all three pointed at a dead port).
- It **honors** Go proxy vars (`HTTPS_PROXY`/`HTTP_PROXY`) and CA-trust vars
  (`SSL_CERT_FILE`/`CACERT_PATH`/`NODE_EXTRA_CA_CERTS`).
- Backend: reached via HTTP **CONNECT** then TLS + **HTTP/2**, REST JSON
  `POST /v1internal:streamGenerateContent?alt=sse` (SSE response). No TLS pinning
  (a mitmproxy CA was accepted in the capture spike).
- The request body (`{model, project, request:{contents:[{parts:[{text}]}]}}`) is **already**
  what `headroom/proxy/handlers/gemini.py:handle_google_cloudcode_stream` compresses.

So compression value is reachable, but only by intercepting `agy`'s TLS — Headroom has
no forward-proxy / CONNECT / certificate-minting capability today (only a reverse proxy
and upstream CA-trust discovery in `ssl_context.py`).

### Two distinct hosts (do not conflate)
- **Allowlist host** = the host `agy` opens `CONNECT` to (capture-verified:
  `daily-cloudcode-pa.googleapis.com`). The terminator matches on this.
- **Upstream host** = where the existing handler re-originates the request. Today that is
  the (wrong) constant `ANTIGRAVITY_DAILY_API_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com"`
  (`gemini.py:28`), corrected under `headroom-30y.4`. These are separate values.

## Decision

**Selective single-host embedded MITM, hosted in the Python proxy.**

A loopback (`127.0.0.1`-only) forward-proxy listener — a **separate `asyncio.start_server`
listener inside the same process** as the FastAPI/uvicorn app (uvicorn does not accept
`CONNECT`), so "one process" holds.

1. It accepts `CONNECT`. If the target host is in the **cloudcode allowlist**, it mints
   (and caches) one leaf certificate signed by a local root CA, terminates TLS, negotiates
   **HTTP/2 via ALPN** (offering `h2` + `http/1.1`) on the **agy-facing** side, parses the
   decrypted request, and hands it to the dispatch adapter.
2. For **every other** `CONNECT`, it performs a raw bidirectional **byte-splice** — no TLS
   termination, no certificate, no inspection.

### Dispatch via hypercorn (T2 — amendment 2026-06-15)
Rather than hand-roll server-side HTTP/2 framing, the decrypted allowlist connection is
served by **hypercorn** running the **existing FastAPI app** in-process on a loopback port.
hypercorn owns TLS termination via a per-SNI cert callback that mints a leaf from the T7
root CA (reusing T8's `mint_leaf`), negotiates **h2 or http/1.1** transparently, and streams
SSE natively. T8's allowlist path therefore **tunnels** the accepted CONNECT to this local
hypercorn HTTPS port instead of terminating TLS itself; T8's blind-tunnel/chain path is
unchanged. The decrypted request hits the same `/v1internal:streamGenerateContent` route →
`handle_google_cloudcode_stream`, so compression + upstream origination are unchanged. This
removes the h2-vs-http/1.1 unknown (an http/1.1-downgrade live test was inconclusive — agy's
OAuth token had expired and mitmproxy over-terminates the non-selective auth path). New dep:
`hypercorn`.

### Upstream-origination ownership (single connection)
The terminator (A2) is **agy-facing only**. It does **not** dial upstream for the allowlist
host. The dispatch adapter (T2) wraps the decrypted request as a Starlette `Request`
(ASGI scope: method/path/query/headers + a `receive()` yielding the decrypted body — the
seam the handler needs, since it reads `_read_request_json(request)`,
`dict(request.headers.items())`, `request.url.query`) and invokes the **existing**
`handle_google_cloudcode_stream`, which remains the **sole** upstream originator (it already
opens the upstream connection via `self.http_client.send(..., stream=True)`). The terminator
splices the handler's `StreamingResponse` (SSE) back over the terminated socket. Exactly one
upstream TLS session per request; the OAuth token is sent upstream once.

### Module invariant (acyclic)
`ca-lifecycle (A1) ← terminator (A2) ← dispatch (T2) → existing handler`. Imports point one
way; the dispatch adapter never reaches back into transport.

`agy` is wrapped by injecting `HTTPS_PROXY=127.0.0.1:<port>` plus a combined CA bundle into
`SSL_CERT_FILE`/`CACERT_PATH`/`NODE_EXTRA_CA_CERTS`.

### Transparency & consent (required)
Wrapping `agy` terminates TLS on its AI connection and makes plaintext `Authorization` /
`x-goog-api-key` visible to the Headroom process. This is categorically different from
base-URL wrapping. Therefore:
- `headroom wrap agy` MUST print a clear one-line disclosure at launch, **before**
  `subprocess.run` and on all non-early-exit paths (via the `env_vars_display` banner): that
  Headroom is intercepting `agy`'s TLS to the **named** cloudcode host
  (`daily-cloudcode-pa.googleapis.com`) via a local, process-scoped CA.
- The docs (`headroom-30y.6`) MUST state this plainly (value-parity, MITM mechanism).
- A `--no-intercept` / `--no-mitm` escape hatch runs `agy` through Headroom in
  byte-splice-only mode (no compression) for users who decline interception.
- `headroom unwrap agy` MUST exist (agy is the first **durable** wrap-only command — it
  writes `mcp_config.json` / `GEMINI.md`; `goose`/`openhands` write nothing and have no
  unwrap). Unwrap removes only Headroom-added entries (merge semantics).

### Enterprise / corporate-proxy coexistence (required, v1 = chain)
`agy` honors a single `HTTPS_PROXY` and one CA bundle, which Headroom overwrites. **v1 commits
to chaining** (not documented-unsupported):
- detect a pre-existing user `HTTPS_PROXY` and **chain** to it — the terminator forwards
  non-allowlist CONNECTs verbatim through the corporate proxy (preserving its proxy-auth
  headers, never TLS-terminating the chained leg), instead of dialing direct; and
- merge any pre-existing corporate CA (from the user's `SSL_CERT_FILE`/`NODE_EXTRA_CA_CERTS`
  or system store) into the combined bundle so the real internet still validates. Only x509
  objects with `basicConstraints CA:TRUE` are merged (do not blindly concatenate arbitrary
  user-pointed PEM, which would widen `agy`'s trust beyond intended roots).

If chaining setup fails, fail-fast with a clear message rather than silently clobbering the
corporate path.

### Fail-open observability (required)
Fail-closed (forward original bytes on compression/dispatch error) keeps `agy` working, but
must never silently nullify the product's value. The design MUST:
- emit a one-line **stderr warning on the first** fail-open occurrence per session
  (compression degraded to passthrough), and
- print an **end-of-session summary** (compressed exchanges vs passthrough count / observed
  token-compression ratio).

The live smoke (T12) already asserts compression is *observed*, not merely error-free; these
signals extend that to the user's normal runtime.

### Properties
- **Performance:** exactly one TLS termination, only on the AI host; all other traffic is a
  zero-parse byte-splice. No second process, no double-TLS, no double-HTTP/2 reframe, no
  extra network hop. Existing handler reused.
- **Security:** see threat model. Interception surface limited to the AI host; root CA
  process-scoped and never in the OS trust store; the **upstream** (Google-facing) leg keeps
  **full** certificate verification against system roots — MITM on the agy-facing side never
  implies trust-anything upstream.
- **Stability:** fail-closed — any compression/dispatch error forwards the original bytes so
  `agy` never breaks; fail-fast on security-critical setup (CA generation, port bind).

## Alternatives considered

| # | Alternative | Verdict | Reason |
|---|---|---|---|
| A | **Embedded single-process MITM, Python** | **CHOSEN** | One process, reuses the Starlette-coupled handler; `cryptography` + `h2` available. Lowest effort-adjusted cost. |
| B | Embedded MITM in the Rust proxy (`crates/headroom-proxy`) | **N/A (resolved, headroom-30y.11)** | The Rust proxy crate is a standalone port that **no `wrap` command launches** — every agent (claude/codex/aider/goose/openhands/openclaw/gemini/agy) runs through the Python proxy (`_start_proxy` → `python -m headroom.cli proxy`). The crate is also client-only (no rustls server / `rcgen` / CONNECT acceptor). Porting the MITM stack to a proxy that carries no wrap traffic is effort for a dead path; agy MITM is **Python-only by design**. The `wrap agy` Rust-backend hard-fail (below) is the enforced contract. No silent drift: documented here. (The Rust **core** — `headroom-core` smart_crusher + `auth_mode` agy classification — already has its agy parity via PyO3.) |
| C | Single-host reverse target via `HTTPS_PROXY`, no per-host MITM | Rejected | The capture shows `agy` uses `CONNECT` + TLS; a passive reverse target without TLS termination cannot read the body. |
| D | `mitmproxy` sidecar | Rejected | Second process + double TLS termination + double HTTP/2 reframe per SSE request + heavyweight dep — a middleman that erodes the latency value proposition. |
| — | Full dynamic per-SNI MITM (intercept all hosts) | Rejected | Needless interception surface / security risk; only one upstream host matters. |

## CA threat model

- Root CA generated once, stored `~/.headroom/ca/` (dir `0700`, key `0600`), regenerated on
  expiry; `basicConstraints` CA:TRUE, `pathlen:0`. On regeneration, old leaf certs and the
  old combined bundle are deleted.
- The CA is **never** added to the OS/system trust store. Injected **only** into the wrapped
  `agy` process environment.
- The combined bundle (= system roots + Headroom CA cert + any pre-existing corporate CA;
  public certs only, no key) is written under `~/.headroom` with `0600` perms (not a
  predictable world-writable temp path); perms asserted after write.
- Leaf certs minted for the cloudcode allowlist host(s), validity ≤ 72h, SAN/EKU
  constrained to that host + `serverAuth` only, cached (bound = allowlist size + 1 — the
  extra slot holds the `headroom.internal` placeholder leaf, below). A non-served placeholder
  leaf is minted once at dispatch start to satisfy `ssl.SSLContext.load_cert_chain` before the
  SNI callback exists; it is never put on the wire (see dispatch trust-boundary enforcement).
- **Dispatch trust-boundary enforcement (allowlist at the SNI + authority layer).** The
  dispatch hypercorn listener is itself a loopback HTTPS port; a local process could connect
  directly and request a leaf for any SNI. Enforced in two layers: (1) the per-SNI
  `set_servername_callback` rejects any `server_name` that is `None` or (lowercased) not in
  the allowlist with `ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME` **before** any mint/cache/swap;
  (2) a mandatory post-handshake ASGI `host`/`:authority` guard (`make_host_guard`) returns
  421 for absent/duplicate/non-allowlisted Host — covering the no-SNI/placeholder path where
  OpenSSL may skip the SNI callback. The dispatch allowlist is the same single value wired
  into the CONNECT terminator (no drift).
- **Leaf private key handling:** `load_cert_chain_in_memory` (`headroom/proxy/agy_ca.py`) is
  used at all three `load_cert_chain` call sites (terminator `_build_server_ssl_context`;
  dispatch placeholder init; dispatch `_sni_callback`). Primary path (Linux, `os.memfd_create`
  available): combined cert+key PEM is written into an anonymous `memfd_create("hr_leaf")`
  file descriptor and loaded via `/proc/self/fd/{fd}`; the fd is closed after load so no file
  ever exists on a filesystem. Fallback path (`memfd_create` absent or `/proc` inaccessible,
  e.g., certain containers): `tempfile.mkstemp` creates a 0600 temp file; perms are asserted
  via `_assert_perms`; `load_cert_chain` reads it; `os.unlink` removes it in a `finally`
  block even if load raises. Leaf private keys are **never** added to any trust store and
  **never** persist beyond the single `load_cert_chain` call.
- `~/.headroom` (the bundle's parent dir) is `0700`; the CA store `~/.headroom/ca/` is `0700`
  with key `0600`; the combined bundle file is `0600`. All perms asserted after write.
- Listener bound to `127.0.0.1` only; `NO_PROXY=127.0.0.1,localhost` loop-guard so the
  terminator can never CONNECT to itself.
- **SSL-bypass interaction:** `_inject_ssl_bypass` (called unconditionally inside
  `_launch_tool` at `wrap.py:2378`, with no `agent_type` param today) blanks
  `SSL_CERT_FILE`/`CURL_CA_BUNDLE` and sets `NODE_TLS_REJECT_UNAUTHORIZED=0` when
  `HEADROOM_SSL_VERIFY=false`. It is made **agent-aware**: for `agy` it must not blank the
  CA vars and must not set the bypass flags. For the **Go** binary `agy` the concrete
  downgrade vector is **CA-var blanking** (`SSL_CERT_FILE=""` erases the injected bundle);
  `NODE_TLS_REJECT_UNAUTHORIZED` is a Node var inert for `agy` but is still exempted for
  hygiene. Other agents' bypass behavior stays byte-identical (regression-tested).
- Plaintext `Authorization` / `x-goog-api-key` post-termination are routed only through the
  existing `redact_for_wire_debug` redactor (helpers.py — covers both keys); the request auth
  is not persisted in the semantic cache (verified: cache keys on messages+model, stores
  response headers only). No parallel log sink is introduced.

## Files touched (regression-audit surface)
- New: `headroom/proxy/` CA-lifecycle, terminator, dispatch-adapter modules.
- Edited (shared): `headroom/cli/wrap.py` (`agy()` + `unwrap agy` + agent-aware
  `_inject_ssl_bypass` + `_launch_tool` threading); `headroom/proxy/handlers/gemini.py:28`
  (host const + resolver, via T4). Handler `gemini.py:740` reused, not modified internally.

## Consequences
- `agy` shipped wrap-only (like `goose`/`openhands`), not added to `ToolTarget`; but it is the
  first wrap-only command with durable on-disk state, so it gains an `unwrap` command.
- HTTP/2 negotiated on the agy-facing side (`h2` sans-io server); upstream leg uses the
  handler's existing httpx h2 client.
- If the Rust proxy is the active backend, `headroom wrap agy` hard-fails with a clear
  "unsupported on Rust backend" message rather than mis-route. This is the enforced contract.
- The Rust proxy port (`crates/headroom-proxy`) gets no `agy` support — **resolved N/A**
  (headroom-30y.11): it carries no `wrap` traffic for any agent, so agy MITM is Python-only by
  design. Documented, not silently dropped.

## Retrieve MCP transport: stdio child, not url-MCP

agy 1.0.10 added `url` support in `mcp_config.json`, allowing an MCP server to be addressed
by HTTP URL instead of a stdio subprocess. The headroom retrieve server (`AgyRetrieveServer`,
`headroom/proxy/agy_retrieve.py`) is a **plain-HTTP/REST loopback** server; it does **not**
implement the MCP-over-HTTP (streamable HTTP) transport. Registering it as a `url`-type entry
would require adding an MCP-HTTP transport layer to the retrieve server for **zero added
capability** — the stdio child (`headroom mcp serve`) already satisfies all retrieve use cases,
and the per-run ephemeral listener is reverted on teardown with no dead pointer left in
`mcp_config.json`.

**Decision:** keep the retrieve integration as a stdio child; do not add an MCP-HTTP transport
to `AgyRetrieveServer`. Revisit only if agy deprecates stdio MCP support.

## Cross-platform status

The agy slice runs on Windows and is **CI-gated** on it: the `agy-windows` job
(`.github/workflows/ci.yml`, `windows-latest`) runs the full agy suite plus
`test_wrap_agy.py` on every code change.

Platform specifics:
- `_assert_perms` is a no-op on non-POSIX platforms (no `os.chmod`/`stat` crash on Windows).
- Atomic bundle writes use `os.replace`; `_write_secure` ORs in `os.O_BINARY`
  (0 on POSIX) so PEM bytes are written verbatim, not CRLF-translated, on Windows.
- System trust source: POSIX/macOS read the detected on-disk CA bundle; Windows has
  no single bundle file, so `_system_trust_pem()` enumerates the ROOT+CA cert stores
  via stdlib `ssl.enum_certificates`, run through the same CA:TRUE filter (no leaf
  trusted as an anchor; no `certifi` dependency).
- Loopback sockets set `SO_REUSEADDR` only on POSIX; on Windows that flag would let
  another local process bind the same port and intercept decrypted traffic, so Windows
  uses `SO_EXCLUSIVEADDRUSE` instead.

**Leaf private-key posture differs by platform (security-relevant):**
- **Linux:** the leaf key is loaded from an anonymous `memfd` and **never touches the
  filesystem**.
- **Windows / macOS (no `memfd`):** the leaf key is written to a `mkstemp` file and
  unlinked immediately after `load_cert_chain`. On POSIX the file is `0600`; on Windows
  POSIX mode bits are not enforceable, so protection comes from the temp directory's
  ACL. **Verified** on `windows-latest` via `icacls`: an `hr_leaf_*.pem` mkstemp file in
  `%LOCALAPPDATA%\Temp` grants Full control only to the owning user, `NT AUTHORITY\SYSTEM`,
  and `BUILTIN\Administrators` — no `Users`/`Everyone`/`Authenticated Users` entry, i.e.
  user-scoped, not world-readable (Administrators can read any file on any OS — unavoidable).
  The guarantee is therefore "owner-only ACL (inherited from `%TEMP%`) + immediate unlink",
  **not** the Linux "never on disk" invariant. The residual exposure is the brief on-disk
  window, mitigated by the immediate unlink; this is a deliberate, documented degradation.
