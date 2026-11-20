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

1. It accepts `CONNECT` and normalizes the target host (`normalize_host`: lowercase, strip a
   trailing root dot and any `:port`). If that host is in the **cloudcode allowlist**, the
   terminator answers `200 Connection Established` and byte-splices the raw connection to the
   in-process **hypercorn** dispatch server on loopback. It does not terminate TLS itself.
2. For **every other** `CONNECT`, it performs a raw bidirectional **byte-splice** — no TLS
   termination, no certificate, no inspection.

Both paths splice bytes; only the destination differs. `AgyDispatchServer` terminates TLS for
the allowlisted host, minting a leaf per SNI from the local root CA (`mint_leaf`, cached in
`_LeafCache`), negotiating **h2 or http/1.1** via ALPN, and serving the **existing FastAPI
app** — so the decrypted request reaches the same `/v1internal:streamGenerateContent` route →
`handle_google_cloudcode_stream`, and compression and upstream origination are unchanged.
Serving the app under hypercorn rather than hand-rolling server-side HTTP/2 framing also
removes the h2-vs-http/1.1 unknown (an http/1.1-downgrade live test was inconclusive: agy's
OAuth token had expired, and mitmproxy over-terminates the non-selective auth path). New dep:
`hypercorn`.

*(Superseded 2026-07-31: earlier revisions had the terminator mint a leaf and terminate TLS
in-process, with the hypercorn dispatch server recorded here as a later amendment. Production
always tunnelled to dispatch, so that code path survived only in tests and has been deleted
along with `_upgrade_to_tls_server`, `_build_server_ssl_context` and `DispatchCallback`.)*

### Host normalization is one invariant, not four checks
Four places compare a host against the allowlist: the `CONNECT` target, the dispatch SNI
callback, the post-handshake `Host` guard, and `cloudcode_host_base` on the passthrough path.
All four compare the output of `normalize_host`. When they disagreed, `CloudCode-PA.googleapis.com`
passed the SNI and Host guards but failed the exact-match `CONNECT` check, so the connection
fell through to the blind tunnel: the request still worked, but skipped termination and
compression with no signal that anything had been bypassed. Silent bypass is worse than a
hard failure, which is why the normalization belongs in one function that all four call.

`normalize_host` lowercases, strips a trailing root dot, and strips `:port` **only when the
suffix is all digits**. `example.com:abc` names no port, so it stays whole and fails the
allowlist as it should; requiring exactly one colon leaves IPv6 literals such as `::1` alone.

### Blind-tunnel targets are restricted
The terminator is an unauthenticated `CONNECT` proxy on loopback for the life of an agy
session, so anything running as the user can drive it. `_resolve_tunnel_target` refuses two
destinations and returns the vetted address the tunnel then dials:

- **the terminator's own port** — `CONNECT 127.0.0.1:<terminator_port>` makes it tunnel into
  itself, costing two file descriptors per nesting level until they run out;
- **link-local addresses** — `169.254.0.0/16` carries the cloud instance-metadata service.

The check runs on the resolved addresses rather than the literal, so a name that resolves to
`127.0.0.1` is caught too, and dialling the vetted address means no second lookup can
substitute another. Other loopback ports stay reachable on purpose: a local process can open
them directly, so refusing them would buy nothing and break plain local tunnelling.

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

**The request goes back to the host agy chose.** `_resolve_cloudcode_base_url` takes the
`CONNECT` host (carried through as the `Host` header) and re-originates to that same host when
it is allowlisted, falling back to the default backend otherwise. This matters because the
allowlist holds two hosts: resolving every antigravity request to one default sent a request
addressed to `cloudcode-pa.googleapis.com` — and the OAuth bearer with it — to
`daily-cloudcode-pa.googleapis.com` instead. An explicit `HEADROOM_ANTIGRAVITY_API_URL` still
wins over both, since an operator setting it is choosing the backend deliberately.

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
  non-allowlist CONNECTs through the corporate proxy (never TLS-terminating the chained
  leg), instead of dialing direct. The child is handed a userinfo-free loopback URL and
  sends no `Proxy-Authorization` header of its own, so a chained CONNECT reached `407`
  until this ADR's v1.1 update: the terminator now derives `Proxy-Authorization` from
  `HTTPS_PROXY`'s own `user:pass@` userinfo (percent-decoded, `http`/`https` schemes
  only) and sends it, taking precedence over any inbound header. This is sent in
  cleartext to an `http://` upstream proxy, matching curl/Go/requests; and
- merge any pre-existing corporate CA (from the user's `SSL_CERT_FILE`/`NODE_EXTRA_CA_CERTS`
  or system store) into the combined bundle so the real internet still validates. Only x509
  objects with `basicConstraints CA:TRUE` are merged (do not blindly concatenate arbitrary
  user-pointed PEM, which would widen `agy`'s trust beyond intended roots).

Chaining failures are reported per connection (`403` for a loopback upstream proxy, `502`
when it cannot be reached) rather than pre-flighted at launch.

An `https://` upstream proxy is chained to over TLS — `ssl.create_default_context()`
(default certificate validation, no bypass knob), SNI set to the proxy's own hostname
(the tunnelled target's TLS handshake and SNI travel separately, inside the tunnel), and
ALPN pinned to `http/1.1` so a proxy that would otherwise negotiate `h2` cannot leave the
terminator writing a CONNECT frame into an HTTP/2 connection. There is deliberately no
override: an operator whose corporate TLS proxy presents an internal-CA certificate that
isn't merged into the combined bundle goes from working (accidentally, over plaintext) to
a `502`, surfaced per connection rather than silently downgraded.

### Fail-open observability (required)
Failing open (forward original bytes on compression/dispatch error) keeps `agy` working, but
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
- **Stability:** fail-open — any compression/dispatch error forwards the original bytes so
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
  used at both `load_cert_chain` call sites (dispatch placeholder init; dispatch
  `_sni_callback`). The terminator has none: it byte-splices to the dispatch server and never
  terminates TLS itself. Primary path (Linux, `os.memfd_create`
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
- **No TLS-verification bypass:** an earlier revision of this work added an
  `HEADROOM_SSL_VERIFY=false` switch that blanked `SSL_CERT_FILE`/`CURL_CA_BUNDLE` and set
  `NODE_TLS_REJECT_UNAUTHORIZED=0` for launched agents, with `agy` exempted so the injected
  bundle survived. It has been removed: upstream ships no such switch, and a PR that adds
  TLS interception must not also add a way to turn verification off.
- Plaintext `Authorization` / `x-goog-api-key` post-termination are routed only through the
  existing `redact_for_wire_debug` redactor (helpers.py — covers both keys); the request auth
  is not persisted in the semantic cache (verified: cache keys on messages+model, stores
  response headers only). No parallel log sink is introduced.

## Files touched (regression-audit surface)
- New: `headroom/proxy/` CA-lifecycle, terminator, dispatch-adapter modules.
- Edited (shared): `headroom/cli/wrap.py` (`agy()` + `unwrap agy` +
  `_launch_tool` threading); `headroom/proxy/handlers/gemini.py:28`
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
`headroom/proxy/agy_retrieve.py`) is `AgyDispatchServer(plain_http=True)`: the same hypercorn
lifecycle serving the same FastAPI app without SNI TLS. A separate guard requires a loopback
peer, exactly one loopback Host, and an allowed retrieval GET/POST route. It
answers plain HTTP on loopback and does **not** implement the MCP-over-HTTP (streamable HTTP)
transport. Registering it as a `url`-type entry
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

## Savings & dashboard integration (per-project attribution)

`headroom wrap agy` runs its selective-MITM dispatch as an in-process `create_app()`
inside the wrap process — a **separate OS process** from the long-running shared proxy
that renders the savings dashboard. The dashboard reads that shared process's *in-memory*
metrics (`m.tokens_saved_total`, `m.savings_tracker.stats_preview()`), so agy's savings,
recorded in agy's own process, never reached it. Two consequences were reported on
PR #1044: no agy savings on the dashboard, and no agy project row in Per-Project Savings.

Resolution (two parts):

- **Per-project attribution.** agy is a Go binary with no header/base-URL knob, so the
  project label cannot be injected via the child's env (as it is for Claude/Codex). It is
  injected at the MITM boundary instead: `make_host_guard` stamps `x-headroom-project`
  (the launch-directory basename, computed once) onto every intercepted request *after*
  the Host allowlist check — the trust boundary is unchanged.

- **Cross-process savings via a durable event inbox.** agy does not write shared savings
  state directly. In the agy process, `HEADROOM_SAVINGS_PATH`, `HEADROOM_SAVINGS_EVENTS_PATH`,
  and `HEADROOM_OTEL_METRICS_ENABLED=0` are redirected to a throwaway temp dir, and each
  request emits one event file into `~/.headroom/savings.d/` carrying the exact
  `PrometheusMetrics.record_request` arguments. The shared proxy drains that inbox (a
  periodic task plus an opportunistic drain on `/stats`) and **replays each event through
  its own `record_request` funnel** — the single funnel that already updates every
  dashboard surface (token/$ heroes, per-project rows, history, CSV, ledger). Because the
  proxy replay is the sole writer of shared savings state, each agy request is counted
  once. Delivery is **at-least-once** with a best-effort processed-id journal: savings are
  estimates, so a rare double-count in the crash window between record and unlink is
  accepted rather than paying for a transactional store. For users who never run agy the
  inbox is empty and the dashboard is byte-identical to before.

## Third-party tool parity (code memory)

`headroom wrap agy` sets up the same code memory as every other client: **Serena is the
engine**, registered via `AgyRegistrar` with a verify-then-remove `initialize` handshake and
a ledger record so `unwrap agy` removes it cleanly; user-managed entries are preserved.
tokensave and the CLI context tools (rtk, lean-ctx) were retired upstream, so `wrap agy`
installs neither — it only *removes* what earlier releases left behind
(`_disable_tokensave_mcp`, `headroom.context_tool_cleanup`). `--no-tokensave` survives as a
hidden no-op flag; `--code-graph` no longer registers an MCP server for agy at all, it
forwards to the proxy's live code-graph watcher exactly as `wrap claude` does.

**MCP parity in all modes.** An earlier build of agy (~1.0.5) hung indefinitely in
`--print` mode whenever any MCP server was configured, so print mode used to register no MCP.
That hang was **fixed in agy 1.0.16** (re-verified 2026-07-05: Serena and the headroom
retrieve server both answer in ~4s in print mode). agy therefore now wires MCP tooling
**identically in print and interactive mode** — Serena plus the headroom retrieve MCP —
giving agy first-class MCP parity in every mode, like any
other client. Live-verified: `wrap agy -p` wires Serena + retrieve
(handshake-verified) and completes in ~10s. Because the fix is agy-side, `wrap agy` still
runs a runtime `agy --version` preflight before wiring print-mode MCP (headroom-37g.37): an
agy older than 1.0.16, or one whose version can't be detected, is treated as unsafe by
default, so print-mode MCP registration is suppressed and only ledger-owned entries are removed. Remaining MCP entries or
malformed config block the print run with upgrade/interactive guidance.

## functionResponse bulk compression (CCR) — where the savings actually come from

The savings-plumbing above only surfaces savings that a compressor produced; for agy the
compressor initially produced ~zero. Root cause: agy's request bulk lives in
`contents[].parts[].functionResponse.response` — the tool-output leaves the coding agent
resends every turn (file reads, greps, command output). Headroom's message-level
compressors never touched those leaves, so a large agy session compressed almost nothing
(PR #1044: "704 → 718" — compression *inflated* tokens and reverted).

**Design — uniform, deterministic, recoverable.** Every `functionResponse.response` string
leaf across ALL of `contents` (history + tail) above a marker-derived token floor is
replaced by a deterministic CCR marker; the original is cached under
`SHA-256(original)[:24]` and recovered on demand via the injected `headroom_retrieve` MCP
tool. Key properties:

- **Uniform, not live-tail-only.** agy is a MITM that never rewrites the client's own
  history, and it resends the full history each turn. A live-zone/recency boundary (compress
  cold history, keep the tail verbatim) is therefore **cache-incoherent** here: the same leaf
  appears compressed in one turn and verbatim the next, so the model re-diffs it every turn.
  Compressing every leaf identically each turn keeps the cross-turn byte-image stable.
- **Retrieved-content exemption (anti-thrash).** A leaf whose hash the model already fetched
  this turn (a `headroom_retrieve` / `call_mcp_tool` call carrying that 24-hex hash in its
  args) is left verbatim — otherwise the re-sent, just-expanded original would be
  re-compressed into the same marker and the model would retrieve it forever. This mirrors
  the retrieve-call suppression the OpenAI/Anthropic paths already do (keyed by hash, since
  agy has no call_id).
- **Envelope exemption (name-independent).** The hash-in-args exemption above recognizes the
  retrieve call by name/args; on agy the retrieve-result functionResponse carries an opaque
  name and args of just `{hash}`, so that path misses it and the retrieve *output*
  (`{hash, source, original_content, …}`) would itself re-compress into a marker the model
  re-retrieves. The compressor therefore also detects the retrieve-result envelope by
  **content** — value-bearing `hash` (24-hex) + `source` (`local`/`proxy`) anchors,
  independent of tool name — and never compresses that leaf (L1), plus adds its resolved hash
  to the retrieved set so the resent original is exempt too (L2). Verified live: one retrieval
  per hash, no thrash.
- **Default + escape hatch.** `HEADROOM_AGY_FR_MODE` selects `ccr` (default, real savings)
  or `lossless` (a safety floor that never emits markers). The WU4 efficacy trial gated the
  default: ccr ships because it delivers material savings while `headroom_retrieve` is wired;
  a lossless downgrade warns loudly if retrieve is not wired so markers can never become
  unrecoverable silently.
- **Revert-independent accounting.** Savings are recorded from the compression decision, not
  from whether the upstream later reverts — each turn independently avoided sending those
  bytes.

## SSE output-token accounting (Cloud Code Assist response-envelope unwrap)

Cloud Code Assist wraps streaming responses in a `response` envelope
(`{"response": {"usageMetadata": {…}}}`), mirroring the request-side wrap. Both gemini SSE
usage parsers read `usageMetadata` at the top level only, so agy's `candidatesTokenCount`
never parsed and every turn logged "Could not parse output_tokens from SSE, estimating N
from B bytes" — output tokens on the dashboard/ledger were a `bytes//40` estimate. The
gemini branches now unwrap the envelope when top-level `usageMetadata` is absent; native
Gemini (top-level) chunks are unaffected.
