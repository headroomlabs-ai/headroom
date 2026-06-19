# Claude Code + Vertex AI + Headroom compression — does it work?

*A plain-English deep code review. Last updated 2026-06-18.*

## TL;DR (read this first)

**Yes, Headroom can compress Claude traffic that goes to Vertex AI — but only if you
set it up by hand, and only if you work around one real bug.** There is **no
turnkey path today**: `headroom wrap claude` does not know anything about Vertex,
and the one backend flag everybody is told to use (`--backend litellm-vertex`) is
broken.

Two setups actually work (both need manual environment variables):

- **Setup A — keep Claude Code in Vertex mode**, and point its Vertex URL at
  Headroom. Headroom compresses, then forwards to real Vertex.
- **Setup B — run Claude Code in normal Anthropic mode**, and let Headroom be the
  translator that talks to Vertex (`--backend litellm-vertex_ai`).

Everything in the *middle* (compression, request/response translation, streaming,
tool calls) is implemented correctly. The gaps are all at the **edges**: how the
client is pointed at Headroom, one mis-named backend, and a few env vars Headroom
never sets for you.

> Correction to an earlier claim: it is **not** true that "the Python proxy just
> passes Vertex through without compressing." For the Anthropic publisher it runs
> the full compression pipeline. That earlier statement was based on an incomplete
> read of the routing code; the verified behavior is in this doc.

---

## The thing we're trying to do

An enterprise runs **Claude Code**, but their Claude models live on **Google
Vertex AI** (not the direct Anthropic API). They want **Headroom** in the middle so
their prompts get compressed (fewer input tokens = lower cost), without changing
the answers.

For that to happen, three things must all be true:

1. **The client's traffic must actually reach Headroom** (the proxy must be in the path).
2. **Headroom must compress it.**
3. **Headroom must forward it to Vertex correctly** (right URL, right auth, right body shape) and translate the answer back so Claude Code understands it.

This review checks all three.

---

## The map: where Claude-on-Vertex can run, and what compresses

There are **two proxies** in this repo and **three** possible routes. Only some compress.

| Route | What it is | Compresses? | Notes |
|---|---|---|---|
| **Python proxy, native Vertex `:rawPredict`** (publisher = `anthropic`) | Client sends a real Vertex request to Headroom | ✅ **Yes** | Runs the full Anthropic compression pipeline, keeps the Vertex body shape, forwards the client's own Google token. `proxy_routes.py:648` |
| **Python proxy, `--backend litellm-vertex_ai`** | Client speaks plain Anthropic; Headroom translates to Vertex | ✅ **Yes** (correct string only) | Full Anthropic↔Vertex translation incl. streaming + tools. **`litellm-vertex` is broken — must use `litellm-vertex_ai`.** |
| **Rust proxy, native Vertex `:rawPredict`** | A separate `headroom-proxy` binary | ✅ **Yes** | Correct and well-built — **but never run by `headroom proxy`/`wrap`.** Dead code for normal users. |
| **Python proxy, passthrough** (any *other* publisher) | Generic verbatim forward | ❌ No | Only used for non-Anthropic, non-Google publishers. `openai.py:6014` |

**Key takeaway:** the *compression engine* for Vertex+Claude exists and works in the
Python proxy. The problems are getting traffic into it and one naming bug.

---

## Does it work end-to-end? The honest answer

**Through `headroom wrap claude` with zero extra setup: no.** `wrap claude` only
sets `ANTHROPIC_BASE_URL`. If Claude Code is in Vertex mode it ignores that and
talks straight to Google — Headroom is never in the path. And `wrap claude` has no
`--backend`/`--region` flags and sets no Vertex environment variables.

**With manual setup: yes, one of two ways.** Both are below. Both work *around*
issues, not because the product wires them for you.

---

## Setup A — Claude Code stays in Vertex mode (recommended for Vertex shops)

Idea: Claude Code keeps using its native Vertex mode and its own Google login.
You just tell it "send Vertex requests to Headroom instead of straight to Google,"
and you tell Headroom where the real Vertex endpoint is.

```bash
# 1) Run Headroom, telling it the real Vertex endpoint (match your region!)
headroom proxy --port 8787 \
  --vertex-api-url https://us-east5-aiplatform.googleapis.com   # use YOUR region

# 2) Run Claude Code in Vertex mode, but point its Vertex base URL at Headroom
export CLAUDE_CODE_USE_VERTEX=1
export ANTHROPIC_VERTEX_PROJECT_ID=<your-gcp-project>
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_BASE_URL=http://127.0.0.1:8787   # <-- the redirect that makes it work
claude
```

What happens: Claude Code → `ANTHROPIC_VERTEX_BASE_URL` (Headroom) → Headroom
matches the `:rawPredict` route, sees `publisher=anthropic`, **compresses**, then
forwards to the real Vertex endpoint using Claude Code's own Google token.

Caveats: you must set `--vertex-api-url` to your region (see Issue #6), and
`wrap claude` won't set `ANTHROPIC_VERTEX_BASE_URL` for you (Issue #3).

---

## Setup B — Claude Code in normal Anthropic mode; Headroom talks to Vertex

Idea: Claude Code thinks it's talking to plain Anthropic. Headroom holds the Google
credentials and is the one that actually talks to Vertex.

```bash
# Headroom does the Vertex talking — note the backend name carefully
export HEADROOM_BACKEND=litellm-vertex_ai        # NOT "litellm-vertex" (that's broken — Issue #1)
export HEADROOM_REGION=us-east5                  # becomes the Vertex location
export VERTEXAI_PROJECT=<your-gcp-project>        # Headroom does NOT set this for you (Issue #4)
export GOOGLE_APPLICATION_CREDENTIALS=/path/sa.json   # or use `gcloud auth application-default login`
export ANTHROPIC_API_KEY=placeholder-not-used    # Claude Code needs *a* key to start (Issue #5)

# Do NOT set CLAUDE_CODE_USE_VERTEX here — Claude Code must stay in normal mode
headroom wrap claude
```

What happens: Claude Code → Headroom (plain Anthropic `/v1/messages`) →
**compresses** → LiteLLM converts to Vertex and calls Claude on Vertex → converts
the answer back to Anthropic shape → Claude Code reads it.

Caveats: the backend-name bug (Issue #1), the missing project env (Issue #4), and
this path has **no automated tests** (Issue #8) — smoke-test it before relying on it.

---

## How to verify compression is really happening

1. Open the dashboard: `http://localhost:8787/dashboard` — "tokens saved" should
   climb as you use Claude Code.
2. Or check response headers on a request: `x-headroom-tokens-before`,
   `x-headroom-tokens-after`, `x-headroom-tokens-saved`.
3. Confirm it actually hit Vertex (proxy logs show a `vertex_ai/claude-…` model or
   a Vertex host, not `api.anthropic.com`).

If `tokens-saved` is 0 on large prompts, compression isn't running — re-check the
setup against the issues below.

---

## Every issue we found (the full list)

Severity: **BROKEN** = doesn't work; **GAP** = works only with manual workaround;
**BUG** = wrong behavior in an edge case; **HOUSEKEEPING** = confusing but harmless.

### 1. BROKEN — `--backend litellm-vertex` never reaches Vertex
The backend name is turned into a provider by chopping off `litellm-`, so
`litellm-vertex` becomes the provider `vertex`. But the Vertex integration is keyed
on `vertex_ai`, not `vertex`. So Headroom falls back to a generic "unknown
provider" mode: it builds the wrong model name (`vertex/claude-…` instead of
`vertex_ai/claude-…`), **ignores the region**, and mishandles auth.
**You must use `--backend litellm-vertex_ai`.** Worse: every help message and the
`wrap` example tell users the broken `litellm-vertex`.
*Where:* `providers/registry.py:174-178`, `backends/litellm.py:291,326-336,681-682`;
help text at `cli/proxy.py:524`, `cli/wrap.py:3645`, `proxy/server.py:3913`.
*Fix (small):* alias `vertex` → `vertex_ai` in `create_proxy_backend`, or add a
`"vertex"` entry to the provider registry. Then fix the help text.

### 2. GAP — `headroom wrap claude` has no Vertex support
The `claude` wrap command has no `--backend` and no `--region` (the `aider` wrap
command has both). It only ever sets `ANTHROPIC_BASE_URL`. So there's no flag to
turn on a Vertex backend for Claude Code — you must pre-export env vars.
*Where:* `cli/wrap.py:2780-2819` (vs `cli/wrap.py:3612,3615` for aider).
*Fix:* add `--backend`/`--region` to `wrap claude`, mirroring `aider`.

### 3. GAP — Vertex-mode Claude Code bypasses the proxy, and Headroom never sets the fix
With `CLAUDE_CODE_USE_VERTEX=1`, Claude Code ignores `ANTHROPIC_BASE_URL` and goes
straight to Google. There **is** a documented override — `ANTHROPIC_VERTEX_BASE_URL`
— that points Claude Code's Vertex traffic at a gateway. But Headroom never sets it
(0 references in the codebase). So the proxy has the right routes, but nothing
connects the client to them automatically.
*Where:* repo-wide grep for `ANTHROPIC_VERTEX_BASE_URL` = 0 hits.
*Fix:* in a Vertex-aware `wrap claude`, set `ANTHROPIC_VERTEX_BASE_URL` to the proxy.

### 4. GAP — the GCP project is never passed to LiteLLM
For Setup B, Headroom passes the region to LiteLLM but never the project. You must
export `VERTEXAI_PROJECT` (or `GOOGLE_CLOUD_PROJECT`) yourself or it fails.
*Where:* `backends/litellm.py:682` sets only `vertex_location`.
*Fix:* thread a project config/env through to the LiteLLM call.

### 5. GAP — no placeholder API key for Claude Code
With a custom `ANTHROPIC_BASE_URL`, Claude Code needs *an* `ANTHROPIC_API_KEY` to
start, even though the proxy uses Google creds upstream. `wrap claude` never sets a
placeholder, so the user must.
*Where:* `cli/wrap.py:2984-2988`.
*Fix:* set a placeholder key (or `ANTHROPIC_AUTH_TOKEN`) when launching.

### 6. BUG — region/host mismatch
Headroom pins the Vertex host to one region (default `us-central1`) but throws away
the region in the client's request path. If your client targets, say,
`europe-west1` while Headroom is on the default host, the request goes to the wrong
region unless you set `--vertex-api-url` to match.
*Where:* `copilot_auth.py:936` (host = base + path, no region reconciliation),
`providers/proxy_routes.py:647` (path `location` is discarded), `registry.py:16`.
*Fix:* derive the upstream host from the request path's `location`, or validate they match.

### 7. HOUSEKEEPING — the Rust Vertex proxy is correct but unwired
There's a second, Rust proxy (`crates/headroom-proxy/src/vertex/`) that compresses
Vertex traffic correctly. But `headroom proxy` and `headroom wrap` run the **Python**
server and never call it — it's a separate binary you'd run by hand. This is a
frequent source of "but I thought Vertex compression was added" confusion: it was,
in Rust, on a path nobody runs by default.
*Where:* `crates/headroom-proxy/Cargo.toml` (`[[bin]]`), no Python→Rust bridge to it.
*Fix:* either document that the Rust proxy is separate, or wire/retire it.

### 8. GAP — no tests for the Vertex compression paths
No test instantiates the LiteLLM Vertex backend with a mocked Vertex call, and the
native-Vertex compression route isn't covered against a real Vertex shape. The
translation code is correct by inspection, but unproven by CI.
*Fix:* add a mocked round-trip test (Anthropic in → compressed → Vertex call asserted → Anthropic out).

### 9. HOUSEKEEPING — stale Rust doc comment
`crates/headroom-proxy/src/vertex/mod.rs:42-47` describes a "synthetic model
injection" strategy the code no longer implements. Doc only; behavior is correct.

---

## What's actually solid (so we don't over-correct)

These were verified and are **correct**:

- **Native Vertex `:rawPredict` for `publisher=anthropic` compresses** and preserves
  the Vertex body shape (keeps `anthropic_version`, never injects `model`).
  `proxy_routes.py:648`, `anthropic.py:604-606,1941-1949`.
- **LiteLLM translation is real and complete** (with the `vertex_ai` provider):
  response is rebuilt into Anthropic shape (`backends/litellm.py:575-628`), streaming
  emits proper Anthropic SSE events (`streaming.py:1344-1472`, `litellm.py:736-947`),
  and tool calls round-trip both directions (`litellm.py:525-565,593-602`).
- **Compression runs before the backend dispatch** (`anthropic.py:1671` then
  `:1781`), so the backend always gets the compressed body.
- **Auth is forwarded correctly** on the native path — the client's Google token is
  passed through untouched (`copilot_auth.py:1156-1157`).

---

## Recommended fixes, smallest-first

1. **Fix the backend name (Issue #1)** — one-line alias `vertex`→`vertex_ai`, then
   correct the help text. This is the highest-impact, lowest-effort fix; it turns the
   *documented* command from broken to working.
2. **Add `--backend`/`--region` to `wrap claude` (Issue #2)** — copy from `aider`.
3. **Add a Vertex mode to `wrap claude` (Issues #3, #5)** — detect/set
   `ANTHROPIC_VERTEX_BASE_URL` → proxy, set a placeholder API key, and configure the
   proxy's Vertex upstream — so Setup A becomes one command.
4. **Pass the GCP project (Issue #4)** and **reconcile the region/host (Issue #6).**
5. **Add a mocked round-trip test (Issue #8).**
6. **Decide the Rust proxy's fate (Issue #7)** — document-as-separate or wire it in.

After #1–#3, the honest customer message becomes: *"`headroom wrap claude` works
with Vertex out of the box."* Until then, it's *"works with a documented manual
setup."*
