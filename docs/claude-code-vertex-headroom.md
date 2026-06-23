# Claude Code + Google Vertex AI, with Headroom compression

*Validated end-to-end on 2026-06-19 (Claude Code 2.1.181, Headroom 0.27.0).*

This is the **working, tested** way to run **Claude Code** against **Claude models on
Google Vertex AI** with **Headroom compressing the context** in the middle.

## TL;DR

Run Claude Code in **normal Anthropic mode** (NOT Vertex mode) pointed at a local
Headroom proxy, and let **Headroom** be the thing that talks to Vertex:

```
Claude Code  ──ANTHROPIC_BASE_URL──▶  Headroom proxy  ──LiteLLM (vertex_ai)──▶  Vertex AI
 (normal mode)     (plain http)        (compresses)         (your GCP ADC)        (Claude)
```

Two non-obvious requirements make the difference between "works" and "silently does nothing":

1. **`pip install "google-cloud-aiplatform>=1.38"`** into the proxy's environment —
   LiteLLM's `vertex_ai` provider needs it, or every request 500s with
   `No module named 'vertexai'`.
2. **Start the proxy with `--code-aware`** — coding sessions are mostly *source code*,
   which routes to the AST/code-aware compressor. It is **disabled by default**, so
   without this flag compression no-ops on code and you see `tokens_saved: 0`.

## Why not "just point Claude Code's Vertex URL at Headroom"?

That approach (Vertex mode + `ANTHROPIC_VERTEX_BASE_URL`=proxy) **does not work** with
Claude Code today. In Vertex mode Claude Code runs a **client-side `probeVertexModel`
check before any request**. When `ANTHROPIC_VERTEX_BASE_URL` points at a non-Google
host, that probe fails *instantly* (no network call is made) with a misleading
`"The model … is not available on your vertex deployment"`, and the proxy never
receives a byte. This is a Claude Code limitation, not a Headroom bug. The native
`:rawPredict` passthrough in Headroom is correct and compresses (verified by direct
curl) — but the client won't route to it. So we use the Anthropic-mode path below.

## Prerequisites

- **Google Cloud auth (ADC).** Run once: `gcloud auth application-default login`
  (and `gcloud config set project <PROJECT>`). The proxy uses ADC to call Vertex; no
  API key is held by Headroom. A service-account JSON via
  `GOOGLE_APPLICATION_CREDENTIALS` works too.
- **Vertex Claude quota** for the model + location you intend to use. Confirm with a
  direct call before involving Headroom:
  ```bash
  ACCESS_TOKEN="$(gcloud auth application-default print-access-token)"
  curl -sS -X POST \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" -H "Content-Type: application/json" \
    "https://aiplatform.googleapis.com/v1/projects/<PROJECT>/locations/global/publishers/anthropic/models/claude-sonnet-4-6:rawPredict" \
    -d '{"anthropic_version":"vertex-2023-10-16","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}'
  ```
  HTTP 200 → good. 429 → model exists but no quota in that location. 404 → model not
  enabled in that project/location.
- **Headroom ML extra** for compression: `pip install "google-cloud-aiplatform>=1.38"`
  plus the Kompress ML stack (`torch`, `transformers`, `onnxruntime` — the
  `headroom-ai[ml]` extra). The `kompress-v2-base` model downloads from Hugging Face
  on first use.

## Terminal 1 — start the Headroom proxy (Vertex backend)

```bash
cd /path/to/headroom
source .venv/bin/activate

export VERTEXAI_PROJECT=<YOUR_GCP_PROJECT>
export GOOGLE_CLOUD_PROJECT=<YOUR_GCP_PROJECT>
export VERTEXAI_LOCATION=global              # match where your quota lives

headroom proxy --port 8787 \
  --backend litellm-vertex_ai \              # NOTE: the _ai suffix is required
  --region global \                          # becomes LiteLLM vertex_location
  --code-aware                               # REQUIRED for code compression
```

On startup, confirm components loaded: `curl -s localhost:8787/debug/warmup` should
show `kompress: loaded`, `code_aware: loaded`, `tree_sitter: loaded`,
`smart_crusher: loaded`.

## Terminal 2 — run Claude Code (normal Anthropic mode) against the proxy

```bash
cd /path/to/your/project

export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=sk-placeholder-not-used     # Claude Code needs *a* key to start
export ANTHROPIC_MODEL=claude-sonnet-4-6             # sent to Headroom, mapped to vertex_ai/claude-sonnet-4-6
export ANTHROPIC_SMALL_FAST_MODEL=claude-sonnet-4-6  # pin background model to one you have quota for

# Do NOT set CLAUDE_CODE_USE_VERTEX or ANTHROPIC_VERTEX_BASE_URL — those put Claude
# Code into Vertex mode and trigger the broken probe described above.

claude
```

Claude Code now talks plain Anthropic `/v1/messages` to Headroom; Headroom compresses
and forwards to Vertex via LiteLLM, then translates the answer back.

## Verify compression is happening

- Dashboard: <http://localhost:8787/dashboard> — "tokens saved" climbs as you work.
- `curl -s localhost:8787/stats` → `tokens.saved`, and `request_logs[].transforms_applied`
  (look for `router:tool_result:mixed`, `kompress:*`, `code_aware:*`).
- Savings appear on **large tool outputs** (Bash/Grep/web fetches) and accumulate over
  turns. Note: **`Read`/`Glob`/`Grep`/`Write`/`Edit` outputs are protected from the
  ContentRouter by default** (safest for coding agents); stale `Read`s are handled
  separately by the Read-lifecycle system. So the biggest wins come from non-excluded
  large outputs and multi-turn sessions, not single one-shot reads.

## What `--code-aware` does — and what it never touches

**What it does.** Code-Aware is an **AST (tree-sitter) compressor for source code that
passes through the proxy inside a request**. It parses the code, **keeps the structure
that matters** — imports, function/class signatures, type annotations, error handlers —
and **shrinks the less-important function bodies**, always emitting **syntactically
valid code** (the output still parses). Languages: Python, JS, TS (tier 1); Go, Rust,
Java, C, C++ (tier 2). The original is **stored for retrieval (CCR, ~5-minute TTL)**, so
if the model needs the exact bytes it can pull them back via the `headroom_retrieve`
tool — compression is reversible, not destructive.

**What it does NOT touch:**

- **Your files on disk.** Headroom is a network proxy: it only rewrites the *request
  body* in flight on the way to Vertex. It never reads, writes, or modifies any local
  file. Code-Aware operates on text that is *already inside the API request*, not on
  your repository.
- **Claude Code's `Read` tool output.** `Read`, `Glob`, `Grep`, `Write`, and `Edit`
  are in `DEFAULT_EXCLUDE_TOOLS` and are **protected from the ContentRouter by default**
  (`protect_recent_reads_fraction = 0.0` ⇒ protect-all). So when Claude Code opens a
  file the normal way, **the model sees it verbatim** — Code-Aware does not alter it.
  (Stale `Read`s — files you later edit — are handled separately and reversibly by the
  Read-lifecycle, replacing the superseded copy with a retrievable marker.)

**Where it actually applies:** code that reaches the model through *other* channels —
most commonly **`Bash` output that prints code** (`cat file.py`, `sed`, `nl`, build
logs with snippets) or large code in results from non-excluded/custom tools. That is
the content that gets AST-compressed. In the validation run, the ~22% savings came
exactly from two `Bash` commands that dumped source files — not from `Read`.

**Net:** with the default config your real file reads and edits go to the model
untouched; Code-Aware only trims bulky *incidental* code (shell dumps, logs, pasted
snippets) and keeps the originals retrievable. Omit `--code-aware` if you want zero
code transformation at all (you lose code compression but keep everything else).

## Model-string notes (Vertex)

- Headroom maps clean ids to Vertex publisher ids: `claude-sonnet-4-6` →
  `vertex_ai/claude-sonnet-4-6` (see `headroom/backends/litellm.py`). Newer models use
  the bare alias; older ones are date-pinned (e.g. `claude-sonnet-4-5@20250929`).
- `--region global` works (LiteLLM targets the `aiplatform.googleapis.com` global
  endpoint). Use a specific region (`us-east5`, `europe-west1`, …) only if that's where
  your quota is.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `500 … No module named 'vertexai'` | LiteLLM vertex provider dep missing | `pip install "google-cloud-aiplatform>=1.38"`, restart proxy |
| `tokens_saved: 0` on code | Code-Aware disabled | start proxy with `--code-aware` |
| `tokens_saved: 0` everywhere, `/debug/warmup` shows `kompress: not installed` | ML extra missing | install `headroom-ai[ml]` (torch/transformers/onnxruntime) |
| `"model … not available on your vertex deployment"`, proxy logs nothing | Claude Code is in Vertex mode (probe) | unset `CLAUDE_CODE_USE_VERTEX` / `ANTHROPIC_VERTEX_BASE_URL`; use Anthropic mode above |
| `429 RESOURCE_EXHAUSTED` | no quota in that location | switch `--region`/`VERTEXAI_LOCATION` to where your quota is |
| `404 Publisher Model not found` | model not enabled in project/location | enable it in Vertex Model Garden / request quota |
