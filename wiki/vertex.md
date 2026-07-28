# Vertex AI (Gemini Enterprise Agent Platform)

*(Note: Vertex AI is now branded as Gemini Enterprise Agent Platform, though underlying APIs remain unchanged.)*

Headroom supports Google Cloud Vertex AI publisher endpoints through the proxy
passthrough surface. Configure the proxy with a regional Vertex base URL, then
send normal Vertex REST requests through Headroom.

Google documents Gemini generation on Vertex with `generateContent` and
`streamGenerateContent`, and the request body uses the Vertex/Gemini `contents`
shape. See Google Cloud's model inference reference:
https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

Google Cloud REST calls authenticate with a bearer access token. Use
**`gcloud auth application-default print-access-token`**: a plain
`gcloud auth print-access-token` user token is rejected by
`aiplatform.googleapis.com` with `401 ACCESS_TOKEN_TYPE_UNSUPPORTED` for many
identities. Application Default Credentials search
`GOOGLE_APPLICATION_CREDENTIALS`, local ADC files, and attached service accounts
in that order. Tokens expire after ~1 hour. See:

- https://docs.cloud.google.com/docs/authentication/rest
- https://docs.cloud.google.com/docs/authentication/application-default-credentials

## Configure

Set the Vertex regional host explicitly:

```bash
headroom proxy --vertex-api-url https://us-central1-aiplatform.googleapis.com
```

The same setting is available through `VERTEX_TARGET_API_URL`. Left unset, the
proxy derives the host from each request's `locations/{location}` segment, so
one proxy serves every region plus `global`.

### Picking a location

Vertex serves each publisher model in only a subset of locations, and this is
the most common source of a confusing `404` during onboarding:

| Model | Where it serves |
| --- | --- |
| `gemini-flash-latest`, `gemini-flash-lite-latest` | `global` only |
| `gemini-3.5-flash` | `global`, `europe-west2`, `asia-northeast1`, `asia-south1`, `asia-southeast1` |
| `claude-sonnet-4-6` (and 4.6-or-older Claude) | `global`, `us-east5`, `europe-west1`, `asia-southeast1` |
| Claude 4.7+ | `global` or the `us`/`eu` multi-region only -- no named regions |

**Gemini 3.x has no US regional endpoint.** Start with `global` and only pin a
region when you need Provisioned Throughput or data residency. Partner models
(Claude, Llama, Mistral) also need a one-time per-project enable in Model
Garden before they serve.

When Vertex rejects a request, Headroom appends a `[headroom] hint: ...` note to
`error.message` (also emitted as an `x-headroom-hint` header and a proxy WARNING)
naming the likely fix.

## Gemini On Vertex

Send Vertex publisher paths through the proxy unchanged:

```bash
LOCATION="global"
MODEL="gemini-flash-latest"
ACCESS_TOKEN="$(gcloud auth application-default print-access-token)"

curl -sS \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8787/v1/projects/PROJECT_ID/locations/${LOCATION}/publishers/google/models/${MODEL}:generateContent \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Summarize this repository in one paragraph."}]
      }
    ]
  }'
```

## Google Gen AI SDK (Proxy)

You can use the official `google-genai` Python SDK pointed directly at Headroom, allowing you to use native extensions, tools, thinking levels, and multi-media components:

```python
import os
from google import genai
from google.genai import types

LOCATION = os.environ.get("LOCATION", "global")
MODEL = os.environ.get("MODEL", "gemini-flash-latest")

client = genai.Client(
    vertexai=True,
    project=os.environ.get("GCP_PROJECT_ID", "your-project"),
    location=LOCATION,
    http_options={"base_url": "http://127.0.0.1:8787"},
)

response = client.models.generate_content(
    model=MODEL,
    contents="Think deeply. Which is heavier: a kg of feathers or a kg of steel?",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=128),
    ),
)
print(response.text)
```

A runnable version of exactly this flow -- it starts the proxy, probes it, and
tears it down -- lives at `examples/vertex_genai_sdk_demo.py`.

Note the `google-genai` SDK only builds `publishers/google/...` paths, so it
cannot reach Claude on Vertex; use the `publishers/anthropic/...:rawPredict`
route below for that.

Supported passthrough actions:

- `generateContent`
- `streamGenerateContent`
- `countTokens`

## Anthropic Publisher On Vertex

Headroom also forwards Anthropic publisher calls on Vertex:

- `rawPredict`
- `streamRawPredict`

The Python proxy preserves caller-supplied Google bearer auth. The native Rust
proxy path additionally resolves GCP ADC and injects the bearer token for the
Anthropic publisher route.

These native routes are a straight passthrough: they need **no** `--backend`
flag and no extra beyond `[proxy]`.

`--backend vertex` (aliases: `vertex_ai`, `litellm-vertex`, `google-vertex`) is a
different mode. It routes Anthropic *Messages* traffic — `/v1/messages` — through
LiteLLM, and needs the Vertex SDK:

```bash
pip install "headroom-ai[proxy,vertex]"
```

That extra is kept out of `[proxy]` because `google-cloud-aiplatform` and its
transitive tree add roughly 175 MB, and only this backend uses it. Select the
backend without it and the proxy refuses to start rather than failing on every
request:

```text
Cannot start proxy: Vertex backend selected but the Vertex SDK is missing. ...
```

Do not combine the two. With `--backend vertex` set, native publisher requests
are re-routed through LiteLLM, which takes the project, region and model from
its own configuration (`VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`) and ignores the
ones in your URL — so a request that works without the flag can come back `404`
for a model you never named.

## Claude Code with Headroom compression (validated)

To run **Claude Code** against Claude-on-Vertex **with Headroom compressing the
context**, use the dedicated, tested runbook:

➡️ **[Claude Code + Vertex + Headroom](https://docs.headroomlabs.ai/docs/claude-code-vertex)**

Short version: run Claude Code in **normal Anthropic mode** (`ANTHROPIC_BASE_URL`
→ the proxy) and start the proxy with `--backend litellm-vertex_ai --region <loc>
--code-aware`; Headroom holds the GCP ADC creds and calls Vertex.

> ⚠️ Do **not** put Claude Code into Vertex mode and point `ANTHROPIC_VERTEX_BASE_URL`
> at the proxy. Claude Code's client-side model probe rejects any non-Google Vertex
> URL before sending a request ("model … not available on your vertex deployment"),
> so the proxy is never reached. Use the Anthropic-mode runbook above instead.
>
> ⚠️ Two easy-to-miss requirements: `pip install "headroom-ai[proxy,vertex]"`
> (the LiteLLM `vertex_ai` provider needs the Vertex SDK) and the `--code-aware`
> flag (code compression is off by default). Without the first the proxy refuses
> to start; without the second you get `tokens_saved: 0`.
