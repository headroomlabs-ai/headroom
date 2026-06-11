# Vertex AI

Headroom supports Google Cloud Vertex AI publisher endpoints through the proxy
passthrough surface. Configure the proxy with a regional Vertex base URL, then
send normal Vertex REST requests through Headroom.

Google documents Gemini generation on Vertex with `generateContent` and
`streamGenerateContent`, and the request body uses the Vertex/Gemini `contents`
shape. See Google Cloud's model inference reference:
https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

Google Cloud REST calls authenticate with a bearer access token. For local
development, Google documents both `gcloud auth print-access-token` and
`gcloud auth application-default print-access-token`; Application Default
Credentials search `GOOGLE_APPLICATION_CREDENTIALS`, local ADC files, and
attached service accounts in that order. See:

- https://docs.cloud.google.com/docs/authentication/rest
- https://docs.cloud.google.com/docs/authentication/application-default-credentials

## Configure

Set the Vertex regional host explicitly:

```bash
headroom proxy --vertex-api-url https://us-central1-aiplatform.googleapis.com
```

The same setting is available through `VERTEX_TARGET_API_URL`.

## Gemini On Vertex

Send Vertex publisher paths through the proxy unchanged:

```bash
ACCESS_TOKEN="$(gcloud auth print-access-token)"

curl -sS \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8787/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Summarize this repository in one paragraph."}]
      }
    ]
  }'
```

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
