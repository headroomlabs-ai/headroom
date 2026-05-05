# AWS Bedrock — Operator Guide

Headroom's Rust proxy ships a native AWS Bedrock InvokeModel surface. After Phase D (PRs D1–D3), Anthropic-on-Bedrock requests are signed, compressed, and observed by the proxy directly — no LiteLLM Python shim on the request path.

This document covers how to deploy the Bedrock-native surface, how compression policy is applied, and how to read the Prometheus metrics the proxy exports.

## What's in scope

| Capability | Status |
|---|---|
| `POST /model/{model}/invoke` | PR-D1 — native Rust handler |
| `POST /model/{model}/converse` | PR-D1 — same handler (Bedrock accepts both paths for the Anthropic envelope) |
| `POST /model/{model}/invoke-with-response-stream` | PR-D2 — binary EventStream parsed and translated to SSE |
| AWS SigV4 signing (post-compression) | PR-D1 |
| `AuthMode::OAuth` classification | PR-D3 — Bedrock IAM is OAuth-equivalent under the policy matrix |
| Per-model + per-region Prometheus metrics | PR-D3 — exposed at `GET /metrics` |
| OAuth compression policy gates (no auto cache_control, lossless-only) | Phase F PR-F2/F3 (gates the marker D3 wires) |

## AWS credential configuration

The proxy uses the [aws-config default credential chain](https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html), resolved once at startup.

The chain searches in this order, stopping at the first source that yields valid credentials:

1. **Environment variables** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`. Useful for ECS task roles that inject creds via env or for `aws sts assume-role` shells.
2. **Shared credentials file** — `~/.aws/credentials`. Profile selected by `--aws-profile` (or `AWS_PROFILE`). Falls back to `[default]`.
3. **IAM instance profile / IMDS** — when running on EC2.
4. **ECS task role / EKS pod identity** — when running on the AWS-managed compute platforms.
5. **AWS SSO** — `~/.aws/sso/cache/...` when `aws sso login` has been run.

If the chain does NOT resolve any credentials at startup, the proxy logs `event=bedrock_credentials_unavailable` at WARN and continues to start. Bedrock invoke routes will then return `500` with `event=bedrock_credentials_missing` per request — there is **no silent fallback to unsigned requests**, by design.

### Required IAM permissions

The proxy needs:

- `bedrock:InvokeModel` for non-streaming
- `bedrock:InvokeModelWithResponseStream` for streaming

Scope these to the specific model ARNs you intend to use. Example IAM policy snippet:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
      ]
    }
  ]
}
```

## Region configuration

```sh
headroom-proxy \
  --upstream http://unused-when-bedrock-only \
  --bedrock-region us-east-1
```

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--bedrock-region` | `HEADROOM_PROXY_BEDROCK_REGION` (or `AWS_REGION`) | `us-east-1` | Drives both the SigV4 region and the derived endpoint hostname. |
| `--bedrock-endpoint` | `HEADROOM_PROXY_BEDROCK_ENDPOINT` | derived from region | Override for FIPS endpoints (`bedrock-runtime-fips.{region}.amazonaws.com`), VPC endpoints, or local mock servers. |
| `--aws-profile` | `HEADROOM_PROXY_AWS_PROFILE` | unset | Selects the named profile from the shared credentials file. |
| `--enable-bedrock-native` | `HEADROOM_PROXY_ENABLE_BEDROCK_NATIVE` | `true` | Set to `false` to mount no Bedrock routes at all (Bedrock requests will then fall through to the catch-all and fail without SigV4). |

## Supported model IDs

The proxy classifies model IDs by **literal vendor prefix** — no regexes. Any model ID starting with `anthropic.` is treated as Anthropic-shape: the live-zone compression dispatcher runs over the body, the envelope is re-emitted with `anthropic_version` preserved as the first key, and the request is signed with SigV4.

Examples that hit the Anthropic compression path:

- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-5-sonnet-20241022-v2:0`
- `anthropic.claude-3-opus-20240229-v1:0`

Other Bedrock vendors (`amazon.titan-...`, `meta.llama3-...`, `cohere.command-...`, `ai21.j2-...`, `stability.stable-diffusion-...`) are signed and forwarded **without compression** — the proxy does not yet understand their body shapes and would risk corrupting them. These model IDs log `event=bedrock_compression_skipped, reason=non_anthropic_vendor` per request. Full Anthropic envelopes only.

The contract: **any new model ID that AWS adds under the `anthropic.` prefix automatically picks up the full compression + signing pipeline.** No code change in the proxy is needed for new versions of Claude on Bedrock.

## Compression behaviour

Bedrock requests are subject to the **same** live-zone compression rules as direct Anthropic (`/v1/messages`):

- Only the live-zone messages (latest user turn, latest tool/output blocks) are eligible for compression.
- The cache hot zone (older messages, system prompt, tools list) is byte-faithful passthrough.
- The dispatcher only mutates body bytes when at least one block compressed. The byte-equality invariant for unchanged blocks is enforced at `debug_assert!` granularity.

### OAuth policy (PR-D3 → PR-F2/F3)

The Bedrock auth-mode middleware classifies every Bedrock request as `AuthMode::OAuth`. Even when the inbound request has no Authorization header (the common case where the AWS SDK signs after our proxy), the middleware **coerces** to OAuth and emits `event=bedrock_auth_mode_unexpected` at WARN if F1's classifier disagreed — so the divergence is loud, not silent.

Under the OAuth policy matrix (see `docs/auth-modes.md`):

- **No auto-`cache_control` injection.** OAuth subscriptions pin the cache scope to `(account, model, session)`; auto-injecting markers can void cache hits.
- **No auto-`prompt_cache_key`.** Same reasoning.
- **Lossless-only compressors.** Lossy compressors (text rewriting, summarisation) are gated off for OAuth.

PR-D3 lands the classification + the resulting `AuthMode` in `request.extensions()`. PR-F2 and PR-F3 wire the actual policy gates that read it. Until those PRs land, the Bedrock route uses the existing dispatcher (which is a no-op in `compression_mode=off`); the OAuth contract above is the documented forward direction.

### Cache safety

The bytes signed by SigV4 are exactly the bytes Bedrock receives — the signer hashes the post-compression body. There is no "sign before compress" shortcut that would produce a signature mismatched to the wire payload. Compression mutates the body once, then the signer runs once, then the bytes are forwarded once.

## Prometheus metrics

The proxy exposes a `GET /metrics` endpoint that serves the standard Prometheus text-format scrape. Three Bedrock-specific metric families are exported:

| Metric | Type | Labels | Source |
|---|---|---|---|
| `bedrock_invoke_count_total` | Counter | `model`, `region`, `auth_mode` | One increment per `/model/.../invoke` (and `/converse` and `/invoke-with-response-stream`) request. |
| `bedrock_invoke_latency_seconds` | Histogram | `model`, `region` | Observed at request completion (success or failure). |
| `bedrock_eventstream_message_count_total` | Counter | `model`, `region`, `event_type` | One increment per parsed binary EventStream message in the streaming path. `event_type` is the `:event-type` header (`chunk`, `metadata`, `internalServerException`, etc.). |

All labels are bounded by infrastructure config (`region` from `--bedrock-region`, `auth_mode` from the 3-variant enum) or by the path parameter (`model`, supplied by the axum extractor — never by user-controlled body bytes). Cardinality is bounded by deployment fan-out, not by traffic volume.

### Sample PromQL queries

**p99 latency by model:**
```promql
histogram_quantile(
  0.99,
  sum by (model, le) (rate(bedrock_invoke_latency_seconds_bucket[5m]))
)
```

**Request rate by region (RPS):**
```promql
sum by (region) (rate(bedrock_invoke_count_total[1m]))
```

**EventStream message rate by event type (debugging the streaming path):**
```promql
sum by (event_type) (rate(bedrock_eventstream_message_count_total[1m]))
```

**Error breakdown by HTTP status (cross-references the structured logs `event=bedrock_upstream_error`):**
```promql
sum by (model) (rate(bedrock_invoke_count_total{auth_mode="oauth"}[5m]))
  / sum by (model) (rate(bedrock_invoke_latency_seconds_count[5m]))
```

(The denominator is the total observed latency samples — useful for sanity-checking that every counted invoke also got a histogram observation. They should be equal.)

### Structured-log correlation

Every metric increment in the Bedrock path is paired with a `tracing::debug!` log line carrying:

- `event = "metric_recorded"`
- `metric = "bedrock_invoke_count_total" | "bedrock_invoke_latency_seconds" | "bedrock_eventstream_message_count_total"`
- the same labels as the metric

Enable with `RUST_LOG=headroom_proxy::observability=debug` for incident correlation. In normal operation keep this at the default `info` level — debug volume per request is bounded by the same cardinality the metric uses.

## Live cloud validation

The PR-D1, D2, D3 implementations are exercised end-to-end against a wiremock upstream (`crates/headroom-proxy/tests/integration_bedrock_*.rs`). The wiremock-based tests are the canonical correctness gate.

A real Bedrock smoke test (`aws bedrock-runtime invoke-model ...` through the proxy) requires `bedrock:InvokeModel` permissions in the developer's AWS account. Set the proxy upstream to the proxy URL (`http://localhost:8787`) via the AWS SDK's `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` env var:

```sh
AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://localhost:8787 \
  aws bedrock-runtime invoke-model \
    --model-id anthropic.claude-3-haiku-20240307-v1:0 \
    --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}' \
    /tmp/out.json
```

If the SDK signs the request before sending to the proxy, the proxy will see a SigV4 `Authorization` header and classify as OAuth via the standard rule. If the SDK is configured to sign downstream of the proxy (some IAM-instance-profile setups), the proxy still classifies as OAuth via the middleware's coerce-and-log fallback.

## Rollback

Set `--enable-bedrock-native=false` to unmount all Bedrock routes; the catch-all proxy then forwards Bedrock requests unchanged to `--upstream`. This is an emergency rollback only — without SigV4 re-signing, the catch-all path will fail closed unless the upstream is itself a Bedrock-aware proxy (e.g., the Python LiteLLM converter on a different port).
