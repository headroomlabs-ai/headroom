# Phase D — Bedrock & Vertex Native Envelopes

**Goal:** Replace the fake LiteLLM-based Bedrock/Vertex paths (which lossy-convert Anthropic↔OpenAI shapes) with native handlers in the Rust proxy. After Phase D, Anthropic-on-Bedrock and Anthropic-on-Vertex preserve `thinking`, `redacted_thinking`, `document`, `search_result`, `image`, `server_tool_use`, `mcp_tool_use` blocks AND benefit from the live-zone compression engine.

**Calendar:** 2 weeks. SigV4 + EventStream are the bulk of the work.

**Shape:** 4 PRs. D1+D2+D3 are AWS Bedrock; D4 is GCP Vertex.

---

## PR-D1 — Native Bedrock InvokeModel route (non-streaming)

**Branch:** `realign-D1-bedrock-native-invoke`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-D1-bedrock-native-invoke`
**Risk:** **HIGH** (new auth + envelope surface)
**LOC:** +1500

### Scope
Eliminate part of P4-37 and P4-39. Add `POST /model/{model}/invoke` route to the Rust proxy. Recognizes the Bedrock envelope (`anthropic_version` body field, no `model` field, AWS SigV4 auth). Forwards the (possibly compressed) request to the Bedrock endpoint with re-signed SigV4. Live-zone compression runs the same as for direct Anthropic.

### Files

**Add:**
- `crates/headroom-proxy/src/bedrock/mod.rs` — module re-exports.
- `crates/headroom-proxy/src/bedrock/sigv4.rs` — AWS SigV4 signing. Use the `aws-sigv4` crate. Sign over the (possibly modified) request body bytes. Critical: sign **after** Headroom finishes mutating the body, so the signature matches what Bedrock receives.
- `crates/headroom-proxy/src/bedrock/invoke.rs` — POST handler for `/model/{model}/invoke`. Detects `anthropic.claude-*` model IDs; routes to live-zone compression for Anthropic shape; signs and forwards.
- `crates/headroom-proxy/src/bedrock/envelope.rs` — `BedrockEnvelope` struct: parses `{"anthropic_version": "...", ...rest_of_anthropic_body}`. Re-emits in Bedrock shape with `anthropic_version` preserved as the first key.

**Modify:**
- `crates/headroom-proxy/src/lib.rs` — route `/model/{model}/invoke` and `/model/{model}/converse` (POST) to the new handler.
- `crates/headroom-proxy/src/config.rs` — add `--bedrock-region` flag (default `us-east-1`) and AWS credential config (uses `aws-config` crate's default chain).
- `Cargo.toml` workspace — add `aws-sigv4`, `aws-config`, `aws-credential-types`.

**Tests added:**
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::native_envelope_round_trip_byte_equal`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::sigv4_signed_correctly_after_compression`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::thinking_block_preserved_through_bedrock`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::redacted_thinking_preserved`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::document_block_preserved`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::tool_result_array_with_image_preserved`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::stop_sequence_null_only_when_present`
- `crates/headroom-proxy/tests/integration_bedrock_invoke.rs::tool_use_input_byte_equal_preserves_key_order`

### Acceptance criteria

- All new tests pass.
- Manual test against a real Bedrock endpoint (developer's AWS account) succeeds.
- Existing fake Bedrock path (`headroom/backends/litellm.py`) still works in Python; this PR adds the Rust path alongside.

### Blocked by

PR-C1.

### Blocks

PR-D2, PR-D3, PR-H2.

### Rollback

`git revert`. Bedrock requests fall back to Python LiteLLM converter (the fake path). No regression for users who weren't using Rust Bedrock.

### Notes

- The SigV4 signing scope: `host`, `x-amz-date`, `x-amz-content-sha256` headers + canonical request body. Compute the body hash AFTER any compression mutations.
- `accept-encoding` is preserved end-to-end (PAYG/OAuth/subscription all preserve it for Bedrock — there's no legacy CLI to mimic; the Bedrock SDK negotiates compression natively).

---

## PR-D2 — Bedrock streaming via binary EventStream

**Branch:** `realign-D2-bedrock-event-stream`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-D2-bedrock-event-stream`
**Risk:** **HIGH** (binary protocol, not SSE)
**LOC:** +1100

### Scope
Add `POST /model/{model}/invoke-with-response-stream` route. Bedrock's streaming uses **binary EventStream** (vnd.amazon.eventstream content type), not SSE. Build a parser/forwarder for it. Translate to Anthropic SSE for Anthropic-shape responses (so the existing `AnthropicStreamState` from PR-C1 can run telemetry).

### Files

**Add:**
- `crates/headroom-proxy/src/bedrock/eventstream.rs` — EventStream binary parser. Format: 12-byte prelude (length + headers length + CRC32 of prelude), N bytes of headers, payload, 4-byte CRC32 of message. Parse incrementally; yield `EventStreamMessage { headers: HashMap, payload: Bytes }`.
- `crates/headroom-proxy/src/bedrock/eventstream_to_sse.rs` — for Anthropic-shape Bedrock responses, each `EventStreamMessage` whose `:event-type` header is `chunk` carries an Anthropic SSE event in its payload. Re-emit as SSE to the client. (Or pass through as EventStream — choose based on the `Accept` header from the client.)
- `crates/headroom-proxy/src/bedrock/invoke_streaming.rs` — POST handler.

**Modify:**
- `crates/headroom-proxy/src/lib.rs` — route `/model/{model}/invoke-with-response-stream`.
- `crates/headroom-proxy/src/sse/anthropic.rs` — accept events from EventStream-translated source.

**Tests added:**
- `crates/headroom-proxy/tests/integration_bedrock_streaming.rs::eventstream_parses_correctly`
- `crates/headroom-proxy/tests/integration_bedrock_streaming.rs::eventstream_translated_to_sse`
- `crates/headroom-proxy/tests/integration_bedrock_streaming.rs::usage_extracted_from_translated_stream`
- `crates/headroom-proxy/tests/integration_bedrock_streaming.rs::client_can_choose_eventstream_or_sse`
- Property test: `proptest! { fn eventstream_parser_no_panic(bytes in any::<Vec<u8>>()) { let _ = parse(bytes); } }`

### Acceptance criteria

- All tests pass.
- Manual test against real Bedrock streaming endpoint succeeds.

### Blocked by

PR-D1.

### Blocks

PR-H2.

### Rollback

`git revert`. Streaming Bedrock falls back to Python LiteLLM.

---

## PR-D3 — Bedrock-side observability + auth-mode integration

**Branch:** `realign-D3-bedrock-observability`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-D3-bedrock-observability`
**Risk:** **LOW**
**LOC:** +400

### Scope
Per-Bedrock-model metrics, region tagging, IAM role attribution, and integration with auth-mode policy (Bedrock IAM = "oauth" mode by default; passthrough-prefer compression).

### Files

**Modify:**
- `crates/headroom-proxy/src/bedrock/invoke.rs` — auth_mode classification: when an inbound request hits `/model/.../invoke`, classify as `AuthMode::OAuth` for compression policy.
- `crates/headroom-proxy/src/observability/prometheus.rs` — add `bedrock_invoke_count_total{model, region}`, `bedrock_invoke_latency_seconds`, `bedrock_eventstream_message_count_total`.

**Add:**
- `docs/bedrock.md` — operator docs: how to configure AWS credentials, what models are supported (any `anthropic.claude-*`), what compression behavior to expect (live-zone-only, lossless preferred).

**Tests added:**
- `crates/headroom-proxy/tests/integration_bedrock_authmode.rs::bedrock_classified_as_oauth`
- `crates/headroom-proxy/tests/integration_bedrock_authmode.rs::oauth_policy_passthrough_prefer`

### Acceptance criteria

- Tests pass.
- Prometheus scrape includes Bedrock metrics.

### Blocked by

PR-D2, PR-F1 (auth-mode helper).

### Blocks

PR-H2.

### Rollback

`git revert`. Loses Bedrock observability; functional path unchanged.

---

## PR-D4 — Native Vertex publisher path

**Branch:** `realign-D4-vertex-native`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-D4-vertex-native`
**Risk:** **HIGH** (new auth + envelope surface)
**LOC:** +1300

### Scope
Eliminate P4-38, P4-39 (Vertex parts). Add `POST /v1beta1/projects/{project}/locations/{loc}/publishers/anthropic/models/{model}:rawPredict` and `:streamRawPredict` routes. Vertex auth is GCP ADC (Application Default Credentials) → bearer token. Envelope: `anthropic_version` body field, no `model` field, GCP auth header.

### Files

**Add:**
- `crates/headroom-proxy/src/vertex/mod.rs` — module re-exports.
- `crates/headroom-proxy/src/vertex/adc.rs` — GCP ADC bearer token resolution. Use `gcp_auth` crate.
- `crates/headroom-proxy/src/vertex/raw_predict.rs` — POST handler.
- `crates/headroom-proxy/src/vertex/stream_raw_predict.rs` — streaming handler. Vertex uses SSE for streaming (unlike Bedrock); the existing `AnthropicStreamState` from PR-C1 works directly.

**Modify:**
- `crates/headroom-proxy/src/lib.rs` — route Vertex paths.
- `Cargo.toml` workspace — add `gcp_auth`.

**Tests added:**
- `crates/headroom-proxy/tests/integration_vertex_raw_predict.rs::native_envelope_round_trip_byte_equal`
- `crates/headroom-proxy/tests/integration_vertex_raw_predict.rs::adc_bearer_token_signed_correctly`
- `crates/headroom-proxy/tests/integration_vertex_raw_predict.rs::thinking_block_preserved`
- `crates/headroom-proxy/tests/integration_vertex_raw_predict.rs::stream_raw_predict_sse_handled`

### Acceptance criteria

- All tests pass.
- Manual test against a real Vertex endpoint succeeds.

### Blocked by

PR-C1, PR-D1 (envelope pattern).

### Blocks

PR-H2.

### Rollback

`git revert`. Vertex requests fall back to LiteLLM Python. No regression for non-Rust-Vertex users.

---

## Phase D acceptance summary

After all 4 PRs land:

- ✅ Native Bedrock `/model/{model}/invoke` route in Rust
- ✅ Native Bedrock `/model/{model}/invoke-with-response-stream` (binary EventStream parsed and translated)
- ✅ SigV4 signing post-compression
- ✅ All Anthropic block types preserved through Bedrock (thinking, redacted_thinking, document, search_result, image, server_tool_use, mcp_tool_use)
- ✅ `stop_sequence: null` no longer hardcoded
- ✅ `tool_calls.function.arguments` preserved as string
- ✅ Native Vertex `:rawPredict` and `:streamRawPredict` routes
- ✅ ADC bearer token resolution
- ✅ Bedrock/Vertex classified as `AuthMode::OAuth` (passthrough-prefer compression)
- ✅ Per-Bedrock-model and per-Vertex-model Prometheus metrics

**Phase D retires P4-37, P4-38, P4-39, P4-43.** Marketplace BYOC pitch (per project memory) becomes real.

After Phase D, the LiteLLM Python converter is no longer on the request path for Bedrock/Vertex — Phase H deletes it.
