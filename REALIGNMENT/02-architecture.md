# 02 — Realigned Target Architecture

The Rust-only proxy after Phase H. Each subsystem documented with its scope, invariants, file layout, and what it explicitly does NOT do.

---

## 2.1 Request lifecycle (Rust, post-Phase-C)

```
                                Client request
                                      │
                                      ▼
            ┌──────────────────────────────────────────────┐
            │ headroom-proxy (axum)                        │
            │                                              │
            │  1. classify_auth_mode(headers)              │  ← Phase F
            │     → "payg" | "oauth" | "subscription"      │
            │                                              │
            │  2. strip x-headroom-* from upstream-bound   │  ← Phase A (PR-A5)
            │                                              │
            │  3. byte-buffer body via RawValue            │  ← Phase A (PR-A4)
            │     (numeric precision preserved)            │
            │                                              │
            │  4. honor cache_control markers              │  ← Phase A (PR-A4)
            │     → frozen_message_count                   │
            │                                              │
            │  5. live_zone_compress(body, frozen_count,   │  ← Phase B
            │                       auth_mode)             │
            │     ├─ identify live-zone blocks             │
            │     ├─ per-block content-type detection      │
            │     ├─ dispatch to type-aware compressor     │
            │     ├─ token-validate; fallback to original  │
            │     ├─ CCR: hash-key, store, marker          │
            │     └─ replace block bytes in-place          │
            │                                              │
            │  6. tool_def_normalize(body)                 │  ← Phase E (PR-E1, E2)
            │     ├─ alpha-sort tools[]                    │
            │     └─ recursive-sort JSON Schema keys       │
            │                                              │
            │  7. cache_control_auto_place(body)           │  ← Phase E (PR-E3)
            │     (Anthropic; up to 4 ephemeral)           │
            │                                              │
            │  8. prompt_cache_key_inject(body)            │  ← Phase E (PR-E4)
            │     (OpenAI; only if not customer-set)       │
            │                                              │
            │  9. forward via reqwest with original bytes  │
            │     for unmodified envelope (RawValue diff)  │
            │                                              │
            │ 10. SSE response: byte-level state machine   │  ← Phase C (PR-C1)
            │     ├─ track blocks/items by id              │
            │     ├─ all delta types handled               │
            │     ├─ mid-stream error/ping/drop surfaced   │
            │     └─ pure passthrough to client            │
            │                                              │
            │ 11. usage telemetry (cache_read,             │  ← Phase G
            │     cache_creation, output_tokens, etc.)     │
            └──────────────────────────────────────────────┘
                                      │
                                      ▼
                                Upstream provider
```

---

## 2.2 The cache-safety invariants (every PR enforces)

### Invariant I1 — Byte-faithful passthrough on unmutated bytes
For every request, the bytes sent to upstream are byte-equal (SHA-256) to the bytes received from the client, **modulo only the byte ranges that a transform explicitly modified**. No re-serialization through a `Value` type. No JSON-prettifier whitespace insertion. No `\uXXXX` ASCII escaping of UTF-8 user content.

**Implementation:** `serde_json::value::RawValue` for `messages[*]` entries; modified messages get fresh serialization, retained messages forward as exact byte copies. Workspace `Cargo.toml` adds `arbitrary_precision` + `raw_value` features.

**Test gate:** `proxy_byte_faithful_anthropic_sha256` — record a real Anthropic `/v1/messages` payload, send it through the proxy with compression off, assert SHA-256 byte-equal at the upstream mock.

### Invariant I2 — Cache hot zone never modified
The following are never mutated by Headroom:
- `system` (string or block list)
- `tools[*]` (other than alpha-sorting and JSON Schema key sorting in Phase E — both deterministic)
- Any message at index < `frozen_message_count`
- Reasoning items with `encrypted_content`
- Thinking blocks with `signature`
- `redacted_thinking.data`
- Compaction items (`{"type": "compaction", "encrypted_content": ...}`)

**Implementation:** `live_zone_compress` walks `messages` from the tail, identifies live-zone blocks (latest user message, latest tool_result, latest function_call_output, latest local_shell_call_output, latest apply_patch_call_output), and ONLY modifies bytes within those blocks.

**Test gate:** `cache_hot_zone_unchanged_under_compression` — fixture with system + tools + 5 historical turns + new tool_result; assert system + tools + first 5 turns bytes equal at upstream.

### Invariant I3 — Append-only
Once a message has appeared in any prior request to upstream, its bytes are frozen. Compression operates on the live zone (latest turn) only.

**Implementation:** `frozen_message_count` is the floor; any compressor that touches index < `frozen_message_count` is rejected at compile time (Rust trait constraint) or runtime (Python assertion).

**Test gate:** `append_only_invariant_under_recompression` — same input bytes through the compressor twice produces byte-equal output; retained messages are byte-equal across the two runs.

### Invariant I4 — Determinism
For the same `(input bytes, frozen_count, auth_mode)`, the compressor produces byte-equal output. No timestamps, no random seeds, no time-dependent decisions.

**Implementation:**
- TOIN is observation-only (Phase B PR-B5); it never alters request-time decisions.
- All hashing is BLAKE3 / SHA-256 with stable input ordering.
- Sort orders are explicit (`BTreeMap` for output, never `HashMap`).
- No `Instant::now()` in any compression code path.

**Test gate:** Property test — for arbitrary valid input, `compress(input) == compress(compress(input).original)` (idempotence on already-compressed); `compress(input) == compress(input)` (run-to-run determinism).

### Invariant I5 — Token-aware, not byte-aware
Every compression is validated post-compression with a tokenizer. If `compressed.tokens >= original.tokens`, the original is forwarded.

**Implementation:** Phase B PR-B4. Per-content-type byte thresholds: code>2KB, JSON>1KB, logs>500B, plain text>5KB. Below threshold = no compression attempted (overhead exceeds savings).

**Test gate:** `proptest_compression_token_count_non_increasing` — for arbitrary valid inputs from a strategy, `tokens(output) ≤ tokens(input)`.

### Invariant I6 — Position-preserving
Compression never reorders blocks within a content array, never splits one block into multiple, never adds inline metadata fields to existing blocks.

**Implementation:** Compressor signature is `fn(block: &mut Block) -> Result<()>` — operates in place. Block type, `tool_use_id` / `call_id`, `is_error`, all sibling fields preserved.

**Side-channel metadata:** A separate marker block (text-type, sibling) carries CCR retrieval directives. Never an extra field on the original block.

### Invariant I7 — Tool definitions normalized, not compressed
Tools are sorted alphabetically by name; JSON Schema keys are sorted recursively; description whitespace is normalized. The bytes of each tool definition's `input_schema.properties[*].description` are otherwise preserved.

**Implementation:** Phase E PR-E1, PR-E2.

### Invariant I8 — `signature`, `encrypted_content`, `redacted_thinking.data` are sacrosanct
These are passthrough only. Never inspected, never decoded, never transformed.

**Implementation:** Compressor block-type dispatch has explicit no-op arms for these types. The Bedrock/Vertex native paths (Phase D) preserve them unlike the LiteLLM converter.

### Invariant I9 — TOIN observes, never mutates request bytes
TOIN's pattern stats grow across requests. Recommendations are published to disk between deploys. The compressor reads recommendations at startup, not per-request.

**Implementation:** Phase B PR-B5. TOIN's in-memory state writes are append-only; reads never block compression.

### Invariant I10 — Auth mode gates compression policy
PAYG: aggressive (full live-zone compression, CCR, tool injection, Phase 3 stabilization). OAuth: passthrough-prefer (live-zone lossless only, no auto-`cache_control`, no auto-`prompt_cache_key`, no `X-Forwarded-*`). Subscription: stealth-prefer (everything OAuth does PLUS preserve `accept-encoding`, never inject `X-Headroom-*` upstream, never mutate `User-Agent`).

**Implementation:** Phase F PR-F1, PR-F2.

---

## 2.3 The compressor module layout (post-Phase-B)

```
crates/headroom-core/src/
├── lib.rs                     # public surface
├── tokenizer/                 # KEEP (HF + tiktoken impls)
│   ├── mod.rs
│   ├── hf_impl.rs
│   ├── tiktoken_impl.rs
│   ├── estimator.rs
│   └── registry.rs
├── ccr.rs                     # KEEP, hardened (persistent backend)
├── signals/                   # KEEP — drives live-zone consumers
│   ├── mod.rs
│   ├── line_importance.rs
│   ├── keyword_detector.rs
│   └── tiered.rs
├── transforms/                # the compressors
│   ├── mod.rs
│   ├── safety.rs              # MOVED from context/safety.rs (Phase B)
│   ├── live_zone.rs           # NEW — live-zone block dispatcher (Phase B)
│   ├── content_detector.rs    # KEEP
│   ├── detection.rs           # KEEP
│   ├── magika_detector.rs     # KEEP
│   ├── unidiff_detector.rs    # KEEP
│   ├── adaptive_sizer.rs      # KEEP
│   ├── anchor_selector.rs     # KEEP
│   ├── tag_protector.rs       # KEEP
│   ├── log_compressor.rs      # KEEP
│   ├── search_compressor.rs   # KEEP
│   ├── diff_compressor.rs     # KEEP
│   ├── kompress_compressor.rs # NEW — Phase H Rust port via `ort` crate
│   ├── smart_crusher/         # KEEP (25 files, correctly scoped)
│   └── pipeline/              # SHRUNK — only the live-zone orchestrator
│       ├── mod.rs
│       ├── orchestrator.rs    # rewrite to live-zone-only
│       ├── traits.rs          # LosslessTransform / LossyTransform
│       └── offloads/          # KEEP — JSON, log, search, diff offloads
└── auth_mode.rs               # NEW — Phase F (classify_auth_mode helper)

# DELETED in Phase B:
# context/                     ← except safety.rs which moved
# scoring/
# relevance/
```

```
crates/headroom-proxy/src/
├── lib.rs
├── main.rs
├── config.rs
├── error.rs
├── proxy.rs                   # Phase A: pure passthrough on /v1/messages
                               # Phase C: + /v1/chat/completions, /v1/responses
├── headers.rs                 # Phase F: conditional X-Forwarded-*
├── websocket.rs               # Phase C: WS Codex flow
├── sse/                       # NEW — Phase C
│   ├── mod.rs
│   ├── parser.rs              # byte-level state machine
│   ├── anthropic.rs           # 4-event dance + delta types
│   ├── openai_chat.rs         # tool_call accumulation
│   └── openai_responses.rs    # output items + reasoning summary
├── compression/
│   ├── mod.rs                 # routing by path × auth_mode
│   ├── live_zone_anthropic.rs # NEW (Phase B)
│   ├── live_zone_openai.rs    # NEW (Phase C)
│   ├── tool_def_normalize.rs  # NEW (Phase E)
│   ├── cache_control.rs       # NEW (Phase E)
│   └── model_limits.rs        # KEEP
├── bedrock/                   # NEW — Phase D
│   ├── mod.rs
│   ├── sigv4.rs
│   ├── invoke.rs
│   └── eventstream.rs
├── vertex/                    # NEW — Phase D
│   ├── mod.rs
│   ├── adc.rs
│   └── stream_raw_predict.rs
└── observability/             # NEW — Phase G
    ├── mod.rs
    ├── prometheus.rs
    ├── cache_hit_rate.rs
    └── compression_ratio.rs

# DELETED:
# compression/icm.rs           ← Phase A PR-A1
# compression/anthropic.rs     ← Phase A PR-A1 (replaced with live_zone_anthropic.rs in Phase B)
```

---

## 2.4 The auth-mode policy matrix (Phase F)

| Policy aspect                       | PAYG          | OAuth                    | Subscription              |
|---|---|---|---|
| Live-zone compression               | aggressive    | lossless-only            | lossless-only             |
| CCR enabled                         | yes           | yes                      | yes (long-session)        |
| Tool def alpha-sort                 | yes           | yes                      | yes                       |
| JSON Schema key sort                | yes           | yes                      | yes                       |
| Auto `cache_control` placement      | yes           | NO (could void scope)    | NO                        |
| Auto `prompt_cache_key` injection   | yes (OpenAI)  | NO                       | NO                        |
| `anthropic-beta` mutation           | NO            | NO                       | NO                        |
| `X-Headroom-*` upstream             | NO            | NO                       | NO                        |
| `X-Forwarded-*` upstream            | yes           | yes                      | NO                        |
| `User-Agent` rewrite                | NO            | NO                       | NO                        |
| `accept-encoding` strip             | OK            | OK                       | NO (preserve)             |
| Lossy compressors (LLMLingua)       | OK            | NO                       | NO                        |
| Memory injection                    | live-zone tail | live-zone tail (gated)  | live-zone tail (gated)    |
| TOIN aggregation key               | (mode, model) | (mode, model)            | (mode, model)             |
| `Authorization` log redaction       | first 12 chars | first 12 chars          | first 12 chars            |

---

## 2.5 Preserved primitives detail

### TOIN (post-Phase-B-PR-B5)

```rust
// Strict observation-only.
pub trait Telemetry {
    fn record_compression(
        &self,
        auth_mode: AuthMode,
        model: ModelFamily,
        structure_hash: StructureHash,
        outcome: CompressionOutcome,
    );
    // No request-time hint API. Period.
}

// Recommendations published between deploys via:
//   $ cargo run -p headroom-toin-publish -- --auth-mode payg --model claude-3-7-sonnet
// Output: recommendations.toml committed to repo, loaded by compressor at startup.
```

### CCR (post-Phase-B-PR-B7)

```rust
pub trait CcrStore: Send + Sync {
    fn put(&self, hash: ContentHash, original: Bytes, ttl: Duration) -> Result<()>;
    fn get(&self, hash: ContentHash) -> Result<Option<Bytes>>;
    fn purge_expired(&self) -> usize;
}

pub struct SqliteCcrStore { ... }   // primary backend
pub struct RedisCcrStore { ... }    // optional, for multi-worker

// `ccr_retrieve` tool registered on every request for sessions that ever did CCR.
// Marker injection format: `<<ccr:HASH>>` appended to compressed block content.
// Markers are deterministic (hash is content-addressed); replay-safe.
```

### Kompress-base (post-Phase-H-PR-H4 Rust port)

```rust
// Plain-text §8.6 compressor. Used only as a last resort, only on live-zone
// user-message text exceeding 5KB.
pub struct KompressCompressor {
    // ONNX runtime via `ort` crate. Model deterministic for fixed weights.
    session: ort::Session,
    threshold_bytes: usize,
}

impl LossyTransform for KompressCompressor { ... }
```

---

## 2.6 What this architecture explicitly does NOT do

- Does NOT drop messages from history. Ever. ICM is gone.
- Does NOT modify `system`, `tools`, or any old turn.
- Does NOT inject Headroom's own tools into customer prompts unless CCR has already fired in this session (and then always, never toggling).
- Does NOT consult TOIN at request time. Recommendations are loaded at startup only.
- Does NOT shell out to RTK from the proxy. RTK lives on the wrap-CLI side (project-decided 2026-05-01).
- Does NOT translate Anthropic ↔ OpenAI shapes. Each provider has its own native handler. Bedrock and Vertex have native envelopes (Phase D).
- Does NOT compress on `/v1/responses/compact` or `/v1/conversations` (different shapes; passthrough only).
- Does NOT rewrite request headers except to strip `x-headroom-*` from upstream-bound headers and add conditional `X-Forwarded-*` (PAYG/OAuth only).
- Does NOT add `User-Agent` headers. The customer's UA passes through verbatim.
- Does NOT compress images, base64 blobs, or audio (out of scope for this realignment).
- Does NOT modify `tool_use.input` JSON key order, `tool_calls.function.arguments` string contents, `phase` field, V4A patches, `local_shell_call.action.command` argv arrays, or any encrypted/redacted/compaction content.
