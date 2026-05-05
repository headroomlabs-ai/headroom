# Phase C — Rust Proxy Paths

**Goal:** Port the remaining proxy surfaces to Rust. After Phase C, the Rust proxy handles `/v1/messages`, `/v1/chat/completions`, `/v1/responses` (HTTP + streaming), with a byte-level SSE state machine that handles every wire-format quirk the guide enumerates.

**Calendar:** 3 weeks. Mostly sequential (each PR builds on the SSE parser).

**Shape:** 5 PRs.

---

## PR-C1 — Byte-level SSE parser with full state machine

**Branch:** `realign-C1-rust-sse-parser`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-C1-rust-sse-parser`
**Risk:** **HIGH** (foundational; many wire-format quirks; UTF-8 split-byte handling)
**LOC:** +1500

### Scope
Eliminate P1-8, P1-9, P1-14, P1-15, P1-17, P4-48. Build the byte-level SSE parser in `crates/headroom-proxy/src/sse/`. Three parsers (one per provider × API), all sharing a common framing layer. Per-stream state (no module-level state). Models the streaming state machines from guide §5 exactly.

### Files

**Add:**
- `crates/headroom-proxy/src/sse/mod.rs` — module re-exports.
- `crates/headroom-proxy/src/sse/framing.rs` — byte-level framing. Reads `bytes::Bytes` chunks, accumulates into a `BytesMut` buffer, finds `\n\n` event terminators in bytes (not strings), yields complete events as `(event_name: Option<String>, data: Bytes)`. Decodes UTF-8 per complete event, never per chunk. Handles `: ping` keepalives (skip silently). Handles `[DONE]` literal.
- `crates/headroom-proxy/src/sse/anthropic.rs` — Anthropic stream state machine per guide §5.1:
  ```rust
  pub struct AnthropicStreamState {
      pub message_id: Option<String>,
      pub model: Option<String>,
      pub blocks: HashMap<usize, BlockState>,  // keyed by index
      pub current_block_index: Option<usize>,
      pub stop_reason: Option<String>,
      pub usage: UsageBuilder,
      pub status: StreamStatus,
  }

  pub struct BlockState {
      pub block_type: String,
      pub text_buffer: String,
      pub partial_json: String,
      pub signature: Option<String>,
      pub citations: Vec<Citation>,
      pub metadata: serde_json::Value,
      pub complete: bool,
  }

  impl AnthropicStreamState {
      pub fn apply(&mut self, event: SseEvent) -> Result<()>;
  }
  ```
  Handlers for `message_start`, `content_block_start`, `content_block_delta` (switching on `delta.type`: `text_delta` / `thinking_delta` / `input_json_delta` / `citations_delta` / `signature_delta`), `content_block_stop`, `message_delta`, `message_stop`, `error`, `ping`.
- `crates/headroom-proxy/src/sse/openai_chat.rs` — OpenAI Chat Completions state machine per guide §5.2. `ChunkState`, `ChoiceState`, `ToolCallState`. Handles `[DONE]` and `stream_options.include_usage` final chunk.
- `crates/headroom-proxy/src/sse/openai_responses.rs` — OpenAI Responses state machine per guide §5.3. `ResponseState`, `ItemState` keyed by `id` (not position) for out-of-order completion. Handlers for `response.created`, `output_item.added/done`, `content_part.added/done`, `output_text.delta/done`, `function_call_arguments.delta/done`, `reasoning_summary.delta/done`, `response.completed/failed/incomplete`.

**Modify:**
- `crates/headroom-proxy/src/proxy.rs` — when forwarding a streaming response, the state machine runs in parallel with the byte-passthrough (so client gets raw bytes immediately; state machine populates telemetry without blocking the stream).

**Tests added:**
- `crates/headroom-proxy/tests/sse_framing.rs::utf8_split_emoji_across_chunks_preserved`
- `crates/headroom-proxy/tests/sse_framing.rs::single_newline_does_not_emit_event`
- `crates/headroom-proxy/tests/sse_framing.rs::double_newline_emits_event`
- `crates/headroom-proxy/tests/sse_framing.rs::ping_keepalive_skipped`
- `crates/headroom-proxy/tests/sse_framing.rs::done_sentinel_detected`
- `crates/headroom-proxy/tests/sse_framing.rs::trailing_data_after_done_tolerated`
- `crates/headroom-proxy/tests/sse_anthropic.rs::four_event_dance_text_block`
- `crates/headroom-proxy/tests/sse_anthropic.rs::thinking_delta_accumulated`
- `crates/headroom-proxy/tests/sse_anthropic.rs::signature_delta_preserved_byte_equal`
- `crates/headroom-proxy/tests/sse_anthropic.rs::input_json_delta_concatenated_parsed_at_stop`
- `crates/headroom-proxy/tests/sse_anthropic.rs::citations_delta_accumulated`
- `crates/headroom-proxy/tests/sse_anthropic.rs::message_delta_finalizes_stop_reason_and_output_tokens`
- `crates/headroom-proxy/tests/sse_anthropic.rs::mid_stream_error_event_handled`
- `crates/headroom-proxy/tests/sse_anthropic.rs::interleaved_blocks_by_index`
- `crates/headroom-proxy/tests/sse_openai_chat.rs::tool_call_id_and_name_only_first_chunk`
- `crates/headroom-proxy/tests/sse_openai_chat.rs::tool_call_arguments_concatenated`
- `crates/headroom-proxy/tests/sse_openai_chat.rs::usage_in_final_chunk_when_include_usage_set`
- `crates/headroom-proxy/tests/sse_openai_chat.rs::refusal_field_handled`
- `crates/headroom-proxy/tests/sse_openai_responses.rs::out_of_order_item_completion_by_id`
- `crates/headroom-proxy/tests/sse_openai_responses.rs::reasoning_summary_accumulated`
- `crates/headroom-proxy/tests/sse_openai_responses.rs::function_call_arguments_string_preserved`
- Property test: `proptest! { fn sse_parser_no_panic_on_arbitrary_bytes(bytes in any::<Vec<u8>>()) { let _ = parse(bytes); } }`

### Acceptance criteria

- All new tests pass.
- Property test: 100K random byte sequences never panic the parser.
- Real-traffic shadow test: feed a recorded production Anthropic stream through both the Rust parser and the Python parser; assert telemetry agrees on `usage` totals.

### Blocked by

PR-A1.

### Blocks

PR-C2, PR-C3, PR-C4.

### Rollback

`git revert`. SSE parsing returns to byte-passthrough (Phase A state). Telemetry less rich but no functional regression.

---

## PR-C2 — `/v1/chat/completions` handler in Rust

**Branch:** `realign-C2-rust-chat-completions`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-C2-rust-chat-completions`
**Risk:** **HIGH** (new endpoint surface)
**LOC:** +1200

### Scope
Add `/v1/chat/completions` to `crates/headroom-proxy`. Handles request-body shape, live-zone compression dispatch (assistant `tool_calls` and `tool` role messages — equivalents of Anthropic's tool_result), and the streaming state machine from PR-C1. Adds Phase E PR-E1/E2 tool-def normalization gate (no-op until Phase E).

### Files

**Add:**
- `crates/headroom-proxy/src/handlers/chat_completions.rs` — POST handler.
  ```rust
  async fn handle_chat_completions(
      State(state): State<AppState>,
      headers: HeaderMap,
      body: Bytes,
  ) -> Result<Response>;
  ```
- `crates/headroom-proxy/src/compression/live_zone_openai.rs` — OpenAI live-zone dispatcher. Live zone for Chat Completions: latest `tool` role message's `content`; latest `user` message's text content. Compress per type-aware dispatch (same compressors as Anthropic; reused).

**Modify:**
- `crates/headroom-proxy/src/lib.rs` — route `/v1/chat/completions` (POST) to the new handler.
- `crates/headroom-proxy/src/compression/mod.rs` — add OpenAI Chat dispatch path.

**Tests added:**
- `crates/headroom-proxy/tests/integration_chat_completions.rs::passthrough_no_compression_byte_equal`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::tool_message_compressed`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::n_greater_than_one_passthrough`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::stream_options_include_usage_preserved`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::tool_choice_change_passthrough_no_mutation`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::refusal_field_in_response_handled`
- `crates/headroom-proxy/tests/integration_chat_completions.rs::streaming_tool_call_argument_accumulation`

### Acceptance criteria

- All new tests pass.
- A real Chat Completions request through the Rust proxy produces byte-equal upstream bytes when compression is off.
- Streaming tool_call accumulation works for the `delta.tool_calls[].function.arguments` pattern.

### Blocked by

PR-C1, PR-B3, PR-B4.

### Blocks

PR-C3, PR-H1.

### Rollback

`git revert`. `/v1/chat/completions` still flows through the Python proxy (Phase H hasn't deleted it yet).

---

## PR-C3 — `/v1/responses` handler in Rust (HTTP)

**Branch:** `realign-C3-rust-responses-http`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-C3-rust-responses-http`
**Risk:** **HIGH**
**LOC:** +1500

### Scope
Add `/v1/responses` HTTP handler. Item-shape passthrough preservation for every Responses item type (V4A patches, `local_shell_call.action.command` argv, Codex `phase`, `compaction`, MCP items, computer_use, `image_generation_call`, server-side tool results). Live-zone compression for `function_call_output`, `local_shell_call_output`, `apply_patch_call_output` (only when >2KB).

### Files

**Add:**
- `crates/headroom-proxy/src/handlers/responses.rs` — POST handler.
- `crates/headroom-proxy/src/compression/live_zone_responses.rs` — Responses live-zone dispatcher. Live zone: latest `function_call_output.output`, latest `local_shell_call_output.output`, latest `apply_patch_call_output.output`, latest `user` message text content.
- `crates/headroom-proxy/src/responses_items.rs` — explicit per-item-type enum and passthrough rules:
  ```rust
  pub enum ResponseItem {
      Message { phase: Option<String>, .. },
      Reasoning { encrypted_content: Option<String>, .. },  // passthrough only
      FunctionCall { call_id: String, arguments: String, .. },  // arguments stays as string
      LocalShellCall { command: Vec<String>, .. },  // argv array preserved
      ApplyPatchCall { operation: ApplyPatchOperation, .. },  // V4A diff verbatim
      Compaction { encrypted_content: String, .. },  // passthrough only
      McpCall { .. } | McpListTools { .. } | McpApprovalRequest { .. },  // passthrough
      ComputerCall { .. } | ComputerCallOutput { .. },
      WebSearchCall { .. } | FileSearchCall { .. } | CodeInterpreterCall { .. },
      ImageGenerationCall { .. },
      ToolSearchCall { .. },
      CustomToolCall { .. },
      Unknown { type_: String, raw: Box<RawValue> },  // log warning; preserve verbatim
  }
  ```

**Modify:**
- `crates/headroom-proxy/src/lib.rs` — route `/v1/responses` (POST) to the new handler.

**Tests added:**
- `crates/headroom-proxy/tests/integration_responses.rs::v4a_patch_byte_equal_through_proxy`
- `crates/headroom-proxy/tests/integration_responses.rs::local_shell_call_command_argv_array_preserved`
- `crates/headroom-proxy/tests/integration_responses.rs::codex_phase_commentary_preserved`
- `crates/headroom-proxy/tests/integration_responses.rs::codex_phase_final_answer_preserved`
- `crates/headroom-proxy/tests/integration_responses.rs::compaction_item_byte_equal`
- `crates/headroom-proxy/tests/integration_responses.rs::reasoning_encrypted_content_byte_equal`
- `crates/headroom-proxy/tests/integration_responses.rs::function_call_arguments_string_preserved`
- `crates/headroom-proxy/tests/integration_responses.rs::call_id_referenced_not_id`
- `crates/headroom-proxy/tests/integration_responses.rs::apply_patch_output_below_2kb_no_compression`
- `crates/headroom-proxy/tests/integration_responses.rs::apply_patch_output_above_2kb_compressed`
- `crates/headroom-proxy/tests/integration_responses.rs::local_shell_output_compressed`
- `crates/headroom-proxy/tests/integration_responses.rs::mcp_tool_call_byte_equal`
- `crates/headroom-proxy/tests/integration_responses.rs::computer_call_byte_equal`
- `crates/headroom-proxy/tests/integration_responses.rs::image_generation_call_no_log_redaction_in_test_mode`
- `crates/headroom-proxy/tests/integration_responses.rs::unknown_item_type_logged_warning_byte_equal`

### Acceptance criteria

- All new tests pass.
- A representative Responses request with reasoning + function_call + local_shell + apply_patch + custom items round-trips byte-equal modulo compressed live-zone outputs.

### Blocked by

PR-C1, PR-C2.

### Blocks

PR-C4.

### Rollback

`git revert`. `/v1/responses` still flows through Python.

---

## PR-C4 — `/v1/responses` streaming + Conversations API awareness

**Branch:** `realign-C4-rust-responses-streaming`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-C4-rust-responses-streaming`
**Risk:** **MEDIUM-HIGH**
**LOC:** +800

### Scope
Streaming for `/v1/responses` using the SSE state machine from PR-C1. Plus first-class awareness of the Conversations API (P4-40) — when `conversation: {"id": "conv_..."}` is in the body, the local view is incomplete; tokenizer must adjust or skip compression decisions.

### Files

**Modify:**
- `crates/headroom-proxy/src/handlers/responses.rs` — when `Accept: text/event-stream`, route to streaming handler. The streaming handler runs the `OpenAIResponsesStreamState` machine in parallel with byte-passthrough.
- `crates/headroom-proxy/src/sse/openai_responses.rs` — add usage extraction from `response.completed`.

**Add:**
- `crates/headroom-proxy/src/conversations.rs` — detect `conversation: {"id": "conv_..."}` in request body. When present, log a warning and disable live-zone compression for that request (until Phase 4 cross-request shared cache lands). Telemetry: `proxy_conversations_api_request_count_total`.

**Tests added:**
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::reasoning_summary_streamed_correctly`
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::function_call_arguments_streamed_byte_equal`
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::out_of_order_items_handled_by_id`
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::response_completed_usage_captured`
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::response_failed_handled`
- `crates/headroom-proxy/tests/integration_responses_streaming.rs::response_incomplete_with_max_output_tokens_reason`
- `crates/headroom-proxy/tests/integration_conversations.rs::conversation_id_present_skips_compression_warns`

### Acceptance criteria

- All new tests pass.
- The Conversations API warning appears in logs at `INFO` level with a `conversation_id` field.

### Blocked by

PR-C3.

### Blocks

PR-H1.

### Rollback

`git revert`. Streaming Responses routes through Python.

---

## PR-C5 — `responses_converter.py` retirement (Rust handles it natively)

**Branch:** `realign-C5-retire-responses-converter`
**Worktree:** `~/claude-projects/headroom-worktrees/realign-C5-retire-responses-converter`
**Risk:** **LOW** (cleanup; Rust handler from PR-C3/C4 covers this surface)
**LOC:** -267 / +20

### Scope
Delete `headroom/proxy/responses_converter.py` (the Anthropic↔OpenAI Responses↔Chat Completions converter that mishandled `phase`, multi-text-part rebuild, etc.). The Rust handler from PR-C3 handles `/v1/responses` natively without converting between shapes. After this PR lands, no Python code is on the `/v1/responses` request path.

### Files

**Delete:**
- `headroom/proxy/responses_converter.py`

**Modify:**
- `headroom/proxy/handlers/openai.py` — remove imports of `responses_converter`. The compression dispatch path that called the converter to convert Responses items to Chat-Completions messages for compression is gone; Rust handles compression natively.
- `tests/test_responses_converter*.py` — delete all (Rust tests at `crates/headroom-proxy/tests/integration_responses.rs` cover the surface).

### Acceptance criteria

- `pytest -x` green.
- `git grep responses_converter headroom/` returns nothing.

### Blocked by

PR-C3, PR-C4.

### Blocks

PR-H1.

### Rollback

`git revert`. Python converter returns; Rust handler stays in place; both run side-by-side temporarily — but the Rust path is canonical.

---

## Phase C acceptance summary

After all 5 PRs land:

- ✅ Byte-level SSE parser with full state machine (handles UTF-8 split, ping, [DONE], all delta types, mid-stream errors)
- ✅ `/v1/chat/completions` handled in Rust
- ✅ `/v1/responses` HTTP handled in Rust
- ✅ `/v1/responses` streaming handled in Rust (out-of-order items, all event types)
- ✅ Conversations API awareness (warns + skips compression)
- ✅ All Responses item types (V4A, local_shell, phase, compaction, MCP, computer, image_gen, etc.) preserved byte-equal
- ✅ `responses_converter.py` deleted

**Phase C retires P1-8 through P1-12, P1-14 through P1-17, P4-40, P4-42 through P4-44, P4-47, P4-48, P0-7 (final), P5-51.**
