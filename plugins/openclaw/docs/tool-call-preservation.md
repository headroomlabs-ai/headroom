# Tool-call preservation — bug report & fix plan

**Scope:** `@headroom-ai/openclaw` plugin only (`plugins/openclaw/`).  
**Out of scope for this PR:** Headroom proxy/runtime Python source (`headroom/`). Headroom **config/env** on the running proxy is operator-controlled and may be documented as mitigations, but code changes stay in the plugin.  
**OpenClaw config:** allowed in follow-up PRs; **do not enable new Headroom features in production until plugin fixes land.**

---

## Summary

Agents using OpenClaw with the Headroom context engine report **mangled or missing tool calls and tool results** — especially `view_image`, `browser`, `read`, and large JSON tool outputs. Investigation shows the primary damage happens in the **plugin’s message conversion layer** (`convert.ts`) and in **per-turn `assemble()`**, before compressed context reaches the model. A second compression pass may occur when gateway provider routing is enabled, but the plugin must not destroy tool payloads on the way in.

This document records bugs, reproduction steps, proposed plugin-only fixes, and the test plan for the upcoming PR.

---

## Architecture (plugin boundary)

```
OpenClaw SQLite transcript (AgentMessage[])
        │
        ▼
  HeadroomContextEngine.assemble()          ← plugins/openclaw/src/engine.ts
        │
        ├─ normalizeAgentMessages()         ← convert.ts
        ├─ agentToOpenAI()                  ← convert.ts  ⚠ primary loss point
        ├─ headroom-ai compress() → POST /v1/compress   ← proxy (read-only for us)
        └─ openAIToAgent()                  ← convert.ts  ⚠ secondary loss point
        │
        ▼
  Model turn (may also route via gatewayProviderIds → proxy live path)
```

**What we control:** `convert.ts`, `engine.ts`, `compaction.ts` (when hygiene/headroom compaction enabled), plugin config schema, tests.  
**What we do not change:** `headroom/transforms/content_router.py`, proxy handlers, `DEFAULT_EXCLUDE_TOOLS` in Headroom core.

---

## Bugs discovered

### BUG-1 — Image tool results stripped in `agentToOpenAI()` (Critical)

**Location:** `src/convert.ts` — `agentToOpenAI()` toolResult branch + `extractText()`

**Behavior:** When a `toolResult` message contains content blocks `{ type: "image", data, mimeType }` (typical for `view_image`), conversion uses `extractText()` which only handles `text` and nested `tool_result` blocks. Image blocks become empty strings.

**Symptom:** Agent calls `view_image` → tool succeeds locally → after `assemble()` the model sees an empty tool result → agent claims “file not found”, retries, or hallucinates screenshot content.

**Evidence:** Production Godot session showed alternating success/failure on paths that existed on disk; failures correlated with turns where `assemble()` compressed (`tokensSaved > 0`).

---

### BUG-2 — Tool name not passed to OpenAI `tool` role (Critical)

**Location:** `src/convert.ts` — `agentToOpenAI()` lines ~151–160

**Behavior:** OpenAI-format tool messages are emitted with `tool_call_id` but **no `name` field**, even though `toolName` lives on the AgentMessage and in `_headroomMeta`.

**Symptom:** Headroom proxy protect/exclude logic maps `tool_call_id` → name from assistant `tool_calls`. Orphan or mis-ordered tool results cannot be matched to OpenClaw tool names (`view_image`, `browser`, `read`, …). Tool outputs that Claude Code would protect as `Read` are lossy-compressed for OpenClaw.

**Note:** Fixing this in the plugin improves compress requests; full protect-list behavior still depends on proxy env (`HEADROOM_EXCLUDE_TOOLS`), which is **config-only**, not source edits.

---

### BUG-3 — `openAIToAgent()` flattens tool results to text-only (Critical)

**Location:** `src/convert.ts` — `openAIToAgent()` tool branch (~256–277)

**Behavior:** After compression, tool messages are rebuilt as `[{ type: "text", text: ... }]` only. Image blocks and structured block arrays are not restored.

**Symptom:** Even if the proxy preserved content, the round-trip through `openAIToAgent()` destroys multimodal tool results permanently for that turn.

---

### BUG-4 — Unknown assistant content blocks silently dropped (High)

**Location:** `src/convert.ts` — `normalizeAssistantContent()` (~371–397)

**Behavior:** Blocks that are not `text`, `thinking`, `toolCall`, or `tool_use` return `[]` from `flatMap` — **silently removed**.

**Symptom:** Assistant turns with provider-specific blocks (e.g. image refs, custom metadata blocks) lose tool calls or text when normalization runs before compress or on budget skip paths.

---

### BUG-5 — Misleading CCR system prompt on marker-free assemble (Medium)

**Location:** `src/engine.ts` — `assemble()` systemPromptAddition (~262–264)

**Behavior:** When `tokensSaved > 100`, plugin injects: *“Use headroom_retrieve with the hash to get full details.”* Per-turn `compress()` via `headroom-ai` SDK often uses **marker-free / lossy** `/v1/compress` with **no `ccrHashes`**.

**Symptom:** Model invokes `headroom_retrieve` for content that was already lossy-compressed with no recoverable hash → agent spirals, blames Headroom, burns turns.

---

### BUG-6 — Double compression when gateway routing enabled (Medium)

**Location:** `src/engine.ts` (`assemble()`) + `src/gateway-config.ts` (in-memory provider rewrite)

**Behavior:** With `gatewayProviderIds` set (e.g. `minimax-portal`, `openrouter`, `opencode-go`), each turn is compressed in `assemble()` **and** again on the live provider request through the proxy.

**Symptom:** Tool outputs compressed twice; errors amplified. OpenClaw tool names still not on Headroom’s default exclude list (core concern — mitigated via proxy env, not source).

**Plugin fix angle:** Skip or soften `assemble()` compress when gateway routing is active, or pass explicit compress config that protects recent tool turns (plugin-side request body only).

---

### BUG-7 — Durable hygiene uses aggressive lossy config (High when enabled)

**Location:** `src/compaction.ts` — `compressDurableTranscript()` (~204–212)

**Behavior:** `mode: "lossy_inline"`, `protect_recent: 0`, `compress_user_messages: true` for SQLite rewrites.

**Symptom:** Permanent transcript corruption for tool-heavy sessions when `transcriptHygiene: true` or `persistentCompaction: "hybrid"|"headroom"`.

**Current production config:** `transcriptHygiene: false`, `persistentCompaction: "openclaw"` — **this path is inactive** but must be fixed before re-enabling hygiene.

---

### BUG-8 — User messages flattened to plain text in `agentToOpenAI()` (Medium)

**Location:** `src/convert.ts` — user branch (~81–89)

**Behavior:** User turns with embedded `tool_result` blocks (Anthropic-shaped) are flattened via `extractText()`, dropping structure.

**Symptom:** Rare in pure OpenClaw sessions; breaks when adapters emit nested tool_result in user role.

---

## Reproduction steps

### Repro A — Empty `view_image` result after assemble (BUG-1, BUG-3)

**Prerequisites:** Headroom plugin enabled, proxy reachable, context large enough to trigger compress (`roughTokens >= (tokenBudget − assembleReserveTokens) × assembleSkipBudgetRatio`, defaults 20000 / 0.7) or force by lowering budget in test.

1. Start OpenClaw with `plugins.slots.contextEngine: "headroom"` and `headroom.config.proxyUrl` pointing at a running proxy.
2. Open a dashboard session with a vision-capable model.
3. Run a turn that produces a **toolResult with image blocks** (e.g. `view_image` on a PNG under workspace or media roots).
4. Observe transcript before turn N+1: tool result contains `[{ type: "image", ... }]`.
5. Trigger `assemble()` on turn N+1 (large context or integration test calling `engine.assemble()` directly).

**Expected (correct):** Image blocks preserved or replaced with explicit lossy summary + optional CCR hash.  
**Actual (bug):** `agentToOpenAI` → empty string content; `openAIToAgent` → `[{ type: "text", text: "" }]`.

**Automated repro (plugin unit test — to add):**

```typescript
const messages = [{
  role: "toolResult",
  toolCallId: "call_1",
  toolName: "view_image",
  content: [{ type: "image", data: "<base64>", mimeType: "image/png" }],
}];
const roundTrip = openAIToAgent(agentToOpenAI(messages));
// Assert image block survives
```

---

### Repro B — Tool name missing on compress request (BUG-2)

1. Capture HTTP body to `POST /v1/compress` during `assemble()` (proxy access log or test mock).
2. Use transcript with assistant `toolCall` + matching `toolResult` where `toolName: "browser"`.
3. Inspect OpenAI messages in request body.

**Expected:** `{ role: "tool", name: "browser", tool_call_id: "...", content: "..." }`  
**Actual:** `name` field absent; proxy cannot apply tool-specific protection.

---

### Repro C — False CCR retrieve hint (BUG-5)

1. Run session until `assemble()` logs tokens saved > 100.
2. Check returned `systemPromptAddition` for headroom_retrieve instruction.
3. Inspect compress response for `ccr_hashes` / markers in tool content.

**Expected:** Hint only when hashes exist.  
**Actual:** Hint appears with marker-free lossy compress → model calls `headroom_retrieve` → empty or error.

---

### Repro D — Double compression with gateway routing (BUG-6)

**Prerequisites:** `gatewayProviderIds: ["openrouter"]` (or similar), proxy stats enabled.

1. Enable gateway routing in plugin config.
2. Run multi-tool session until both assemble and upstream request fire.
3. Compare proxy logs: `/v1/compress` call from assemble **and** transform activity on live chat completion.

**Expected (ideal):** Single compression point or idempotent second pass with tool protection.  
**Actual:** Tool blobs compressed twice; OpenClaw-named tools not excluded unless proxy env configured.

---

### Repro E — Hygiene destroys tool pairs (BUG-7) — **do not run in production**

1. Set `persistentCompaction: "hybrid"`, `transcriptHygiene.enabled: true`.
2. Large tool-heavy session → wait for turn-end hygiene or run `/compact`.
3. Inspect SQLite transcript entries for toolCall/toolResult pairs.

**Expected:** Replace-only, tool structure preserved.  
**Actual:** Lossy inline rewrite; recent tool turns not protected (`protect_recent: 0`).

---

## Fix plan (plugin only)

### Phase 1 — `convert.ts` ✅

| Task | Description |
|------|-------------|
| **1a** | Non-text blocks never reach the proxy. `agentToOpenAI` emits text verbatim and a short `[headroom-omitted <type> …]` placeholder per image/tool envelope (`content-blocks.ts`). Base64 is neither tokenized, counted in `tokens_before`, nor eligible for lossy rewrite |
| **1b** | Set `name` on OpenAI `tool` messages from `toolName` |
| **1c** | `openAIToAgent(compressed, { originals })` restores images, `toolCall` / `thinking` / unknown blocks, `toolName`, `toolCallId`, `isError`, timestamps and assistant metadata from the **local originals** (`original-lookup.ts`): tool messages by unique `tool_call_id`, others by echoed `hrIndex` hint or position. Compressed text is folded into the first text block; non-text blocks keep their position (`mergeCompressedTextIntoBlocks`) |
| **1d** | `normalizeAssistantContent` passes unknown blocks through instead of dropping them |
| **1e** | Without originals (stock-compatible call), behaviour degrades to text-only — documented and test-locked |
| **1f** | `estimateRoughTokens` charges a fixed ~1.5k tokens per image (vision pricing) instead of `base64.length / 4`, so image-heavy sessions do not trip the budget short-circuit on every turn |

Design note: an earlier iteration of this branch carried block arrays inside a `__HR_TOOL_BLOCKS__` content prefix and in `_headroomMeta`. That shipped every image **twice** as base64 text to the proxy (a 60 KB image inflated `tokens_before` from ~10k to ~50k in a live probe) and relied on the proxy echoing `_headroomMeta`, which it does not do consistently (observed `isError` loss in the mega-80 live run). Both problems are removed by the originals-based restore.

### Phase 2 — `engine.ts` ✅

| Task | Description |
|------|-------------|
| **2a** | Gate `systemPromptAddition` on `result.ccrHashes?.length > 0` (or equivalent SDK field) |
| **2b** | Pass `assembleCompressConfig` (default `protect_recent: 2`) to `compress()` via headroom-ai SDK |
| **2c** | `skipAssembleWhenGatewayRouted` (default `false`) skips assemble **only when `runtimeSettings.model.provider` is gateway-routed** (`assemble-skip.ts`). Non-routed providers still compress; unknown provider → skip (honours operator intent) |
| **2d** | `assemble()` calls `openAIToAgent(result.messages, { originals: params.messages })` |

### Phase 3 — `compaction.ts` ✅

| Task | Description |
|------|-------------|
| **3a** | Before durable rewrite, skip messages containing `toolCall` / `toolResult` / `image` / user `tool_result` blocks |
| **3b** | Durable config uses `protect_recent: 2` (via shared `compress-request-config.ts`) |
| **3c** | Do **not** enable hygiene/hybrid in production until staging validation passes |

### Phase 4 — Config & docs ✅

| Task | Description |
|------|-------------|
| **4a** | Proxy env mitigations documented in README (`HEADROOM_EXCLUDE_TOOLS`, `HEADROOM_PROTECT_TOOL_RESULTS`) |
| **4b** | Plugin config keys `assembleCompressConfig`, `skipAssembleWhenGatewayRouted` in `openclaw.plugin.json` |
| **4c** | Optional OpenClaw `openclaw.json` tuning after staging (see below) |

---

## Testing plan

### Unit tests (`test/convert.test.ts`, `test/content-blocks.test.ts`, `test/original-lookup.test.ts`, `test/assemble-skip.test.ts`)

- [x] Image bytes never appear in the `agentToOpenAI` output (placeholder only)
- [x] Image toolResult round-trip preserves `{ type: "image" }` with `_headroomMeta` echoed **and** stripped
- [x] Tool message includes `name: "view_image"` / `"browser"`
- [x] Mixed assistant content: text + toolCall + thinking preserved; multi-text assistants do not duplicate text
- [x] Tool results restored by `tool_call_id` when the proxy dropped other messages
- [x] `isError` restored from originals; fallback heuristic only fires on the OpenClaw envelope (`status` + `tool`)
- [x] Unknown block type passthrough
- [x] Provider-aware gateway skip (routed / not routed / unknown)
- [x] Regression: existing text toolResult tests still pass

### Engine tests (`test/engine.test.ts`)

- [x] `systemPromptAddition` absent when no `ccrHashes`
- [x] `systemPromptAddition` present when mock compress returns hashes
- [x] Budget skip path still returns normalized messages without stripping tools

### Compaction tests (`test/compaction.test.ts`)

- [x] `planHeadroomCompaction` does not replace entries whose payload contains image blocks
- [x] Durable compress requests `protect_recent: 2`
- [x] Forced / compress-result truncate re-aligns the cut to a turn start and drops orphan tool results (`truncate-boundary.ts`); prefix tool pairs before the cut are dropped whole

### Manual / integration (staging)

1. Godot or Finetune session: `browser` screenshot → `view_image` → model describes image correctly after 500k+ context.
2. Proxy log: `/v1/compress` request bodies include `tool.name` for OpenClaw tools and `config.protect_recent: 2`.
3. No `headroom_retrieve` spiral when compress saves tokens without hashes.
4. With `gatewayProviderIds` enabled, set `skipAssembleWhenGatewayRouted: true` and confirm single compression path on live provider requests for routed providers, while a direct provider still logs `Assembled: … tokens`.
5. Proxy log: `/v1/compress` request bodies for `view_image` turns are a few hundred bytes (placeholder), not the image size.

### Suggested staging `openclaw.json` (after plugin reload)

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "config": {
          "assembleCompressConfig": { "protect_recent": 2 },
          "skipAssembleWhenGatewayRouted": true
        }
      }
    }
  }
}
```

Proxy process (operator env, not OpenClaw):

```bash
export HEADROOM_EXCLUDE_TOOLS="view_image,browser,read,exec"
export HEADROOM_PROTECT_TOOL_RESULTS=1
```

---

## PR checklist

- [x] All changes under `plugins/openclaw/src/` and `plugins/openclaw/test/`
- [x] No edits to `headroom/` Python runtime
- [x] README / this doc updated; proxy env mitigations documented as operator config
- [x] No request to enable `transcriptHygiene` or `hybrid` in production OpenClaw config until Phase 3 merges
- [x] `npm test` in `plugins/openclaw` green
- [x] `npm run build` produces updated `dist/`

---

## References (plugin files)

| File | Role |
|------|------|
| `src/convert.ts` | AgentMessage ↔ OpenAI conversion |
| `src/engine.ts` | `assemble()`, `compact()`, `maintain()` |
| `src/compaction.ts` | Durable `/v1/compress` + SQLite rewrite |
| `src/gateway-config.ts` | In-memory provider routing |
| `test/convert.test.ts` | Conversion tests (currently text-only) |

---

## Out of scope (explicit)

- Changing Headroom `DEFAULT_EXCLUDE_TOOLS` or ContentRouter source
- OpenClaw core / dist modifications
- Enabling hybrid transcript hygiene in user `openclaw.json` as part of this PR
