# TypeScript SDK

The Headroom TypeScript SDK lets any JavaScript or TypeScript application compress LLM messages before sending them to a model. It saves tokens, reduces costs, and fits more context into every request.

## Install

```bash
npm install headroom-ai
```

Requires a running [Headroom proxy](proxy.md) or Headroom Cloud API key.

## Quick Start

```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, { model: 'gpt-4o' });
console.log(`Saved ${result.tokensSaved} tokens`);

const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: result.messages,
});
```

## How It Works

The TypeScript SDK is an HTTP client. When you call `compress()`, it sends your messages to the Headroom proxy's `POST /v1/compress` endpoint. The proxy runs the full compression pipeline (SmartCrusher, ContentRouter, CacheAligner, etc.) and returns compressed messages. No compression logic runs in Node.js — all the heavy lifting happens in the proxy.

```
Your TypeScript App
    │
    │  compress(messages)
    ▼
headroom-ai (npm)  ← HTTP client
    │
    │  POST /v1/compress
    ▼
Headroom Proxy / Cloud  ← compression pipeline (Python)
    │
    │  compressed messages
    ▼
Your TypeScript App
    │
    │  openai.chat.completions.create(compressed)
    ▼
LLM Provider
```

## Core API: `compress()`

```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, {
  model: 'gpt-4o',                      // model name (for token counting)
  baseUrl: 'http://localhost:8787',      // proxy URL (default)
  apiKey: 'hr_...',                      // Headroom Cloud key
  timeout: 30000,                        // ms (default)
  fallback: true,                        // return uncompressed if proxy down (default)
  retries: 1,                            // retry on transient errors (default)
});

result.messages          // compressed messages (same format as input)
result.tokensBefore      // original token count
result.tokensAfter       // compressed token count
result.tokensSaved       // tokens removed
result.compressionRatio  // tokensAfter / tokensBefore
result.transformsApplied // e.g. ['router:smart_crusher:0.35']
result.compressed        // false if fallback kicked in
```

Messages use standard OpenAI chat format: `{ role, content, tool_calls?, tool_call_id? }`.

### Environment Variables

Instead of passing options, set environment variables:

- `HEADROOM_BASE_URL` — proxy or cloud URL (default: `http://localhost:8787`)
- `HEADROOM_API_KEY` — Headroom Cloud API key

## Reusable Client

For apps making many calls, create a client once and reuse it:

```typescript
import { HeadroomClient } from 'headroom-ai';

const client = new HeadroomClient({
  baseUrl: 'http://localhost:8787',
  apiKey: 'hr_...',
});

const r1 = await client.compress(messages1, { model: 'gpt-4o' });
const r2 = await client.compress(messages2, { model: 'gpt-4o' });
```

## Framework Adapters

### Vercel AI SDK

The Headroom middleware plugs directly into Vercel AI SDK's `wrapLanguageModel()`:

```typescript
import { headroomMiddleware } from 'headroom-ai/vercel-ai';
import { wrapLanguageModel, generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const model = wrapLanguageModel({
  model: openai('gpt-4o'),
  middleware: headroomMiddleware(),
});

// All calls through this model are automatically compressed
const { text } = await generateText({ model, messages });
```

The middleware intercepts messages in the `transformParams` hook, converts Vercel's internal format to OpenAI format, compresses via the proxy, and converts back. Your app code doesn't change.

You can also compress Vercel messages directly:

```typescript
import { compressVercelMessages } from 'headroom-ai/vercel-ai';

const result = await compressVercelMessages(modelMessages, { model: 'gpt-4o' });
// result.messages is in Vercel ModelMessage[] format
```

### OpenAI SDK

Wrap your OpenAI client to auto-compress messages on every `chat.completions.create()` call:

```typescript
import { withHeadroom } from 'headroom-ai/openai';
import OpenAI from 'openai';

const client = withHeadroom(new OpenAI());

// Messages are compressed before sending — transparent to your code
const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: longConversation,
});
```

Only `chat.completions.create()` is intercepted. All other methods (embeddings, images, audio) pass through unchanged.

### Anthropic SDK

Same pattern for the Anthropic client:

```typescript
import { withHeadroom } from 'headroom-ai/anthropic';
import Anthropic from '@anthropic-ai/sdk';

const client = withHeadroom(new Anthropic());

const response = await client.messages.create({
  model: 'claude-sonnet-4-5-20250929',
  messages: longConversation,
  max_tokens: 1024,
});
```

Only `messages.create()` is intercepted. The adapter converts between Anthropic's content block format and OpenAI format automatically.

## Error Handling

```typescript
import { compress, HeadroomConnectionError, HeadroomAuthError } from 'headroom-ai';

try {
  const result = await compress(messages, { model: 'gpt-4o', fallback: false });
} catch (error) {
  if (error instanceof HeadroomAuthError) {
    // Invalid API key (401)
  } else if (error instanceof HeadroomConnectionError) {
    // Proxy unreachable
  }
}
```

With `fallback: true` (the default), connection errors and 5xx responses return the original messages uncompressed instead of throwing. Auth errors (401) and client errors (400) always throw.

## Fallback Behavior

By default, `compress()` never blocks your app. If the proxy is unreachable:

| Scenario | `fallback: true` (default) | `fallback: false` |
|----------|---------------------------|-------------------|
| Proxy unreachable | Returns uncompressed, `compressed: false` | Throws `HeadroomConnectionError` |
| Proxy 503 error | Returns uncompressed after retries | Throws `HeadroomCompressError` |
| Invalid API key (401) | Throws `HeadroomAuthError` | Throws `HeadroomAuthError` |
| Bad request (400) | Throws `HeadroomCompressError` | Throws `HeadroomCompressError` |

## Zero Dependencies

The `headroom-ai` package has no runtime dependencies. Framework SDKs (Vercel AI, OpenAI, Anthropic) are optional peer dependencies — only install what you use.

## OpenClaw Plugin

The TypeScript SDK powers the [`headroom-openclaw`](https://www.npmjs.com/package/headroom-openclaw) plugin for [OpenClaw](https://github.com/openclaw/openclaw) agents. The plugin uses `HeadroomClient` internally to compress context during the `assemble()` lifecycle hook. The preferred install flow is `headroom wrap openclaw`; the direct plugin command is `openclaw plugins install --dangerously-force-unsafe-install headroom-ai/openclaw`. See the [plugin source](https://github.com/chopratejas/headroom/tree/main/plugins/openclaw) for details.

## Comparison with Python SDK

| Feature | Python SDK | TypeScript SDK |
|---------|-----------|---------------|
| `compress()` | Native (runs locally) | HTTP client (calls proxy) |
| Proxy | Built-in server | Connects to proxy |
| Vercel AI SDK | N/A | Middleware adapter |
| OpenAI SDK | `HeadroomClient` wrapper | `withHeadroom()` wrapper |
| Anthropic SDK | `HeadroomClient` wrapper | `withHeadroom()` wrapper |
| LangChain | `HeadroomChatModel` | Use `compress()` directly |
| Memory system | Full (SQLite + HNSW) | Not yet (use proxy) |
| MCP server | Built-in | Not yet |
| CLI tools | `headroom proxy`, `headroom wrap`, etc. | N/A (use Python CLI) |
