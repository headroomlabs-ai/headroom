# headroom-ai examples

Runnable examples showing how to use the Headroom TypeScript SDK for context compression.

## Prerequisites

- Node.js 18+
- A running Headroom proxy (`pip install "headroom-ai[proxy]" && headroom proxy`)
- `OPENAI_API_KEY` set in your environment (most examples use OpenAI)
- Optional: `ANTHROPIC_API_KEY` for Anthropic examples

## Running

```bash
cd sdk/typescript
npm install
npx tsx examples/<filename>.ts
```

## Examples

### Vercel AI SDK

| Example | Description |
|---------|-------------|
| [with-headroom-vercel.ts](with-headroom-vercel.ts) | One-liner `withHeadroom(openai('gpt-4o'))` — simplest integration |
| [streaming-chat.ts](streaming-chat.ts) | `withHeadroom` + `streamText` for real-time streaming |
| [tool-calling-agent.ts](tool-calling-agent.ts) | Multi-step agent with tools, context auto-compressed each step |
| [structured-output.ts](structured-output.ts) | Extract structured data with `Output.object()` from compressed context |
| [middleware-composition.ts](middleware-composition.ts) | Stack `headroomMiddleware` with other middlewares (`extractReasoningMiddleware`) |
| [multi-provider.ts](multi-provider.ts) | Same compression across GPT-4o and GPT-4o-mini |

### Core SDK

| Example | Description |
|---------|-------------|
| [basic-compress.ts](basic-compress.ts) | `compress()` function — compress then send to any LLM |
| [simulation-dry-run.ts](simulation-dry-run.ts) | `simulate()` — see what compression would do without calling the LLM |
| [hooks-custom-compression.ts](hooks-custom-compression.ts) | `CompressionHooks` — customize with pre/post hooks and per-message biases |
| [shared-context-multi-agent.ts](shared-context-multi-agent.ts) | `SharedContext` — compressed handoff between agents (70-90% savings) |
| [ccr-retrieve.ts](ccr-retrieve.ts) | CCR — retrieve original content after compression (lossless) |

### Native SDK Adapters

| Example | Description |
|---------|-------------|
| [openai-anthropic-adapters.ts](openai-anthropic-adapters.ts) | `withHeadroom` for native OpenAI and Anthropic SDKs (no Vercel AI SDK) |
