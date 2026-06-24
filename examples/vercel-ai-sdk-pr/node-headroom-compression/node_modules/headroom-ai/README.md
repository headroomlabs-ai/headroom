# headroom-ai

Compress LLM context. Save tokens. Fit more into every request.

## Install

```bash
npm install headroom-ai
```

## Quick Start

```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, { model: 'gpt-4o' });
console.log(`Saved ${result.tokensSaved} tokens (${((1 - result.compressionRatio) * 100).toFixed(0)}%)`);

// Use compressed messages with any LLM client
const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: result.messages,
});
```

Requires a running Headroom proxy (`headroom proxy`) or Headroom Cloud API key.

## Framework Adapters

### Vercel AI SDK

```typescript
import { headroomMiddleware } from 'headroom-ai/vercel-ai';
import { wrapLanguageModel, generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const model = wrapLanguageModel({
  model: openai('gpt-4o'),
  middleware: headroomMiddleware(),
});

const { text } = await generateText({ model, messages });
```

### OpenAI SDK

```typescript
import { withHeadroom } from 'headroom-ai/openai';
import OpenAI from 'openai';

const client = withHeadroom(new OpenAI());
const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: longConversation,
});
```

### Anthropic SDK

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

## Configuration

```typescript
import { compress } from 'headroom-ai';

const result = await compress(messages, {
  model: 'gpt-4o',
  baseUrl: 'http://localhost:8787',  // or https://api.headroom.ai
  apiKey: 'hr_...',                   // for Headroom Cloud
  timeout: 30000,                     // ms
  fallback: true,                     // return uncompressed if proxy is down (default)
  retries: 1,                         // retry on transient failures (default)
});
```

Or use environment variables:
- `HEADROOM_BASE_URL` — proxy/cloud URL
- `HEADROOM_API_KEY` — Cloud API key

## Reusable Client

```typescript
import { HeadroomClient } from 'headroom-ai';

const client = new HeadroomClient({
  baseUrl: 'http://localhost:8787',
  apiKey: 'hr_...',
});

// Reuse across many calls
const r1 = await client.compress(messages1, { model: 'gpt-4o' });
const r2 = await client.compress(messages2, { model: 'gpt-4o' });
```

## License

Apache-2.0
