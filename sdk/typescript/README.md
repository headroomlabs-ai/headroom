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
import { withHeadroom } from 'headroom-ai/vercel-ai';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

const model = withHeadroom(openai('gpt-4o'));
const { text } = await generateText({ model, messages });
```

<details>
<summary>Advanced: using middleware directly</summary>

```typescript
import { headroomMiddleware } from 'headroom-ai/vercel-ai';
import { wrapLanguageModel } from 'ai';

const model = wrapLanguageModel({
  model: openai('gpt-4o'),
  middleware: headroomMiddleware({ baseUrl: 'http://localhost:8787' }),
});
```

</details>

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

### Google Gemini

```typescript
import { withHeadroom } from 'headroom-ai/gemini';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
const model = withHeadroom(genAI.getGenerativeModel({ model: 'gemini-2.0-flash' }));

const result = await model.generateContent({
  contents: longConversation,
});
```

## HeadroomClient

The full client provides direct access to the proxy's OpenAI and Anthropic passthrough endpoints, plus metrics, CCR, and observability.

```typescript
import { HeadroomClient } from 'headroom-ai';

const client = new HeadroomClient({
  baseUrl: 'http://localhost:8787',
  providerApiKey: process.env.OPENAI_API_KEY,
  config: {
    smartCrusher: { enabled: true, maxItemsAfterCrush: 10 },
    ccr: { enabled: true },
  },
});
```

### Chat Completions (OpenAI-style)

```typescript
const response = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: longConversation,
  headroomMode: 'optimize',
});
```

### Messages (Anthropic-style)

```typescript
const response = await client.messages.create({
  model: 'claude-sonnet-4-5-20250929',
  messages: longConversation,
  max_tokens: 1024,
  headroomMode: 'optimize',
});
```

### Direct Compression

```typescript
const result = await client.compress(messages, { model: 'gpt-4o', tokenBudget: 4000 });
```

## Simulation (Dry Run)

See what compression would do without calling the LLM.

```typescript
import { simulate } from 'headroom-ai';

const sim = await simulate(messages, { model: 'gpt-4o' });
console.log(`Would save ${sim.tokensSaved} tokens (${sim.estimatedSavings})`);
console.log('Transforms:', sim.transforms);
console.log('Waste signals:', sim.wasteSignals);
console.log('Cache alignment:', sim.cacheAlignmentScore);
```

Also available on the client:

```typescript
const sim = await client.chat.completions.simulate({
  model: 'gpt-4o',
  messages,
});
```

## Compression Hooks

Customize compression with pre/post hooks — matching the Python `CompressionHooks` API.

```typescript
import { compress, CompressionHooks } from 'headroom-ai';
import type { CompressContext, CompressEvent } from 'headroom-ai';

class MyHooks extends CompressionHooks {
  // Modify messages before compression
  preCompress(messages: any[], ctx: CompressContext) {
    return [{ role: 'system', content: 'Always preserve error details.' }, ...messages];
  }

  // Set per-message importance biases
  computeBiases(messages: any[], ctx: CompressContext) {
    return { 0: 2.0 }; // preserve first message
  }

  // Observe compression results
  postCompress(event: CompressEvent) {
    console.log(`Saved ${event.tokensSaved} tokens via ${event.transformsApplied.join(', ')}`);
  }
}

const result = await compress(messages, { model: 'gpt-4o', hooks: new MyHooks() });
```

## SharedContext (Multi-Agent)

Compressed inter-agent context sharing — matching the Python `SharedContext` API.

```typescript
import { SharedContext } from 'headroom-ai';

const ctx = new SharedContext({ model: 'gpt-4o', ttl: 3600, maxEntries: 100 });

// Agent A stores data (automatically compressed)
const entry = await ctx.put('research', bigAgentOutput, { agent: 'researcher' });
console.log(`Compressed: ${entry.savingsPercent.toFixed(0)}% savings`);

// Agent B reads it (~80% smaller)
const summary = ctx.get('research');

// Agent B gets original if needed
const full = ctx.get('research', { full: true });

// Stats
const stats = ctx.stats();
console.log(`${stats.entries} entries, ${stats.totalTokensSaved} tokens saved`);
```

## CCR Retrieve (Compress-Cache-Retrieve)

Retrieve original content when the LLM needs full details.

```typescript
const result = await client.compress(messages, { model: 'gpt-4o' });

// Later, when the LLM calls headroom_retrieve:
for (const hash of result.ccrHashes) {
  const original = await client.retrieve(hash);
  console.log(`${original.originalTokens} original tokens for ${original.toolName}`);
}

// Search within compressed content
const search = await client.retrieve('abc123', { query: 'error logs' });

// Handle LLM tool calls in an agent loop
const toolResult = await client.handleToolCall({
  toolCall: assistantMessage.tool_calls[0],
  provider: 'openai',
});
```

## Metrics & Observability

```typescript
// Proxy health
const health = await client.health();
// → { status: 'healthy', version: '0.5.18', config: { optimize: true, ... } }

// Proxy stats
const stats = await client.proxyStats();
// → { requests: { total, cached, failed }, tokens: { saved, savingsPercent }, ... }

// Request metrics
const metrics = await client.getMetrics({ model: 'gpt-4o', limit: 10 });

// Summary
const summary = await client.getSummary();

// Validate setup
const validation = await client.validateSetup();

// Clear cache
await client.clearCache();

// Prometheus metrics
const prom = await client.prometheusMetrics();
```

## Telemetry, Feedback & TOIN

Access the proxy's learning systems.

```typescript
// Telemetry
const telemetry = await client.telemetry.getStats();
const tools = await client.telemetry.getTools();

// Feedback — per-tool compression hints
const hints = await client.feedback.getHints('list_servers');
// → { hints: { maxItems: 8, skipCompression: false, preserveFields: ['id', 'status'] } }

// TOIN (Tool Output Intelligence Network)
const toinStats = await client.toin.getStats();
const patterns = await client.toin.getPatterns(20);
```

## Configuration Types

Full TypeScript interfaces for every Python config dataclass.

```typescript
import type { HeadroomConfig, SmartCrusherConfig, CCRConfig } from 'headroom-ai';

const config: HeadroomConfig = {
  defaultMode: 'optimize',
  smartCrusher: {
    enabled: true,
    minItemsToAnalyze: 5,
    maxItemsAfterCrush: 10,
    varianceThreshold: 2.0,
    relevance: { tier: 'hybrid', relevanceThreshold: 0.25 },
    anchor: { anchorBudgetPct: 0.25 },
  },
  ccr: { enabled: true, injectTool: true },
  cacheOptimizer: { enabled: true, autoDetectProvider: true },
  intelligentContext: { enabled: true, useImportanceScoring: true },
};

const client = new HeadroomClient({ config });
```

## Error Handling

Full error hierarchy matching the Python SDK.

```typescript
import {
  HeadroomError,
  HeadroomConnectionError,
  HeadroomAuthError,
  HeadroomCompressError,
  ConfigurationError,
  ProviderError,
  StorageError,
  TokenizationError,
  CacheError,
  ValidationError,
  TransformError,
} from 'headroom-ai';

try {
  await client.compress(messages);
} catch (err) {
  if (err instanceof HeadroomAuthError) {
    console.error('Auth failed — check HEADROOM_API_KEY');
  } else if (err instanceof HeadroomCompressError) {
    console.error(`Compression error ${err.statusCode}: ${err.errorType}`);
  } else if (err instanceof ConfigurationError) {
    console.error('Bad config:', err.details);
  }
}
```

## Format Detection & Conversion

Auto-detects and converts between OpenAI, Anthropic, Vercel AI SDK, and Gemini formats.

```typescript
import { detectFormat, toOpenAI, fromOpenAI } from 'headroom-ai';

const format = detectFormat(messages); // 'openai' | 'anthropic' | 'vercel' | 'gemini'
const openaiMessages = toOpenAI(messages);
const back = fromOpenAI(openaiMessages, format);
```

The `compress()` function handles this automatically — pass any format and get the same format back.

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
  tokenBudget: 4000,                  // compress to fit this limit
  hooks: new MyHooks(),               // pre/post compression hooks
});
```

Or use environment variables:
- `HEADROOM_BASE_URL` — proxy/cloud URL
- `HEADROOM_API_KEY` — Cloud API key

## Utilities

```typescript
// Case conversion for proxy communication
import { deepCamelCase, deepSnakeCase } from 'headroom-ai';

const tsObj = deepCamelCase({ tokens_before: 100 }); // { tokensBefore: 100 }
const pyObj = deepSnakeCase({ tokensBefore: 100 });   // { tokens_before: 100 }

// SSE stream parsing
import { parseSSE, collectStream } from 'headroom-ai';

// Hook helpers
import { extractUserQuery, countTurns, extractToolCalls } from 'headroom-ai';
```

## License

Apache-2.0
