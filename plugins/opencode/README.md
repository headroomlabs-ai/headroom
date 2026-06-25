# headroom-opencode

OpenCode integration helpers for Headroom. The package supports two integration paths:

1. Provider config helpers used by `headroom wrap opencode` and persistent installs.
2. A native OpenCode plugin that installs Headroom transport interception and exposes the retrieve tool.

## Install

```bash
npm install headroom-opencode
```

## Provider Config Helpers

Use these helpers when you need to generate OpenCode config that routes a `headroom` provider through a running Headroom proxy.

```ts
import {
  buildOpencodeConfigContent,
  createHeadroomProvider,
} from "headroom-opencode";

const provider = createHeadroomProvider({ proxyPort: 8787 });
const config = buildOpencodeConfigContent({
  proxyPort: 8787,
  defaultModel: "claude-sonnet-4-6",
});

console.log(provider.provider.headroom.npm);
console.log(config.model);
```

The generated provider uses `@ai-sdk/openai-compatible` and points model requests at `http://127.0.0.1:<port>/v1`.

## Native OpenCode Plugin

Use `HeadroomPlugin` when OpenCode should intercept provider traffic in-process and expose Headroom tooling from a plugin.

```ts
import { HeadroomPlugin } from "headroom-opencode";

export default async function plugin(input) {
  return HeadroomPlugin(input, {
    proxyUrl: process.env.HEADROOM_PROXY_URL ?? "http://127.0.0.1:8787",
  });
}
```

`HeadroomPlugin`:

- installs Headroom transport interception for OpenCode provider traffic.
- exposes the `headroom_retrieve` tool.
- publishes `HEADROOM_PROXY_URL` in the plugin output env.
- defaults to `http://127.0.0.1:8787` when no proxy URL is supplied.

## Retrieve Tool

```ts
import { createHeadroomRetrieveTool } from "headroom-opencode";

const retrieve = createHeadroomRetrieveTool({
  proxyBaseUrl: "http://127.0.0.1:8787",
});

const result = await retrieve.execute({
  hash: "0123456789abcdef01234567",
  query: "needle",
});
```

The tool calls `/v1/retrieve/<hash>` on the Headroom proxy.

## Compression Helper

```ts
import { compressWithHeadroom } from "headroom-opencode";

const result = await compressWithHeadroom(
  [{ role: "user", content: "Summarize this file" }],
  { model: "gpt-4o", proxyUrl: "http://127.0.0.1:8787" },
);

console.log(`Saved ${result.tokensSaved} tokens`);
```

## Models

| Model | Context | Output |
|---|---:|---:|
| `claude-sonnet-4-6` | 200K | 16K |
| `claude-opus-4-6` | 200K | 16K |
| `claude-haiku-4-5-20251001` | 200K | 8K |
| `gpt-4o` | 128K | 16K |
| `gpt-4.1` | 1M | 32K |

The provider config exposes these as `headroom/<model>` and defaults to `headroom/claude-sonnet-4-6`.

## Environment

| Variable | Used by | Description |
|---|---|---|
| `HEADROOM_PROXY_URL` | Native plugin | Proxy URL used by `HeadroomPlugin` |
| `OPENCODE_CONFIG_CONTENT` | OpenCode wrapper | Generated OpenCode provider, model, and MCP config |

## License

Apache-2.0
