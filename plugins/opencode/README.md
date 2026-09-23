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

The package's default export is an OpenCode plugin that loads on both OpenCode 1.x and 2.x. Register it in `opencode.json`:

```json
{
  "plugin": [
    ["headroom-opencode", { "proxyUrl": "http://127.0.0.1:8787" }]
  ]
}
```

OpenCode 2.x migrates the 1.x `plugin` key automatically. To configure it the 2.x way, use `"plugins": [{ "package": "headroom-opencode", "options": { ... } }]`.

The default export is `{ id: "headroom", server, setup }`. OpenCode 1.x calls `server` (`HeadroomPlugin`) and 2.x calls `setup` (`headroomSetup`); each ignores the other's field. Both entry points:

- install Headroom transport interception for OpenCode provider traffic.
- expose the `headroom_retrieve` tool (a direct tool on 2.x, not only through code mode).
- add `HEADROOM_ACTIVE`, `HEADROOM_PROXY_URL`, `HEADROOM_PROJECT` (and `HEADROOM_BACKEND` when set) to shell command environments.
- default to `HEADROOM_PROXY_URL`, then `http://127.0.0.1:8787`, when no `proxyUrl` option is supplied.

On 2.x, setup returns a cleanup that releases the transport, so hot reloads do not stack interceptors.

OpenCode 2.x keeps its own reference to `fetch` from before plugins load, so patching `fetch` alone does not catch its model traffic. On 2.x the plugin also rewrites each remote model that uses the OpenAI chat-completions or responses format so that it points at the proxy. Headers name the real upstream, as the transport does. Models on other formats (Anthropic messages, Google) are left alone. `headroom wrap opencode` routes native Anthropic and OpenAI through the proxy separately, via their `baseURL`.

## Retrieve Tool

```ts
import { createHeadroomRetrieveTool } from "headroom-opencode";

const retrieve = createHeadroomRetrieveTool({
  proxyBaseUrl: "http://127.0.0.1:8787",
});

const result = await retrieve.execute({
  hash: "0123456789abcdef01234567",
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
