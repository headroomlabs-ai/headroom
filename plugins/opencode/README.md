# headroom-opencode

OpenCode integration helpers for Headroom. The package supports two integration paths:

1. Provider config helpers used by `headroom wrap opencode` and persistent installs.
2. A native OpenCode plugin that installs Headroom transport interception and exposes the retrieve tool.

The policy adapter implements the external-plugin direction discussed in
[headroom#3279](https://github.com/headroomlabs-ai/headroom/issues/3279).

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
- enforces optional native tool policies, with shell/HTTP transport checks as defense in depth.
- defaults to `http://127.0.0.1:8787` when no proxy URL is supplied.

### Tool policy enforcement

Pass `toolPolicy` to `HeadroomPlugin` (or set `HEADROOM_TOOL_POLICY_JSON`) to preflight outbound HTTP requests and child-process shell launches before they execute.

```ts
import { HeadroomPlugin } from "headroom-opencode";

export default async function plugin(input) {
  return HeadroomPlugin(input, {
    proxyUrl: "http://127.0.0.1:8787",
    toolPolicy: {
      version: 1,
      mode: "enforce",
      rules: [
        {
          id: "deny-direct-openai",
          scope: "http",
          action: "deny",
          domain: "api.openai.com",
          reason: "force egress through approved gateways",
        },
        {
          id: "approve-curl",
          scope: "shell",
          action: "require_approval",
          command: "curl",
        },
      ],
    },
  });
}
```

You can also keep the policy outside the plugin code:

```bash
export HEADROOM_TOOL_POLICY_PATH=~/.headroom/config/tool_policy.json
```

Or load it from an authenticated policy service:

```bash
export HEADROOM_TOOL_POLICY_URL=https://policy.example.com/v1/headroom
export HEADROOM_TOOL_POLICY_TOKEN="$POLICY_READ_TOKEN"
export HEADROOM_TOOL_POLICY_REFRESH_SECONDS=900
```

Or commit a repo-local policy file:

```text
<repo>/.headroom/tool_policy.json
```

```json
{
  "version": 1,
  "mode": "enforce",
  "defaultAction": "allow",
  "rules": [
    {
      "id": "deny-direct-openai",
      "scope": "http",
      "action": "deny",
      "domain": "api.openai.com",
      "reason": "force egress through approved gateways"
    },
    {
      "id": "ask-before-curl-post",
      "scope": "shell",
      "action": "require_approval",
      "command": "curl",
      "argsPattern": "(^|\\s)-X\\s+POST\\b"
    }
  ]
}
```

Behavior:

- scopes: `shell`, `http`, and cross-cutting `tool_call`
- actions: `allow`, `deny`, `require_approval`
- control precedence: explicit plugin policy → `HEADROOM_TOOL_POLICY_JSON` → `HEADROOM_TOOL_POLICY_PATH` → `HEADROOM_TOOL_POLICY_URL` → `~/.headroom/config/tool_policy.json` → nearest repo `.headroom/tool_policy.json`
- deterministic precedence: first matching rule wins
- matchers: `tool`, `command`, `argsPattern`, `cwdPattern`, `envKeys`, `domain`, `urlPattern`
- `report_only` mode logs the decision but allows the operation
- decisions are appended to `~/.headroom/tool_policy_audit.jsonl` and emitted as structured JSON lines on stderr
- native `tool.execute.before` enforcement covers shell and non-shell OpenCode tools; transport interception remains advisory defense in depth
- remote policies use ETag revalidation and a credential-bound five-minute atomic cache; refresh can be extended to one hour
- an unavailable or invalid remote policy fails closed after the cache expires

`require_approval` currently fails closed in the OpenCode transport because there is no interactive approval callback yet.

### Enforcement authority and acknowledgements

This plugin treats OpenCode's native tool lifecycle as the execution authority. The
`tool.execute.before` hook issues a preflight decision bound to the OpenCode caller,
session, call/task, tool name, effective tool working directory, and a hash of canonical
arguments. Before returning an allowed preflight, the adapter deeply freezes the
authorized argument graph and hook output so later hooks cannot replace the bound
request. A blocked decision receives an immediate
`headroom_tool_policy_enforcement_acknowledgement` because the adapter commits to
throwing. An allowed preflight receives no acknowledgement until OpenCode invokes
`tool.execute.after` with matching final arguments. Argument or working-directory
mutation between the hooks produces an integrity error and no acknowledgement.

A blocked acknowledgement proves only that the adapter prevented that specific bound
invocation. An allowed acknowledgement proves that OpenCode completed the matching
native tool lifecycle; neither is a claim about similar work attempted elsewhere.

Transport interception (`fetch`, `http`, and `child_process`) is labeled
`authority: "advisory"`. It is defense in depth for operations originating inside the
plugin process, but it cannot prove that an external executor prevented an effect and
therefore never emits an enforcement acknowledgement.

Native decisions use this envelope:

```json
{
  "version": 1,
  "decisionId": "<sha256>",
  "authority": "authoritative",
  "requestHash": "<request-hash>",
  "binding": {
    "caller": "opencode",
    "adapter": "tool.execute.before",
    "sessionID": "<session>",
    "taskID": "<call>",
    "callID": "<call>",
    "toolName": "bash",
    "cwd": "/workspace",
    "canonicalArgsHash": "<arguments-hash>"
  }
}
```

Only an acknowledgement with the same `decisionId`, `requestHash`, and complete binding
belongs to that decision. Missing or mismatched acknowledgements must not be interpreted
as enforcement.

Shell commands that use executable-name escaping, Bash/PowerShell/cmd expansion,
computed invocation, process substitution, control structures, `eval`, `exec`,
`source`, or command substitution cannot be statically authorized when a
restrictive shell policy is present. They fail closed unless a matching rule already
blocks the statically identified command. Because the process transport is global, a
second OpenCode workspace with a different policy context is rejected rather than
replacing the active workspace's policy.

Example audit line:

```json
{
  "event": "headroom_tool_policy_decision",
  "scope": "http",
  "action": "deny",
  "effective_action": "deny",
  "matched_rule": "deny-direct-openai",
  "resource": "https://api.openai.com/v1/responses"
}
```

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
| `HEADROOM_TOOL_POLICY_JSON` | Native plugin / child Node processes | Optional JSON policy document for native tool and transport checks |
| `HEADROOM_TOOL_POLICY_PATH` | Native plugin / child Node processes | Optional path to a shared JSON policy file; falls back to repo-local `.headroom/tool_policy.json` or `~/.headroom/config/tool_policy.json` when unset |
| `HEADROOM_TOOL_POLICY_URL` | Native plugin | HTTPS endpoint returning a versioned policy JSON document |
| `HEADROOM_TOOL_POLICY_TOKEN` | Native plugin | Optional bearer token for the remote policy service |
| `HEADROOM_TOOL_POLICY_REFRESH_SECONDS` | Native plugin | Cache refresh interval from 300 through 3600 seconds; invalid values use 300 |

## License

Apache-2.0
