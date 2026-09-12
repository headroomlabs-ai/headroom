# Headroom for OpenCode

Route OpenCode model traffic through Headroom, expose Headroom retrieval, and apply
policy to native tool calls before they execute.

This package supports:

- **Individuals:** keep a policy in one repository or in a personal config directory.
- **Teams:** commit a reviewed policy with the repository.
- **Enterprises:** retrieve policy from an authenticated HTTPS service with local,
  credential-bound caching and fail-closed outage behavior.

The policy adapter follows the external-plugin design discussed in
[headroom#3279](https://github.com/headroomlabs-ai/headroom/issues/3279).

## What the plugin does

| Capability | Behavior |
|---|---|
| Provider routing | Routes supported model traffic through a Headroom proxy. |
| Retrieval | Adds the `headroom_retrieve` OpenCode tool. |
| Native policy | Evaluates `tool.execute.before` for shell and non-shell tools. |
| Defense in depth | Evaluates in-process HTTP and child-process activity as advisory checks. |
| Audit | Writes structured, secret-safe JSON lines to stderr and disk. |

Policy enforcement is optional. With no policy configured, existing OpenCode tool
behavior is unchanged.

## Getting started

### Prerequisites

- OpenCode with plugin support
- Node.js 18 or newer
- A running Headroom proxy, normally at `http://127.0.0.1:8787`

### 1. Install the plugin dependency

Create `.opencode/package.json` in your project:

```json
{
  "dependencies": {
    "headroom-opencode": "^0.37.0"
  }
}
```

OpenCode installs dependencies declared in `.opencode/package.json` when it starts.

### 2. Register the plugin

Create `.opencode/plugins/headroom.ts`:

```ts
import { HeadroomPlugin } from "headroom-opencode";

export default async function headroom(input) {
  return HeadroomPlugin(input, {
    proxyUrl: process.env.HEADROOM_PROXY_URL ?? "http://127.0.0.1:8787",
  });
}
```

Local plugins in `.opencode/plugins/` are loaded automatically. The wrapper above
exports only the plugin factory, which is required by OpenCode's plugin loader.

### 3. Add a starter policy

Create `.headroom/tool_policy.json`:

```json
{
  "version": 1,
  "mode": "enforce",
  "defaultAction": "allow",
  "rules": [
    {
      "id": "deny-direct-provider-egress",
      "scope": "http",
      "action": "deny",
      "domain": [
        "api.openai.com",
        "api.anthropic.com"
      ],
      "reason": "use the approved Headroom gateway"
    },
    {
      "id": "deny-curl-post",
      "scope": "shell",
      "action": "deny",
      "command": "curl",
      "argsPattern": "(^|\\s)(-X|--request)\\s+POST\\b",
      "reason": "direct HTTP writes are not allowed"
    }
  ]
}
```

### 4. Verify the setup

Restart OpenCode from the project directory, then ask it to:

1. Run `echo headroom-policy-ok`. The command should be allowed.
2. Run `curl -X POST https://example.com`. The command should be denied before execution.
3. Inspect `~/.headroom/tool_policy_audit.jsonl`. It should contain a bound native
   decision and a terminal acknowledgement.

Do not test a deny rule against a production endpoint. Use a non-sensitive hostname
such as `example.com`.

## Individual setup

### Repository-specific policy

Use `<repo>/.headroom/tool_policy.json` when a policy belongs to one project. This is
the simplest setup and can be reviewed like other project configuration.

### Personal policy across repositories

Store a policy at:

```text
~/.headroom/config/tool_policy.json
```

The global policy takes precedence over repository policy. Use it for non-negotiable
personal defaults, such as blocking credential-management commands in every project.

Example:

```json
{
  "version": 1,
  "mode": "enforce",
  "defaultAction": "allow",
  "rules": [
    {
      "id": "protect-ssh-directory",
      "scope": "tool_call",
      "action": "deny",
      "tool": [
        "write",
        "edit"
      ],
      "argsPattern": "[\\\\/]\\.ssh[\\\\/]",
      "reason": "SSH configuration requires manual changes"
    }
  ]
}
```

To use a different file without changing the plugin wrapper:

```bash
export HEADROOM_TOOL_POLICY_PATH="$HOME/policies/opencode.json"
opencode
```

PowerShell:

```powershell
$env:HEADROOM_TOOL_POLICY_PATH = "$HOME\policies\opencode.json"
opencode
```

### Test a policy before enforcing it

Set `"mode": "report_only"` while developing rules. Matching deny and approval rules
are audited but operations continue. Review the audit log, then change the mode to
`"enforce"`.

`report_only` is an observation mode, not a security boundary.

## Team setup

For a shared repository:

1. Commit `.opencode/plugins/headroom.ts`.
2. Commit `.opencode/package.json`.
3. Commit `.headroom/tool_policy.json`.
4. Require code-owner review for `.headroom/**` and `.opencode/**`.
5. Start with `report_only`, review representative audit output, then switch to
   `enforce`.

Prefer stable rule IDs because audit consumers use them to group decisions over time.
Put narrow exceptions before broad rules because the first matching rule wins.

Example default-deny policy:

```json
{
  "version": 1,
  "mode": "enforce",
  "defaultAction": "deny",
  "rules": [
    {
      "id": "allow-read-tools",
      "scope": "tool_call",
      "action": "allow",
      "tool": [
        "read",
        "glob",
        "grep"
      ]
    },
    {
      "id": "allow-safe-shell",
      "scope": "shell",
      "action": "allow",
      "command": [
        "git",
        "npm"
      ],
      "argsPattern": "^(git (status|diff|log)|npm (test|run (build|typecheck)))\\b"
    }
  ]
}
```

For compound shell commands, an allow rule must cover every executable candidate.
Unsupported dynamic shell grammar fails closed under a restrictive policy. Prefer
direct OpenCode tool calls over complex shell programs.

## Enterprise setup

### Recommended architecture

```text
OpenCode
  -> Headroom OpenCode plugin
      -> authenticated policy service
      -> credential-bound local cache
      -> native tool lifecycle enforcement
      -> JSONL audit collector
```

Run the policy service separately from the Headroom model proxy. Give agents a
read-only policy credential scoped to the tenant, environment, and policy set.

### Configure a remote policy

Set these variables in the environment that launches OpenCode:

```bash
export HEADROOM_TOOL_POLICY_URL="https://policy.example.com/v1/opencode"
export HEADROOM_TOOL_POLICY_TOKEN="$POLICY_READ_TOKEN"
export HEADROOM_TOOL_POLICY_REFRESH_SECONDS=900
opencode
```

PowerShell:

```powershell
$env:HEADROOM_TOOL_POLICY_URL = "https://policy.example.com/v1/opencode"
$env:HEADROOM_TOOL_POLICY_TOKEN = $env:POLICY_READ_TOKEN
$env:HEADROOM_TOOL_POLICY_REFRESH_SECONDS = "900"
opencode
```

The URL and bearer token remain in the plugin process. They are removed from OpenCode
shell environments and wrapped child-process environments. Do not place credentials in
the policy document.

### Policy service contract

The endpoint must:

- accept `GET`;
- return a version 1 policy JSON document;
- return a valid JSON body; `Content-Type: application/json` is strongly recommended;
- use HTTPS, except loopback HTTP for local development;
- return an `ETag` when revalidation is supported;
- honor `If-None-Match` with `304 Not Modified`;
- keep the response body at or below 1 MiB;
- avoid redirects, which the client rejects.

Example response:

```http
HTTP/1.1 200 OK
Content-Type: application/json
ETag: "policy-42"

{
  "version": 1,
  "mode": "enforce",
  "defaultAction": "deny",
  "rules": [
    {
      "id": "allow-approved-ci",
      "scope": "shell",
      "action": "allow",
      "command": "npm",
      "argsPattern": "^npm (test|run build)\\b"
    }
  ]
}
```

### Cache and outage behavior

- Refresh defaults to 300 seconds and can be configured from 300 through 3600 seconds.
- Missing, non-integer, or out-of-range refresh values use 300 seconds rather than
  being clamped.
- Cache identity includes both the policy URL and bearer token, preventing reuse across
  credentials or tenants.
- Cache writes are atomic and stored under `~/.headroom/policy-cache/`.
- A fresh cache avoids a network request until its refresh interval elapses.
- After cache expiry, an unavailable, invalid, oversized, or redirected policy fails
  closed.
- A second workspace with a different policy or credential context in the same process
  is rejected rather than replacing the active policy.

When `HEADROOM_STATELESS` is `1`, `true`, `yes`, or `on`, disk audit and cache reads
and writes are disabled. Remote policy remains fail-closed, but there is no persisted
cache fallback across requests or restarts; the policy service must remain available
when a refresh is required.

For recovery, restore the service before the cache expires, correct the local policy
source, or remove policy configuration to return deliberately to the plugin's
unconfigured behavior. Do not delete an expired cache as an outage workaround; that
does not create an allow fallback.

### Rollout checklist

1. Define stable rule IDs and owners.
2. Deploy the endpoint behind authenticated HTTPS.
3. Use a short-lived, read-only credential.
4. Roll out in `report_only`.
5. Collect audit records across representative repositories and operating systems.
6. Resolve false positives, especially dynamic shell usage.
7. Switch a pilot group to `enforce`.
8. Test endpoint outage and expired-cache behavior.
9. Expand rollout and monitor `unknown` terminal outcomes.
10. Document the disable and credential-rotation procedures.

### Audit collection

By default, records are appended to:

```text
~/.headroom/tool_policy_audit.jsonl
```

Set `HEADROOM_WORKSPACE_DIR` to move audit and cache storage to a managed directory.
Unless `HEADROOM_CONFIG_DIR` is set separately, this also moves the global policy to
`<HEADROOM_WORKSPACE_DIR>/config/tool_policy.json`. Records are also emitted as JSON
lines on stderr for collection by a supervisor.

Audit output intentionally contains safe resource summaries, such as a command name or
hostname, plus a hash of the full resource. Signed URLs, bearer tokens, and full command
arguments are not written.

## Policy reference

### Source precedence

The first configured source wins:

1. `toolPolicy` passed directly to `HeadroomPlugin`
2. `HEADROOM_TOOL_POLICY_JSON`
3. `HEADROOM_TOOL_POLICY_PATH`
4. `HEADROOM_TOOL_POLICY_URL`
5. global policy file (by default `~/.headroom/config/tool_policy.json`)
6. nearest `.headroom/tool_policy.json`

An invalid selected source is an error. The plugin does not silently fall through to a
lower-precedence source.

`HEADROOM_CONFIG_DIR` changes only the global policy directory. For example,
`HEADROOM_CONFIG_DIR=/etc/headroom` reads `/etc/headroom/tool_policy.json`.
`HEADROOM_WORKSPACE_DIR` changes the audit/cache root and, when
`HEADROOM_CONFIG_DIR` is unset, changes global policy lookup to
`<HEADROOM_WORKSPACE_DIR>/config/tool_policy.json`.

### Document fields

| Field | Required | Description |
|---|---:|---|
| `version` | No | Must be `1` when present. |
| `mode` | No | `enforce` (default) or `report_only`. |
| `defaultAction` | No | `allow` (default) or `deny`. |
| `rules` | Yes | Ordered list; first matching rule wins. |

### Rule fields

| Field | Applies to | Description |
|---|---|---|
| `id` | all | Stable audit identifier. Generated when omitted. |
| `scope` | all | `tool_call`, `shell`, or `http`. |
| `action` | all | `allow`, `deny`, or `require_approval`. |
| `reason` | all | Operator-facing explanation. |
| `tool` | tool/shell | Tool name or list of names. |
| `command` | shell | Executable basename, path, or list. |
| `argsPattern` | tool/shell | JavaScript regular expression over canonical arguments or command text. |
| `cwdPattern` | tool/shell | JavaScript regular expression over normalized effective cwd. |
| `envKeys` | tool/shell | Required environment key names; values are never matched or audited. |
| `domain` | HTTP | Exact hostname or `*.example.com`. |
| `urlPattern` | HTTP | JavaScript regular expression over the full URL. |

`require_approval` currently fails closed because OpenCode does not expose an
interactive approval callback to this plugin.

### Shell safety model

The shell evaluator supports direct, statically identifiable executable invocations,
including common wrappers such as `sudo` and `env`. Under a restrictive policy, it
fails closed on grammar that can construct or hide an executable at runtime:

- executable-name escaping;
- Bash, PowerShell, or cmd variable expansion;
- command and process substitution;
- computed invocation;
- shell control structures;
- `eval`, `exec`, `source`, and equivalent commands.

This deliberately favors a false denial over authorizing an executable that the policy
could not identify.

## Authority and terminal outcomes

OpenCode's native tool lifecycle is the authoritative path:

1. `tool.execute.before` evaluates policy.
2. The decision is bound to caller, session, call/task, tool, normalized cwd, and a
   hash of canonical arguments.
3. A denied call emits `effect: "blocked"` and the plugin throws.
4. An allowed call has its argument graph frozen before execution.
5. A matching `tool.execute.after` emits `effect: "allowed"`.

OpenCode currently invokes `tool.execute.after` only after successful tool execution;
it has no error hook. The plugin therefore retains allowed preflights in a bounded,
expiring store:

- default capacity: 1,024 preflights;
- default lifetime: 5 minutes;
- capacity eviction: `effect: "unknown", reason: "capacity_evicted"`;
- timeout: `effect: "unknown", reason: "postflight_timeout"`;
- mismatched final binding: `effect: "unknown", reason: "postflight_mismatch"`;
- ambiguous reuse of a retired argument graph:
  `effect: "unknown", reason: "ambiguous_reused_call"`;
- plugin shutdown: `effect: "unknown", reason: "plugin_disposed"`;
- reused session/call identity: `effect: "unknown", reason: "call_replaced"`.

`unknown` is a terminal audit outcome, not proof that execution succeeded or failed.
It distinguishes an unobserved postflight from a bypassed or missing record without
claiming an effect the host did not expose.

The bounds can be tuned in the wrapper:

```ts
return HeadroomPlugin(input, {
  pendingPreflightTtlMs: 5 * 60 * 1000,
  maxPendingPreflights: 1024,
});
```

Only an acknowledgement with the same `decisionId`, `requestHash`, and complete
binding belongs to a decision.

HTTP and child-process interception is labeled `authority: "advisory"`. It provides
defense in depth for activity originating inside the plugin process, but it cannot
prove that an external executor prevented an effect and never emits an enforcement
acknowledgement.

## Troubleshooting

### Policy is not loading

Check sources in precedence order. A machine-global policy shadows repository policy.
Validate the selected file as JSON and restart OpenCode after changing launch
environment variables.

### Commands fail with "dynamic or escaped shell execution"

The command uses grammar that cannot be statically authorized. Replace it with direct
OpenCode tool calls or split it into simple executable invocations. Do not add a broad
allow rule to bypass the parser.

### Remote policy fails closed

Check HTTPS certificate validity, endpoint reachability, response size, JSON schema,
and credential scope. Confirm the endpoint does not redirect. Restore service or
correct configuration; an expired cache is never treated as permission to continue.

### Many terminal outcomes are `unknown`

`postflight_timeout` usually indicates tool failure, cancellation, host shutdown, or a
missing OpenCode postflight. `capacity_evicted` indicates sustained outstanding calls;
investigate host behavior before increasing the limit. Correlate by `sessionID`,
`callID`, and `decisionId`.

### Another workspace is rejected

One Node process cannot safely host incompatible global transport policies. Run the
workspaces in separate OpenCode processes or give them the same policy context.

## Provider config helpers

Use these helpers to generate OpenCode provider configuration that routes a `headroom`
provider through a local proxy:

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

## Retrieve and compression helpers

```ts
import {
  compressWithHeadroom,
  createHeadroomRetrieveTool,
} from "headroom-opencode";

const retrieve = createHeadroomRetrieveTool({
  proxyBaseUrl: "http://127.0.0.1:8787",
});
const original = await retrieve.execute({
  hash: "0123456789abcdef01234567",
});

const compressed = await compressWithHeadroom(
  [{ role: "user", content: "Summarize this file" }],
  { model: "gpt-4o", proxyUrl: "http://127.0.0.1:8787" },
);

console.log(original);
console.log(`Saved ${compressed.tokensSaved} tokens`);
```

## Environment variables

| Variable | Description |
|---|---|
| `HEADROOM_PROXY_URL` | Proxy URL used by `HeadroomPlugin`. |
| `HEADROOM_BASE_URL` | Backward-compatible proxy URL fallback when `HEADROOM_PROXY_URL` is unset. |
| `HEADROOM_TOOL_POLICY_JSON` | Inline JSON policy document. |
| `HEADROOM_TOOL_POLICY_PATH` | Path to a policy JSON file. |
| `HEADROOM_TOOL_POLICY_URL` | HTTPS endpoint returning policy JSON. |
| `HEADROOM_TOOL_POLICY_TOKEN` | Optional bearer token for the policy endpoint. |
| `HEADROOM_TOOL_POLICY_REFRESH_SECONDS` | Remote refresh interval, 300-3600 seconds. |
| `HEADROOM_CONFIG_DIR` | Global policy directory; reads `<dir>/tool_policy.json`. |
| `HEADROOM_WORKSPACE_DIR` | Audit/cache root and fallback global policy root. |
| `HEADROOM_STATELESS` | Disables disk audit and cache writes when truthy. |
| `OPENCODE_CONFIG_CONTENT` | Generated OpenCode provider/model/MCP configuration. |

## License

Apache-2.0
