# Headroom for Pi and Oh My Pi

<p align="center">
  <img src="banner.png" alt="HEADROOM — The context compression layer for AI agents" width="820">
</p>

Provider-independent context compression for [Pi](https://github.com/earendil-works/pi) and [Oh My Pi](https://github.com/can1357/oh-my-pi). The extension prepares large tool results in the background, substitutes only previously prepared results in the model-facing context, and preserves the host session transcript byte-for-byte.

## Requirements

- Node.js 20 or newer
- Pi 0.80 or newer, or OMP 17.1 or newer
- A running Headroom proxy, or the Headroom CLI to launch Pi/OMP with one

The packed-host lifecycle gate runs against Pi `0.80.10`, `0.82.1`, and `0.84.1`, plus OMP `17.1.8`. These are the exact tested host versions for this release.

Headroom's official wrappers start or reuse the proxy and pass its endpoint to the extension:

```bash
uv run headroom wrap pi
uv run headroom wrap omp
```

Install the extension first as shown below. The OMP wrapper still applies its existing Anthropic inference route; `HEADROOM_PI_BASE_URL` separately enables provider-independent extension compression. For an unwrapped host session, start the proxy manually:

```bash
headroom proxy --port 8787 --stateless --no-embedding-server
```

## Durable setup

The first durable release is current-user/global only; project-local durable setup is not supported. The lifecycle test covers Headroom `0.34.0` and `0.35.0` exact extension pins, upgrades, and rollback, using isolated Pi/OMP command contracts. Packed-host CI separately covers Pi `0.80.10`, `0.82.1`, and `0.84.1`, plus OMP `17.1.8`.

After the matching extension version is published:

```bash
headroom init -g pi
headroom init -g omp
```

Both hosts share one loopback proxy, config, and scheduled task. Re-running either command is idempotent. To roll back, install the older released Headroom CLI and rerun the same commands; durable init pins the extension to that CLI's exact release version:

```bash
uv tool install --force "headroom-ai==0.34.0"
headroom init -g pi
headroom init -g omp
```

A source/dev Headroom build cannot durable-init an unpublished extension version. It fails rather than installing `latest`. The extension fails open when the shared loopback proxy is unavailable.

From a Headroom source checkout:

```bash
cd plugins/pi
npm ci
npm run build
pi --extension "$PWD/src/index.ts"
omp --extension "$PWD/src/index.ts"
```

### Persistent OMP development setup

The extension does not spawn a subprocess itself; wrappers and durable installs own proxy lifecycle. To load the extension from a source checkout in every normal OMP session and keep the local proxy running under the operating system's service supervisor:

1. Install the source dependencies from the repository root:

   ```bash
   uv sync --extra proxy
   npm --prefix plugins/pi ci
   npm --prefix plugins/pi run build
   ```

2. Add the source entry to the existing `extensions` list in `~/.omp/agent/config.yml`:

   ```yaml
   extensions:
     - /absolute/path/to/headroom/plugins/pi/src/index.ts
   ```

   A named OMP profile uses `~/.omp/profiles/<name>/agent/config.yml` instead.

3. Install and start a persistent proxy from the checkout:

   ```bash
   uv run headroom install apply \
     --preset persistent-service \
     --runtime python \
     --scope provider \
     --providers manual \
     --profile omp-dev \
     --port 8787 \
     --mode token \
     --no-telemetry
   ```

   This command starts the checkout's Python environment without changing provider configuration. On macOS, it installs a `LaunchAgent` with `RunAtLoad` and `KeepAlive`.

4. Restart OMP, then verify both layers:

   ```bash
   uv run headroom install status --profile omp-dev
   ```

   ```text
   /headroom health
   /headroom status
   ```

OMP loads TypeScript extension changes when a new session starts. Restart the persistent proxy after changing Python source:

```bash
uv run headroom install restart --profile omp-dev
```

Remove the development service with `uv run headroom install remove --profile omp-dev`; remove the source entry from OMP's `config.yml` separately.

The default configuration needs no file or environment variables. New sessions connect to `http://127.0.0.1:8787` and fail open: if Headroom is unavailable, original tool results continue to the model unchanged.

## Remove

```bash
headroom init -g remove pi
headroom init -g remove omp
```

Removal uninstalls only packages recorded as Headroom-owned. Removing one host preserves the shared proxy, config, and task for the other. Removing the final native host restores or removes owned config and tears down its runtime state. Durable OMP operations never touch `models.yml` or its wrapper backup: OMP's native compression remains separate from `headroom wrap omp` Anthropic inference routing and `headroom unwrap omp` cleanup.

## Verify

In Pi or OMP:

```text
/headroom health
/headroom status
```

A healthy default setup reports `Headroom online` in the status footer until a compression is accepted, then:

```text
Headroom saved 8,593 tokens this session
```

That number is this Pi/OMP session's accepted savings, not a project lifetime total and not the last model request. It updates when a large unprotected tool result is accepted, not after every chat. `/reload` and `/resume` restore it from a sidecar next to the Pi session file. `/new` starts at zero.

`/headroom status` still shows endpoint, last transform, and last error.

## Commands

| Command | Purpose |
| --- | --- |
| `/headroom` | Show compact session status (same text as the footer). |
| `/headroom status` | Show health, endpoint, thresholds, queue, cache, latest transform, and last error. |
| `/headroom stats` | Separate prepared-entry counters from substitutions and savings applied to the latest transform. |
| `/headroom health` | Run an explicit proxy health check. |
| `/headroom on` | Enable substitution for the current session. |
| `/headroom off` | Disable substitution for the current session. |

Prepared counters advance when Headroom accepts a background compression result; they represent savings potential. Latest-transform counters describe only substitutions applied to the most recent model-facing context. Neither counter is provider billing data, and model-inference requests that bypass the proxy remain absent from Headroom's inference-proxy statistics.

When compressed context contains a `Retrieve more: hash=...` marker, the model can call `headroom_retrieve` to recover the exact source content associated with that marker. A local bounded LRU cache serves retrieval first; a cache miss falls back to the proxy's `/v1/retrieve` endpoint.

## Configuration

Optional JSON file:

```text
~/.headroom/integrations/pi-extension.json
```

Example local configuration:

```json
{
  "enabled": true,
  "baseUrl": "http://127.0.0.1:8787",
  "allowRemote": false,
  "remoteHosts": [],
  "minContextTokens": 20000,
  "minResultChars": 4000,
  "protectRecentToolResults": 2,
  "protectedTools": ["read", "edit", "write", "ask", "todo", "headroom_retrieve"],
  "maxCacheBytes": 67108864
}
```

Environment variables override file values:

| Variable | Value |
| --- | --- |
| `HEADROOM_PI_ENABLED` | `true`, `false`, `1`, or `0` |
| `HEADROOM_PI_BASE_URL` | HTTP or HTTPS URL |
| `HEADROOM_PI_ALLOW_REMOTE` | `true`, `false`, `1`, or `0` |
| `HEADROOM_PI_REMOTE_HOSTS` | Comma-separated exact hostnames |
| `HEADROOM_PI_MIN_CONTEXT_TOKENS` | Positive integer |
| `HEADROOM_PI_MIN_RESULT_CHARS` | Positive integer |
| `HEADROOM_PI_PROTECT_RECENT_TOOL_RESULTS` | Positive integer |
| `HEADROOM_PI_PROTECTED_TOOLS` | Comma-separated tool names |
| `HEADROOM_PI_MAX_CACHE_BYTES` | Positive integer |

Invalid fields fall back individually and appear under `config warnings` in `/headroom status`.

### Remote proxy opt-in

Non-loopback endpoints require both explicit remote enablement and an exact hostname allowlist. Redirects are rejected.

```json
{
  "baseUrl": "https://headroom.internal.example:9443",
  "allowRemote": true,
  "remoteHosts": ["headroom.internal.example"]
}
```

The allowlist contains hostnames only, without schemes, ports, paths, credentials, or wildcards. `/headroom status` shows the effective endpoint and allowlist. Prefer TLS and network-level authentication for remote deployments; the extension does not transmit provider credentials.

## Compression policy

A result becomes eligible only when all of these are true:

- Headroom is enabled for the session.
- The active model context is at least `minContextTokens`.
- The result contains exactly one text block with at least `minResultChars` characters.
- The tool is not protected.
- The result is older than the newest `protectRecentToolResults` tool results.

Compression is asynchronous and cache-only on the context path. The `context` event performs no network, subprocess, or disk I/O. Unknown models remain eligible and are labeled `unknown`; provider/model switches do not invalidate prepared entries.

## Failure behavior

- Proxy unavailable or unhealthy: keep raw context and retry health checks with bounded backoff.
- Queue full, timeout, malformed response, or insufficient savings: reject that candidate and keep raw context.
- Retrieval cache miss: query the proxy, then return an actionable miss if unavailable.
- Session shutdown: abort queued and active work.
- Extension callback error: fail open without interrupting the host.

## Development verification

With a local Headroom proxy running on port 8787:

```bash
npm ci
npm test
npm run typecheck
npm run build
npm run test:live
npm run benchmark:live
npm run test:hosts
npm pack --dry-run
```

`test:live` validates real compression, CCR retrieval, provider-independent substitution, offline fail-open, and recovery. `benchmark:live` emits a seeded `HEADROOM_BENCHMARK_JSON` record with the fixture, policy, environment, prepared-entry metrics, latest-transform metrics, raw-history invariant, and hot-path timings. `test:hosts` packs and installs the tarball into isolated homes, then drives deterministic multi-turn lifecycle checks through both real Pi and OMP processes.

Each benchmark run reports a cache-isolating nonce. Set `HEADROOM_BENCHMARK_SEED` and `HEADROOM_BENCHMARK_NONCE` to replay an exact fixture; leaving the nonce unset prevents a prior proxy cache entry from contaminating repeated measurements.
