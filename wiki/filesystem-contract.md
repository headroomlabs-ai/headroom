# Filesystem Contract

Headroom writes configuration, runtime state, logs, and caches to a small
set of well-known paths under the user's home directory. This page is the
source of truth for where those paths live, how to override them, and how
they behave inside Docker containers.

## Two-root model

| Variable | Default | Purpose | Typical access |
|---|---|---|---|
| `HEADROOM_CONFIG_DIR` | `~/.headroom/config` | User/admin-authored configuration (model catalogs, plugin settings, etc.) | Read-mostly |
| `HEADROOM_WORKSPACE_DIR` | `~/.headroom` | Runtime state written by the proxy and CLI (savings, logs, memory DB, telemetry, caches) | Read-write |

Both variables are recognized by the Python proxy / CLI and the npm SDK.
They are **additive** — every pre-existing per-resource env var
(`HEADROOM_SAVINGS_PATH`, `HEADROOM_TOIN_PATH`,
`HEADROOM_SUBSCRIPTION_STATE_PATH`, `HEADROOM_MODEL_LIMITS`, ...)
continues to work with identical semantics.

## Precedence

For every per-resource helper, resolution follows this order:

```
explicit argument
    │ falls through when None/""
    ▼
per-resource env var (e.g. HEADROOM_SAVINGS_PATH)
    │ falls through when unset/blank
    ▼
derived from canonical root
    │ e.g. ${HEADROOM_WORKSPACE_DIR}/proxy_savings.json
    ▼
default (e.g. ~/.headroom/proxy_savings.json)
```

Examples:

- `HEADROOM_WORKSPACE_DIR=/mnt/state` → savings land at
  `/mnt/state/proxy_savings.json` unless `HEADROOM_SAVINGS_PATH` overrides.
- `HEADROOM_SAVINGS_PATH=/custom/savings.json` always wins, even when
  `HEADROOM_WORKSPACE_DIR` is set.
- Unset both and the default is `~/.headroom/proxy_savings.json`.

## Bucket assignments

### Workspace bucket (`HEADROOM_WORKSPACE_DIR`)

| Resource | Default path | Legacy env var |
|---|---|---|
| Proxy savings ledger | `${WORKSPACE_DIR}/proxy_savings.json` | `HEADROOM_SAVINGS_PATH` |
| TOIN telemetry JSON | `${WORKSPACE_DIR}/toin.json` | `HEADROOM_TOIN_PATH` |
| Subscription tracker state | `${WORKSPACE_DIR}/subscription_state.json` | `HEADROOM_SUBSCRIPTION_STATE_PATH` |
| Memory SQLite | `${WORKSPACE_DIR}/memory.db` | CLI `--memory-db-path` |
| Native memory directory | `${WORKSPACE_DIR}/memories/` | `MemoryConfig.native_memory_dir` |
| License cache | `${WORKSPACE_DIR}/license_cache.json` | — |
| Session stats JSONL | `${WORKSPACE_DIR}/session_stats.jsonl` | — |
| Memory sync state | `${WORKSPACE_DIR}/sync_state.json` | — |
| Memory bridge state | `${WORKSPACE_DIR}/bridge_state.json` | — |
| Proxy log directory | `${WORKSPACE_DIR}/logs/` | — |
| HTTP 400 debug dumps | `${WORKSPACE_DIR}/logs/debug_400/` | — |
| Vendored `rtk` binary | `${WORKSPACE_DIR}/bin/rtk[.exe]` | — |
| Deployment profiles | `${WORKSPACE_DIR}/deploy/` | — |
| Beacon lock file | `${WORKSPACE_DIR}/.beacon_lock_<port>` | — |

### Config bucket (`HEADROOM_CONFIG_DIR`)

| Resource | Default path | Legacy env var |
|---|---|---|
| Models catalog | `${CONFIG_DIR}/models.json` | `HEADROOM_MODEL_LIMITS` (content override) |
| Plugin settings | `${CONFIG_DIR}/plugins/<name>/...` | — |

### Backward compatibility — models.json

`models.json` historically lived at `~/.headroom/models.json` (i.e. in the
workspace root, not in `config/`). For a seamless migration the Python
providers check **both** locations in this order:

1. `${HEADROOM_CONFIG_DIR}/models.json` (new canonical location)
2. `${HEADROOM_WORKSPACE_DIR}/models.json` (legacy fallback)

Existing installs continue to work unchanged. New installs are encouraged
to put `models.json` in the config bucket.

## Plugin authors

Two helpers give plugins isolated, per-plugin directories under both
roots:

### Python

```python
from headroom import paths

cfg_dir = paths.plugin_config_dir("my-plugin")
# → ~/.headroom/config/plugins/my-plugin

state_dir = paths.plugin_workspace_dir("my-plugin")
# → ~/.headroom/plugins/my-plugin

cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "settings.json").write_text("{}")
```

### npm SDK

```typescript
import { pluginConfigDir, pluginWorkspaceDir } from "@headroom/sdk";

const cfgDir = pluginConfigDir("my-plugin");
const stateDir = pluginWorkspaceDir("my-plugin");
```

Plugin-author helpers reject names containing `/` or `\` to keep the
namespace flat.

## Docker naming overlap: `HEADROOM_WORKSPACE` vs `HEADROOM_WORKSPACE_DIR`

These are **two different variables** with different semantics, both
retained for backward compatibility:

| Variable | Scope | Meaning |
|---|---|---|
| `HEADROOM_WORKSPACE` | Host-side (Docker) | Directory to bind-mount into the container as `/workspace` (equivalent to CWD in native runs). Used by `docker-compose.native.yml`. |
| `HEADROOM_WORKSPACE_DIR` | Inside-the-container | Canonical Headroom state root. Resolves to `/tmp/headroom-home/.headroom` inside the official container image, which in turn bind-mounts to `${HOME}/.headroom` on the host. |

The official Docker bootstrap (compose file, `scripts/install.sh`, and the
Python `install` command) sets `HEADROOM_WORKSPACE_DIR` and
`HEADROOM_CONFIG_DIR` inside the container so the proxy resolves state to
the bind-mounted path without any user action.

## Project-scoped `.headroom/` directories

A few code paths deliberately use **project-local** `.headroom/` paths
resolved relative to the current working directory rather than the
canonical workspace root:

- `headroom/proxy/server.py` — project-scoped memory DB default
- `headroom/memory/mcp_server.py` — project-scoped memory DB default
- `headroom/cli/wrap.py` — project-scoped memory and hook artifacts

These **do not obey** `HEADROOM_WORKSPACE_DIR`. This is intentional: it
preserves the "project memory lives in the project directory" invariant
documented in [memory.md](memory.md). Users who want a single centrally
located memory store can pass `--memory-db-path <path>` explicitly or set
the path via the plugin API.

## Legacy per-resource env vars

Every legacy env var continues to work with its original semantics (raw
string in, raw string out — no tilde expansion, no path-separator
normalization), ensuring byte-for-byte backward compatibility.

Full list:

- `HEADROOM_SAVINGS_PATH`
- `HEADROOM_TOIN_PATH`
- `HEADROOM_SUBSCRIPTION_STATE_PATH`
- `HEADROOM_MODEL_LIMITS` (content override — JSON string or file path)

## See also

- [configuration.md](configuration.md) — general configuration reference
- [docker-install.md](docker-install.md) — Docker install details
- [persistent-installs.md](persistent-installs.md) — persistent
  deployment profiles
- [memory.md](memory.md) — memory-system paths and project scoping
