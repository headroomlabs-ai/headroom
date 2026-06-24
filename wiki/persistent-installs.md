# Persistent Installs

Headroom can now be installed as a durable local runtime instead of only being started ad hoc with `headroom proxy` or `headroom wrap ...`.

Use the Python-native `headroom install` CLI when you want supported tools to keep talking to an always-on proxy at `http://127.0.0.1:8787` and have `wrap` reuse or recover that deployment instead of starting a second ephemeral proxy.

## Runtime matrix

| Mode | What stays running | Primary entrypoint |
|---|---|---|
| Persistent Service | Native background service | `headroom install apply --preset persistent-service` |
| Persistent Task | Scheduled watchdog + on-demand runner | `headroom install apply --preset persistent-task` |
| Persistent Docker | Restartable Docker container | `headroom install apply --preset persistent-docker` |
| On-Demand CLI (Python) | Nothing after command exits | `headroom proxy` |
| On-Demand CLI (Docker) | Nothing after container exits | Docker-native wrapper / compose CLI |
| Wrapped (Python) | Proxy lasts for wrapped session | `headroom wrap ...` |
| Wrapped (Docker) | Containerized proxy + host tool session | Docker-native wrapper |

## Quick examples

### Persistent service on the local machine

```bash
headroom install apply --preset persistent-service --providers auto
headroom install status
```

This installs a background service on the current machine, applies persistent tool wiring, and keeps the proxy healthy on port `8787`.

### Persistent watchdog task

```bash
headroom install apply --preset persistent-task --providers manual --target claude --target codex
```

This installs a scheduled recovery path instead of a traditional always-running service.

### Persistent Docker

```bash
headroom install apply --preset persistent-docker --scope user --providers auto
```

This uses Docker's restart policy instead of an OS supervisor.

If you are using the Docker-native host wrapper instead of a Python install, you can now use `headroom install apply|status|start|stop|restart|remove` for the `persistent-docker` preset directly from the installed wrapper. Service/task installs and provider/user/system mutation flows still belong to the Python-native CLI.

## Command surface

```text
headroom install apply
headroom install status
headroom install start
headroom install stop
headroom install restart
headroom install remove
```

`apply` creates or updates a named deployment profile, stores its manifest under `~/.headroom/deploy/<profile>/manifest.json`, applies reversible configuration changes, and starts the selected runtime.

## Presets and runtime kinds

### Presets

- `persistent-service` -> native service supervisor
- `persistent-task` -> scheduled watchdog / recovery supervisor
- `persistent-docker` -> Docker restart policy with no extra OS supervisor

### Runtime kinds

- `--runtime python` runs `headroom proxy` directly
- `--runtime docker` runs Headroom inside Docker while keeping the deployment managed locally

For `persistent-docker`, the runtime is always Docker.

## Configuration scopes

| Scope | What changes |
|---|---|
| `provider` | Tool-specific config surfaces where Headroom can make a precise reversible edit |
| `user` | User-level shell or environment surfaces |
| `system` | Machine-wide shell or environment surfaces |

### Provider scope today

Provider scope is intentionally conservative. The current direct adapters are:

- Claude Code -> `~/.claude/settings.json` `env`
- Codex -> managed block in `~/.codex/config.toml`
- OpenClaw -> existing `wrap openclaw` / `unwrap openclaw` flow

For Copilot, Aider, Cursor, and broader env-driven setups, prefer `--scope user` or `--scope system`.

## Provider selection

| Option | Meaning |
|---|---|
| `--providers auto` | Detect supported tools on the host and configure the best available defaults |
| `--providers all` | Configure all known targets |
| `--providers manual --target ...` | Configure only the named tools |

Examples:

```bash
headroom install apply --providers auto
headroom install apply --providers all --scope user
headroom install apply --providers manual --target claude --target copilot
```

## Health and wrap behavior

Persistent deployments publish the same `readyz` and `health` endpoints as ad hoc proxy runs.

`/health` now also exposes deployment metadata when the proxy was launched through the install subsystem:

```json
{
  "deployment": {
    "profile": "default",
    "preset": "persistent-service",
    "runtime": "python",
    "supervisor": "service",
    "scope": "user"
  }
}
```

The Python-native `headroom wrap ...` flow checks for a matching persistent deployment on the requested port before it starts a new ephemeral proxy. If an installed deployment exists but is stopped or unhealthy, it attempts to recover it first.

The Docker-native host wrapper does **not** yet reuse or recover persistent profiles automatically; it still starts a fresh proxy container unless you opt into `--no-proxy`.

## Docker-native relationship

The Docker-native host wrapper and the Python install CLI solve different layers of the runtime story:

- [Docker-Native Install](docker-install.md) -> containerized on-demand CLI, wrapped host-tool flows, and Docker-native `persistent-docker` lifecycle commands
- `headroom install ...` -> full persistent service, task, and Docker lifecycle management, including provider/user/system mutation

For a no-Python persistent Docker workflow, use the compose-managed proxy path from `docker/docker-compose.native.yml`:

```bash
export HEADROOM_HOST_HOME="$HOME"
export HEADROOM_WORKSPACE="$PWD"
docker compose -f docker/docker-compose.native.yml up -d proxy
```

That keeps `localhost:8787` stable and restarts the proxy automatically.

> **Note:** `HEADROOM_WORKSPACE` (the host-side bind-mount source used
> by the compose file) is **not** the same variable as
> `HEADROOM_WORKSPACE_DIR` (the canonical Headroom state root inside
> the container). Both are retained; the compose file sets the latter
> automatically. See [Filesystem Contract](filesystem-contract.md) for
> the full bucket model.

## Related guides

- [CLI Reference](cli.md)
- [Docker-Native Install](docker-install.md)
- [Proxy Server](proxy.md)
- [macOS LaunchAgent](macos-deployment.md)
