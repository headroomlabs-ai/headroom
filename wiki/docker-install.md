# Docker-Native Install

Run Headroom without installing Python or Node.js on the host. The install scripts add a native `headroom` wrapper that keeps **Headroom itself** in Docker while orchestrating the rest of your workflow on the host OS.

## One-line install

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.sh | bash
```

### macOS (bash 4.3+)

```bash
curl -fsSL https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.sh | "$(brew --prefix bash)/bin/bash"
```

Stock `/bin/bash` on macOS is 3.2, so install a newer bash first (for example via Homebrew) and run the installer with that shell. The installed wrapper pins that same bash interpreter so later invocations stay on the supported runtime.

### Windows PowerShell

```powershell
irm https://raw.githubusercontent.com/chopratejas/headroom/main/scripts/install.ps1 | iex
```

## What the installer does

1. Verifies Docker is installed and available.
2. Pulls `ghcr.io/chopratejas/headroom:latest` by default, or reuses / pulls `HEADROOM_DOCKER_IMAGE` when you set a custom image override.
3. Installs a `headroom` wrapper into `~/.local/bin` or `~/bin`.
4. Updates shell startup files so the wrapper directory is on `PATH`.

The wrapper keeps Headroom inside Docker and mounts host state back into the container so native behavior stays consistent:

- project workspace -> `/workspace`
- `~/.headroom`
- `~/.claude`
- `~/.codex`
- `~/.gemini`

Port `8787` stays the default, so `http://localhost:8787` works the same way as a native install.

Published releases also push versioned GHCR tags such as `ghcr.io/chopratejas/headroom:0.5.26`, and those images are built with the same synced package version used for the matching PyPI and npm release.

## How the wrapper behaves

### Native Headroom commands

These run directly inside the container:

```bash
headroom proxy
headroom learn
headroom mcp install
headroom memory list
```

For `proxy`, the wrapper publishes the selected port back to the host:

```bash
docker run --rm -it \
  -p 8787:8787 \
  -v "$PWD:/workspace" \
  -w /workspace \
  ghcr.io/chopratejas/headroom:latest \
  headroom proxy --host 0.0.0.0 --port 8787
```

### `wrap` commands

`wrap` is host-oriented in Docker-native mode:

- the wrapper starts the Headroom proxy in Docker
- container-side prep writes Headroom config, memory, and `rtk` guidance into mounted host files
- the target CLI itself is launched on the host by the wrapper

Supported host wrap flows:

- `headroom wrap claude`
- `headroom wrap codex`
- `headroom wrap aider`
- `headroom wrap cursor`
- `headroom wrap openclaw`
- `headroom unwrap openclaw`

OpenClaw remains host-native in Docker-native mode:

- the host must already have the `openclaw` CLI installed
- `headroom wrap openclaw` installs/configures the Headroom plugin through the host `openclaw` CLI
- plugin auto-start still launches the installed host `headroom` wrapper from `PATH`, which then runs Headroom in Docker
- local plugin source mode (`--plugin-path`) is also supported, but it may require host `npm` when build steps are needed

## Persistent Docker lifecycle from the native wrapper

The Docker-native `headroom` wrapper now exposes the persistent Docker lifecycle directly:

```bash
headroom install apply --profile default --preset persistent-docker
headroom install status
headroom install restart
headroom install remove
```

In Docker-native mode this surface is intentionally scoped to **persistent-docker**:

- supported: `apply`, `status`, `start`, `stop`, `restart`, `remove`
- supported flags: `--profile`, `--port`, `--backend`, `--anyllm-provider`, `--region`, `--mode`, `--memory`, `--no-telemetry`, `--image`
- not supported: `persistent-service`, `persistent-task`, or provider/user/system mutation flags such as `--scope`, `--providers`, and `--target`

Those broader lifecycle and config-mutation flows still belong to the Python-native `headroom install ...` command.

Persistent Docker deployments launched by the wrapper also tag the proxy process with deployment metadata, so `/health` reports the active `profile`, `preset`, `runtime`, `supervisor`, and `scope` the same way the Python install subsystem does.

## Docker Compose support

Use `docker/docker-compose.native.yml` when you want an explicit compose-managed proxy or CLI shell, or when you prefer compose over the native wrapper's `headroom install ...` surface.

### Persistent Docker runtime

The `proxy` service now uses `restart: unless-stopped`, so compose can act as the always-on Docker runtime for Headroom:

```bash
export HEADROOM_HOST_HOME="$HOME"
export HEADROOM_WORKSPACE="$PWD"
docker compose -f docker/docker-compose.native.yml up -d proxy
```

```powershell
$env:HEADROOM_HOST_HOME = $HOME
$env:HEADROOM_WORKSPACE = (Get-Location).Path
docker compose -f docker/docker-compose.native.yml up -d proxy
```

This remains a supported persistent-Docker path when you want the proxy managed explicitly through Compose instead of the installed wrapper.

#### `HEADROOM_WORKSPACE` vs `HEADROOM_WORKSPACE_DIR`

These are two different variables — both are set by the compose file,
and both are retained for backward compatibility:

- **`HEADROOM_WORKSPACE`** (host-side) is the directory the compose file
  bind-mounts into the container as `/workspace`. It behaves like CWD
  in a native (non-Docker) run.
- **`HEADROOM_WORKSPACE_DIR`** (inside-the-container) is the canonical
  Headroom state root — part of the [filesystem contract][fs]
  introduced in issue #175. The compose file sets it to
  `/tmp/headroom-home/.headroom` so the proxy resolves savings, logs,
  TOIN, and memory under the bind-mounted `${HOME}/.headroom`.

You do not need to set `HEADROOM_WORKSPACE_DIR` manually when using the
shipped compose file — it is already in the `environment:` block.

[fs]: filesystem-contract.md

### macOS / Linux

```bash
export HEADROOM_HOST_HOME="$HOME"
export HEADROOM_WORKSPACE="$PWD"
docker compose -f docker/docker-compose.native.yml up proxy
```

### Windows PowerShell

```powershell
$env:HEADROOM_HOST_HOME = $HOME
$env:HEADROOM_WORKSPACE = (Get-Location).Path
docker compose -f docker/docker-compose.native.yml up proxy
```

You can also run one-off CLI commands through compose:

```bash
docker compose -f docker/docker-compose.native.yml run --rm cli learn
docker compose -f docker/docker-compose.native.yml run --rm cli mcp install
```

## Environment passthrough

The wrapper forwards Headroom and provider environment variables into the container, including common prefixes such as:

- `HEADROOM_`
- `ANTHROPIC_`
- `OPENAI_`
- `GEMINI_`
- `AWS_`
- `GOOGLE_` / `GOOGLE_CLOUD_`
- `AZURE_`
- `OTEL_`

That keeps provider auth and runtime config working without maintaining a separate env file for the container.

## Notes

- Docker is the only required Headroom runtime dependency on the host.
- Wrapped tools like Claude Code, Codex CLI, Aider, and Cursor still run on the host when you use `headroom wrap ...`.
- The install scripts are idempotent: rerunning them refreshes the wrapper and image without duplicating shell profile blocks.
- For persistent service and task installs, use the Python-native `headroom install ...` workflow described in [Persistent Installs](persistent-installs.md).
- For Docker-native `headroom install ...`, the wrapper persists its profile manifest under `~/.headroom/deploy/<profile>/`.
