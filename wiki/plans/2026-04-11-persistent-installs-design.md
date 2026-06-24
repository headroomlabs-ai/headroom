# Persistent Deployments / Installs Design

## Problem

Headroom already supports session-oriented usage through `headroom proxy`, `headroom wrap ...`, and the Docker-native wrapper scripts, but there is no first-class way to install Headroom as a durable background runtime. That leaves users to hand-roll launch agents, services, scheduled tasks, or Docker restart policies, and it keeps direct tool usage (`claude`, `codex`, `copilot`, `openclaw`, etc.) tied to explicit `wrap` commands.

The new feature should make Headroom deployable as a persistent local runtime while keeping the existing on-demand and wrapped flows intact.

## Goals

- Support these runtime/install modes as one coherent system:
  - Persistent Service
  - Persistent Task
  - Persistent Docker
  - On-Demand CLI (Python)
  - On-Demand CLI (Docker)
  - Wrapped (Python)
  - Wrapped (Docker)
- Support install target selection modes:
  - Auto-Detect
  - All
  - Manual Select
- Support configuration scopes:
  - Provider
  - User
  - System
- Keep `wrap` idempotent and persistent-aware.
- Preserve local defaults such as `localhost:8787`.

## Architecture

Introduce a new shared deployment subsystem under `headroom.install`.

Core model:

- `execution_mode`: `persistent | on_demand | wrapped`
- `runtime_kind`: `python | docker`
- `supervisor_kind`: `service | task | none`

These three axes normalize all seven user-facing runtime modes without duplicating logic across CLI commands, install scripts, and platform-specific deployment adapters.

The subsystem centers on a persisted deployment manifest in `~/.headroom/deploy/` that records:

- resolved proxy configuration
- runtime type
- supervisor type
- configured tool targets
- applied config mutations
- generated artifact paths
- health URL and port

## Command model

Add a new public `headroom install` group:

- `headroom install apply`
- `headroom install status`
- `headroom install start`
- `headroom install stop`
- `headroom install restart`
- `headroom install remove`

Add hidden helper commands for artifact runners and health recovery:

- `headroom install agent run --profile <name>`
- `headroom install agent ensure --profile <name>`

Platform supervisors should register the hidden agent entrypoint rather than raw `headroom proxy ...` so restart, health polling, and manifest handling live in one place.

## Runtime adapters

- `PythonRuntimeAdapter`: launches `headroom proxy` directly.
- `DockerRuntimeAdapter`: launches a detached or foreground Docker container with the existing host mounts and loopback-only port publishing.

Persistent Docker uses the same deployment manifest and status semantics as service/task installs, but the child runtime is Docker-managed rather than OS-supervised.

## Supervisor adapters

- Linux
  - Service: systemd unit
  - Task: cron watchdog + reboot/start entry
- macOS
  - Service: LaunchDaemon / LaunchAgent variant
  - Task: launchd user agent or cron-style watchdog where appropriate
- Windows
  - Service: Windows Service wrapper
  - Task: Scheduled Task startup + periodic health-check task

Each adapter renders artifacts into `~/.headroom/deploy/` and stores enough metadata for clean removal.

## Tool target configuration

Provider-level configuration should be target-specific and reversible.

Initial target adapters:

- Claude Code
  - write `env` keys into Claude settings JSON where appropriate
- Codex
  - manage a marked block or targeted settings in `~/.codex/config.toml`
- Copilot CLI
  - configure BYOK environment surfaces using persistent env strategy
- OpenClaw
  - reuse existing OpenClaw config/plugin merge logic where possible
- Aider / Cursor
  - use env-based integration first, with tool-specific config only where a stable supported surface exists

All non-marker edits must store previous values in the deployment manifest so uninstall removes only Headroom-managed changes.

## Wrap behavior

`headroom wrap ...` should consult the active deployment manifest before starting a new proxy. If a compatible persistent deployment is present and healthy, `wrap` should reuse it and only perform any remaining tool-specific preparation. If the deployment exists but is unhealthy, `wrap` should attempt to recover it through the install subsystem before falling back to an ephemeral proxy.

## Docs strategy

The public docs should be reframed around:

- runtime mode
- lifecycle mode
- configuration scope
- direct-use vs `wrap`

Create a new top-level guide at `docs/persistent-installs.md` and update the existing Docker install, proxy, CLI, getting-started, quickstart, configuration, troubleshooting, and integration docs to reflect the broader runtime story.

## Risks

- Windows service installation is the highest-risk platform path and needs strong test isolation.
- Provider-specific config mutation must remain conservative and reversible.
- `/health` should gain deployment metadata without breaking the existing `config` payload shape expected by current tests and docs.
