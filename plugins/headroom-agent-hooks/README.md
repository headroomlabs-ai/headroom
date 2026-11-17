# Headroom agent hooks

This plugin exposes lightweight startup hooks for Claude Code and GitHub Copilot CLI.

The hooks resolve and call:

```bash
headroom init hook ensure
```

That hidden helper checks for a matching durable `headroom init` deployment and starts it if needed.

When `CLAUDE_PLUGIN_ROOT` points to a readable launcher, resolution order is
`HEADROOM_BIN`, `headroom` on `PATH`, then `headroom` and
`headroom.exe` in `$HOME/.local/bin`, `$HOME/.local/share/uv/tools/headroom-ai/bin`,
`${PIPX_HOME:-$HOME/.local/pipx}/venvs/headroom-ai/bin`, `/opt/homebrew/bin`, and
`/usr/local/bin`. If no executable is found, `HEADROOM_PYTHON`, `python3`, and
`python` are checked for an importable `headroom` module and invoked as
`python -m headroom.cli init hook ensure`.

Set `HEADROOM_BIN` or `HEADROOM_PYTHON` to override discovery. An absent CLI emits
one actionable diagnostic and exits successfully, so the host hook remains
nonblocking. Without a readable launcher, the hook runs bare
`headroom init hook ensure` from `PATH` with none of the discovery above. Copilot
CLI on Windows uses the manifest's `powershell` entry, which does the same.
