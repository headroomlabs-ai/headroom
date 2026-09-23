"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from headroom._subprocess import run
from headroom.mcp_registry.install import DEFAULT_PROXY_URL

from .config import HEADROOM_OPENCODE_PLUGIN, headroom_provider_entry


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode integrations."""
    return f"http://127.0.0.1:{port}/v1"


# OpenCode 2.x subcommands. Of these only the interactive ones (the root TUI,
# ``run`` and ``mini``) get ``--standalone``; the rest are management commands.
_OPENCODE_SUBCOMMANDS = frozenset(
    {
        "acp",
        "api",
        "auth",
        "debug",
        "mcp",
        "mini",
        "models",
        "plugin",
        "run",
        "serve",
        "service",
        "session",
        "stats",
        "uninstall",
        "update",
        "upgrade",
    }
)
_OPENCODE_STANDALONE_SUBCOMMANDS = frozenset({"run", "mini"})


def opencode_major_version(binary: str) -> int | None:
    """Return the major version of the ``opencode`` binary, or ``None``.

    Parses ``opencode --version`` (1.x prints ``1.18.32``, 2.x prints
    ``opencode v2.0.12``). Returns ``None`` on any failure — missing binary,
    non-zero exit, timeout, unparseable output — so callers treat the version
    as unknown and change nothing. Never raises.
    """
    try:
        proc = run([binary, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(proc, "returncode", 0) != 0:
        return None
    output = f"{getattr(proc, 'stdout', '') or ''} {getattr(proc, 'stderr', '') or ''}"
    # Not `\b`: in "v2.0.12" the "v" and "2" are both word characters.
    match = re.search(r"(?<!\d)(\d+)\.\d+\.\d+", output)
    return int(match.group(1)) if match else None


def with_opencode_standalone(args: Sequence[str], major_version: int | None) -> tuple[str, ...]:
    """Add ``--standalone`` to an interactive OpenCode 2.x launch.

    OpenCode 2.x attaches to a shared background service by default. When one
    is already running, it was started with its own environment, so the
    ``OPENCODE_CONFIG_CONTENT`` wrap sets (Headroom provider routing and the
    transport plugin) never reaches it. ``--standalone`` runs a private server
    that reads this launch's environment.

    Leaves ``args`` alone for OpenCode 1.x or an unknown version, when the user
    already chose ``--standalone`` or ``--server``, and for management
    subcommands (``models``, ``auth``, ...). The flag goes right after ``run``
    or ``mini``, and first for the root TUI.
    """
    args = tuple(args)
    if major_version is None or major_version < 2:
        return args
    if any(arg in ("--standalone", "--server") or arg.startswith("--server=") for arg in args):
        return args
    for index, arg in enumerate(args):
        if arg in _OPENCODE_SUBCOMMANDS:
            if arg not in _OPENCODE_STANDALONE_SUBCOMMANDS:
                return args
            return (*args[: index + 1], "--standalone", *args[index + 1 :])
    return ("--standalone", *args)


def headroom_opencode_plugin_path() -> str | None:
    """Return the absolute path to the built OpenCode transport plugin directory, or None.

    OpenCode 2.x only loads a configured local plugin from a directory — a file
    path is dropped with "configured plugin path must be a directory" — and
    resolves ``<dir>/server.*`` then ``<dir>/index.*``. OpenCode 1.x resolves a
    directory without ``package.json`` to its ``index.*`` as well. The
    ``index.js`` in each candidate directory default-exports a plugin object
    carrying both the 1.x ``server`` and the 2.x ``setup`` entry points.

    Resolution order:

    1. ``HEADROOM_OPENCODE_PLUGIN_PATH`` env override (a directory for 2.x).
    2. A repo-checkout build (``plugins/opencode/dist/``) — external-deps
       build, resolvable because the checkout has node_modules.
    3. The self-contained bundle shipped inside the wheel
       (``headroom/providers/opencode/_dist/``, built by
       ``npm run build:standalone``) — every dependency inlined, so it loads
       from site-packages where no node_modules exists (verified against
       opencode 1.18.32 and 2.0.12).

    Returns ``None`` only when none of the three exist, in which case wrap
    falls back to the native-provider baseURL override, which already covers
    Anthropic/OpenAI.
    """
    override = os.environ.get("HEADROOM_OPENCODE_PLUGIN_PATH", "").strip()
    if override:
        return override if Path(override).exists() else None
    # runtime.py → opencode → providers → headroom → <repo root>
    repo_candidate = Path(__file__).resolve().parents[3] / "plugins" / "opencode" / "dist"
    if (repo_candidate / "index.js").is_file():
        return str(repo_candidate)
    packaged = Path(__file__).resolve().parent / "_dist"
    return str(packaged) if (packaged / "index.js").is_file() else None


def build_opencode_config_content(
    *,
    port: int,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> dict[str, object]:
    """Build JSON payload for ``OPENCODE_CONFIG_CONTENT``.

    Two complementary routing layers (both verified against opencode 1.17):

    1. **Native-provider baseURL override** — points OpenCode's built-in
       ``anthropic`` / ``openai`` providers at the proxy. Keeps native provider
       identity (model metadata, output-token limits) and reuses the user's own
       API keys (env / ``opencode auth``); the proxy forwards upstream by path
       (``/v1/messages`` → Anthropic, ``/v1/chat/completions`` → OpenAI). This
       is the reliable always-on layer and the only one shipped pip-only
       installs need.

    2. **Transparent transport plugin** — when the local plugin is built, it is
       loaded by absolute path and patches ``fetch``/``http`` to reroute *every*
       provider's traffic through the proxy, tagging the real upstream via
       ``x-headroom-base-url``. This covers providers we don't name (Gemini,
       Copilot, custom gateways) and providers added mid-session. The plugin
       self-configures from ``HEADROOM_PROXY_URL`` (set in :func:`build_launch_env`).
       Loopback URLs are not double-routed, so it coexists with layer 1.

    ponytail: config-level ``options.baseURL`` is reliable where the env-var
    override (``ANTHROPIC_BASE_URL``) is not — verified against opencode 1.17.
    """
    base_url = proxy_base_url(port)
    config: dict[str, object] = {
        "provider": {
            "anthropic": {"options": {"baseURL": base_url}},
            "openai": {"options": {"baseURL": base_url}},
            "headroom": headroom_provider_entry(port),
        }
    }
    if include_mcp:
        proxy_url = f"http://127.0.0.1:{port}"
        mcp_entry: dict[str, object] = {
            "type": "local",
            "command": ["headroom", "mcp", "serve"],
            "enabled": True,
        }
        if proxy_url != DEFAULT_PROXY_URL:
            mcp_entry["environment"] = {"HEADROOM_PROXY_URL": proxy_url}
        config["mcp"] = {
            "headroom": mcp_entry,
        }
    if include_plugin:
        plugin_path = headroom_opencode_plugin_path()
        if plugin_path:
            # Plain absolute-path string; the plugin reads HEADROOM_PROXY_URL
            # from the launch env (build_launch_env sets it).
            config["plugin"] = [plugin_path]
    return config


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
    *,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for launching OpenCode through Headroom.

    ``OPENCODE_CONFIG_CONTENT`` carries Headroom provider/MCP/plugin config.
    Existing provider/base URL environment variables are preserved. When the
    transport plugin is loaded, ``HEADROOM_PROXY_URL`` tells it which proxy to
    route to.
    """
    env = dict(environ or os.environ)

    config_content = build_opencode_config_content(
        port=port,
        include_mcp=include_mcp,
        include_plugin=include_plugin,
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content, separators=(",", ":"))

    display = ["OPENCODE_CONFIG_CONTENT={provider: headroom}"]
    if "plugin" in config_content:
        env["HEADROOM_PROXY_URL"] = f"http://127.0.0.1:{port}"
        display.append(f"plugin={HEADROOM_OPENCODE_PLUGIN}")

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    return env, display
