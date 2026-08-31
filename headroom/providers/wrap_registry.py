"""Declarative registry of env-var wrap targets.

Extension seam for ``headroom wrap``: a tool whose integration is fully
described by data — the binary to launch, which environment variables point at
the local proxy, and the upstream defaults — is registered here as a
:class:`WrapTarget` instead of a hand-written command in ``cli/wrap.py``.
``cli/wrap.py`` generates one click command per entry, so adding a new
env-var tool is a registry entry, not a new command body. Mirrors the
declarative-route seam (:mod:`headroom.providers.route_specs`): the core
defines the contract and the launch flow; the entry supplies the data.

Tools that need imperative setup (settings files, token exchange, MCP
registrars, config rendering) do not fit this contract and keep their
bespoke commands.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from headroom.providers.claude.runtime import proxy_base_url as _anthropic_proxy_base_url
from headroom.providers.codex.runtime import proxy_base_url as _openai_proxy_base_url
from headroom.proxy.project_policy import with_project_prefix

# How an EnvVar's URL is shaped. ``openai_v1`` ends in ``/v1`` (OpenAI-style
# clients append ``/chat/completions``); ``anthropic`` is a bare origin
# (Anthropic clients append ``/v1/messages``).
UrlStyle = Literal["openai_v1", "anthropic"]

_STYLE_BUILDERS = {
    "openai_v1": _openai_proxy_base_url,
    "anthropic": _anthropic_proxy_base_url,
}


@dataclass(frozen=True, slots=True)
class EnvVar:
    """One environment variable a wrap target needs pointed at the proxy.

    ``display`` controls whether the assignment is echoed in the wrap banner;
    hidden aliases (e.g. ``OPENAI_API_BASE`` next to ``OPENAI_BASE_URL``) are
    set but not shown, matching the hand-written commands they replaced.
    """

    key: str
    style: UrlStyle
    display: bool = True


@dataclass(frozen=True, slots=True)
class WrapTarget:
    """Everything needed to generate a ``headroom wrap <name>`` command."""

    name: str
    binaries: tuple[str, ...]
    install_hint: str
    env_vars: tuple[EnvVar, ...]
    help_text: str
    # Encode the launch directory as a /p/<name> base-URL prefix so the proxy
    # can attribute savings per project (for tools that can't send headers).
    project_prefix: bool = True
    # Upstream defaults handed to the proxy at startup.
    openai_api_url: str | None = None
    anthropic_api_url: str | None = None
    # Whether the generated command exposes --backend/--anyllm-provider/--region.
    backend_options: bool = True
    agent_type: str = ""
    tool_label: str = ""

    def __post_init__(self) -> None:
        if not self.agent_type:
            object.__setattr__(self, "agent_type", self.name)
        if not self.tool_label:
            object.__setattr__(self, "tool_label", self.name.upper())


def build_launch_env(
    target: WrapTarget,
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the launch environment for ``target`` routed through the proxy."""
    env = dict(environ if environ is not None else os.environ)
    display: list[str] = []
    for var in target.env_vars:
        url = _STYLE_BUILDERS[var.style](port)
        if target.project_prefix:
            url = with_project_prefix(url, project)
        env[var.key] = url
        if var.display:
            display.append(f"{var.key}={url}")
    return env, display


WRAP_TARGETS: dict[str, WrapTarget] = {
    target.name: target
    for target in (
        WrapTarget(
            name="goose",
            binaries=("goose",),
            install_hint="Install Goose: https://block.github.io/goose/",
            env_vars=(
                EnvVar("OPENAI_BASE_URL", "openai_v1"),
                EnvVar("OPENAI_API_BASE", "openai_v1", display=False),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
            ),
            project_prefix=False,
            help_text=(
                "Launch Goose (Block) CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_BASE_URL and ANTHROPIC_BASE_URL to route Goose's API calls\n"
                "through Headroom.\n"
                "\n"
                "\b\n"
                "Uninstall: there is no ``headroom unwrap goose`` subcommand — nothing is\n"
                "written to the project.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap goose                          # Start proxy + goose\n"
                "    headroom wrap goose -- session               # Start a Goose session\n"
                "    headroom wrap goose -- --provider anthropic  # Pass args to goose"
            ),
        ),
        WrapTarget(
            name="openhands",
            binaries=("openhands",),
            install_hint="Install OpenHands: https://docs.all-hands.dev/",
            env_vars=(
                EnvVar("OPENAI_BASE_URL", "openai_v1"),
                EnvVar("OPENAI_API_BASE", "openai_v1", display=False),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
                # OpenHands' generic LLM provider config reads LLM_BASE_URL.
                EnvVar("LLM_BASE_URL", "openai_v1"),
            ),
            project_prefix=False,
            help_text=(
                "Launch OpenHands CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_BASE_URL / ANTHROPIC_BASE_URL to route OpenHands' API calls\n"
                "through Headroom. Nothing is written to disk, so there is nothing to undo.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap openhands                # Start proxy + openhands\n"
                "    headroom wrap openhands -- --task ...  # Pass args to openhands"
            ),
        ),
        WrapTarget(
            name="openclaude",
            binaries=("openclaude",),
            install_hint="Install OpenClaude before running `headroom wrap openclaude`.",
            env_vars=(
                EnvVar("OPENAI_API_BASE", "openai_v1"),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
            ),
            help_text=(
                "Launch OpenClaude through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets OPENAI_API_BASE and ANTHROPIC_BASE_URL to route API calls\n"
                "through Headroom.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap openclaude               # Start proxy + openclaude\n"
                "    headroom wrap openclaude -- --help     # Pass args to openclaude"
            ),
        ),
    )
}


def get_wrap_target(name: str) -> WrapTarget:
    """Return the registered target for ``name``; raises ``KeyError`` if unknown."""
    return WRAP_TARGETS[name]
