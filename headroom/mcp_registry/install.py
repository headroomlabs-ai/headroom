"""Top-level orchestration: register Headroom MCP across detected agents."""

from __future__ import annotations

from collections.abc import Iterable

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .claude import ClaudeRegistrar
from .codex import CodexRegistrar
from .opencode import OpenCodeRegistrar

#: Default proxy URL used when none is given.
DEFAULT_PROXY_URL = "http://127.0.0.1:8787"


def get_all_registrars() -> list[MCPRegistrar]:
    """Return one instance of every registrar implemented today.

    The list grows as we add adapters for Cursor, Continue, Cline, etc.
    """
    return [ClaudeRegistrar(), CodexRegistrar(), OpenCodeRegistrar()]


def build_headroom_spec(proxy_url: str = DEFAULT_PROXY_URL) -> ServerSpec:
    """Construct the canonical :class:`ServerSpec` for the headroom server.

    The spec is identical across agents — every JSON/TOML registrar
    serializes the same shape into its own format.
    """
    env: dict[str, str] = {}
    if proxy_url and proxy_url != DEFAULT_PROXY_URL:
        env["HEADROOM_PROXY_URL"] = proxy_url
    return ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env=env,
    )


def build_serena_spec(context: str) -> ServerSpec:
    """Construct the canonical Serena MCP server spec for an agent context."""
    return ServerSpec(
        name="serena",
        command="uvx",
        args=(
            "--from",
            "git+https://github.com/oraios/serena",
            "serena",
            "start-mcp-server",
            "--project-from-cwd",
            "--context",
            context,
        ),
    )


def install_everywhere(
    proxy_url: str = DEFAULT_PROXY_URL,
    *,
    agents: Iterable[str] | None = None,
    force: bool = False,
    registrars: Iterable[MCPRegistrar] | None = None,
) -> dict[str, RegisterResult]:
    """Install the headroom MCP server into every detected agent.

    Args:
        proxy_url: URL the MCP server should contact for retrieval.
        agents: If given, only install into agents whose ``name`` matches.
        force: Pass through to each registrar — overwrites mismatched config.
        registrars: Inject a custom registrar list (test seam).

    Returns:
        Dict keyed by registrar name. Includes :attr:`RegisterStatus.NOT_DETECTED`
        entries for agents we know about that aren't installed locally.
    """
    spec = build_headroom_spec(proxy_url)
    selected = list(registrars) if registrars is not None else get_all_registrars()

    if agents is not None:
        agent_set = set(agents)
        selected = [r for r in selected if r.name in agent_set]

    results: dict[str, RegisterResult] = {}
    for registrar in selected:
        if not registrar.detect():
            results[registrar.name] = RegisterResult(
                RegisterStatus.NOT_DETECTED,
                f"{registrar.display_name} not found on this system",
            )
            continue
        results[registrar.name] = registrar.register_server(spec, force=force)

    return results
