"""Declarative registry of env-var wrap targets.

A tool whose integration is fully described by data — the binary to launch
and which environment variables point at the local proxy — is registered here
as a :class:`WrapTarget` instead of a hand-written command in ``cli/wrap.py``.
``cli/wrap.py`` generates one ``headroom wrap <name>`` command per entry, so
adding such a tool is a registry entry, not a new command body. Mirrors the
declarative-route seam (:mod:`headroom.providers.route_specs`).

Tools that need imperative setup (settings files, token exchange, MCP
registrars, config rendering) or whose launch env is shared with
``headroom install`` (aider) keep their own commands.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from headroom.providers.claude import proxy_base_url as claude_proxy_base_url
from headroom.providers.codex import proxy_base_url as codex_proxy_base_url
from headroom.proxy.project_context import with_project_prefix

# How an EnvVar's URL is shaped: ``openai_v1`` ends in ``/v1`` (OpenAI-style
# clients append ``/chat/completions``); ``anthropic`` is a bare origin
# (Anthropic clients append ``/v1/messages``); ``bare_origin`` is for tools
# that append their own full path prefix (IBM Bob appends
# ``/inference/v1/chat/completions``) — a ``/v1`` base would double it.
UrlStyle = Literal["openai_v1", "anthropic", "bare_origin"]

_STYLE_BUILDERS = {
    "openai_v1": codex_proxy_base_url,
    "anthropic": claude_proxy_base_url,
    # Same URL as the anthropic base today; a distinct style documents why
    # there is no /v1 and keeps bare-origin tools right if that base changes.
    "bare_origin": claude_proxy_base_url,
}


@dataclass(frozen=True, slots=True)
class EnvVar:
    """One environment variable a wrap target needs pointed at the proxy.

    ``display`` controls whether the assignment is echoed in the wrap banner;
    hidden aliases (e.g. ``OPENAI_API_BASE`` next to ``OPENAI_BASE_URL``) are
    set but not shown.
    """

    key: str
    style: UrlStyle
    display: bool = True


@dataclass(frozen=True, slots=True)
class WrapTarget:
    """Everything needed to generate a ``headroom wrap <name>`` command."""

    name: str
    binary: str
    install_hint: str
    env_vars: tuple[EnvVar, ...]
    help_text: str
    # Encode the launch directory as a /p/<name> base-URL prefix so the proxy
    # can attribute savings per project (for tools that can't send headers).
    project_prefix: bool = True
    # OpenAI-family upstream handed to the proxy at startup.
    openai_api_url: str | None = None
    # Nonstandard chat-completions paths the tool posts to. Each becomes a
    # proxy route to ``handle_openai_chat`` (see ``route_specs``) so the
    # traffic is compressed instead of hitting the uncompressed catch-all.
    extra_chat_routes: tuple[str, ...] = ()
    # Non-chat gateway paths the tool builds itself against the bare origin.
    # The catch-all forwards these to the upstream ORIGIN verbatim; joining
    # them onto ``openai_api_url``'s path doubles or misroots them.
    origin_passthrough_prefixes: tuple[str, ...] = ()
    # (path, key) pairs removed from JSON responses on origin passthrough.
    origin_passthrough_strip_json_keys: tuple[tuple[str, str], ...] = ()
    # Proxy mode exported as HEADROOM_MODE when the user has not set one.
    # Only affects a proxy this wrap starts; wrap warns when a reused proxy
    # runs a different mode.
    default_mode: str | None = None


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
            binary="goose",
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
            binary="openhands",
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
            name="bob",
            binary="bob",
            install_hint="Install IBM Bob CLI: npm install -g bobshell",
            env_vars=(
                # Bob resolves its gateway as config.gatewayUrl ?? BOB_GATEWAY_URL
                # ?? default, so this reroutes inference without touching
                # ~/.bob/settings. Bob appends /inference/v1/... itself.
                EnvVar("BOB_GATEWAY_URL", "bare_origin"),
            ),
            # Carries /inference/v1 so the proxy's _normalize_api_url (strips
            # /v1) and handle_openai_chat (re-appends /v1/chat/completions)
            # compose back into the path IBM serves.
            openai_api_url="https://api.us-east.bob.ibm.com/inference/v1",
            extra_chat_routes=("/inference/v1/chat/completions",),
            origin_passthrough_prefixes=("/inference/", "/admin/"),
            # Bob 2.0.1 rewrites its gateway host from region_domain in the
            # /admin/v1/profile response while keeping the proxied port, so
            # every later request targets api.<region>:<proxy-port>. With the
            # key absent it keeps its configured gateway URL (the proxy).
            origin_passthrough_strip_json_keys=(("/admin/v1/profile", "region_domain"),),
            # Bob bills flat per token (no prompt-cache discount to protect),
            # so token mode converts compression 1:1 into dollars.
            default_mode="token",
            help_text=(
                "Launch IBM Bob CLI through Headroom proxy.\n"
                "\n"
                "\b\n"
                "Sets ``BOB_GATEWAY_URL`` so Bob routes inference traffic through\n"
                "Headroom while keeping its own ``Authorization: apikey ...``\n"
                "credential and its ~/.bob/settings files untouched.\n"
                "\n"
                "\b\n"
                "Bob bills flat per token, so token mode converts compression 1:1\n"
                "into dollars. Token mode is the default for bob; set HEADROOM_MODE\n"
                "to override:\n"
                "    HEADROOM_MODE=cache headroom wrap bob\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap bob                          # Start proxy + bob\n"
                '    headroom wrap bob -- run "fix the bug"     # Pass args to bob\n'
                "    headroom wrap bob --port 9999              # Custom proxy port"
            ),
        ),
        WrapTarget(
            name="openclaude",
            binary="openclaude",
            install_hint="Install OpenClaude before running `headroom wrap openclaude`.",
            # Same env shape as `wrap aider`.
            env_vars=(
                EnvVar("OPENAI_API_BASE", "openai_v1"),
                EnvVar("ANTHROPIC_BASE_URL", "anthropic"),
            ),
            help_text=(
                "Launch OpenClaude through Headroom proxy.\n"
                "\n"
                "\b\n"
                "OpenClaude is a prose-format coding CLI (like Aider / Cline); it speaks\n"
                "OpenAI- and Anthropic-compatible HTTP, so wrap routes both base URLs\n"
                "through the local proxy — same env shape as `wrap aider`.\n"
                "\n"
                "\b\n"
                "Examples:\n"
                "    headroom wrap openclaude                         # Start proxy + openclaude\n"
                "    headroom wrap openclaude -- --model gpt-4o       # Pass args to openclaude"
            ),
        ),
    )
}


# ---------------------------------------------------------------------------
# Origin passthrough
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _OriginRules:
    prefixes: tuple[str, ...]
    strip_keys: tuple[tuple[str, str], ...]


# (scheme, netloc) of a target's upstream -> its passthrough rules.
_ORIGIN_RULES: dict[tuple[str, str], _OriginRules] = {
    (url.scheme, url.netloc): _OriginRules(
        target.origin_passthrough_prefixes, target.origin_passthrough_strip_json_keys
    )
    for target in WRAP_TARGETS.values()
    if target.openai_api_url
    and (target.origin_passthrough_prefixes or target.origin_passthrough_strip_json_keys)
    for url in (urlsplit(target.openai_api_url),)
}


def _origin_rules(base_url: str | None) -> tuple[str, _OriginRules] | None:
    if not base_url:
        return None
    base = urlsplit(base_url)
    rules = _ORIGIN_RULES.get((base.scheme, base.netloc))
    return (f"{base.scheme}://{base.netloc}", rules) if rules else None


def resolve_origin_passthrough_url(base_url: str | None, path: str) -> str | None:
    """Origin-rooted upstream URL for a declared origin-passthrough path.

    When ``base_url`` points at a registered target's gateway host and ``path``
    starts with one of its ``origin_passthrough_prefixes``, forward the path
    verbatim to that origin. None means the caller keeps its base+path join.
    """
    found = _origin_rules(base_url)
    if found and any(path.startswith(prefix) for prefix in found[1].prefixes):
        return found[0] + path
    return None


def _strip_json_key(obj: object, key: str) -> bool:
    """Remove ``key`` from every dict in ``obj`` in place; True when removed."""
    removed = False
    if isinstance(obj, dict):
        if key in obj:
            del obj[key]
            removed = True
        for value in obj.values():
            removed = _strip_json_key(value, key) or removed
    elif isinstance(obj, list):
        for value in obj:
            removed = _strip_json_key(value, key) or removed
    return removed


def strip_origin_passthrough_response_keys(
    base_url: str | None, path: str, body: bytes
) -> bytes | None:
    """Filtered JSON body for an origin-passthrough response, or None.

    None when nothing is declared for this host and path, the body is not
    JSON, or no declared key was present.
    """
    found = _origin_rules(base_url)
    if not found:
        return None
    keys = [
        key
        for declared, key in found[1].strip_keys
        if path == declared or path.startswith(declared.rstrip("/") + "/")
    ]
    if not keys:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    changed = False
    for key in keys:
        changed = _strip_json_key(payload, key) or changed
    if not changed:
        return None
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
