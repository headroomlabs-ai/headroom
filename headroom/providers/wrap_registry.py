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
# (Anthropic clients append ``/v1/messages``); ``bare_origin`` is for tools
# that append their own full path prefix to the URL they are handed (e.g. IBM
# Bob appends ``/inference/v1/chat/completions``) — handing those a ``/v1``
# base would produce a doubled prefix.
UrlStyle = Literal["openai_v1", "anthropic", "bare_origin"]

_STYLE_BUILDERS = {
    "openai_v1": _openai_proxy_base_url,
    "anthropic": _anthropic_proxy_base_url,
    # Same shape as the anthropic base today (http://127.0.0.1:<port>), but a
    # distinct style: it documents *why* the URL has no /v1, and keeps targets
    # honest if the anthropic base ever grows a suffix.
    "bare_origin": _anthropic_proxy_base_url,
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
    # Nonstandard chat-completions paths the tool posts to. Each is registered
    # as a proxy route delegating to ``handle_openai_chat`` (see
    # ``route_specs.OPENAI_HANDLER_ROUTES``) so the traffic is compressed
    # instead of falling through to the uncompressed catch-all passthrough.
    extra_chat_routes: tuple[str, ...] = ()
    # Non-chat gateway paths the tool sends under its own prefixes (it appends
    # full paths to the bare origin it is handed). The proxy's catch-all
    # passthrough must forward these to the upstream ORIGIN with the inbound
    # path verbatim; appending them to ``openai_api_url``'s path produces a
    # doubled or misrooted URL (e.g. ``.../inference/inference/v1/model/info``,
    # ``.../inference/admin/v1/profile`` — both 403 at IBM's edge).
    origin_passthrough_prefixes: tuple[str, ...] = ()
    # (path, key) pairs to strip from JSON responses on this target's origin
    # passthrough. Bob 2.0.1's ``resolveBaseUrl`` rewrites its gateway
    # hostname from ``region_domain`` in the ``/admin/v1/profile`` response
    # while keeping the proxied port, so every request after the first profile
    # fetch targets ``api.<region>:<proxy-port>`` — unreachable. With the key
    # absent, bob falls back to its configured gateway URL (the proxy).
    origin_passthrough_strip_json_keys: tuple[tuple[str, str], ...] = ()
    # Preferred proxy mode when the user has not chosen one. The generated
    # wrap command exports it as HEADROOM_MODE before proxy startup; an
    # explicit HEADROOM_MODE in the environment always wins. Only affects a
    # proxy this wrap starts — an already-running proxy is reused with the
    # mode it booted with (/health does not expose mode for a mismatch check).
    default_mode: str | None = None
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
            name="bob",
            binaries=("bob",),
            install_hint="Install IBM Bob CLI: npm install -g bobshell",
            env_vars=(
                # Bob resolves its gateway as config.gatewayUrl ?? BOB_GATEWAY_URL
                # ?? default, so this env var reroutes inference without touching
                # ~/.bob/settings. Bob appends /inference/v1/... itself.
                EnvVar("BOB_GATEWAY_URL", "bare_origin"),
            ),
            # DEFAULT_API_URL carries the /inference/v1 suffix so the proxy's
            # _normalize_api_url (strips /v1) and handle_openai_chat (re-appends
            # /v1/chat/completions) compose back into the path IBM serves.
            openai_api_url="https://api.us-east.bob.ibm.com/inference/v1",
            extra_chat_routes=("/inference/v1/chat/completions",),
            origin_passthrough_prefixes=("/inference/", "/admin/"),
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
                "Mode matters more for Bob than for most agents: its traffic is ~46%\n"
                "system prompt and ~44% tool output, and Bob bills flat per token, so\n"
                "token mode converts compression 1:1 into dollars. Token mode is the\n"
                "default for bob; set HEADROOM_MODE to override:\n"
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


def resolve_origin_passthrough_url(base_url: str | None, path: str) -> str | None:
    """Origin-rooted upstream URL for a declared origin-passthrough path.

    When ``base_url`` points at a registered target's gateway host and ``path``
    starts with one of that target's ``origin_passthrough_prefixes``, the tool
    built the path itself against a bare origin — forward it verbatim to that
    origin. Returns None when no target matches (caller keeps its normal
    base+path join).
    """
    if not base_url:
        return None
    from urllib.parse import urlsplit

    base = urlsplit(base_url)
    for target in WRAP_TARGETS.values():
        if not target.origin_passthrough_prefixes or not target.openai_api_url:
            continue
        declared = urlsplit(target.openai_api_url)
        if (base.scheme, base.netloc) != (declared.scheme, declared.netloc):
            continue
        if any(path.startswith(prefix) for prefix in target.origin_passthrough_prefixes):
            return f"{base.scheme}://{base.netloc}{path}"
    return None


_MISSING = object()


def _strip_json_key(obj: object, key: str) -> bool:
    """Remove ``key`` from every dict in ``obj`` in place; True when removed."""
    removed = False
    if isinstance(obj, dict):
        if obj.pop(key, _MISSING) is not _MISSING:
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

    Applies the ``origin_passthrough_strip_json_keys`` declarations of the
    target whose gateway host matches ``base_url``. Returns None when nothing
    is declared for this path, the body is not JSON, or no declared key was
    present.
    """
    if not base_url:
        return None
    from urllib.parse import urlsplit

    base = urlsplit(base_url)
    keys = [
        key
        for target in WRAP_TARGETS.values()
        if target.origin_passthrough_strip_json_keys and target.openai_api_url
        for declared_path, key in target.origin_passthrough_strip_json_keys
        if (base.scheme, base.netloc)
        == (urlsplit(target.openai_api_url).scheme, urlsplit(target.openai_api_url).netloc)
        and (path == declared_path or path.startswith(declared_path.rstrip("/") + "/"))
    ]
    if not keys:
        return None
    import json

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
