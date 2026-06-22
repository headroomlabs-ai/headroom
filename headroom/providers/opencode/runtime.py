"""OpenCode provider runtime.

Provider discovery, upstream mapping, config overlay generation,
and launch environment construction for ``headroom wrap opencode``.

Zero disk mutation in wrap mode — the overlay is injected via the
``OPENCODE_CONFIG_CONTENT`` environment variable. Cleanup is handled
by process exit.

Routing modes
-------------
Two routing strategies are supported:

**single** (default)
  A single Headroom proxy on one port.  A lightweight ESM fetch shim
  (``shim.mjs``) is injected via ``NODE_OPTIONS=--import=...`` so that
  *every* outbound AI-provider call — including providers added
  mid-session via ``/connect`` — is intercepted and routed through the
  proxy.  The original upstream origin is passed as the
  ``x-headroom-base-url`` header; proxy route handlers read it and
  forward to the real provider.  Providers that were known at launch
  are also listed in the ``OPENCODE_CONFIG_CONTENT`` overlay so
  OpenCode's model-picker shows real provider names (``deepseek``,
  ``opencode-go``) rather than a generic wrapper name.

**multi** (``--routing-mode multi``)
  One Headroom proxy process per unique upstream URL.  Each proxy's
  OpenAI and Anthropic API targets are set to its upstream so the
  proxy's existing handlers (compression, CCR, caching) operate
  correctly.  OpenCode's ``baseURL`` is pointed at the port for that
  upstream.  Zero proxy handler changes required.  Does not cover
  providers added mid-session.

Provider routing strategy
-------------------------
OpenCode supports 75+ providers.  Our overlay routes each provider
through a Headroom proxy while preserving the provider's real upstream
URL.  The overlay does **not** set ``apiKey`` — OpenCode resolves
keys from ``auth.json``, environment variables, and lower-precedence
config files.

**Direct-call providers** — routable via Headroom.
  The user's machine makes API calls directly to the provider.  Each
  provider is assigned a proxy with the correct upstream target.

**User-configured custom providers** — dynamically discovered.
  Providers explicitly configured in ``opencode.json`` are
  auto-detected and routed alongside the standard providers.

**OpenCode-managed providers** — routed via dedicated proxy.
  ``opencode`` (Zen) and ``opencode-go`` (Go) are treated as
  separate providers with their own upstreams at
  ``opencode.ai/zen/v1`` and ``opencode.ai/zen/go/v1`` respectively.
  Both use the Anthropic-compatible ``/v1/messages`` endpoint for
  some models and the OpenAI-compatible ``/v1/chat/completions``
  endpoint for others — the shared proxy handles both paths.

**Providers that cannot be routed.**
  ``github-copilot`` and ``gitlab`` use OAuth to their respective
  hosts — they cannot be re-pointed to a local proxy.

Per-project attribution uses a ``/p/<name>`` path prefix inserted
*before* the ``/v1`` path segment so the proxy's
``split_project_path`` can strip it:

  ``http://127.0.0.1:{port}/p/headroom/v1`` -> ``.../p/headroom/v1/messages``
"""

from __future__ import annotations

import importlib.resources
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Known upstream defaults for standard providers.
# Providers not listed here are looked up from opencode.json or auth.json.
# ---------------------------------------------------------------------------

_KNOWN_UPSTREAMS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "opencode": "https://opencode.ai/zen/v1",
    "opencode-go": "https://opencode.ai/zen/go/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "cohere": "https://api.cohere.ai/v1",
}

# Providers routed through the shared proxy's Anthropic handler
# in single-proxy mode (handle_anthropic_messages accepts
# upstream_base_url).
_ANTHROPIC_HANDLER_PROVIDERS: set[str] = {
    "anthropic", "opencode", "opencode-go",
}

# Providers routed through the shared proxy's Gemini handler
# in single-proxy mode (handle_gemini_generate_content accepts
# upstream_base_url for non-streaming).
_GEMINI_HANDLER_PROVIDERS: set[str] = {"google", "gemini"}


# ---------------------------------------------------------------------------
# JSONC comment stripper (stdlib only — no commentjson dependency)
# ---------------------------------------------------------------------------

_JSONC_LINE_COMMENT = re.compile(r"//.*$", re.MULTILINE)
_JSONC_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_JSONC_STRING = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


def _strip_jsonc_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` style comments from *text*.

    String literals are preserved — comments inside strings are left alone.
    Handles escape sequences (``\"``, ``\\``) correctly.  This means
    URLs in strings (e.g. ``"baseURL": "https://api.deepseek.com"``) are
    never corrupted.
    """
    strings: dict[str, str] = {}
    counter = 0

    def _replace(m: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"\x00JSONC_S{counter}\x00"
        strings[placeholder] = m.group(0)
        counter += 1
        return placeholder

    text = _JSONC_STRING.sub(_replace, text)

    text = _JSONC_LINE_COMMENT.sub("", text)
    text = _JSONC_BLOCK_COMMENT.sub("", text)

    for placeholder, original in strings.items():
        text = text.replace(placeholder, original)

    return text


# ---------------------------------------------------------------------------
# System-level config paths
# ---------------------------------------------------------------------------

_USER_CONFIG_CANDIDATES: tuple[Path, ...] = (
    Path.home() / ".config" / "opencode" / "opencode.json",
    Path.home() / ".config" / "opencode" / "opencode.jsonc",
    Path("opencode.json"),
    Path("opencode.jsonc"),
)

_AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"


def _opencode_home_dir() -> Path:
    """Derive the OpenCode config home directory from common env vars."""
    from headroom.providers.opencode.config import _opencode_home_dir as _home
    return _home()


def _user_config_path() -> Path | None:
    """Return the first available user-level OpenCode config or None."""
    for candidate in _USER_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Provider discovery
# ---------------------------------------------------------------------------


def discover_user_providers() -> dict[str, dict[str, object]]:
    """Return all providers declared in the user's ``opencode.json(c)``.

    Custom providers with ``@ai-sdk/openai-compatible``, ``@ai-sdk/anthropic``,
    or ``@ai-sdk/openai`` are included.  Standard providers without explicit
    config entries are NOT included — they must be discovered from
    ``auth.json`` instead.
    """
    result: dict[str, dict[str, object]] = {}
    config_path = _user_config_path()
    if not config_path:
        return result

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError:
        return result

    try:
        data = json.loads(_strip_jsonc_comments(raw))
    except (json.JSONDecodeError, ValueError):
        return result

    provider = data.get("provider", {})
    if not isinstance(provider, dict):
        return result

    for name, entry in provider.items():
        if not isinstance(entry, dict):
            continue
        result[str(name)] = entry

    return result


def build_provider_upstream_map() -> dict[str, str]:
    """Build a provider-name -> upstream-URL mapping.

    Sources (in precedence order):
      1. The user's opencode.json (custom providers with explicit baseURL).
      2. auth.json entries with ``type == "api"`` that have a known upstream.
      3. The _KNOWN_UPSTREAMS table.

    Providers that use OAuth (github-copilot, gitlab) are excluded.
    Providers whose upstream URL resolves to localhost (leftover from a
    prior install) are also excluded to avoid routing loops.
    """
    result: dict[str, str] = {}

    # --- source 1: user-configured custom providers ---
    for name, entry in discover_user_providers().items():
        options = entry.get("options", {})
        if not isinstance(options, dict):
            continue
        base_url = options.get("baseURL")
        if not isinstance(base_url, str) or not base_url.strip():
            continue
        result[name] = base_url

    # --- source 2: auth.json entries with known upstreams ---
    try:
        raw_auth = _AUTH_PATH.read_text(encoding="utf-8")
    except OSError:
        raw_auth = "{}"

    try:
        auth_data = json.loads(raw_auth)
    except (json.JSONDecodeError, ValueError):
        auth_data = {}

    for name, entry in auth_data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "api":
            continue
        if name in result:
            continue  # config entry takes precedence
        upstream = _KNOWN_UPSTREAMS.get(name)
        if upstream:
            result[name] = upstream

    # --- filter localhost entries (leftover install artifacts) ---
    result = {
        n: u for n, u in result.items()
        if "localhost" not in u and "127.0.0.1" not in u
    }

    return result


def has_zen_auth() -> bool:
    """True when auth.json contains an api key for ``opencode`` or ``opencode-go``."""
    try:
        raw = _AUTH_PATH.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    for name in ("opencode", "opencode-go"):
        entry = data.get(name)
        if isinstance(entry, dict) and entry.get("type") == "api":
            return True

    return False


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def proxy_base_url(port: int, project: str | None = None) -> str:
    """Build the base URL that OpenCode should use for a proxy on *port*.

    When *project* is given the ``/p/<project>`` prefix is inserted before
    ``/v1`` so the proxy can attribute savings to the project.
    """
    if project:
        encoded = quote(project, safe="")
        return f"http://127.0.0.1:{port}/p/{encoded}/v1"
    return f"http://127.0.0.1:{port}/v1"


# ---------------------------------------------------------------------------
# Config overlay generation
# ---------------------------------------------------------------------------


def _build_overlay_body_multi(
    port_map: dict[str, int],
    project: str | None,
) -> dict[str, dict[str, object]]:
    """Build the ``provider`` block for multi-proxy mode.

    Each provider gets its own dedicated proxy port.  No
    ``x-headroom-base-url`` header is needed because each proxy is
    configured with that provider's exact upstream URL.
    """
    overlay: dict[str, dict[str, object]] = {}
    for name, port in sorted(port_map.items()):
        overlay[name] = {
            "options": {"baseURL": proxy_base_url(port, project)},
        }
    return overlay


def _build_overlay_body_single(
    provider_map: dict[str, str],
    shared_port: int,
    project: str | None,
) -> dict[str, dict[str, object]]:
    """Build the ``provider`` block for single-proxy mode.

    All providers share *shared_port*.  The ``x-headroom-base-url``
    header tells the proxy which upstream to target.  The header is
    stripped by ``_strip_internal_headers`` before the upstream call.

    For providers whose handler supports ``upstream_base_url``
    (Anthropic, Gemini), the handler receives the dynamic upstream
    URL and applies full compression/CCR/caching.  For all others
    (OpenAI, DeepSeek, etc.), the request goes through
    ``handle_passthrough`` which is a correct byte-level forwarder.
    """
    overlay: dict[str, dict[str, object]] = {}

    for name, upstream_url in sorted(provider_map.items()):
        overlay[name] = {
            "options": {
                "baseURL": proxy_base_url(shared_port, project),
                "headers": {"x-headroom-base-url": upstream_url},
            },
        }

    return overlay


def build_overlay(
    provider_map: dict[str, str],
    shared_port: int,
    project: str | None = None,
    *,
    routing_mode: str = "multi",
    port_map: dict[str, int] | None = None,
) -> dict[str, object]:
    """Return the OpenCode config overlay dict (``{"provider": {...}}``).

    In **multi** mode each unique upstream URL gets its own port.
    *port_map* must be provided with a ``{provider_name: port}``
    mapping.  The caller is responsible for assigning those ports (via
    ``allocate_ports``).

    In **single** mode all passthrough providers share *shared_port*
    and use the ``x-headroom-base-url`` header for routing.  Providers
    that need handler-specific processing get a dedicated proxy.
    *port_map* is ignored.
    """
    if routing_mode == "single":
        body = _build_overlay_body_single(provider_map, shared_port, project)
    else:
        if port_map is None:
            port_map = dict.fromkeys(provider_map, shared_port)
        body = _build_overlay_body_multi(port_map, project)

    return {"provider": body}


# ---------------------------------------------------------------------------
# Launch environment
# ---------------------------------------------------------------------------


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
    *,
    routing_mode: str = "single",
    include_mcp: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Build the process environment for ``headroom wrap opencode``.

    Returns ``(env_dict, display_lines)``.  The display lines are
    human-readable summaries of the env vars we set, for the CLI banner.

    In single-proxy mode (default) a fetch shim is injected via
    ``NODE_OPTIONS=--import=<shim.mjs>`` so that *all* outbound AI calls —
    including providers added mid-session — route through the proxy.
    ``OPENCODE_CONFIG_CONTENT`` still lists known providers so the model
    picker shows real provider names.

    The overlay does NOT modify OpenCode's on-disk config.
    """
    env = dict(environ or os.environ)
    provider_map = build_provider_upstream_map()

    overlay = build_overlay(
        provider_map, port, project, routing_mode=routing_mode,
    )

    config_content = json.dumps(overlay, separators=(",", ":"))
    env["OPENCODE_CONFIG_CONTENT"] = config_content

    display = [
        f"OPENCODE_CONFIG_CONTENT={{provider: {','.join(provider_map.keys())}}}"
    ]

    # In single-proxy mode inject the fetch shim so mid-session /connect
    # providers are also captured (shim intercepts at the HTTP boundary).
    if routing_mode == "single":
        proxy_url = f"http://127.0.0.1:{port}"
        env["HEADROOM_PROXY_URL"] = proxy_url

        shim_path = (
            importlib.resources.files("headroom.providers.opencode")
            .joinpath("shim.mjs")
        )
        existing_node_opts = env.get("NODE_OPTIONS", "")
        import_flag = f"--import={shim_path}"
        env["NODE_OPTIONS"] = (
            f"{import_flag} {existing_node_opts}".strip()
            if existing_node_opts
            else import_flag
        )
        display.append(f"HEADROOM_PROXY_URL={proxy_url} (fetch shim active)")

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    # Preserve existing OPENAI_BASE_URL / ANTHROPIC_BASE_URL if set by
    # the user — don't clobber. OpenCode resolves auth from auth.json
    # regardless, so these are only needed as fallbacks and should NOT
    # be set automatically.
    for _fallback_key in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        if _fallback_key in env:
            display.append(f"{_fallback_key}={env[_fallback_key]} (preserved)")

    return env, display
