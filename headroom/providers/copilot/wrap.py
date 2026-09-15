"""Copilot wrapper provider helpers."""

from __future__ import annotations

import glob
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from headroom.proxy.project_context import with_project_prefix

#: Hosts a corporate HTTP(S) proxy must never be asked to reach on our behalf.
#: The Copilot CLI honours ``HTTP_PROXY``/``HTTPS_PROXY`` (and applies its own
#: ``proxyUrl`` setting to those variables), so without an explicit exemption a
#: fleet-wide proxy setting sends the loopback hop to the corporate proxy, which
#: cannot connect to ``127.0.0.1`` on the developer's machine.
LOOPBACK_NO_PROXY_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")


def ensure_loopback_no_proxy(env: dict[str, str]) -> dict[str, str]:
    """Exempt the loopback proxy hop from any HTTP(S) proxy configuration.

    Idempotent: hosts already listed are not duplicated. ``NO_PROXY`` is always
    written; a pre-existing lowercase ``no_proxy`` is kept in sync. Readers
    disagree on which spelling wins (undici prefers lowercase, reqwest and Go
    prefer uppercase), so both are seeded from the union of the two — writing a
    fresh ``NO_PROXY`` that lacked the entries of an existing ``no_proxy`` would
    silently drop the corporate exemptions for anything the CLI spawns. Mutates
    and returns ``env`` for call-site convenience.
    """
    entries: list[str] = []
    for variable in ("NO_PROXY", "no_proxy"):
        for entry in (env.get(variable) or "").split(","):
            entry = entry.strip()
            if entry and entry not in entries:
                entries.append(entry)
    for host in LOOPBACK_NO_PROXY_HOSTS:
        if host not in entries:
            entries.append(host)
    merged = ",".join(entries)
    env["NO_PROXY"] = merged
    if "no_proxy" in env:
        env["no_proxy"] = merged
    return env


def copilot_home(environ: Mapping[str, str] | None = None, *, windows: bool | None = None) -> Path:
    """Return the Copilot CLI config directory (``$COPILOT_HOME`` or ``~/.copilot``).

    Matches Node's ``os.homedir()``, which the CLI uses: ``USERPROFILE`` on
    Windows (``HOME`` is ignored there, even when Git-for-Windows sets it),
    ``HOME`` elsewhere. ``windows`` defaults to the running platform.
    """
    env = environ if environ is not None else os.environ
    configured = (env.get("COPILOT_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser()
    on_windows = os.name == "nt" if windows is None else windows
    home = env.get("USERPROFILE") if on_windows else env.get("HOME")
    return (Path(home) if home else Path.home()) / ".copilot"


#: Settings key the CLI consults for its Copilot API host. Undocumented, but the
#: 1.0.8x bundle resolves ``settings.copilotUrl || COPILOT_API_URL ||
#: token.endpoints.api`` — the file beats the environment variable Headroom sets.
COPILOT_URL_SETTING_KEY = "copilotUrl"


def read_copilot_url_settings(environ: Mapping[str, str] | None = None) -> list[tuple[Path, str]]:
    """Return every ``(file, value)`` pin of the CLI's Copilot API host.

    Both ``settings.json`` (the current user-settings file) and the legacy
    ``config.json`` are consulted: the CLI migrates keys from the latter at
    startup but still reports a legacy value as shadowing the new one when both
    exist, so a pin in either file can be the effective one. Unreadable or
    malformed files contribute nothing — the CLI would ignore them too, so they
    cannot redirect its traffic.
    """
    home = copilot_home(environ)
    pins: list[tuple[Path, str]] = []
    for name in ("settings.json", "config.json"):
        path = home / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        value = payload.get(COPILOT_URL_SETTING_KEY)
        if isinstance(value, str) and value.strip():
            pins.append((path, value.strip()))
    return pins


def read_copilot_url_setting(environ: Mapping[str, str] | None = None) -> tuple[Path, str] | None:
    """Return the first ``copilotUrl`` pin (``settings.json`` before ``config.json``), if any."""
    pins = read_copilot_url_settings(environ)
    return pins[0] if pins else None


def _origin(url: str) -> str:
    """Return ``scheme://host[:port]`` lowercased, so two spellings of one proxy compare equal."""
    parsed = urllib.parse.urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/").lower()
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def check_copilot_url_setting(base_url: str, *, environ: Mapping[str, str] | None = None) -> None:
    """Refuse a native launch that the CLI's own ``copilotUrl`` setting would bypass.

    Headroom points the CLI at itself through ``COPILOT_API_URL``. The CLI ranks
    the ``copilotUrl`` settings key above that variable, so a pre-existing pin
    makes the wrapper print a successful launch while every request goes
    straight to the pinned host. Fail closed on any pin whose origin is not this
    proxy. A pin on the same origin but a different path (a durable install that
    wrote the bare proxy URL, without this launch's ``/p/<project>`` prefix)
    still routes through Headroom; it is accepted with a note, because only the
    per-project attribution differs.
    """
    proxy_origin = _origin(base_url)
    foreign: list[tuple[Path, str]] = []
    same_origin_other_path: list[tuple[Path, str]] = []
    for path, value in read_copilot_url_settings(environ):
        if _origin(value) != proxy_origin:
            foreign.append((path, value))
        elif value.rstrip("/").lower() != base_url.rstrip("/").lower():
            same_origin_other_path.append((path, value))
    if foreign:
        described = "; ".join(
            f"{path} sets {COPILOT_URL_SETTING_KEY}={value!r}" for path, value in foreign
        )
        raise click.ClickException(
            f"{described}. The Copilot CLI prefers that setting over COPILOT_API_URL, so this "
            f"launch would bypass Headroom. Remove the key, or set it to {proxy_origin!r} to "
            "route every project through this proxy."
        )
    for path, value in same_origin_other_path:
        click.echo(
            f"  Note: {path} pins {COPILOT_URL_SETTING_KEY}={value!r}; the CLI will use that URL "
            f"instead of {base_url!r}, so per-project savings attribution follows the pinned path."
        )


def resolve_provider_type(
    backend: str | None, provider_type: str, environ: Mapping[str, str] | None = None
) -> str:
    """Resolve Copilot BYOK provider type for the current proxy backend."""
    if provider_type != "auto":
        return provider_type

    env = environ or os.environ
    # Check COPILOT_PROVIDER_TYPE env var before falling back to backend default.
    env_type = env.get("COPILOT_PROVIDER_TYPE")
    if env_type in {"anthropic", "openai"}:
        return env_type
    effective_backend = backend or env.get("HEADROOM_BACKEND") or "anthropic"
    return "anthropic" if effective_backend == "anthropic" else "openai"


def query_proxy_config(port: int) -> dict[str, Any] | None:
    """Query the running proxy's feature configuration via /health."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None

    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return config


def detect_running_proxy_backend(port: int) -> str | None:
    """Read the backend of an already-running proxy from its health endpoint."""
    config = query_proxy_config(port)
    if config is None:
        return None
    backend = config.get("backend")
    return backend if isinstance(backend, str) else None


def validate_configuration(
    *,
    provider_type: str,
    wire_api: str | None,
    backend: str | None,
) -> None:
    """Validate Copilot BYOK provider and wire-api settings."""
    if provider_type == "anthropic" and wire_api is not None:
        raise click.ClickException(
            "--wire-api is only valid when Copilot is using the openai provider type."
        )
    if wire_api == "responses" and backend not in (None, "anthropic"):
        raise click.ClickException(
            "--wire-api responses is not supported with translated backends; use completions."
        )


#: Copilot virtual model names that map to native auto-routing.
#: Forwarding these to BYOK endpoints causes a 400; they must be stripped.
_AUTO_MODEL_ALIASES: frozenset[str] = frozenset({"auto"})


def is_auto_model(model: str | None) -> bool:
    """Return True when the model name is a Copilot auto-routing alias.

    ``model auto`` is a virtual model ID that Copilot resolves internally.
    It is **not** a valid model string for BYOK providers (Anthropic, OpenAI)
    and causes a ``400 The requested model is not supported`` error if forwarded
    verbatim.  This helper centralises the detection so both the CLI and the
    proxy layer can guard against it.
    """
    if not model:
        return False
    return model.strip().lower() in _AUTO_MODEL_ALIASES


def strip_auto_model_args(copilot_args: tuple[str, ...]) -> tuple[str, ...]:
    """Remove ``--model auto`` (and ``--model=auto``) from Copilot CLI args.

    Used in the subscription/OAuth path: when the user passes ``--model auto``
    to ``headroom wrap copilot --subscription``, we strip it before launching
    Copilot so the CLI falls back to its own native automatic model selection
    instead of sending the unsupported ``auto`` string to the BYOK API.
    """
    result: list[str] = []
    i = 0
    while i < len(copilot_args):
        arg = copilot_args[i]
        if arg == "--model" and i + 1 < len(copilot_args):
            if is_auto_model(copilot_args[i + 1]):
                i += 2  # skip both --model and auto
                continue
        elif arg.startswith("--model=") and is_auto_model(arg.split("=", 1)[1]):
            i += 1  # skip --model=auto
            continue
        result.append(arg)
        i += 1
    return tuple(result)


def _normalized_model_name(model: str | None) -> str:
    """Return a lowercase model name without provider/path prefixes."""
    if not model:
        return ""
    value = model.strip().lower()
    for separator in ("/", ":"):
        if separator in value:
            value = value.rsplit(separator, 1)[-1]
    return value


def model_prefers_responses_api(model: str | None) -> bool:
    """Return True for OpenAI reasoning models served via /responses."""
    value = _normalized_model_name(model)
    return value.startswith(("gpt-5", "o1", "o3"))


def copilot_model_from_args(
    copilot_args: tuple[str, ...],
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the Copilot model from CLI args or environment variables."""
    for idx, arg in enumerate(copilot_args):
        if arg == "--model" and idx + 1 < len(copilot_args):
            return copilot_args[idx + 1]
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]

    source = env or os.environ
    return source.get("COPILOT_MODEL") or source.get("COPILOT_PROVIDER_MODEL_ID")


def default_wire_api_for_model(model: str | None) -> str:
    """Choose the Copilot OpenAI-compatible wire API for a model."""
    return "responses" if model_prefers_responses_api(model) else "completions"


def provider_key_source(provider_type: str) -> str:
    """Return the preferred provider key variable for the selected provider type."""
    return "ANTHROPIC_API_KEY" if provider_type == "anthropic" else "OPENAI_API_KEY"


COPILOT_NATIVE_API_URL_ENV = "COPILOT_API_URL"

# Any survivor keeps Copilot in its single-model BYOK lane, defeating native
# model routing while making the launch look superficially successful.
COPILOT_BYOK_ENV_VARS: tuple[str, ...] = (
    "COPILOT_PROVIDER_BASE_URL",
    "COPILOT_PROVIDER_TYPE",
    "COPILOT_PROVIDER_API_KEY",
    "COPILOT_PROVIDER_BEARER_TOKEN",
    "COPILOT_PROVIDER_WIRE_API",
    "COPILOT_PROVIDER_TRANSPORT",
    "COPILOT_PROVIDER_AZURE_API_VERSION",
    "COPILOT_PROVIDER_MODEL_ID",
    "COPILOT_PROVIDER_WIRE_MODEL",
    "COPILOT_PROVIDER_MODEL_LIMITS_ID",
    "COPILOT_PROVIDER_MAX_PROMPT_TOKENS",
    "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS",
    "COPILOT_PROVIDER_HEADERS",
)


def build_native_launch_env(
    *,
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Redirect Copilot's native API surface through Headroom, not BYOK."""
    env = dict(environ if environ is not None else os.environ)
    base_url = with_project_prefix(f"http://127.0.0.1:{port}", project)
    env[COPILOT_NATIVE_API_URL_ENV] = base_url
    for variable in COPILOT_BYOK_ENV_VARS:
        env.pop(variable, None)
    ensure_loopback_no_proxy(env)
    return env, [
        f"{COPILOT_NATIVE_API_URL_ENV}={base_url}",
        "COPILOT_AUTH_MODE=github-native",
    ]


def _copilot_bundle_candidates(copilot_bin: str | None) -> list[str]:
    """Return ``app.js`` paths that belong to the CLI binary actually on PATH.

    The CLI stopped being a single ``app.js`` under a ``pkg`` directory: since
    the 1.0.8x line ``@github/copilot`` is an npm shim (``npm-loader.js``) that
    spawns a per-platform package (``@github/copilot-<os>-<arch>``) holding a
    native ``copilot`` binary next to the JavaScript ``app.js`` it embeds. Both
    layouts are covered by looking beside the resolved binary and in sibling
    ``copilot-*`` packages of the directory it lives in.
    """
    if not copilot_bin:
        return []
    try:
        resolved = os.path.realpath(copilot_bin)
    except OSError:
        return []
    bin_dir = os.path.dirname(resolved)
    parent = os.path.dirname(bin_dir)
    candidates: list[str] = [
        # A native platform binary sits next to its own bundle.
        os.path.join(bin_dir, "app.js"),
        # `npm-loader.js` (resolved through the `bin/copilot` symlink) lives in
        # the shim package; the platform packages are its siblings under the
        # same `@github/` scope directory.
        *sorted(glob.glob(os.path.join(parent, "copilot-*", "app.js"))),
        # Windows npm shims (`copilot.cmd`, `copilot.ps1`) are scripts in the
        # npm prefix, not symlinks, so the packages hang off that directory.
        *sorted(glob.glob(os.path.join(bin_dir, "node_modules", "@github", "copilot-*", "app.js"))),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for path in candidates:
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            unique.append(path)
    return unique


def _bundle_mentions_native_api_url(path: str) -> bool | None:
    """Return True/False for a readable bundle, None when it cannot be read.

    Reads in chunks, carrying the tail of each chunk into the next so a match
    that straddles a chunk boundary is not missed — a miss here now refuses the
    launch rather than merely leaving the verdict unknown.
    """
    needle = COPILOT_NATIVE_API_URL_ENV
    carry = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as bundle:
            while chunk := bundle.read(1 << 20):
                window = carry + chunk
                if needle in window:
                    return True
                carry = window[-(len(needle) - 1) :]
    except OSError:
        return None
    return False


def native_api_url_supported(
    *, environ: Mapping[str, str] | None = None, copilot_bin: str | None = None
) -> bool | None:
    """Best-effort tri-state probe for the CLI's native API URL override.

    Returns True when a bundle references ``COPILOT_API_URL``, False when at
    least one bundle was read and none did, and None when no bundle could be
    located — the caller then proceeds but cannot promise the redirect took.
    ``copilot_bin`` (the binary about to be launched) decides on its own when its
    bundle can be read: a stale installer copy under the legacy ``pkg`` roots
    must not vouch for a build that dropped the hook. Those roots are consulted
    only when no readable bundle sits beside the launched binary.
    """
    env = environ if environ is not None else os.environ
    home = env.get("HOME") or os.path.expanduser("~")
    local = env.get("LOCALAPPDATA") or home
    roots = (
        os.path.join(local, "copilot", "pkg"),
        os.path.join(home, ".local", "share", "copilot", "pkg"),
    )
    launched_verdicts = [
        _bundle_mentions_native_api_url(path) for path in _copilot_bundle_candidates(copilot_bin)
    ]
    if True in launched_verdicts:
        return True
    if False in launched_verdicts:
        return False
    found_bundle = False
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "app.js" not in filenames:
                continue
            found_bundle = True
            if _bundle_mentions_native_api_url(os.path.join(dirpath, "app.js")):
                return True
    return False if found_bundle else None


def build_launch_env(
    *,
    port: int,
    provider_type: str,
    wire_api: str | None,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the Copilot BYOK environment for the selected provider type.

    A durable ``headroom install`` exports ``COPILOT_API_URL`` into the shell so
    the native lane works everywhere; this lane routes chat through the BYOK
    variables instead and drops that hook, so the CLI's ancillary CAPI calls
    are not sent to a proxy that may not be running.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because the Copilot CLI cannot send custom headers; the
    proxy strips it and attributes savings per project.
    """
    # Distinguish "caller passed nothing" (use os.environ) from "caller
    # explicitly passed an empty dict" (start fresh — the test/CLI is in
    # charge of which keys to seed). The previous `environ or os.environ`
    # collapsed those two cases because `bool({}) is False`.
    env = dict(environ if environ is not None else os.environ)
    env.pop(COPILOT_NATIVE_API_URL_ENV, None)
    env["COPILOT_PROVIDER_TYPE"] = provider_type
    env.pop("COPILOT_PROVIDER_WIRE_API", None)
    ensure_loopback_no_proxy(env)

    if not env.get("COPILOT_PROVIDER_API_KEY"):
        key = env.get(provider_key_source(provider_type), "")
        if key:
            env["COPILOT_PROVIDER_API_KEY"] = key

    if provider_type == "anthropic":
        base_url = with_project_prefix(f"http://127.0.0.1:{port}", project)
        env["COPILOT_PROVIDER_BASE_URL"] = base_url
        return env, [
            "COPILOT_PROVIDER_TYPE=anthropic",
            f"COPILOT_PROVIDER_BASE_URL={base_url}",
        ]

    effective_wire_api = wire_api or "completions"
    base_url = with_project_prefix(f"http://127.0.0.1:{port}/v1", project)
    env["COPILOT_PROVIDER_BASE_URL"] = base_url
    env["COPILOT_PROVIDER_WIRE_API"] = effective_wire_api
    return env, [
        "COPILOT_PROVIDER_TYPE=openai",
        f"COPILOT_PROVIDER_BASE_URL={base_url}",
        f"COPILOT_PROVIDER_WIRE_API={effective_wire_api}",
    ]


def model_configured(copilot_args: tuple[str, ...], env: Mapping[str, str]) -> bool:
    """Return True when Copilot BYOK model selection is configured (non-auto).

    ``--model auto`` is **not** considered configured for BYOK purposes: it is
    a virtual Copilot routing token that has no meaning to external providers
    such as Anthropic or OpenAI, and forwarding it causes a 400.  Returning
    ``False`` here ensures the BYOK "model required" warning is still shown
    when the user mistakenly passes ``--model auto`` in BYOK mode.
    """
    model = copilot_model_from_args(copilot_args, env)
    if model is None or is_auto_model(model):
        return False
    return True
