"""Copilot wrapper provider helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import click

from headroom.proxy.project_context import with_project_prefix


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
    """Choose the Copilot OpenAI-compatible wire API for a model, by name."""
    return "responses" if model_prefers_responses_api(model) else "completions"


def resolve_wire_api_for_model(
    model: str | None,
    *,
    api_url: str | None = None,
    token: str | None = None,
    timeout: float = 4.0,
) -> str:
    """Pick the wire API from the model's published endpoints when possible.

    The launcher pins ``COPILOT_PROVIDER_WIRE_API`` for the whole session, so
    guessing it from the model name can make a perfectly valid main model
    unusable. ``mai-code-1-flash-picker`` is served **only** on ``/responses``
    but does not match ``gpt-5*/o1*/o3*``, so the name heuristic pins the
    session to ``completions`` and every turn fails with
    ``400 model "mai-code-1-flash-picker" is not accessible via the
    /chat/completions endpoint`` -- reproduced end-to-end through the real
    Copilot CLI.

    Asking the upstream which endpoints serve the model removes the guess. This
    runs once at launch, is bounded by ``timeout``, and falls back to
    :func:`default_wire_api_for_model` on any failure, so a slow or unreachable
    ``/models`` only costs a few seconds and never blocks a launch.
    """
    fallback = default_wire_api_for_model(model)
    if not model or not api_url or not token:
        return fallback

    try:
        import httpx

        from headroom.copilot_auth import _copilot_chat_header_defaults
        from headroom.models.copilot_catalog import catalog_enabled, parse_models_payload

        if not catalog_enabled():
            return fallback

        headers = {"Authorization": f"Bearer {token}", **_copilot_chat_header_defaults()}
        response = httpx.get(f"{api_url.rstrip('/')}/models", headers=headers, timeout=timeout)
        if response.status_code != 200:
            return fallback
        card = parse_models_payload(response.json()).get(model)
    except Exception:  # noqa: BLE001 — launch must never fail on discovery
        return fallback

    if card is None or not card.constrains_endpoints():
        return fallback
    # Prefer the model's native wire so the main model never rides the buffered
    # bridge for every turn; only fall back when it publishes neither.
    if "/responses" in card.endpoints and "/chat/completions" not in card.endpoints:
        return "responses"
    if "/chat/completions" in card.endpoints and "/responses" not in card.endpoints:
        return "completions"
    if "/responses" in card.endpoints or "/chat/completions" in card.endpoints:
        return fallback if fallback in ("responses", "completions") else "completions"
    return fallback


#: The env var that redirects the Copilot CLI's **native** (GitHub-authenticated)
#: Copilot-API surface at a different host. Undocumented -- it is absent from
#: ``copilot help environment`` -- but structurally embedded in the shipped CLI:
#: every native call resolves its base URL through a helper that checks it first,
#: and the CLI logs "Using COPILOT_API_URL from environment" when it fires.
#:
#: This is the interposition point that BYOK is not. BYOK *replaces* Copilot's
#: model routing and accepts exactly one model, so the picker collapses to that
#: model and the main agent cannot be switched in-session. Redirecting the native
#: surface instead leaves Copilot's own routing intact -- the CLI still fetches
#: ``GET /models``, so the picker stays populated and ``/model`` keeps working --
#: while every request still travels through Headroom.
COPILOT_NATIVE_API_URL_ENV = "COPILOT_API_URL"

#: Prefix on every model id the VS Code chat-model config registers.
#:
#: VS Code's chat model picker keys a model on its **bare id**, ignoring which
#: provider contributed it. A BYOK entry that reuses Copilot's own id is
#: therefore treated as the same model as Copilot's native one, and the picker
#: renders that identity exactly once -- as the native entry. Every model the
#: user has recently used or GitHub has featured is already shown that way, so
#: precisely those models lost their Headroom twin: the compressed route was
#: invisible for the models people reach for most.
#:
#: Registering ``headroom--claude-opus-5`` keeps the two identities distinct in
#: the picker. The prefix is stripped again in the proxy
#: (``resolve_copilot_model_id``) so the id that reaches Copilot is the real one,
#: and savings are still attributed to the real model.
VSCODE_MODEL_ID_PREFIX = "headroom--"

#: Paths the native CLI drives that are *not* inference and must pass through
#: untouched. Recorded here for documentation; the proxy's catch-all already
#: forwards them.
COPILOT_NATIVE_PASSTHROUGH_PATHS: tuple[str, ...] = (
    "/models",
    "/models/session",
    "/models/session/intent",
    "/mcp/readonly",
    "/agents",
)


def native_api_url_supported(
    *, environ: Mapping[str, str] | None = None, home: str | None = None
) -> bool | None:
    """Best-effort check that the installed CLI honours ``COPILOT_API_URL``.

    Returns ``True`` when the shipped application bundle references the variable,
    ``False`` when a bundle was found and does not, and ``None`` when no bundle
    could be located (unknown -- caller should proceed with a note rather than
    refuse).

    Worth checking because the variable is undocumented: if a future CLI drops
    it, the failure is *silent* in the worst way -- the CLI would talk straight to
    GitHub, everything would appear to work, and Headroom would simply never see
    the traffic. A cheap pre-launch signal beats discovering that from a savings
    report that reads zero.
    """
    env = environ if environ is not None else os.environ
    resolved_home = home if home is not None else os.path.expanduser("~")
    local = env.get("LOCALAPPDATA") or env.get("HOME") or resolved_home
    roots = [
        os.path.join(local, "copilot", "pkg"),
        os.path.join(resolved_home, ".local", "share", "copilot", "pkg"),
    ]

    # Answer from the NEWEST bundle only. ORing over every installed version is
    # unsafe in the direction that matters: an old bundle that still has the
    # variable would vouch for a new one that dropped it, which is precisely the
    # silent-bypass case this probe exists to catch.
    bundles: list[tuple[tuple[int, ...], str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if "app.js" in filenames:
                bundles.append((_version_key(dirpath), os.path.join(dirpath, "app.js")))
    if not bundles:
        return None  # nothing to inspect: unknown, not unsupported

    bundles.sort()
    for _key, path in reversed(bundles):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                # Overlap successive reads so the needle cannot be missed by
                # landing across a chunk boundary (it silently returned False for
                # a supported CLI, which then hard-refused the launch).
                carry = ""
                overlap = len(COPILOT_NATIVE_API_URL_ENV) - 1
                while chunk := fh.read(1 << 20):
                    if COPILOT_NATIVE_API_URL_ENV in carry + chunk:
                        return True
                    carry = chunk[-overlap:] if overlap else ""
        except OSError:
            continue
        # The newest readable bundle is authoritative; do not let older ones vote.
        return False
    # Every located bundle failed to open, or failed mid-read (e.g. deleted or
    # locked concurrently) -- unknown, not confirmed unsupported, since no
    # bundle's content was ever actually inspected.
    return None


def _version_key(path: str) -> tuple[int, ...]:
    """Sortable version tuple from a bundle directory name (``.../1.0.77``)."""
    name = os.path.basename(path.rstrip(os.sep))
    parts: list[int] = []
    for piece in name.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def provider_key_source(provider_type: str) -> str:
    """Return the preferred provider key variable for the selected provider type."""
    return "ANTHROPIC_API_KEY" if provider_type == "anthropic" else "OPENAI_API_KEY"


def build_launch_env(
    *,
    port: int,
    provider_type: str,
    wire_api: str | None,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the Copilot BYOK environment for the selected provider type.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because the Copilot CLI cannot send custom headers; the
    proxy strips it and attributes savings per project.
    """
    # Distinguish "caller passed nothing" (use os.environ) from "caller
    # explicitly passed an empty dict" (start fresh — the test/CLI is in
    # charge of which keys to seed). The previous `environ or os.environ`
    # collapsed those two cases because `bool({}) is False`.
    env = dict(environ if environ is not None else os.environ)
    env["COPILOT_PROVIDER_TYPE"] = provider_type
    env.pop("COPILOT_PROVIDER_WIRE_API", None)

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


#: BYOK provider variables that must be absent in native mode. Leaving any one of
#: them set keeps the CLI in BYOK, which is exactly the single-model behaviour
#: native mode exists to escape -- and the failure would look like "native mode
#: silently did nothing".
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
    """Build the launch environment for native (non-BYOK) Copilot routing.

    Sets :data:`COPILOT_NATIVE_API_URL_ENV` to the local proxy and **clears every
    BYOK variable**, so the CLI keeps its own GitHub auth and its own model
    routing. The consequences are the point of the whole mode: the model picker
    stays fully populated, ``/model`` switches the main agent mid-session,
    ``--model auto`` works again, and a custom agent's ``model:`` frontmatter is
    honoured -- none of which BYOK can offer.

    The ``/p/<project>`` prefix is preserved because the Copilot CLI cannot send
    custom headers; per-project savings attribution rides in the base URL, and
    the native surface keeps the prefix on every request.
    """
    env = dict(environ if environ is not None else os.environ)
    base_url = with_project_prefix(f"http://127.0.0.1:{port}", project)
    env[COPILOT_NATIVE_API_URL_ENV] = base_url
    for var in COPILOT_BYOK_ENV_VARS:
        env.pop(var, None)
    return env, [
        f"{COPILOT_NATIVE_API_URL_ENV}={base_url}",
        "COPILOT_AUTH_MODE=github-native",
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
