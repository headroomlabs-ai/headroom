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
    """Choose the Copilot OpenAI-compatible wire API for a model."""
    return "responses" if model_prefers_responses_api(model) else "completions"


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
