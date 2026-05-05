"""Copilot wrapper provider helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import click


def resolve_provider_type(
    backend: str | None, provider_type: str, environ: Mapping[str, str] | None = None
) -> str:
    """Resolve Copilot BYOK provider type for the current proxy backend."""
    if provider_type != "auto":
        return provider_type

    env = environ or os.environ
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


def provider_key_source(provider_type: str) -> str:
    """Return the preferred provider key variable for the selected provider type."""
    return "ANTHROPIC_API_KEY" if provider_type == "anthropic" else "OPENAI_API_KEY"


def build_launch_env(
    *,
    port: int,
    provider_type: str,
    wire_api: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the Copilot BYOK environment for the selected provider type."""
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
        base_url = f"http://127.0.0.1:{port}"
        env["COPILOT_PROVIDER_BASE_URL"] = base_url
        return env, [
            "COPILOT_PROVIDER_TYPE=anthropic",
            f"COPILOT_PROVIDER_BASE_URL={base_url}",
        ]

    effective_wire_api = wire_api or "completions"
    base_url = f"http://127.0.0.1:{port}/v1"
    env["COPILOT_PROVIDER_BASE_URL"] = base_url
    env["COPILOT_PROVIDER_WIRE_API"] = effective_wire_api
    return env, [
        "COPILOT_PROVIDER_TYPE=openai",
        f"COPILOT_PROVIDER_BASE_URL={base_url}",
        f"COPILOT_PROVIDER_WIRE_API={effective_wire_api}",
    ]


def model_configured(copilot_args: tuple[str, ...], env: Mapping[str, str]) -> bool:
    """Return True when Copilot BYOK model selection is configured."""
    if env.get("COPILOT_MODEL") or env.get("COPILOT_PROVIDER_MODEL_ID"):
        return True

    for idx, arg in enumerate(copilot_args):
        if arg == "--model" and idx + 1 < len(copilot_args):
            return True
        if arg.startswith("--model="):
            return True

    return False
