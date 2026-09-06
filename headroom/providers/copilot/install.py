"""Copilot install-time helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .wrap import (
    COPILOT_NATIVE_API_URL_ENV,
    build_launch_env,
    build_native_launch_env,
    resolve_provider_type,
)

#: Backends that speak Anthropic/OpenAI wire formats natively and can therefore
#: forward the Copilot CLI's own GitHub-authenticated traffic to GitHub's API.
#: Translated backends (any-llm, LiteLLM, Bedrock, Vertex) cannot serve
#: GitHub's hosted models, so they keep the single-model BYOK lane.
_NATIVE_CAPABLE_BACKENDS: frozenset[str] = frozenset({"", "anthropic"})


def install_uses_native_lane(backend: str, environ: Mapping[str, str] | None = None) -> bool:
    """Decide between the native (GitHub-hosted) and BYOK lanes for an install.

    Native is the default: an installed Copilot CLI keeps its model picker and
    Auto mode, and the proxy forwards the developer's own Copilot token to
    GitHub. BYOK is chosen only when the operator has stated that intent with
    ``COPILOT_PROVIDER_API_KEY`` in the install-time environment, or when the
    proxy backend is a translated one that cannot reach GitHub's hosted API.
    """
    env = environ if environ is not None else os.environ
    if (env.get("COPILOT_PROVIDER_API_KEY") or "").strip():
        return False
    return (backend or "").strip().lower() in _NATIVE_CAPABLE_BACKENDS


def build_install_env(
    *, port: int, backend: str, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the persistent install environment for Copilot.

    Native lane: ``COPILOT_API_URL`` only. The CLI resolves its API host as
    ``settings.copilotUrl || COPILOT_API_URL || token.endpoints.api``, so this
    variable redirects GitHub-authenticated traffic through the proxy without
    touching model selection. The proxy recognises the CLI's Copilot bearer
    token and forwards it to GitHub even when its OpenAI target is the stock
    ``api.openai.com``, so no target pin is needed in the manifest.
    """
    if install_uses_native_lane(backend, environ):
        env, _lines = build_native_launch_env(port=port, environ={})
        return {COPILOT_NATIVE_API_URL_ENV: env[COPILOT_NATIVE_API_URL_ENV]}

    provider_type = resolve_provider_type(backend, "auto", {"HEADROOM_BACKEND": backend})
    env, _lines = build_launch_env(
        port=port,
        provider_type=provider_type,
        wire_api=None,
        environ={},
    )
    return {
        key: env[key]
        for key in (
            "COPILOT_PROVIDER_TYPE",
            "COPILOT_PROVIDER_BASE_URL",
            "COPILOT_PROVIDER_WIRE_API",
        )
        if key in env
    }
