"""Runtime helpers for Mistral Vibe integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from headroom.providers.codex import proxy_base_url as codex_proxy_base_url
from headroom.proxy.project_context import with_project_prefix


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Mistral Vibe through the local proxy.

    Mistral Vibe uses a provider configuration system with `api_base` field.
    It supports overriding providers via the `VIBE_PROVIDERS` environment variable
    as a JSON array. When routing through Headroom, we set the mistral provider's
    `api_base` to the local proxy URL. The proxy will then forward requests to
    the actual Mistral API.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because Vibe cannot send custom headers; the proxy
    strips it and attributes savings per project.
    """
    env = dict(environ or os.environ)
    # NOTE: When used with a persistent Headroom deployment (via headroom install),
    # the proxy process runs with environment variables captured at its startup.
    # Mistral Vibe reads MISTRAL_API_KEY from its own process environment (via
    # api_key_env_var below). This will eventually expire, remove and restarting the install resolves
    base_url = with_project_prefix(codex_proxy_base_url(port), project)

    # Build the providers JSON with mistral provider pointing to Headroom proxy
    # We need to override the default mistral provider's api_base
    providers = [
        {
            "name": "mistral",
            "api_base": base_url,
            "api_key_env_var": "MISTRAL_API_KEY",
            "browser_auth_base_url": "https://console.mistral.ai",
            "browser_auth_api_base_url": "https://console.mistral.ai/api",
            "backend": "mistral",
        }
    ]

    env["VIBE_PROVIDERS"] = json.dumps(providers)

    return env, [f"VIBE_PROVIDERS={json.dumps(providers)}"]
