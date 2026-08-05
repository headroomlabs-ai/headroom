"""Runtime helpers for Kimi CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.providers.codex import proxy_base_url as codex_proxy_base_url
from headroom.proxy.project_context import with_project_prefix


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Kimi CLI through the local proxy.

    Kimi CLI (``kimi`` / ``kimi-cli``) talks to its managed coding endpoint with
    an OpenAI-compatible ``/chat/completions`` client (``kosong``'s ``Kimi``
    provider wraps ``AsyncOpenAI``). Its base URL is overridable via the
    ``KIMI_BASE_URL`` environment variable, so we point it at the local proxy.
    The proxy forwards the request — including Kimi's own OAuth ``Authorization``
    bearer (passthrough auth mode) — to the real upstream configured by
    ``--openai-api-url`` (``https://api.kimi.com/coding/v1``).

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because the Kimi base-URL override cannot carry custom
    headers; the proxy strips it and attributes savings per project.
    """
    env = dict(environ or os.environ)
    base_url = with_project_prefix(codex_proxy_base_url(port), project)
    env["KIMI_BASE_URL"] = base_url
    return env, [f"KIMI_BASE_URL={base_url}"]
