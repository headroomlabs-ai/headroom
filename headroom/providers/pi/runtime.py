"""Runtime helpers for Pi coding agent integrations."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.providers.claude import proxy_base_url as anthropic_proxy_base_url
from headroom.providers.codex import proxy_base_url as openai_proxy_base_url

DEFAULT_PROJECT_HEADER_NAME = "X-Headroom-Project"


def gemini_proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Gemini integrations."""
    return f"http://127.0.0.1:{port}/v1beta"


def vertex_proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Vertex integrations."""
    return f"http://127.0.0.1:{port}"


def proxy_provider_configs(
    port: int,
    *,
    project: str | None = None,
    project_header: str = DEFAULT_PROJECT_HEADER_NAME,
) -> dict[str, dict[str, object]]:
    """Provider overrides used by the transient Pi extension."""
    headers: dict[str, str] = {}
    if project:
        headers[project_header] = project

    def _config(base_url: str) -> dict[str, object]:
        config: dict[str, object] = {"baseUrl": base_url}
        if headers:
            config["headers"] = headers
        return config

    openai_base = openai_proxy_base_url(port)
    return {
        "anthropic": _config(anthropic_proxy_base_url(port)),
        "google": _config(gemini_proxy_base_url(port)),
        "google-vertex": _config(vertex_proxy_base_url(port)),
        "openai": _config(openai_base),
        "openai-codex": _config(openai_base),
    }


def build_proxy_extension_source(
    port: int,
    *,
    project: str | None = None,
    project_header: str = DEFAULT_PROJECT_HEADER_NAME,
) -> str:
    """Build a Pi extension that routes built-in providers through Headroom."""
    providers_json = json.dumps(
        proxy_provider_configs(port, project=project, project_header=project_header),
        indent=2,
    )
    return (
        "export default function (pi) {\n"
        f"  const providers = {providers_json};\n"
        "  for (const [name, config] of Object.entries(providers)) {\n"
        "    pi.registerProvider(name, config);\n"
        "  }\n"
        "}\n"
    )


def write_proxy_extension(
    directory: Path,
    port: int,
    *,
    project: str | None = None,
    project_header: str = DEFAULT_PROJECT_HEADER_NAME,
) -> Path:
    """Write the transient Pi provider override extension."""
    extension_path = directory / "headroom-pi-provider.ts"
    extension_path.write_text(
        build_proxy_extension_source(port, project=project, project_header=project_header),
        encoding="utf-8",
    )
    return extension_path
