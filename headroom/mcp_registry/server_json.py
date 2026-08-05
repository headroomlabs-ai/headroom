"""Canonical MCP publication metadata for the Headroom server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .install import build_headroom_spec

SCHEMA_URL = "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"
SERVER_NAME = "io.github.headroomlabs-ai/headroom"
SERVER_TITLE = "Headroom"
SERVER_DESCRIPTION = (
    "Headroom MCP server for compression, retrieval, and stats in MCP-compatible hosts."
)
PYPI_OWNERSHIP_MARKER = f"<!-- mcp-name: {SERVER_NAME} -->"
WEBSITE_URL = "https://headroomlabs-ai.github.io/headroom/"
REPOSITORY_URL = "https://github.com/headroomlabs-ai/headroom"
REPOSITORY_ID = "1129940957"
PYPI_REGISTRY_URL = "https://pypi.org"


@dataclass(frozen=True)
class ProjectMetadata:
    """Package metadata needed for the published MCP descriptor."""

    package_name: str
    version: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_metadata(pyproject_path: Path | None = None) -> ProjectMetadata:
    """Load the publishable package metadata from ``pyproject.toml``."""

    path = pyproject_path or (_project_root() / "pyproject.toml")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data["project"]
    return ProjectMetadata(
        package_name=project["name"],
        version=project["version"],
    )


def _build_runtime_contract() -> tuple[str, tuple[str, str]]:
    """Return the canonical CLI entrypoint plus the MCP subcommand tail."""

    spec = build_headroom_spec()
    if spec.name != "headroom":
        raise ValueError(f"unexpected MCP server name: {spec.name}")
    runtime_tail = spec.args[-2:]
    if runtime_tail != ("mcp", "serve"):
        raise ValueError(f"unexpected MCP launch tail: {spec.args}")
    return spec.name, (runtime_tail[0], runtime_tail[1])


def _build_mcp_package_spec(metadata: ProjectMetadata) -> str:
    """Return the PyPI requirement needed to launch ``headroom mcp serve``."""

    return f"{metadata.package_name}[mcp]"


def build_server_json(metadata: ProjectMetadata | None = None) -> dict[str, object]:
    """Build the canonical ``server.json`` payload for Headroom."""

    metadata = metadata or load_project_metadata()
    command_name, runtime_tail = _build_runtime_contract()
    package_spec = _build_mcp_package_spec(metadata)
    return {
        "$schema": SCHEMA_URL,
        "name": SERVER_NAME,
        "description": SERVER_DESCRIPTION,
        "title": SERVER_TITLE,
        "websiteUrl": WEBSITE_URL,
        "repository": {
            "url": REPOSITORY_URL,
            "source": "github",
            "id": REPOSITORY_ID,
        },
        "version": metadata.version,
        "packages": [
            {
                "registryType": "pypi",
                "registryBaseUrl": PYPI_REGISTRY_URL,
                "identifier": metadata.package_name,
                "version": metadata.version,
                "runtimeHint": "uvx",
                # Current uvx needs --from when the package name and script name differ.
                "runtimeArguments": [
                    {
                        "type": "named",
                        "name": "--from",
                        "value": package_spec,
                    }
                ],
                "transport": {
                    "type": "stdio",
                },
                "packageArguments": [
                    {
                        "type": "positional",
                        "value": command_name,
                    },
                    *(
                        {
                            "type": "positional",
                            "value": value,
                        }
                        for value in runtime_tail
                    ),
                ],
            }
        ],
    }


def render_server_json(metadata: ProjectMetadata | None = None) -> str:
    """Render the canonical ``server.json`` payload with stable formatting."""

    return json.dumps(build_server_json(metadata), indent=2) + "\n"
