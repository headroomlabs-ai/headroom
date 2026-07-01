"""Tests for the canonical MCP server.json descriptor."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.mcp_registry import build_server_json, render_server_json
from headroom.mcp_registry.install import build_headroom_spec
from headroom.mcp_registry.server_json import (
    PYPI_OWNERSHIP_MARKER,
    REPOSITORY_ID,
    REPOSITORY_URL,
    SCHEMA_URL,
    SERVER_DESCRIPTION,
    SERVER_NAME,
    WEBSITE_URL,
    _build_mcp_package_spec,
    load_project_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_build_server_json_uses_project_metadata() -> None:
    metadata = load_project_metadata()
    descriptor = build_server_json(metadata)

    assert descriptor["$schema"] == SCHEMA_URL
    assert descriptor["name"] == SERVER_NAME
    assert descriptor["description"] == SERVER_DESCRIPTION
    assert descriptor["version"] == metadata.version
    assert descriptor["websiteUrl"] == WEBSITE_URL
    assert descriptor["repository"] == {
        "url": REPOSITORY_URL,
        "source": "github",
        "id": REPOSITORY_ID,
    }

    package = descriptor["packages"][0]
    assert package["registryType"] == "pypi"
    assert package["registryBaseUrl"] == "https://pypi.org"
    assert package["identifier"] == metadata.package_name
    assert package["version"] == metadata.version
    assert package["runtimeArguments"] == [
        {
            "type": "named",
            "name": "--from",
            "value": _build_mcp_package_spec(metadata),
        }
    ]


def test_build_server_json_matches_runtime_contract() -> None:
    descriptor = build_server_json()
    runtime = build_headroom_spec()
    package = descriptor["packages"][0]

    assert package["runtimeHint"] == "uvx"
    assert package["runtimeArguments"] == [
        {
            "type": "named",
            "name": "--from",
            "value": _build_mcp_package_spec(load_project_metadata()),
        }
    ]
    assert [arg["value"] for arg in package["packageArguments"]] == [
        runtime.name,
        *runtime.args[-2:],
    ]
    assert package["transport"] == {"type": "stdio"}


def test_root_server_json_matches_builder() -> None:
    artifact = PROJECT_ROOT / "server.json"
    assert artifact.read_text(encoding="utf-8") == render_server_json()
    assert json.loads(artifact.read_text(encoding="utf-8")) == build_server_json()


def test_docs_point_to_canonical_server_json() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    mcp_docs = (PROJECT_ROOT / "docs/content/docs/mcp.mdx").read_text(encoding="utf-8")

    assert PYPI_OWNERSHIP_MARKER in readme
    assert "`server.json`" in readme
    assert "https://github.com/headroomlabs-ai/headroom/blob/main/server.json" in mcp_docs
