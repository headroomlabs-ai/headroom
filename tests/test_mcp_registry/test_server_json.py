"""Tests for the canonical MCP server.json descriptor."""

from __future__ import annotations

from pathlib import Path

from headroom.mcp_registry import build_server_json
from headroom.mcp_registry.install import build_headroom_spec
from headroom.mcp_registry.server_json import (
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

