"""Keep the installed MCP SDK compatible with Headroom's live server API."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def _mcp_requirement(dependencies: list[str]) -> Requirement:
    matches = [Requirement(value) for value in dependencies if Requirement(value).name == "mcp"]
    assert len(matches) == 1, "expected exactly one mcp requirement"
    return matches[0]


def test_all_shipping_mcp_extras_exclude_incompatible_sdk_v2() -> None:
    """SDK 2 removed Server.list_tools()/call_tool(); #2658 owns that port."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project["optional-dependencies"]

    for extra in ("proxy", "mcp"):
        requirement = _mcp_requirement(optional[extra])
        assert requirement.specifier.contains("1.28.1")
        assert not requirement.specifier.contains("2.0.0"), (
            f"the {extra!r} extra admits MCP SDK 2.x before the server port in #2658"
        )
