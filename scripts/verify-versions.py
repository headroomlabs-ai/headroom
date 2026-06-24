#!/usr/bin/env python3
"""Verify all package manifest versions are in sync before publishing."""

import json
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

ROOT = Path(__file__).parent.parent


def _read_json_version(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return str(json.load(f)["version"])


def _read_marketplace_versions(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    versions: dict[str, str] = {}
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        versions[f"{path}:metadata"] = str(metadata.get("version"))
    plugins = payload.get("plugins")
    if isinstance(plugins, list):
        for index, plugin in enumerate(plugins):
            if isinstance(plugin, dict):
                versions[f"{path}:plugins[{index}]"] = str(plugin.get("version"))
    return versions


def main() -> None:
    with open(ROOT / "pyproject.toml", "rb") as f:
        py_ver = tomllib.load(f)["project"]["version"]

    versions = {
        "pyproject.toml": py_ver,
        "plugins/openclaw/package.json": _read_json_version(ROOT / "plugins/openclaw/package.json"),
        "sdk/typescript/package.json": _read_json_version(ROOT / "sdk/typescript/package.json"),
        "plugins/headroom-agent-hooks/.claude-plugin/plugin.json": _read_json_version(
            ROOT / "plugins/headroom-agent-hooks/.claude-plugin/plugin.json"
        ),
        "plugins/headroom-agent-hooks/.github/plugin/plugin.json": _read_json_version(
            ROOT / "plugins/headroom-agent-hooks/.github/plugin/plugin.json"
        ),
    }
    versions.update(_read_marketplace_versions(ROOT / ".claude-plugin/marketplace.json"))
    versions.update(_read_marketplace_versions(ROOT / ".github/plugin/marketplace.json"))

    if not all(v == py_ver for v in versions.values()):
        print("Version mismatch detected:")
        for file, ver in versions.items():
            print(f"  {file}: {ver}")
        print(f"Expected all to be: {py_ver}")
        raise SystemExit(1)

    print(f"All versions aligned at {py_ver}")
    print("Packages:", ", ".join(versions.keys()))


if __name__ == "__main__":
    main()
