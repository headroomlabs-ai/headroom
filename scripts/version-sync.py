#!/usr/bin/env python3
"""Synchronize version across all headroom packages."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def get_version_from_pyproject(root: Path) -> str:
    """Read version from pyproject.toml."""
    pyproject_path = root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def bump_version(version: str, bump_type: str) -> str:
    """Bump version according to bump_type (major, minor, patch)."""
    major, minor, patch = map(int, version.split("."))
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    return f"{major}.{minor}.{patch}"


def update_package_json(file_path: Path, version: str) -> None:
    """Update a package.json version field."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = version
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_plugin_manifest(file_path: Path, version: str) -> None:
    """Update a plugin.json version field."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = version
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_marketplace_manifest(file_path: Path, version: str) -> None:
    """Update marketplace metadata and plugin entry versions."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata["version"] = version
    plugins = data.get("plugins")
    if isinstance(plugins, list):
        for plugin in plugins:
            if isinstance(plugin, dict):
                plugin["version"] = version
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_plugin_versions(root: Path, version: str) -> None:
    """Update marketplace and plugin manifest versions."""
    update_marketplace_manifest(root / ".claude-plugin" / "marketplace.json", version)
    update_marketplace_manifest(root / ".github" / "plugin" / "marketplace.json", version)
    update_plugin_manifest(
        root / "plugins" / "headroom-agent-hooks" / ".claude-plugin" / "plugin.json", version
    )
    update_plugin_manifest(
        root / "plugins" / "headroom-agent-hooks" / ".github" / "plugin" / "plugin.json",
        version,
    )


def update_openclaw_package_json(file_path: Path, version: str, sdk_version: str) -> None:
    """Update openclaw package.json version and headroom-ai dependency range."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = version
    if "dependencies" in data and "headroom-ai" in data["dependencies"]:
        data["dependencies"]["headroom-ai"] = f"^{sdk_version}"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_pyproject_version(root: Path, version: str) -> None:
    """Update pyproject.toml version."""
    pyproject_path = root / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    updated = re.sub(
        r'^version = "[^"]+"',
        f'version = "{version}"',
        content,
        flags=re.MULTILINE,
    )
    pyproject_path.write_text(updated, encoding="utf-8")


def write_release_metadata(root: Path, version: str) -> None:
    """Write .releaseetadata JSON file."""
    metadata = {
        "version": version,
        "packages": {
            "pypi": version,
            "npm-sdk": version,
            "npm-openclaw": version,
            "agent-hooks-plugin": version,
        },
    }
    metadata_path = root / ".releaseetadata"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize version across headroom packages")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root directory of the project",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--version", help="Explicit version to set (e.g., 0.6.0)")
    group.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version from pyproject.toml",
    )
    parser.add_argument(
        "--plugin-manifests-only",
        action="store_true",
        help="Only update marketplace/plugin manifest versions",
    )
    args = parser.parse_args()

    if args.version:
        version = args.version
    elif args.bump:
        base_version = get_version_from_pyproject(args.root)
        version = bump_version(base_version, args.bump)
    else:
        version = get_version_from_pyproject(args.root)

    if args.plugin_manifests_only:
        update_plugin_versions(args.root, version)
        print(f"Plugin versions synchronized to {version}")
        return

    # Update all versioned files
    update_pyproject_version(args.root, version)
    update_openclaw_package_json(
        args.root / "plugins" / "openclaw" / "package.json", version, version
    )
    update_package_json(args.root / "sdk" / "typescript" / "package.json", version)
    update_plugin_versions(args.root, version)
    write_release_metadata(args.root, version)

    print(f"Version synchronized to {version}")


if __name__ == "__main__":
    main()
