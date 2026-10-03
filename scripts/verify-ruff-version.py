#!/usr/bin/env python3
"""Verify that every Ruff execution path uses the dev dependency pin."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
RUFF_PRE_COMMIT_REPO = "https://github.com/astral-sh/ruff-pre-commit"
WORKFLOW_VERSION_COMMAND = "python scripts/verify-ruff-version.py --print-version"
WORKFLOW_INSTALL_REFERENCE = "steps.ruff-version.outputs.version"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return cast(dict[str, Any], tomllib.load(file))


def _authoritative_version() -> str:
    dependencies = _load_toml(ROOT / "pyproject.toml")["project"]["optional-dependencies"]["dev"]
    ruff_requirements = [
        requirement for requirement in dependencies if requirement.startswith("ruff")
    ]
    if len(ruff_requirements) != 1:
        raise ValueError(f"expected one Ruff dev dependency, found {ruff_requirements!r}")

    match = re.fullmatch(r"ruff==([^;,\s]+)", ruff_requirements[0])
    if match is None:
        raise ValueError(
            "pyproject.toml must contain one exact Ruff pin in project.optional-dependencies.dev"
        )
    return match.group(1)


def _locked_version() -> str:
    packages = _load_toml(ROOT / "uv.lock")["package"]
    versions = [str(package["version"]) for package in packages if package["name"] == "ruff"]
    if len(versions) != 1:
        raise ValueError(f"expected one locked Ruff package, found {versions!r}")
    return versions[0]


def _pre_commit_version() -> str:
    lines = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f"- repo: {RUFF_PRE_COMMIT_REPO}":
            continue
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if stripped.startswith("- repo:"):
                break
            if stripped.startswith("rev:"):
                return stripped.removeprefix("rev:").strip().removeprefix("v")
        break
    raise ValueError(f"could not find a rev for {RUFF_PRE_COMMIT_REPO}")


def _workflow_errors() -> list[str]:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    errors = []
    if WORKFLOW_VERSION_COMMAND not in workflow:
        errors.append(f"ci.yml does not run {WORKFLOW_VERSION_COMMAND!r}")
    if WORKFLOW_INSTALL_REFERENCE not in workflow:
        errors.append(f"ci.yml does not install Ruff from {WORKFLOW_INSTALL_REFERENCE!r}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="print the authoritative version after validating every execution path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        authoritative = _authoritative_version()
        versions = {
            "pyproject.toml": authoritative,
            "uv.lock": _locked_version(),
            ".pre-commit-config.yaml": _pre_commit_version(),
        }
        errors = [
            f"{path} uses Ruff {version}, expected {authoritative}"
            for path, version in versions.items()
            if version != authoritative
        ]
        errors.extend(_workflow_errors())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Ruff version verification failed: {exc}")
        raise SystemExit(1) from exc

    if errors:
        print("Ruff version mismatch detected:")
        for message in errors:
            print(f"  {message}")
        raise SystemExit(1)

    if args.print_version:
        print(authoritative)
    else:
        print(f"Ruff versions aligned at {authoritative}")


if __name__ == "__main__":
    main()
