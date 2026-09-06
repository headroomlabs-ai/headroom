#!/usr/bin/env python3
"""Create and verify immutable Headroom candidate manifests.

The workflow-facing CLI deliberately accepts resolved values only. Git refs,
artifact paths, and rollout configuration are validated before they are placed
in the canonical X-001A contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "release" / "contracts"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[^/\s]+/[^/\s]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validator() -> Draft202012Validator:
    schemas = {path.name: _load_json(path) for path in CONTRACT_DIR.glob("*.schema.json")}
    resources = ((schema["$id"], Resource.from_contents(schema)) for schema in schemas.values())
    return Draft202012Validator(
        schemas["candidate-manifest.schema.json"],
        registry=Registry().with_resources(resources),
        format_checker=FormatChecker(),
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _full_sha(value: str) -> str:
    if not FULL_SHA.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a lowercase full 40-character Git SHA")
    return value


def _repository(value: str) -> str:
    if not REPOSITORY.fullmatch(value):
        raise argparse.ArgumentTypeError("must have owner/repository form")
    return value


def _rollout_identity(path: Path) -> dict[str, Any]:
    rollout = _load_json(path)
    allowed = {
        "schema_version",
        "policy_version",
        "channel",
        "registry_digest",
        "snapshot_digest",
        "unsafe_override",
        "qualification_eligible",
        "qualification_ineligible_reason",
        "features",
    }
    unknown = sorted(set(rollout) - allowed)
    if unknown:
        raise ValueError(f"rollout snapshot contains unsupported fields: {unknown}")
    if rollout.get("qualification_eligible") is not True:
        raise ValueError("candidate rollout must be qualification eligible")
    if rollout.get("unsafe_override") is not False:
        raise ValueError("candidate rollout must not use an unsafe override")
    return rollout


def inventory(args: argparse.Namespace) -> None:
    directory = args.directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"runtime payload path is not a directory: {directory}")
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"runtime payload must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            files.append(
                {
                    "filename": relative,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not files:
        raise ValueError("runtime payload directory contains no files")
    args.output.write_text(
        json.dumps(
            {"schema_version": 1, "files": files},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def create(args: argparse.Namespace) -> None:
    artifact = args.artifact.resolve(strict=True)
    if not artifact.is_file() or artifact.stat().st_size < 1:
        raise ValueError(f"candidate artifact is not a non-empty file: {artifact}")

    created_at = args.created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest = {
        "schema_version": 1,
        "candidate_id": args.candidate_id or f"candidate-{args.source_sha[:12]}",
        "created_at": created_at,
        "source": {
            "repository": args.repository,
            "sha": args.source_sha,
            "branch": "main",
        },
        "artifact": {
            "name": args.package,
            "version": args.version,
            "filename": artifact.name,
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
            "media_type": args.media_type,
        },
        "build": {
            "repository": args.repository,
            "workflow": args.workflow,
            "run_id": args.run_id,
            "run_attempt": args.run_attempt,
            "commit_sha": args.producer_sha,
        },
        "rollout": _rollout_identity(args.rollout),
    }
    if args.runtime_payload:
        runtime_payload = args.runtime_payload.resolve(strict=True)
        if not runtime_payload.is_file() or runtime_payload.stat().st_size < 1:
            raise ValueError(f"runtime payload identity is not a non-empty file: {runtime_payload}")
        manifest["runtime_payload_sha256"] = _sha256(runtime_payload)
    _validator().validate(manifest)
    args.output.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def verify(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)
    _validator().validate(manifest)
    artifact = args.artifact.resolve(strict=True)
    expected = manifest["artifact"]
    mismatches: list[str] = []
    if artifact.name != expected["filename"]:
        mismatches.append("artifact filename")
    if artifact.stat().st_size != expected["size_bytes"]:
        mismatches.append("artifact size")
    if _sha256(artifact) != expected["sha256"]:
        mismatches.append("artifact digest")
    if "runtime_payload_sha256" in manifest:
        if args.runtime_payload is None:
            mismatches.append("runtime payload missing")
        elif (
            _sha256(args.runtime_payload.resolve(strict=True)) != manifest["runtime_payload_sha256"]
        ):
            mismatches.append("runtime payload digest")
    elif args.runtime_payload is not None:
        mismatches.append("unexpected runtime payload")
    if args.source_sha and manifest["source"]["sha"] != args.source_sha:
        mismatches.append("source SHA")
    if args.producer_sha and manifest["build"]["commit_sha"] != args.producer_sha:
        mismatches.append("producer SHA")
    if args.repository and manifest["source"]["repository"] != args.repository:
        mismatches.append("source repository")
    if args.package and expected["name"] != args.package:
        mismatches.append("package name")
    if args.version and expected["version"] != args.version:
        mismatches.append("package version")
    if args.workflow and manifest["build"]["workflow"] != args.workflow:
        mismatches.append("producer workflow")
    if args.run_id and manifest["build"]["run_id"] != args.run_id:
        mismatches.append("producer run ID")
    if args.run_attempt and manifest["build"]["run_attempt"] != args.run_attempt:
        mismatches.append("producer run attempt")
    if mismatches:
        raise ValueError("candidate verification failed: " + ", ".join(mismatches))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    payload = commands.add_parser("inventory")
    payload.add_argument("--directory", type=Path, required=True)
    payload.add_argument("--output", type=Path, required=True)
    payload.set_defaults(handler=inventory)

    make = commands.add_parser("create")
    make.add_argument("--artifact", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--source-sha", type=_full_sha, required=True)
    make.add_argument("--producer-sha", type=_full_sha, required=True)
    make.add_argument("--repository", type=_repository, required=True)
    make.add_argument("--package", required=True)
    make.add_argument("--version", required=True)
    make.add_argument("--workflow", required=True)
    make.add_argument("--run-id", type=_positive, required=True)
    make.add_argument("--run-attempt", type=_positive, required=True)
    make.add_argument("--rollout", type=Path, required=True)
    make.add_argument("--runtime-payload", type=Path)
    make.add_argument("--candidate-id")
    make.add_argument("--created-at")
    make.add_argument("--media-type", default="application/octet-stream")
    make.set_defaults(handler=create)

    check = commands.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--artifact", type=Path, required=True)
    check.add_argument("--runtime-payload", type=Path)
    check.add_argument("--source-sha", type=_full_sha)
    check.add_argument("--producer-sha", type=_full_sha)
    check.add_argument("--repository", type=_repository)
    check.add_argument("--package")
    check.add_argument("--version")
    check.add_argument("--workflow")
    check.add_argument("--run-id", type=_positive)
    check.add_argument("--run-attempt", type=_positive)
    check.set_defaults(handler=verify)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
