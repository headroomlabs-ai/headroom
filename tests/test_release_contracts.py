"""Contract tests for release policy documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

from headroom.rollout import resolve_rollout

CONTRACT_DIR = Path(__file__).parents[1] / "release" / "contracts"
EXAMPLE_DIR = CONTRACT_DIR / "examples"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict[str, Any]]:
    return {path.name: _read_json(path) for path in CONTRACT_DIR.glob("*.schema.json")}


@pytest.fixture(scope="module")
def registry(schemas: dict[str, dict[str, Any]]) -> Registry:
    resources = ((schema["$id"], Resource.from_contents(schema)) for schema in schemas.values())
    return Registry().with_resources(resources)


def _validator(
    schema_name: str,
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> Draft202012Validator:
    return Draft202012Validator(
        schemas[schema_name],
        registry=registry,
        format_checker=FormatChecker(),
    )


def _policy() -> dict[str, Any]:
    return _read_json(EXAMPLE_DIR / "policy.valid.json")


def test_policy_schema_is_valid_draft_2020_12(
    schemas: dict[str, dict[str, Any]],
) -> None:
    Draft202012Validator.check_schema(schemas["common.schema.json"])
    Draft202012Validator.check_schema(schemas["policy.schema.json"])


def test_checked_in_policy_example_validates(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    _validator("policy.schema.json", schemas, registry).validate(_policy())


def test_unknown_policy_schema_version_fails_visibly(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _policy()
    document["schema_version"] = 2
    with pytest.raises(ValidationError, match="1 was expected"):
        _validator("policy.schema.json", schemas, registry).validate(document)


def test_policy_rejects_unknown_fields(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _policy()
    document["publish_without_gate"] = True
    with pytest.raises(ValidationError, match="Additional properties"):
        _validator("policy.schema.json", schemas, registry).validate(document)


def test_runtime_rollout_snapshot_matches_common_contract(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://headroom.dev/schemas/release/v1/runtime-rollout-check.schema.json",
        "$ref": "common.schema.json#/$defs/rolloutIdentity",
    }
    Draft202012Validator(
        wrapper,
        registry=registry,
        format_checker=FormatChecker(),
    ).validate(resolve_rollout({}).to_dict())
