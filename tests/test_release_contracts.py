"""Contract tests for immutable release evidence documents."""

from __future__ import annotations

import copy
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


def _example(name: str) -> dict[str, Any]:
    return _read_json(EXAMPLE_DIR / f"{name}.valid.json")


def test_all_schemas_are_valid_draft_2020_12(
    schemas: dict[str, dict[str, Any]],
) -> None:
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "name",
    [
        "candidate-manifest",
        "gate-result",
        "policy",
        "integration-result",
        "benchmark-result-ref",
        "qualification-manifest",
    ],
)
def test_checked_in_examples_validate(
    name: str,
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    _validator(f"{name}.schema.json", schemas, registry).validate(_example(name))


def test_runtime_rollout_snapshot_matches_release_contract(
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


@pytest.mark.parametrize(
    "name",
    [
        "candidate-manifest",
        "gate-result",
        "policy",
        "integration-result",
        "benchmark-result-ref",
        "qualification-manifest",
    ],
)
def test_unknown_schema_versions_fail_visibly(
    name: str,
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example(name)
    document["schema_version"] = 2
    with pytest.raises(ValidationError, match="1 was expected"):
        _validator(f"{name}.schema.json", schemas, registry).validate(document)


def test_gate_cannot_pass_with_reasons(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("gate-result")
    document["reasons"] = ["required benchmark missing"]
    with pytest.raises(ValidationError):
        _validator("gate-result.schema.json", schemas, registry).validate(document)


def test_inconclusive_gate_requires_a_reason(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("gate-result")
    document["status"] = "inconclusive"
    with pytest.raises(ValidationError):
        _validator("gate-result.schema.json", schemas, registry).validate(document)


def test_unsafe_rollout_cannot_qualify_as_pass(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("qualification-manifest")
    document["rollout"].update(
        {
            "unsafe_override": True,
            "qualification_eligible": False,
            "qualification_ineligible_reason": "unsafe_rollout_override_active",
        }
    )
    with pytest.raises(ValidationError):
        _validator("qualification-manifest.schema.json", schemas, registry).validate(document)


def test_revoked_qualification_cannot_remain_pass(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("qualification-manifest")
    document["revocation"] = {
        "revoked_at": "2026-08-13T07:00:00Z",
        "revoked_by": "release-owner",
        "reason": "candidate regression",
    }
    with pytest.raises(ValidationError):
        _validator("qualification-manifest.schema.json", schemas, registry).validate(document)


def test_failed_evidence_cannot_appear_in_passing_qualification(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("qualification-manifest")
    document["evidence"][0]["status"] = "fail"
    with pytest.raises(ValidationError):
        _validator("qualification-manifest.schema.json", schemas, registry).validate(document)


def test_benchmark_identity_mismatch_forces_inconclusive_result(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("benchmark-result-ref")
    document["identity_verification"] = {
        "status": "inconclusive",
        "mismatches": ["a1.artifact_sha256 != b.artifact_sha256"],
    }
    with pytest.raises(ValidationError):
        _validator("benchmark-result-ref.schema.json", schemas, registry).validate(document)

    document["status"] = "inconclusive"
    _validator("benchmark-result-ref.schema.json", schemas, registry).validate(document)


def test_a1_and_b_optimization_states_are_fixed(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    validator = _validator("benchmark-result-ref.schema.json", schemas, registry)
    document = _example("benchmark-result-ref")
    document["experiment"]["a1"]["optimization_enabled"] = True
    with pytest.raises(ValidationError):
        validator.validate(document)

    document = copy.deepcopy(_example("benchmark-result-ref"))
    document["experiment"]["b"]["optimization_enabled"] = False
    with pytest.raises(ValidationError):
        validator.validate(document)


def test_gate_a_rejects_runtime_payload_equivalence(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("gate-result")
    document["runtime_payload_equivalence"] = _runtime_payload_equivalence()
    with pytest.raises(ValidationError):
        _validator("gate-result.schema.json", schemas, registry).validate(document)


def test_gate_b_requires_artifact_or_payload_equivalence(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("gate-result")
    document["gate"] = "gate_b"
    with pytest.raises(ValidationError, match="runtime_payload_equivalence"):
        _validator("gate-result.schema.json", schemas, registry).validate(document)


def test_passing_gate_b_rejects_inconclusive_payload_identity(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("gate-result")
    document["gate"] = "gate_b"
    document["runtime_payload_equivalence"] = _runtime_payload_equivalence()
    document["runtime_payload_equivalence"]["identity_verification"] = {
        "status": "inconclusive",
        "mismatches": ["qualified payload != publication payload"],
    }
    with pytest.raises(ValidationError):
        _validator("gate-result.schema.json", schemas, registry).validate(document)


def _runtime_payload_equivalence() -> dict[str, Any]:
    return {
        "method": "runtime_payload",
        "qualified_artifact_sha256": "sha256:" + "a" * 64,
        "publication_artifact_sha256": "sha256:" + "b" * 64,
        "qualified_runtime_payload_sha256": "sha256:" + "c" * 64,
        "publication_runtime_payload_sha256": "sha256:" + "c" * 64,
        "identity_verification": {"status": "pass", "mismatches": []},
        "verifier": {
            "repository": "headroomlabs-ai/headroom",
            "workflow": "release-policy.yml",
            "run_id": 123456792,
            "run_attempt": 1,
            "commit_sha": "3077ac81e8ef3ddefebbe308ea37a4e9bb2100e6",
        },
    }


def test_contracts_reject_unknown_fields(
    schemas: dict[str, dict[str, Any]],
    registry: Registry,
) -> None:
    document = _example("candidate-manifest")
    document["publish_without_gate"] = True
    with pytest.raises(ValidationError, match="Additional properties"):
        _validator("candidate-manifest.schema.json", schemas, registry).validate(document)
