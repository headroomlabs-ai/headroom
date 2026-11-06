"""Malformed v1 policies must fail before a controller can evaluate them."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource

CONTRACTS = Path(__file__).resolve().parents[1] / "release/contracts"


@pytest.fixture(scope="module")
def validator():
    schemas = [
        json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
        for name in ("common.schema.json", "policy.schema.json")
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas
    )
    return Draft202012Validator(schemas[1], registry=registry, format_checker=FormatChecker())


@pytest.fixture
def policy():
    return json.loads((CONTRACTS / "examples/policy.valid.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("gate", ["gate_a", "gate_b"])
@pytest.mark.parametrize("field", ["required_evidence", "rules"])
def test_release_gate_cannot_define_zero_requirements(validator, policy, gate, field):
    policy["risk_classes"]["R3"][gate][field] = []
    with pytest.raises(ValidationError):
        validator.validate(policy)


@pytest.mark.parametrize("gate", ["gate_a", "gate_b"])
def test_release_policy_rejects_unregistered_rule_kinds(validator, policy, gate):
    policy["risk_classes"]["R3"][gate]["rules"][0]["kind"] = "required_stauts"
    with pytest.raises(ValidationError):
        validator.validate(policy)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"accepted_stauts": "pass"},
        {"accepted_status": "fail"},
        {"accepted_status": "inconclusive"},
        {"accepted_status": "skipped_by_policy"},
        {"accepted_status": 1},
        {"accepted_status": "pass", "ignore_missing": True},
    ],
)
def test_required_status_rule_requires_explicit_passing_status(validator, policy, parameters):
    policy["risk_classes"]["R3"]["gate_a"]["rules"][0]["parameters"] = parameters
    with pytest.raises(ValidationError):
        validator.validate(policy)


@pytest.mark.parametrize("parameters", [{"revoked": False}, {"accepted_status": "pass"}])
def test_not_revoked_rule_rejects_unrecognized_parameters(validator, policy, parameters):
    policy["risk_classes"]["R3"]["gate_b"]["rules"][0]["parameters"] = parameters
    with pytest.raises(ValidationError):
        validator.validate(policy)
