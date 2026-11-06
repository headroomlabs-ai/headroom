from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[2]
PROPOSAL = ROOT / "docs" / "proposals" / "unified-api-gateway"
SCHEMA = PROPOSAL / "gateway-config.schema.json"
EXAMPLES = PROPOSAL / "examples"
REQUIREMENTS = PROPOSAL / "requirements.json"
SCENARIOS = PROPOSAL / "scenarios.json"


def test_all_examples_validate_against_runtime_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    for example in sorted(EXAMPLES.glob("*.json")):
        errors = sorted(validator.iter_errors(json.loads(example.read_text())), key=str)
        assert errors == [], f"{example.name}: {errors}"


def test_every_requirement_has_expected_stable_scenarios() -> None:
    requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))["requirements"]
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]

    assert [item["id"] for item in requirements] == [f"R{index:02d}" for index in range(1, 20)]
    assert [item["id"] for item in scenarios] == [f"T{index:03d}" for index in range(1, 101)]
    assert Counter(item["requirement"] for item in scenarios) == {
        **{f"R{index:02d}": 5 for index in range(1, 19)},
        "R19": 10,
    }
