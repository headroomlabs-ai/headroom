from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.unified_gateway.scenario_coverage import load_scenario_coverage


def test_every_scenario_has_an_executable_or_explicit_external_cell() -> None:
    coverage = load_scenario_coverage()
    assert set(coverage) == {f"T{index:03d}" for index in range(1, 101)}
    assert all(
        cell.status in {"local_test", "unavailable_test", "external_not_run"}
        for cell in coverage.values()
    )
    assert all(
        cell.test_nodes
        for cell in coverage.values()
        if cell.status in {"local_test", "unavailable_test"}
    )


def test_unavailable_capabilities_are_classified_as_negative_evidence() -> None:
    coverage = load_scenario_coverage()
    unavailable = {
        "T005",
        "T023",
        "T031",
        "T033",
        "T065",
        "T074",
        "T078",
        "T080",
        "T085",
        "T087",
        "T088",
        "T089",
        "T091",
        "T099",
    }

    assert {
        scenario_id for scenario_id, cell in coverage.items() if cell.status == "unavailable_test"
    } == unavailable


def test_local_coverage_uses_concrete_pytest_node_ids() -> None:
    coverage = load_scenario_coverage()

    for scenario_id, cell in coverage.items():
        if cell.status not in {"local_test", "unavailable_test"}:
            continue
        assert all("::" in node for node in cell.test_nodes), (
            f"{scenario_id} maps to a broad path instead of a concrete pytest node"
        )


def test_scenario_ledger_status_matches_executable_coverage() -> None:
    scenarios_path = (
        Path(__file__).parents[2] / "docs" / "proposals" / "unified-api-gateway" / "scenarios.json"
    )
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))["scenarios"]
    coverage = load_scenario_coverage()

    expected = {
        "local_test": "proved_locally",
        "unavailable_test": "unavailable_negatively_tested",
        "external_not_run": "external_not_run",
    }
    assert {scenario["id"]: scenario["execution_status"] for scenario in scenarios} == {
        scenario_id: expected[cell.status] for scenario_id, cell in coverage.items()
    }


def test_every_local_coverage_node_is_collectable() -> None:
    nodes = sorted(
        {
            node
            for cell in load_scenario_coverage().values()
            if cell.status in {"local_test", "unavailable_test"}
            for node in cell.test_nodes
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
