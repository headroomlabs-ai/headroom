from __future__ import annotations

import json
from pathlib import Path

MATRIX_PATH = Path("docs/platform-feature-matrix.json")
VALID_STATUSES = {"covered", "partial", "gap", "blocked"}
PLATFORMS = {"linux", "macos", "windows"}


def test_platform_feature_matrix_is_complete() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    assert matrix["schema_version"] == 1
    assert set(matrix["platforms"]) == PLATFORMS
    assert set(matrix["status_values"]) == VALID_STATUSES
    assert matrix["features"], "matrix must list hardening features"

    feature_ids: set[str] = set()
    for feature in matrix["features"]:
        feature_id = feature["id"]
        assert feature_id not in feature_ids, f"duplicate feature id: {feature_id}"
        feature_ids.add(feature_id)
        assert feature["name"]
        assert feature["risk"] in {"install", "runtime", "performance", "cache", "proxy"}
        assert set(feature["platforms"]) == PLATFORMS

        for platform, coverage in feature["platforms"].items():
            status = coverage["status"]
            assert status in VALID_STATUSES, f"{feature_id}/{platform} has invalid status"
            assert coverage["tests"], f"{feature_id}/{platform} must cite tests or workflows"
            for test_ref in coverage["tests"]:
                path = Path(test_ref.split("#", 1)[0])
                assert path.exists(), f"{feature_id}/{platform} cites missing path {test_ref}"
            if status in {"partial", "gap", "blocked"}:
                assert coverage.get("gap"), f"{feature_id}/{platform} must explain {status}"


def test_platform_feature_matrix_covers_issue_1843_regression_areas() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    feature_ids = {feature["id"] for feature in matrix["features"]}

    assert {
        "install_windows_service",
        "single_instance_start",
        "compression_fail_open",
        "proxy_functional_smoke",
        "ccr_persistence",
        "toin_skip_recommendations",
    } <= feature_ids


def test_platform_feature_matrix_sanity_tests_are_enumerated() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    sanity_ids = {item["id"] for item in matrix["sanity_tests"]}

    assert {
        "cli_help",
        "install_paths",
        "runtime_selection",
        "health_startup",
        "compression_backpressure",
        "proxy_route_smoke",
    } <= sanity_ids
    for item in matrix["sanity_tests"]:
        assert item["description"]
        assert item["tests"]
        for test_ref in item["tests"]:
            path = Path(test_ref.split("#", 1)[0])
            assert path.exists(), f"{item['id']} cites missing path {test_ref}"
