"""Tests for manifest assembly (deterministic given injected ``now``)."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_evals.config import Settings
from agent_evals.manifest import build_manifest
from agent_evals.models import ArmSpec


def test_build_manifest_is_deterministic(three_arms: list[ArmSpec]) -> None:
    now = datetime(2026, 6, 15, 9, 30, 0, tzinfo=timezone.utc)
    settings = Settings()
    m = build_manifest(
        settings,
        now=now,
        arms=three_arms,
        benchmark="aider_polyglot",
        benchmark_ref="exercism@abc123",
        harness="aider",
        harness_version="0.50.0",
        headroom_repo_path="/nonexistent-repo",
        agent_evals_repo_path="/nonexistent-repo",
    )
    assert m.experiment_id == "aider_polyglot-20260615T093000Z"
    # git_sha falls back to "unknown" for a non-repo path rather than raising.
    assert m.headroom_git_sha == "unknown"
    assert len(m.arms) == 3
    # seeds default to range(k_runs) when not supplied.
    assert m.seeds == list(range(settings.stats.k_runs))
    assert m.margins == {"ccr": 0.0, "lossy": 2.0}
    assert m.pricing.input_usd_per_1m == settings.pricing.input_usd_per_1m


def test_manifest_json_roundtrip(three_arms: list[ArmSpec]) -> None:
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    m = build_manifest(
        Settings(),
        now=now,
        arms=three_arms,
        benchmark="swebench_verified",
        benchmark_ref="verified@v1",
        harness="openhands",
        harness_version="0.1.0",
        headroom_repo_path=".",
        agent_evals_repo_path=".",
    )
    from agent_evals.models import RunManifest

    again = RunManifest.model_validate_json(m.model_dump_json())
    assert again.experiment_id == m.experiment_id
    assert again.benchmark == "swebench_verified"
