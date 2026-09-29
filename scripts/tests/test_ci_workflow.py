"""Tests for CI workflow hardening contracts."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_sharded_ci_verifies_offline_huggingface_cache_before_pytest() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    verify_step = "Verify offline HuggingFace model cache"
    pytest_step = "Run test shard ${{ matrix.shard }}/4"

    assert verify_step in workflow
    assert "python scripts/ci/verify_hf_model_cache.py" in workflow
    assert workflow.index(verify_step) < workflow.index(pytest_step)


def test_sharded_ci_uploads_only_explicit_coverage_reports() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    upload_step = workflow[workflow.index("Upload coverage shard") :]

    assert "files: coverage-${{ matrix.shard }}.xml" in upload_step
    assert "disable_search: true" in upload_step


def test_sharded_ci_disables_process_killing_hard_watchdog() -> None:
    """Pytest must report a stalled test instead of being hard-exited."""
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))

    assert workflow["jobs"]["test"]["env"]["HEADROOM_HARD_WATCHDOG_SECS"] == "0"
