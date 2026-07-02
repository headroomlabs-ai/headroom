"""Tests for CI workflow hardening contracts."""

from __future__ import annotations

from pathlib import Path


def test_sharded_ci_verifies_offline_huggingface_cache_before_pytest() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    verify_step = "Verify offline HuggingFace model cache"
    pytest_step = "Run test shard ${{ matrix.shard }}/4"

    assert verify_step in workflow
    assert "python scripts/ci/verify_hf_model_cache.py" in workflow
    assert workflow.index(verify_step) < workflow.index(pytest_step)
