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


def test_sharded_ci_uploads_only_explicit_coverage_reports() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    upload_step = workflow[workflow.index("Upload coverage shard") :]

    assert "files: coverage-${{ matrix.shard }}.xml" in upload_step
    assert "disable_search: true" in upload_step


def test_ci_checks_documented_environment_variables_for_source_changes() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "env_docs: ${{ steps.filter.outputs.env_docs }}" in workflow
    assert "env-doc-consistency:" in workflow
    assert "python scripts/ci/verify_documented_env_vars.py" in workflow


def test_docs_workflow_checks_environment_variables_for_documentation_changes() -> None:
    workflow = Path(".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "- 'README.md'" in workflow
    assert "- 'SECURITY.md'" in workflow
    assert "python scripts/ci/verify_documented_env_vars.py" in workflow
