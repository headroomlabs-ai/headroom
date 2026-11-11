"""Tests for CI workflow hardening contracts."""

from __future__ import annotations

import re
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
    assert "if: ${{ !cancelled() }}" in upload_step.split("uses:", 1)[0]


def test_sharded_ci_disables_process_killing_hard_watchdog() -> None:
    """Pytest must report a stalled test instead of being hard-exited."""
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))

    assert workflow["jobs"]["test"]["env"]["HEADROOM_HARD_WATCHDOG_SECS"] == "0"


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    """A tag like actions/checkout@v7 can be moved to other code; a commit SHA cannot."""
    uses = re.compile(r"^\s*(?:-\s*)?uses:\s*[\"']?([^\s\"'#]+)", re.MULTILINE)
    unpinned = [
        f"{path}: {ref}"
        for path in sorted(Path(".github").rglob("*.y*ml"))
        for ref in uses.findall(path.read_text(encoding="utf-8"))
        if not ref.startswith("./") and not re.search(r"@[0-9a-f]{40}$", ref)
    ]

    assert unpinned == [], "pin these to a full commit SHA, version as a comment:\n" + "\n".join(
        unpinned
    )
