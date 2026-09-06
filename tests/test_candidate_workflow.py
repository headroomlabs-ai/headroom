"""Structural safety tests for immutable candidate production."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / ".github" / "workflows" / "candidate-artifact.yml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"


def _job(content: str, name: str, next_name: str | None = None) -> str:
    start = content.index(f"\n  {name}:")
    end = content.index(f"\n  {next_name}:", start) if next_name else len(content)
    return content[start:end]


def test_candidate_validates_full_sha_reachable_from_authoritative_main() -> None:
    content = CANDIDATE.read_text(encoding="utf-8")
    validation = _job(content, "validate-source", "build-and-smoke")
    assert "^[0-9a-f]{40}$" in validation
    assert "refs/heads/main:refs/remotes/origin/main" in validation
    assert 'git cat-file -e "$SOURCE_SHA^{commit}"' in validation
    assert 'git merge-base --is-ancestor "$SOURCE_SHA" refs/remotes/origin/main' in validation
    assert '"$WORKFLOW_REF" != "refs/heads/main"' in validation
    assert 'git merge-base --is-ancestor "$PRODUCER_SHA" refs/remotes/origin/main' in validation
    assert "ref: ${{ inputs.source_sha }}" not in validation
    assert "SOURCE_SHA: ${{ inputs.source_sha }}" in validation


def test_candidate_reuses_release_build_in_nonpublishing_mode() -> None:
    content = CANDIDATE.read_text(encoding="utf-8")
    build = _job(content, "build-and-smoke", "emit-candidate")
    assert "uses: ./.github/workflows/release.yml" in build
    assert "source_sha: ${{ needs.validate-source.outputs.source_sha }}" in build
    assert "candidate_mode: true" in build

    release = RELEASE.read_text(encoding="utf-8")
    assert "workflow_call:" in release
    for current, following in [
        ("publish-pypi", "publish-npm"),
        ("publish-npm", "publish-github-packages"),
        ("publish-github-packages", "publish-docker"),
        ("publish-docker", "create-release"),
    ]:
        assert "inputs.candidate_mode != true" in _job(release, current, following)
    assert "inputs.candidate_mode != true" in _job(release, "create-release")


def test_candidate_entry_point_has_no_mutating_authority_or_publish_step() -> None:
    content = CANDIDATE.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in content
    for forbidden in [
        "contents: write",
        "packages: write",
        "id-token: write",
        "npm publish",
        "gh-action-pypi-publish",
        "docker build",
        "docker push",
        "gh release",
    ]:
        assert forbidden not in content


def test_candidate_binds_payload_inventory_and_verifies_downloaded_bytes() -> None:
    content = CANDIDATE.read_text(encoding="utf-8")
    emit = _job(content, "emit-candidate", "verify-downloaded-candidate")
    verify = _job(content, "verify-downloaded-candidate")
    assert "candidate_manifest.py inventory" in emit
    assert "candidate_manifest.py create" in emit
    assert "--runtime-payload candidate-runtime-payload.json" in emit
    assert "retention-days: 90" in emit
    assert "actions/download-artifact@v8" in verify
    assert "candidate_manifest.py verify" in verify
    assert "candidate/headroom-candidate-${SOURCE_SHA}.tar" in verify


def test_reusable_build_checks_out_the_requested_exact_sha() -> None:
    content = RELEASE.read_text(encoding="utf-8")
    checkout_count = content.count("uses: actions/checkout@v7")
    exact_ref_count = content.count("ref: ${{ inputs.source_sha || github.sha }}")
    assert checkout_count == exact_ref_count
