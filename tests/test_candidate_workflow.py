"""Structural safety tests for immutable candidate production."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
CANDIDATE = ROOT / ".github" / "workflows" / "candidate-artifact.yml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
BUILD = ROOT / ".github" / "workflows" / "release-build.yml"


def test_candidate_call_tree_cannot_request_write_permissions() -> None:
    """GitHub validates nested permissions before evaluating job if guards."""
    pending = [CANDIDATE]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for scope in [workflow, *workflow["jobs"].values()]:
            permissions = scope.get("permissions", {})
            assert isinstance(permissions, dict), path
            assert all(value in ("read", "none") for value in permissions.values()), path
        for job in workflow["jobs"].values():
            assert "secrets" not in job, path
            if "uses" in job:
                assert job["uses"].startswith("./.github/workflows/"), path
                pending.append(ROOT / job["uses"][2:])
    assert RELEASE not in visited
    assert BUILD in visited


def test_shared_build_keeps_all_release_gates_and_version_outputs() -> None:
    build = yaml.safe_load(BUILD.read_text(encoding="utf-8"))
    jobs = build["jobs"]
    assert set(jobs) == {
        "detect-version",
        "build",
        "build-wheels",
        "collect-dist",
        "smoke-import-wheels",
    }
    assert jobs["build"]["needs"] == ["detect-version"]
    assert jobs["build-wheels"]["needs"] == ["detect-version", "build"]
    assert jobs["collect-dist"]["needs"] == ["build", "build-wheels"]
    assert jobs["smoke-import-wheels"]["needs"] == ["build-wheels"]
    for name, job in jobs.items():
        assert not job.get("continue-on-error"), name
        # Only an explicitly requested release dry-run may omit the smoke gate;
        # immutable candidate dispatch has no dry_run input.
        if name == "smoke-import-wheels":
            assert job["if"] == "github.event.inputs.dry_run != 'true'"
        else:
            assert "if" not in job, name
    # PyYAML's YAML 1.1 resolver treats the unquoted GitHub 'on' key as True.
    interface = build.get("on", build.get(True))
    for name in ("version", "npm_version", "canonical", "height", "bump", "previous_tag"):
        assert interface["workflow_call"]["outputs"][name]["value"] == (
            "${{ jobs.detect-version.outputs." + name + " }}"
        )
    release = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))
    assert release["jobs"]["build-and-smoke"]["uses"] == ("./.github/workflows/release-build.yml")
    for name in ("publish-pypi", "publish-npm", "publish-github-packages", "publish-docker"):
        assert release["jobs"][name]["needs"] == ["build-and-smoke"]
    assert "needs.build-and-smoke.result == 'success'" in release["jobs"]["create-release"]["if"]
    assert '".github/workflows/release-build.yml"' in RELEASE.read_text(encoding="utf-8")


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


def test_candidate_reuses_the_build_only_release_workflow() -> None:
    content = CANDIDATE.read_text(encoding="utf-8")
    build = _job(content, "build-and-smoke", "emit-candidate")
    assert "uses: ./.github/workflows/release-build.yml" in build
    assert "source_sha: ${{ needs.validate-source.outputs.source_sha }}" in build
    assert "candidate_mode" not in build
    assert "uses: ./.github/workflows/release-build.yml" in RELEASE.read_text(encoding="utf-8")

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
    content = BUILD.read_text(encoding="utf-8")
    checkout_count = content.count("uses: actions/checkout@v7")
    exact_ref_count = content.count("ref: ${{ inputs.source_sha || github.sha }}")
    assert checkout_count == exact_ref_count
