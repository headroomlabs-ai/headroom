"""Candidate version assignment must depend only on the selected source tree."""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "version,npm_version",
    [
        ("0.38.0", "0.38.0"),
        ("0.38.0a1", "0.38.0-alpha.1"),
        ("0.38.0b2", "0.38.0-beta.2"),
        ("0.38.0rc3", "0.38.0-rc.3"),
    ],
)
def test_historical_candidate_version_ignores_future_tags_and_worktree(
    tmp_path, version, npm_version
):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("config", "user.email", "candidate-test@example.invalid")
    git("config", "user.name", "Candidate regression")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "--quiet", "-m", "chore: first source")
    source = git("rev-parse", "HEAD")
    git("tag", "v0.38.0")
    manifest.write_text('[project]\nversion = "0.39.0"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "--quiet", "-m", "feat: later source")
    git("tag", "v0.39.0")

    env = os.environ.copy()
    for key in ("PYTHONPATH", "GITHUB_OUTPUT", "MANUAL_VER", "LEVEL"):
        env.pop(key, None)
    env["CANDIDATE_SOURCE_SHA"] = source

    def resolve():
        result = subprocess.run(
            [sys.executable, str(ROOT / "headroom/release_version.py")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    before = resolve()
    git("tag", "v99.0.0")
    after = resolve()
    assert before["version"] == after["version"] == version
    assert before["npm_version"] == after["npm_version"] == npm_version
    assert before["canonical"] == after["canonical"] == version


def test_candidate_version_is_resolved_before_calling_shared_build():
    import yaml

    candidate = yaml.safe_load(
        (ROOT / ".github/workflows/candidate-artifact.yml").read_text(encoding="utf-8")
    )
    validation = candidate["jobs"]["validate-source"]
    assert validation["outputs"]["version"] == "${{ steps.version.outputs.version }}"
    version = next(step for step in validation["steps"] if step.get("id") == "version")
    assert version["env"]["CANDIDATE_SOURCE_SHA"] == "${{ inputs.source_sha }}"
    assert version["run"] == "python headroom/release_version.py"
    assert candidate["jobs"]["build-and-smoke"]["with"]["resolved_version"] == (
        "${{ needs.validate-source.outputs.version }}"
    )


@pytest.mark.parametrize("source_sha", ["", "main", "A" * 40, "a" * 39])
def test_invalid_candidate_identity_cannot_fall_back_to_release_detection(tmp_path, source_sha):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CANDIDATE_SOURCE_SHA"] = source_sha
    result = subprocess.run(
        [sys.executable, str(ROOT / "headroom/release_version.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Candidate source must be a lowercase full commit SHA" in result.stderr
    assert "version=" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive-letter tar regression")
def test_snapshot_restore_accepts_windows_runner_temp_path(tmp_path):
    import yaml

    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.exists():
        pytest.skip("Git Bash is required for the Windows runner replay")
    runner_temp = tmp_path / "runner-temp"
    artifact = runner_temp / "release-source"
    artifact.mkdir(parents=True)
    payload = tmp_path / "original.txt"
    payload.write_text("snapshot payload", encoding="utf-8")
    with tarfile.open(artifact / "release-source.tar.gz", "w:gz") as archive:
        archive.add(payload, arcname="restored.txt")
    history = runner_temp / "release-history"
    history.mkdir()
    with tarfile.open(history / "release-history.tar.gz", "w:gz") as archive:
        archive.add(payload, arcname="history-restored.txt")
    destination = tmp_path / "destination"
    destination.mkdir()
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    restore = next(
        step["run"]
        for step in workflow["jobs"]["detect-version"]["steps"]
        if step.get("name") == "Restore exact source and Git history"
    )
    env = os.environ.copy()
    env["RUNNER_TEMP"] = str(runner_temp)
    env["RUNNER_OS"] = "Windows"
    result = subprocess.run(
        [str(bash), "-e", "-c", restore],
        cwd=destination,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "restored.txt").read_text(encoding="utf-8") == "snapshot payload"
    assert (destination / "history-restored.txt").read_text(encoding="utf-8") == "snapshot payload"
