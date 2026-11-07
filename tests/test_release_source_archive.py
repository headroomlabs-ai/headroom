"""Execute source preparation with credential-bearing checkout configuration."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("credential_location", ["remote", "include", "worktree"])
def test_source_archive_preserves_history_without_checkout_configuration(
    tmp_path, credential_location
):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash or not Path(bash).exists():
        pytest.skip("Bash is required to execute the workflow step")
    source = tmp_path / "source"
    source.mkdir()
    runner = tmp_path / "runner"
    runner.mkdir()

    def git(*args, cwd=source):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "Archive regression")
    git("config", "user.email", "archive@example.invalid")
    (source / "payload.txt").write_bytes(b"exact source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "original source")
    selected = git("rev-parse", "HEAD")
    git("tag", "v0.37.0")
    (source / "payload.txt").write_bytes(b"future source\n")
    git("commit", "--quiet", "-am", "future source")
    future = git("rev-parse", "HEAD")
    git("tag", "v0.38.0")
    git("checkout", "--quiet", "--detach", selected)
    sentinel = "dummy-archive-credential"
    included = tmp_path / "credential-config"
    included.write_text(
        f'[http "https://example.invalid/"]\nextraheader = {sentinel}\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = str(tmp_path / "absent")
    if credential_location == "remote":
        git("remote", "add", "origin", f"https://user:{sentinel}@example.invalid/repo")
    elif credential_location == "include":
        git("config", "include.path", str(included))
    else:
        git("config", "extensions.worktreeConfig", "true")
        git("config", "--worktree", "http.https://example.invalid/.extraheader", sentinel)
    original_config = (source / ".git/config").read_bytes()
    original_hook = source / ".git/hooks/post-checkout"
    original_hook.write_bytes(b"dummy credential-bearing original hook\n")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    step = next(s for s in workflow["jobs"]["prepare-source"]["steps"] if "run" in s)
    env["SOURCE_SHA"] = selected
    # Git Bash requires POSIX paths for tar, which interprets drive letters as hosts.
    env["RUNNER_TEMP"] = (
        subprocess.check_output(
            [bash, "-c", 'cygpath -u "$1"', "path", str(runner)], text=True
        ).strip()
        if os.name == "nt"
        else str(runner)
    )
    result = subprocess.run(
        [bash, "-e", "-c", step["run"]], cwd=source, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (source / ".git/config").read_bytes() == original_config
    restored = tmp_path / "restored"
    restored.mkdir()
    with tarfile.open(runner / "release-source.tar.gz") as archive:
        assert not any(
            name == "./.git" or name.startswith("./.git/") for name in archive.getnames()
        )
        archive.extractall(restored, filter="data")
    assert not (restored / ".git").exists()
    with tarfile.open(runner / "release-history.tar.gz") as archive:
        archive.extractall(restored, filter="data")
    assert git("rev-parse", "HEAD", cwd=restored) == selected
    assert git("ls-files", cwd=restored) == "payload.txt"
    assert git("status", "--porcelain", cwd=restored) == ""
    assert git("tag", "--list", cwd=restored) == "v0.37.0\nv0.38.0"
    assert git("log", "--format=%s", cwd=restored) == "original source"
    assert git("show", f"{future}:payload.txt", cwd=restored) == "future source"
    assert (restored / "payload.txt").read_bytes() == b"exact source\n"
    configuration = git("config", "--local", "--list", cwd=restored)
    assert sentinel not in configuration
    assert "include.path" not in configuration
    assert "remote.origin" not in configuration
    assert not (restored / ".git/config.worktree").exists()
    assert not (restored / ".git/hooks/post-checkout").exists()
    assert original_hook.read_bytes() == b"dummy credential-bearing original hook\n"


@pytest.mark.parametrize("job", ["detect-version", "build"])
def test_only_history_consumers_download_the_history_artifact(job):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"][job]["steps"]
    assert any(
        "download-artifact" in step.get("uses", "")
        and step.get("with", {}).get("name") == "release-history"
        for step in steps
    )
    wheels = workflow["jobs"]["build-wheels"]["steps"]
    assert not any(step.get("with", {}).get("name") == "release-history" for step in wheels)
