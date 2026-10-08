"""Exercise workflow version assignment without executing historical helpers."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_version_step(tmp_path, value):
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash or not Path(bash).exists():
        pytest.skip("Bash is required for workflow execution")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    step = next(s for s in workflow["jobs"]["detect-version"]["steps"] if s.get("id") == "ver")
    output = tmp_path / "outputs"
    env = os.environ.copy()
    env["RESOLVED_VERSION"] = value
    env["GITHUB_OUTPUT"] = (
        subprocess.check_output(
            [bash, "-c", 'cygpath -u "$1"', "path", str(output)], text=True
        ).strip()
        if os.name == "nt"
        else str(output)
    )
    result = subprocess.run(
        [bash, "-e", "-c", step["run"]], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    outputs = (
        dict(line.split("=", 1) for line in output.read_text().splitlines())
        if output.exists()
        else {}
    )
    return result, outputs


@pytest.mark.parametrize(
    "version,npm_version",
    [
        ("0.37.0", "0.37.0"),
        ("0.37.0a1", "0.37.0-alpha.1"),
        ("0.37.0b2", "0.37.0-beta.2"),
        ("0.37.0rc3", "0.37.0-rc.3"),
    ],
)
def test_shared_version_step_supports_python_and_npm_prereleases(tmp_path, version, npm_version):
    result, outputs = run_version_step(tmp_path, version)
    assert result.returncode == 0, result.stderr
    assert outputs["version"] == outputs["canonical"] == version
    assert outputs["npm_version"] == npm_version
    assert outputs["height"] == "0"


@pytest.mark.parametrize(
    "value",
    [
        "0.37.0c1",
        "0.37.0r1",
        "0.37.0rc",
        "0.37.0-rc.1",
        "01.37.0",
        "0.37.0rc01",
        "0.37.0\nheight=999",
        "0.37.0;touch injected",
        "0.37.0$(touch injected)",
    ],
)
def test_shared_version_step_rejects_invalid_or_injected_versions(tmp_path, value):
    result, outputs = run_version_step(tmp_path, value)
    assert result.returncode != 0
    assert not outputs
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize("job", ["build", "build-wheels"])
def test_package_synchronization_uses_python_version_for_wheel_metadata(job):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-build.yml").read_text(encoding="utf-8")
    )
    sync = next(
        step for step in workflow["jobs"][job]["steps"] if "version-sync.py" in step.get("run", "")
    )
    assert "--version ${{ needs.detect-version.outputs.version }}" in sync["run"]
    assert "outputs.npm_version" not in sync["run"]
