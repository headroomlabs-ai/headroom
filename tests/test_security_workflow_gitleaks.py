"""Run the Security workflow's gitleaks install script against local archives.

The step downloads a release tarball and must reject one whose SHA-256 does not
match the pinned value before anything is extracted or installed. The test runs
the real ``run:`` script offline: ``curl`` is a stub that copies a local fixture,
and the ``/tmp`` and ``/usr/local/bin`` paths are redirected into ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

CURL_STUB = """#!/usr/bin/env bash
# Copy the local fixture to the path given after -o; ignore the URL.
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; shift; fi
  shift
done
cp "$FIXTURE_ARCHIVE" "$out"
"""

SUDO_STUB = '#!/usr/bin/env bash\nexec "$@"\n'

# macOS ships a sha256sum without --strict; CI (ubuntu) uses GNU coreutils.
SHA256SUM_SHIM = '#!/usr/bin/env bash\nexec shasum -a 256 "$@"\n'


def _install_script() -> str:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "security.yml").read_text())
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    return next(step["run"] for step in steps if step.get("name") == "Install gitleaks")


def _archive(path: Path, banner: str) -> str:
    """Write a tarball holding a fake ``gitleaks`` and return its SHA-256."""
    body = f"#!/usr/bin/env bash\necho {banner}\n".encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("gitleaks")
        info.size = len(body)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(body))
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _run_install(
    tmp_path: Path, served: Path, pinned_sha256: str
) -> subprocess.CompletedProcess[str]:
    work = tmp_path / "work"
    install_dir = tmp_path / "install"
    stubs = tmp_path / "stubs"
    for directory in (work, install_dir, stubs):
        directory.mkdir()
    _write_exe(stubs / "curl", CURL_STUB)
    _write_exe(stubs / "sudo", SUDO_STUB)
    if subprocess.run(["sha256sum", "--strict", "--help"], capture_output=True).returncode != 0:
        _write_exe(stubs / "sha256sum", SHA256SUM_SHIM)

    script = re.sub(r"(?m)^sha256=[0-9a-f]{64}$", f"sha256={pinned_sha256}", _install_script())
    # /tmp first: on Linux, tmp_path itself lives under /tmp.
    script = re.sub(r"/tmp\b", str(work), script).replace("/usr/local/bin/", f"{install_dir}/")
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{stubs}:{install_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
            "FIXTURE_ARCHIVE": str(served),
        },
        check=False,
    )


def test_intact_archive_is_extracted_and_installed(tmp_path: Path) -> None:
    archive = tmp_path / "intact.tar.gz"
    pinned = _archive(archive, "gitleaks-fixture")

    result = _run_install(tmp_path, archive, pinned)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "install" / "gitleaks").is_file()
    assert "gitleaks-fixture" in result.stdout


def test_tampered_archive_fails_before_extraction_or_install(tmp_path: Path) -> None:
    pinned = _archive(tmp_path / "intact.tar.gz", "gitleaks-fixture")
    tampered = tmp_path / "tampered.tar.gz"
    assert _archive(tampered, "tampered-fixture") != pinned

    result = _run_install(tmp_path, tampered, pinned)

    assert result.returncode != 0
    assert not (tmp_path / "work" / "gitleaks").exists(), "archive was extracted"
    assert not (tmp_path / "install" / "gitleaks").exists(), "binary was installed"
    assert "tampered-fixture" not in result.stdout
