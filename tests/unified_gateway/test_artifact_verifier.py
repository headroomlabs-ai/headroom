from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_artifact_verifier_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    wheel = tmp_path / "headroom_test-0-py3-none-any.whl"
    wheel.write_bytes(b"not-a-wheel")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_unified_gateway_artifact.py",
            "--wheel",
            str(wheel),
            "--expected-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode != 0
    assert "wheel digest mismatch" in result.stderr.lower()
