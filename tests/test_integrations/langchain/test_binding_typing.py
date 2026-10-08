"""Check the optional integration with the real LangChain classes in scope."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_binding_guard_typechecks_with_langchain_installed(tmp_path):
    pytest.importorskip("langchain_core")
    pytest.importorskip("mypy")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--follow-imports=silent",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            "headroom/integrations/langchain/chat_model.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
