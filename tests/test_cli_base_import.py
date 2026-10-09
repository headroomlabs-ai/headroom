from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def test_base_cli_import_does_not_require_proxy_dependencies() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        from click.testing import CliRunner


        class BlockProxyDeps(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (
                    fullname in {"fastapi", "uvicorn"}
                    or fullname.startswith("fastapi.")
                    or fullname.startswith("uvicorn.")
                ):
                    raise ImportError(f"blocked optional dependency: {fullname}")
                return None


        sys.meta_path.insert(0, BlockProxyDeps())

        import headroom.cli  # noqa: F401
        from headroom.cli.main import main

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0, result.output
        assert "fastapi" not in sys.modules
        assert "uvicorn" not in sys.modules
        """
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"subprocess unavailable for base CLI import check: {exc}")

    assert result.returncode == 0, result.stdout + result.stderr
