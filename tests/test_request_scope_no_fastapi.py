"""Regression tests for importing request-scope helpers without FastAPI."""

import subprocess
import sys
import textwrap


def test_request_scope_and_project_context_import_without_fastapi() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def import_without_fastapi(name, *args, **kwargs):
            if name == "fastapi" or name.startswith("fastapi."):
                raise ModuleNotFoundError("No module named 'fastapi'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = import_without_fastapi

        import headroom.proxy.request_scope as request_scope
        import headroom.proxy.project_context

        assert not any(
            name == "fastapi" or name.startswith("fastapi.") for name in sys.modules
        )
        scope = {"path": "/a", "raw_path": b"/a"}
        request_scope.normalize_scope_path(scope, "/b c")
        assert scope == {"path": "/b c", "raw_path": b"/b%20c"}
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
