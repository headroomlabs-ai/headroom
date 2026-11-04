"""Proxy-runtime subprocess calls must decode UTF-8 explicitly.

On Windows, ``subprocess.run(text=True)`` without an ``encoding`` decodes
child output with the console code page (cp1252). A child's emoji-laden
output then kills the reader thread with ``UnicodeDecodeError: 'charmap'
codec can't decode byte ...`` (seen in user proxy logs). These tests pin the
``encoding="utf-8"`` kwarg on every proxy-runtime subprocess call.
"""

import subprocess

from headroom.proxy.interceptors import astgrep


def _capture_run(captured, returncode=0, stdout='{"summary": {}}'):
    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return fake_run


def test_ast_grep_subprocess_uses_utf8(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(astgrep.subprocess, "run", _capture_run(captured, returncode=1, stdout=""))

    astgrep._run_ast_grep("/fake/sg", "python", "def foo():\n    pass\n")

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
