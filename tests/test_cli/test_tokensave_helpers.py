"""Coverage for the tokensave binary-resolution and indexing helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.cli import wrap as wrap_cli
from headroom.graph import tokensave_installer as ts

_FAKE_BIN = Path("/usr/local/bin/tokensave")


# ---------------------------------------------------------------------------
# _ensure_tokensave_binary
# ---------------------------------------------------------------------------


def test_ensure_binary_returns_existing_without_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: _FAKE_BIN)

    def _should_not_run(*a, **k):
        raise AssertionError("must not download when binary already present")

    monkeypatch.setattr(ts, "ensure_tokensave", _should_not_run)
    assert wrap_cli._ensure_tokensave_binary() == _FAKE_BIN


def test_ensure_binary_fetches_when_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: None)
    monkeypatch.setattr(ts, "ensure_tokensave", lambda: _FAKE_BIN)
    assert wrap_cli._ensure_tokensave_binary() == _FAKE_BIN
    assert "installed at" in capsys.readouterr().out


def test_ensure_binary_none_prints_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ts, "get_tokensave_path", lambda: None)
    monkeypatch.setattr(ts, "ensure_tokensave", lambda: None)
    assert wrap_cli._ensure_tokensave_binary() is None
    assert "falling back to Serena" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _index_tokensave_project
# ---------------------------------------------------------------------------


def _patch_run(monkeypatch: pytest.MonkeyPatch, result):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(wrap_cli.subprocess, "run", fake_run)
    return calls


def test_index_runs_init_when_no_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _patch_run(monkeypatch, SimpleNamespace(returncode=0, stdout="", stderr=""))
    wrap_cli._index_tokensave_project(_FAKE_BIN)
    assert calls == [[str(_FAKE_BIN), "init"]]
    assert "Code graph: indexed (tokensave)" in capsys.readouterr().out


def test_index_runs_sync_when_db_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".tokensave").mkdir()
    monkeypatch.chdir(tmp_path)
    calls = _patch_run(monkeypatch, SimpleNamespace(returncode=0, stdout="", stderr=""))
    wrap_cli._index_tokensave_project(_FAKE_BIN)
    assert calls == [[str(_FAKE_BIN), "sync"]]


def test_index_nonzero_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_run(monkeypatch, SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    wrap_cli._index_tokensave_project(_FAKE_BIN, verbose=True)
    assert "init failed" in capsys.readouterr().out


def test_index_timeout_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    monkeypatch.chdir(tmp_path)
    _patch_run(monkeypatch, subprocess.TimeoutExpired(cmd="tokensave", timeout=60))
    wrap_cli._index_tokensave_project(_FAKE_BIN)
    assert "timed out" in capsys.readouterr().out


def test_index_exception_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_run(monkeypatch, FileNotFoundError("no binary"))
    # Must not raise even when the binary is missing.
    wrap_cli._index_tokensave_project(_FAKE_BIN, verbose=True)
    assert "indexing skipped" in capsys.readouterr().out
