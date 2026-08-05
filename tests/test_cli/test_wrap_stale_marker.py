from __future__ import annotations

import json
from pathlib import Path

from headroom.cli import doctor as doctor_cli
from headroom.cli import wrap as wrap_cli


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.local.json"


def test_doctor_skips_with_no_marker(tmp_path: Path) -> None:
    result = doctor_cli.check_wrap_marker_staleness(_settings(tmp_path))
    assert result.status == doctor_cli.SKIP


def test_doctor_passes_with_live_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    result = doctor_cli.check_wrap_marker_staleness(path)
    assert result.status == doctor_cli.PASS


def test_doctor_flags_stale_wrap_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    marker_path = wrap_cli._wrap_marker_path(path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["pid"] = 999_999_999
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    result = doctor_cli.check_wrap_marker_staleness(path)
    assert result.status == doctor_cli.WARN
    assert "999999999" in result.summary
    assert "headroom unwrap claude" in result.summary


def test_claude_command_registers_sighup_next_to_sigterm() -> None:
    """`claude()` must catch SIGHUP (terminal close) the same way it catches
    SIGTERM, or a crashed-by-terminal-close wrap session never restores its
    base_url (issue #1768). Full signal delivery isn't practical to exercise
    via CliRunner (would require spawning/killing a real subprocess), so this
    asserts the registration is present in claude()'s source, guarded for
    platforms without SIGHUP.
    """
    import inspect

    src = inspect.getsource(wrap_cli.claude.callback)
    assert 'hasattr(signal, "SIGHUP")' in src
    assert "signal.signal(signal.SIGHUP, cleanup)" in src
