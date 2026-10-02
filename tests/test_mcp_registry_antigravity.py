from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom import fsutil
from headroom.mcp_registry.antigravity import AntigravityRegistrar, _config_candidates
from headroom.mcp_registry.install import build_headroom_spec


def _write_config(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _servers(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["mcpServers"]


def test_antigravity_registrar_register_writes_current_location(tmp_path: Path) -> None:
    registrar = AntigravityRegistrar(home_dir=tmp_path)

    result = registrar.register_server(build_headroom_spec("http://127.0.0.1:9999"))

    assert result.status.value == "registered"
    assert "headroom" in _servers(_config_candidates(tmp_path)[0])


def test_antigravity_unregister_removes_entry_from_every_candidate(
    tmp_path: Path,
) -> None:
    registrar = AntigravityRegistrar(home_dir=tmp_path)
    for candidate in _config_candidates(tmp_path):
        _write_config(candidate, {"headroom": {"command": "headroom"}})

    assert registrar.unregister_server("headroom") is True
    for candidate in _config_candidates(tmp_path):
        assert "headroom" not in _servers(candidate)


def test_antigravity_unregister_reports_failure_when_a_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registrar = AntigravityRegistrar(home_dir=tmp_path)
    candidates = _config_candidates(tmp_path)
    for candidate in candidates:
        _write_config(candidate, {"headroom": {"command": "headroom"}})

    real_write = fsutil.write_text

    def fail_first(path: Path, content: str) -> None:
        if path == candidates[0]:
            raise OSError("read-only filesystem")
        return real_write(path, content)

    monkeypatch.setattr(fsutil, "write_text", fail_first)

    # The second config was cleaned up, but the failed one still registers the
    # server — so this must not report success.
    assert registrar.unregister_server("headroom") is False
    assert "headroom" in _servers(candidates[0])
    assert "headroom" not in _servers(candidates[1])
