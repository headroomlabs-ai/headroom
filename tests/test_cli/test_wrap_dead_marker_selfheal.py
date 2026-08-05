"""Session-start self-heal for a wrap base_url left by a dead proxy (issue #2221).

`headroom wrap claude` persists ANTHROPIC_BASE_URL=<proxy> into project-local
.claude/settings.local.json so cc-daemon conversation workers (which read
settings fresh) also route through the proxy. When the proxy dies via hard
reboot / SIGKILL no cleanup fires, so the stale URL lingers and bricks a later
bare `claude`. These tests cover the port-liveness self-heal that clears it.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.local.json"


def _marker(tmp_path: Path) -> Path:
    return wrap_cli._wrap_marker_path(_settings(tmp_path))


def _closed_port() -> int:
    """A port that is (almost certainly) not accepting connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _listening_port() -> Iterator[int]:
    """A real bound+listening socket; its port answers TCP connects (live proxy)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


# --- _wrap_marker_proxy_is_dead -------------------------------------------


def test_proxy_is_dead_when_port_not_listening() -> None:
    assert wrap_cli._wrap_marker_proxy_is_dead({"port": _closed_port()}) is True


def test_proxy_is_not_dead_when_port_listening() -> None:
    with _listening_port() as port:
        assert wrap_cli._wrap_marker_proxy_is_dead({"port": port}) is False


def test_proxy_is_not_dead_when_no_port_recorded() -> None:
    # No port → fall back to PID-based staleness, so not "dead" by port here.
    assert wrap_cli._wrap_marker_proxy_is_dead({}) is False


# --- _check_and_clear_dead_wrap_marker ------------------------------------


def test_dead_port_restores_previous_value(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    # Live PID (this process) but a dead port — the reboot/SIGKILL case where
    # PID liveness lies and only the port tells the truth.
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    wrap_cli._write_wrap_marker(
        path, port=_closed_port(), key="ANTHROPIC_BASE_URL", previous="http://old.proxy:9000"
    )

    restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
    assert restored == "http://old.proxy:9000"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://old.proxy:9000"
    assert not _marker(tmp_path).exists()


def test_dead_port_removes_key_when_no_previous(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    wrap_cli._write_wrap_marker(path, port=_closed_port(), key="ANTHROPIC_BASE_URL", previous=None)

    restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
    assert restored is None
    # env held only our key, so the now-empty settings file is removed entirely.
    assert not path.exists()
    assert not _marker(tmp_path).exists()


def test_live_port_is_never_cleared(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    with _listening_port() as port:
        wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
        wrap_cli._write_wrap_marker(path, port=port, key="ANTHROPIC_BASE_URL", previous=None)

        restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
        assert restored is None
        # Live wrapped session preserved: base URL and marker both intact.
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        assert _marker(tmp_path).exists()


def test_noop_when_no_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://x"}}), encoding="utf-8")
    assert wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL") is None
    # Untouched — no marker means no evidence the URL is Headroom's to clear.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://x"


def test_noop_when_no_settings_file(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    assert wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL") is None


def test_noop_when_marker_key_mismatch(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    wrap_cli._write_wrap_marker(
        path, port=_closed_port(), key="ANTHROPIC_VERTEX_BASE_URL", previous=None
    )
    # Asked about the default key; marker is for the vertex key → no-op.
    assert wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL") is None
    assert _marker(tmp_path).exists()


# --- retry-hardened liveness (_wrap_proxy_alive) --------------------------


def test_proxy_alive_returns_true_on_first_success() -> None:
    with _listening_port() as port:
        assert wrap_cli._wrap_proxy_alive(port) is True


def test_proxy_alive_false_only_after_all_attempts_fail() -> None:
    assert wrap_cli._wrap_proxy_alive(_closed_port(), attempts=3, delay=0.01) is False


def test_proxy_alive_tolerates_single_transient_failure(monkeypatch) -> None:
    # A live-but-busy proxy whose first TCP connect blips: retry must treat it
    # as ALIVE, not dead, so a live session's base_url is never cleared.
    calls = {"n": 0}

    def _flaky(_port: int) -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # first probe fails, second succeeds

    monkeypatch.setattr(wrap_cli, "_check_proxy", _flaky)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _s: None)
    assert wrap_cli._wrap_proxy_alive(1234) is True


def test_transient_blip_does_not_clear_live_marker(tmp_path: Path, monkeypatch) -> None:
    # End-to-end: a marked session whose proxy blips once at session start must
    # keep its base_url and marker — the retry closes the false-dead hole.
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    wrap_cli._write_wrap_marker(path, port=9191, key="ANTHROPIC_BASE_URL", previous=None)

    calls = {"n": 0}

    def _flaky(_port: int) -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(wrap_cli, "_check_proxy", _flaky)
    monkeypatch.setattr(wrap_cli.time, "sleep", lambda _s: None)

    assert wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL") is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert _marker(tmp_path).exists()


# --- SessionStart self-heal hook install (wrap claude) --------------------

_HOOK_MARKER = "headroom-wrap-selfheal"


def test_wrap_installs_sessionstart_only_selfheal_hook(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    session_start = payload["hooks"]["SessionStart"]
    assert len(session_start) == 1
    entry = session_start[0]
    assert entry["matcher"] == "startup|resume"
    command = entry["hooks"][0]["command"]
    assert _HOOK_MARKER in command
    assert "wrap selfheal" in command
    assert entry["hooks"][0]["timeout"] == 10
    # SessionStart ONLY — never registered on PreToolUse (defect 2 exposure).
    assert "PreToolUse" not in payload["hooks"]


def test_wrap_selfheal_hook_install_is_idempotent(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    marked = [
        entry
        for entry in payload["hooks"]["SessionStart"]
        if any(_HOOK_MARKER in h.get("command", "") for h in entry["hooks"])
    ]
    assert len(marked) == 1


def test_wrap_selfheal_hook_preserves_existing_hooks(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{"command": "mine"}]}]}}
        ),
        encoding="utf-8",
    )
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    commands = [h.get("command") for e in payload["hooks"]["SessionStart"] for h in e["hooks"]]
    assert "mine" in commands
    assert any(_HOOK_MARKER in str(c) for c in commands)


def test_unwrap_removes_selfheal_hook(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)
    assert wrap_cli._remove_claude_wrap_selfheal_hook(path) is True

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        commands = [
            h.get("command")
            for e in (payload.get("hooks", {}).get("SessionStart") or [])
            for h in e.get("hooks", [])
        ]
        assert not any(_HOOK_MARKER in str(c) for c in commands)


def test_unwrap_removal_keeps_unrelated_hooks(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{"command": "mine"}]}]}}
        ),
        encoding="utf-8",
    )
    wrap_cli._ensure_claude_wrap_selfheal_hook(path)
    assert wrap_cli._remove_claude_wrap_selfheal_hook(path) is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    commands = [h.get("command") for e in payload["hooks"]["SessionStart"] for h in e["hooks"]]
    assert commands == ["mine"]


def test_unwrap_removal_noop_without_hook(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    assert wrap_cli._remove_claude_wrap_selfheal_hook(path) is False


# --- self-heal CLI command (installed hook target) ------------------------


def test_selfheal_command_clears_dead_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    wrap_cli._write_wrap_marker(path, port=_closed_port(), key="ANTHROPIC_BASE_URL", previous=None)

    result = CliRunner().invoke(wrap_cli.wrap, ["selfheal", "--marker", "headroom-wrap-selfheal"])
    assert result.exit_code == 0
    # env held only our key, so the now-empty settings file is removed entirely.
    assert not path.exists()
    assert not _marker(tmp_path).exists()


def test_selfheal_command_preserves_live_marker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = _settings(tmp_path)
    with _listening_port() as port:
        wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
        wrap_cli._write_wrap_marker(path, port=port, key="ANTHROPIC_BASE_URL", previous=None)

        result = CliRunner().invoke(
            wrap_cli.wrap, ["selfheal", "--marker", "headroom-wrap-selfheal"]
        )
        assert result.exit_code == 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        assert _marker(tmp_path).exists()


def test_selfheal_never_raises_without_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    wrap_cli._selfheal_dead_wrap_base_url()  # no .claude dir at all — silent no-op
