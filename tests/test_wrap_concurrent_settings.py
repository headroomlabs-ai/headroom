"""Concurrent `headroom wrap` sessions sharing one project's settings (#3205).

`wrap claude` writes ANTHROPIC_BASE_URL into `.claude/settings.local.json` and
restores it on exit. Several sessions in one project run that read-modify-write
concurrently. The write is atomic so the file never tears, but the updates were
still lost against each other:

  * the first session's exit deleted the key while the others were still
    running -- they silently stopped routing through the proxy, kept working,
    and lost every byte of compression with no error anywhere; and
  * a session that started second remembered the *first* session's proxy URL as
    "the original", so its exit wrote a dead proxy back into the file, which
    every later session in that project then failed to connect to.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from headroom.cli import wrap as W


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"FOO": "bar"}}), encoding="utf-8")
    return path


def _env(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("env", {}) if path.exists() else {}


class _Sessions:
    """Drive several wrap sessions with distinct, controllable PIDs."""

    def __init__(self, *pids: int) -> None:
        self.live = set(pids)

    def __enter__(self) -> _Sessions:
        self._patches = [
            mock.patch.object(W, "_pid_alive", lambda pid: pid in self.live),
            mock.patch.object(W, "_identity_mismatch", lambda *a: False),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for p in self._patches:
            p.stop()

    def launch(self, pid: int, url: str, path: Path, port: int | None = None) -> str | None:
        with mock.patch("os.getpid", lambda: pid):
            return W._write_claude_wrap_base_url(url, settings_path=path, port=port)

    def exit(self, pid: int, previous: str | None, path: Path) -> None:
        self.live.discard(pid)
        with mock.patch("os.getpid", lambda: pid):
            W._restore_claude_wrap_base_url(previous, settings_path=path)

    def crash(self, pid: int) -> None:
        """Vanish without running cleanup (SIGKILL, hard reboot)."""
        self.live.discard(pid)


def test_first_session_exiting_leaves_the_others_routed(settings: Path) -> None:
    """The reported symptom: sessions silently stop routing when a sibling exits."""
    with _Sessions(1001, 1002) as s:
        a = s.launch(1001, "http://127.0.0.1:8787", settings)
        s.launch(1002, "http://127.0.0.1:8788", settings)

        s.exit(1001, a, settings)

        assert "ANTHROPIC_BASE_URL" in _env(settings), "surviving session was unrouted"


def test_last_session_out_restores_the_true_original(settings: Path) -> None:
    """A later session must not restore an earlier session's dead proxy URL."""
    with _Sessions(1001, 1002) as s:
        a = s.launch(1001, "http://127.0.0.1:8787", settings)
        b = s.launch(1002, "http://127.0.0.1:8788", settings)

        s.exit(1001, a, settings)
        s.exit(1002, b, settings)

        assert _env(settings) == {"FOO": "bar"}, "stale proxy URL left behind"


def test_a_pre_existing_user_base_url_survives_the_whole_cycle(settings: Path) -> None:
    """A URL the project already had is restored, not deleted."""
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://user-proxy:1234"}}), encoding="utf-8"
    )
    with _Sessions(1001, 1002) as s:
        a = s.launch(1001, "http://127.0.0.1:8787", settings)
        b = s.launch(1002, "http://127.0.0.1:8788", settings)
        s.exit(1001, a, settings)
        s.exit(1002, b, settings)

    assert _env(settings)["ANTHROPIC_BASE_URL"] == "http://user-proxy:1234"


def test_three_sessions_any_exit_order(settings: Path) -> None:
    for order in ([1001, 1002, 1003], [1003, 1001, 1002], [1002, 1003, 1001]):
        settings.write_text(json.dumps({"env": {"FOO": "bar"}}), encoding="utf-8")
        with _Sessions(*order) as s:
            prev = {
                pid: s.launch(pid, f"http://127.0.0.1:{8787 + i}", settings)
                for i, pid in enumerate(order)
            }
            for pid in order[:-1]:
                s.exit(pid, prev[pid], settings)
                assert "ANTHROPIC_BASE_URL" in _env(settings), f"unrouted early in {order}"
            s.exit(order[-1], prev[order[-1]], settings)
        assert _env(settings) == {"FOO": "bar"}, f"residue after {order}"


def test_a_crashed_session_does_not_wedge_the_key(settings: Path) -> None:
    """A SIGKILLed session never releases; its claim must be pruned as dead."""
    with _Sessions(1001, 1002) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings)
        b = s.launch(1002, "http://127.0.0.1:8788", settings)

        s.crash(1001)
        s.exit(1002, b, settings)

    assert _env(settings) == {"FOO": "bar"}
    assert not W._wrap_owners_path(settings).exists()


def test_single_session_behaviour_is_unchanged(settings: Path) -> None:
    with _Sessions(1001) as s:
        a = s.launch(1001, "http://127.0.0.1:8787", settings)
        assert _env(settings)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        s.exit(1001, a, settings)

    assert _env(settings) == {"FOO": "bar"}


def test_restore_without_an_owner_record_still_honours_the_caller(settings: Path) -> None:
    """unwrap and legacy sessions pass the previous value directly."""
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}), encoding="utf-8"
    )
    assert not W._wrap_owners_path(settings).exists()

    W._restore_claude_wrap_base_url("http://legacy:9999", settings_path=settings)

    assert _env(settings)["ANTHROPIC_BASE_URL"] == "http://legacy:9999"


def test_tool_search_key_is_tracked_independently(settings: Path) -> None:
    """Ownership is per key -- the tool-search entry has the same race."""
    with _Sessions(1001, 1002) as s:
        with mock.patch("os.getpid", lambda: 1001):
            a = W._write_claude_wrap_tool_search("auto", settings_path=settings)
        with mock.patch("os.getpid", lambda: 1002):
            W._write_claude_wrap_tool_search("auto", settings_path=settings)

        s.live.discard(1001)
        with mock.patch("os.getpid", lambda: 1001):
            W._restore_claude_wrap_tool_search(a, settings_path=settings)

        assert W._TOOL_SEARCH_ENV in _env(settings), "surviving session lost tool-search"


def test_exit_on_signal_unwinds_so_finally_can_run() -> None:
    """`cleanup` as the handler never unwound; the settings restore never ran."""
    with pytest.raises(SystemExit) as excinfo:
        W._exit_on_signal(15, None)

    assert excinfo.value.code == 143


def test_unwrap_forces_the_restore_past_a_live_session(settings: Path) -> None:
    """`unwrap` is the user asking for their settings back -- it must not no-op.

    Deferring to a live sibling is right for a session exiting on its own, but
    unwrap deferring means the command prints success while leaving the proxy
    URL in the file.
    """
    with _Sessions(1001) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings)

        with mock.patch("os.getpid", lambda: 2002):
            W._restore_claude_wrap_base_url(None, settings_path=settings, force=True)

    assert _env(settings) == {"FOO": "bar"}, "unwrap left the proxy URL behind"
    assert not W._wrap_owners_path(settings).exists(), "unwrap left ownership state behind"


def test_unwrap_restores_the_true_original_not_the_marker_value(settings: Path) -> None:
    """A caller with no claim of its own trusts the record over its marker.

    The single-slot marker is won by the *last* writer, whose `previous` is the
    first session's proxy URL -- restoring that is the #3205 bug via unwrap.
    """
    settings.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://user-proxy:1234"}}), encoding="utf-8"
    )
    with _Sessions(1001, 1002) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings, port=8787)
        s.launch(1002, "http://127.0.0.1:8788", settings, port=8788)

        with mock.patch("os.getpid", lambda: 2002):
            W._restore_claude_wrap_base_url(
                "http://127.0.0.1:8787", settings_path=settings, force=True
            )

    assert _env(settings)["ANTHROPIC_BASE_URL"] == "http://user-proxy:1234"


def test_a_holder_that_outlived_its_proxy_cannot_veto_the_selfheal(settings: Path) -> None:
    """#2221: a wrapper PID can outlive its proxy; its claim must not block."""
    with _Sessions(1001) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings, port=8787)

        # PID 1001 is still alive, but port 8787 has been proven dead.
        W._restore_claude_wrap_base_url(None, settings_path=settings, dead_ports=frozenset({8787}))

        assert _env(settings) == {"FOO": "bar"}, "dead proxy URL survived the self-heal"


def test_exiting_session_hands_its_marker_to_a_survivor(settings: Path) -> None:
    """The marker has one slot; the leaver must not strand or hijack it."""
    with _Sessions(1001, 1002) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings, port=8787)
        b = s.launch(1002, "http://127.0.0.1:8788", settings, port=8788)

        marker = W._read_wrap_marker(settings)
        assert marker is not None and marker["pid"] == 1002, "last writer owns the marker"

        s.exit(1002, b, settings)

        marker = W._read_wrap_marker(settings)
        assert marker is not None, "survivor lost its #2221 self-heal record"
        assert marker["pid"] == 1001, "marker still describes the exited session"
        assert marker["port"] == 8787
        assert marker["previous"] is None, "marker must carry the true original"


def test_the_founding_session_still_honours_an_explicit_previous(settings: Path) -> None:
    """A sole writer observed the pre-wrap value first-hand; do not override it."""
    with _Sessions(1001) as s:
        s.launch(1001, "http://127.0.0.1:8787", settings)
        s.exit(1001, "https://existing-gateway.example.com/v1", settings)

    assert _env(settings)["ANTHROPIC_BASE_URL"] == "https://existing-gateway.example.com/v1"
