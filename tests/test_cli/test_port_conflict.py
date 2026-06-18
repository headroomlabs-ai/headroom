"""Tests for global port conflict detection across all agents."""

from __future__ import annotations

import pytest

from headroom.cli.wrap import _find_available_port


# ---------------------------------------------------------------------------
# _find_available_port — unit tests
# ---------------------------------------------------------------------------


class TestFindAvailablePort:
    def test_first_port_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no ports are occupied, return the start port unchanged."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda _p: False)
        assert _find_available_port(8787) == 8787

    def test_first_port_occupied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When port 8787 is in use, return 8788."""
        monkeypatch.setattr(
            "headroom.cli.wrap._check_proxy", lambda p: p == 8787
        )
        assert _find_available_port(8787) == 8788

    def test_multiple_ports_occupied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ports 8787 and 8788 are occupied, return 8789."""
        occupied = {8787, 8788}
        monkeypatch.setattr(
            "headroom.cli.wrap._check_proxy", lambda p: p in occupied
        )
        assert _find_available_port(8787) == 8789

    def test_no_skip_when_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Port 9000 free → returned unchanged, no skip message."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda _p: False)
        assert _find_available_port(9000) == 9000

    def test_max_attempts_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all ports up to max_attempts are occupied, raise ClickException."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda _p: True)
        import click

        with pytest.raises(click.ClickException, match="No free port"):
            _find_available_port(8787, max_attempts=3)

    def test_all_ports_in_range_occupied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """10 consecutive ports occupied → raises with the right range in message."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda _p: True)
        import click

        with pytest.raises(click.ClickException, match="8787 to 8796"):
            _find_available_port(8787, max_attempts=10)

    def test_gap_in_middle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Port 8787 occupied, 8788 free → returns 8788."""
        occupied = {8787, 8789, 8790}
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda p: p in occupied)
        assert _find_available_port(8787) == 8788

    def test_first_free_after_gap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ports 8787-8790 occupied, 8791 free → returns 8791."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda p: p < 8791)
        assert _find_available_port(8787) == 8791

    def test_empty_port_range_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All free → start port returned immediately (no loop iterations)."""
        monkeypatch.setattr("headroom.cli.wrap._check_proxy", lambda _p: False)
        from headroom.cli.wrap import _check_proxy

        call_count = [0]
        orig = _check_proxy

        def counting_check(p):
            call_count[0] += 1
            return orig(p)

        monkeypatch.setattr("headroom.cli.wrap._check_proxy", counting_check)
        assert _find_available_port(8787) == 8787
        assert call_count[0] == 1, "Should only check once when first port is free"


# ---------------------------------------------------------------------------
# Integration — three code paths that use port conflict detection
# ---------------------------------------------------------------------------


class TestIntegrationPoints:
    """Verify port conflict detection is wired into all three code paths.

    Path A: _launch_tool         — aider, copilot, codex, goose, openhands
    Path B: _run_proxy_only_watcher — cursor, cline, continue
    Path C: claude()             — Claude Code (custom proxy lifecycle)
    """

    def test_launch_tool_uses_find_available_port(self) -> None:
        """_launch_tool calls _find_available_port before _ensure_proxy."""
        import ast, inspect
        from headroom.cli.wrap import _launch_tool

        source = inspect.getsource(_launch_tool)
        # _find_available_port should appear before _ensure_proxy
        find_pos = source.index("_find_available_port")
        ensure_pos = source.index("_ensure_proxy")
        assert find_pos < ensure_pos, (
            "_find_available_port must be called before _ensure_proxy in _launch_tool"
        )

    def test_run_proxy_only_watcher_uses_find_available_port(self) -> None:
        """_run_proxy_only_watcher calls _find_available_port before _ensure_proxy."""
        import ast, inspect
        from headroom.cli.wrap import _run_proxy_only_watcher

        source = inspect.getsource(_run_proxy_only_watcher)
        find_pos = source.index("_find_available_port")
        ensure_pos = source.index("_ensure_proxy")
        assert find_pos < ensure_pos, (
            "_find_available_port must be called before _ensure_proxy in _run_proxy_only_watcher"
        )

    def test_claude_uses_find_available_port(self) -> None:
        """claude() function source code contains _find_available_port before _ensure_proxy."""
        from pathlib import Path
        from importlib import import_module

        wrap_mod = import_module("headroom.cli.wrap")
        wrap_path = Path(wrap_mod.__file__)
        source = wrap_path.read_text()

        claude_start = source.index("def claude(")
        next_def = source.index("\ndef ", claude_start + 1)
        claude_body = source[claude_start:next_def]

        find_pos = claude_body.index("_find_available_port")
        ensure_pos = claude_body.index("_ensure_proxy")
        assert find_pos < ensure_pos, (
            "_find_available_port must be called before _ensure_proxy in claude()"
        )

    def test_all_three_paths_have_gated_by_no_proxy(self) -> None:
        """Each path only runs port conflict detection when not --no-proxy."""
        import ast, inspect
        from pathlib import Path
        from importlib import import_module
        from headroom.cli.wrap import _launch_tool, _run_proxy_only_watcher

        wrap_mod = import_module("headroom.cli.wrap")
        wrap_path = Path(wrap_mod.__file__)
        source = wrap_path.read_text()

        # Path A: _launch_tool
        fn_source = inspect.getsource(_launch_tool)
        assert "not no_proxy" in fn_source, "Path A: _launch_tool missing no_proxy guard"
        assert "_find_available_port" in fn_source, "Path A: _launch_tool missing _find_available_port"

        # Path B: _run_proxy_only_watcher
        fn_source = inspect.getsource(_run_proxy_only_watcher)
        assert "not no_proxy" in fn_source, "Path B: _run_proxy_only_watcher missing no_proxy guard"
        assert "_find_available_port" in fn_source, "Path B: _run_proxy_only_watcher missing _find_available_port"

        # Path C: claude() — Click Command, use file search
        claude_start = source.index("def claude(")
        next_def = source.index("\ndef ", claude_start + 1)
        claude_body = source[claude_start:next_def]
        assert "not no_proxy" in claude_body, "Path C: claude() missing no_proxy guard"
        assert "_find_available_port" in claude_body, "Path C: claude() missing _find_available_port"
