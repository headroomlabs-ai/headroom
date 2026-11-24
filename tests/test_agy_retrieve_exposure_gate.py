"""Exposure gate for agy ``headroom_retrieve`` (headroom-h76.5).

The wrap↔child MCP ``initialize`` handshake proves only that wrap can spawn the
retrieve child; it does NOT prove agy will surface the tool. agy exposes tools
only from its persistent per-tool cache (``<appdata>/mcp/<server>/<tool>.json``),
so a registered-then-reverted entry is rejected at call time as
"Unknown tool: headroom_retrieve". These tests pin the exposure signal that
gates ``HEADROOM_AGY_RETRIEVE_WIRED`` — the flag that keeps ccr compression on —
so unrecoverable markers never ship on a false-positive handshake.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.ccr.mcp_server import CCR_TOOL_NAME
from headroom.cli.wrap import (
    _agy_exposes_retrieve_tool,
    _ccr_backend_is_cross_process,
)
from headroom.mcp_registry import build_headroom_spec
from headroom.mcp_registry.agy import AgyRegistrar


def _registrar(tmp_path: Path) -> AgyRegistrar:
    return AgyRegistrar(home_dir=tmp_path)


def _write_tool_cache(reg: AgyRegistrar, tool: str = CCR_TOOL_NAME) -> None:
    """Simulate agy caching a discovered tool for the headroom server.

    Cache lives under agy's app-data dir (``cache_dir``), decoupled from the
    (migrated) config dir.
    """
    cache = reg.cache_dir / "headroom" / f"{tool}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(f'{{"name": "{tool}"}}')


class TestBackendCrossProcess:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, True),  # unset → default sqlite → shared
            ("", True),
            ("sqlite", True),
            ("redis", True),  # external shared store
            ("memory", False),  # per-process dict → child sees empty store
            ("MEMORY", False),  # case-insensitive
            ("  memory  ", False),  # whitespace-insensitive
        ],
    )
    def test_only_memory_is_process_local(
        self, monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
    ) -> None:
        if value is None:
            monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        else:
            monkeypatch.setenv("HEADROOM_CCR_BACKEND", value)
        assert _ccr_backend_is_cross_process() is expected


class TestExposureSignal:
    """All three conjuncts required: live config entry + tool cache + shared backend."""

    def test_all_present_is_exposed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        reg = _registrar(tmp_path)
        reg.register_server(build_headroom_spec(), force=True)
        _write_tool_cache(reg)
        assert _agy_exposes_retrieve_tool(reg) is True

    def test_missing_config_entry_not_exposed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        reg = _registrar(tmp_path)
        # cache present, but NO mcp_config "headroom" entry (reverted per-run entry)
        _write_tool_cache(reg)
        assert _agy_exposes_retrieve_tool(reg) is False

    def test_missing_tool_cache_not_exposed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        reg = _registrar(tmp_path)
        reg.register_server(build_headroom_spec(), force=True)
        # config entry present, but agy has NOT cached the tool (never discovered)
        assert _agy_exposes_retrieve_tool(reg) is False

    def test_wrong_tool_cached_not_exposed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        reg = _registrar(tmp_path)
        reg.register_server(build_headroom_spec(), force=True)
        # a different tool cached under headroom/ must not count
        _write_tool_cache(reg, tool="something_else")
        assert _agy_exposes_retrieve_tool(reg) is False

    def test_memory_backend_forces_unverified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
        reg = _registrar(tmp_path)
        reg.register_server(build_headroom_spec(), force=True)
        _write_tool_cache(reg)
        # config + cache present, but a per-process store can't resolve hashes
        assert _agy_exposes_retrieve_tool(reg) is False


class TestWiredGate:
    """The ``agy()`` call site sets WIRED only on positive exposure.

    Behavioral spy on the exposure probe: the boolean passed to the downgrade
    warning IS the gated signal that drives ``HEADROOM_AGY_RETRIEVE_WIRED``, so
    asserting on it proves the gate without exec'ing agy.
    """

    @pytest.mark.parametrize("exposed", [True, False])
    def test_wired_follows_exposure(self, monkeypatch: pytest.MonkeyPatch, exposed: bool) -> None:
        import headroom.cli.wrap as wrap_mod

        for key in (
            "HEADROOM_AGY_FR_MODE",
            "HEADROOM_AGY_RETRIEVE_WIRED",
            "HEADROOM_BACKEND",
            "HEADROOM_SAVINGS_PATH",
            "HEADROOM_SAVINGS_EVENTS_PATH",
            "HEADROOM_OTEL_METRICS_ENABLED",
            "HEADROOM_AGY_INBOX_EMIT",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.ensure_root_ca", lambda: (None, None, None, None)
        )
        monkeypatch.setattr("headroom.proxy.agy_ca.build_combined_bundle", lambda: "/dev/null")
        monkeypatch.setattr("headroom.providers.agy.build_agy_env", lambda **kwargs: {})

        class _FakeStats:
            def snapshot_start(self) -> None:
                pass

            def print_summary(self, handler: Any) -> None:
                pass

        monkeypatch.setattr("headroom.providers.agy.stats.AgySessionStats", _FakeStats)
        monkeypatch.setattr("headroom.providers.agy.stats.install_fail_open_handler", lambda: None)
        monkeypatch.setattr(
            "headroom.providers.agy.stats.remove_fail_open_handler",
            lambda handler: None,
        )

        class _FakeRegistrar:
            name = "agy"

            def unregister_server(self, name: str) -> bool:
                return False

        monkeypatch.setattr("headroom.mcp_registry.agy.AgyRegistrar", _FakeRegistrar)
        monkeypatch.setattr("headroom.cli.wrap._disable_tokensave_mcp", lambda *a, **k: None)
        monkeypatch.setattr("headroom.cli.wrap._disable_serena_mcp", lambda *a, **k: None)

        fake_servers = SimpleNamespace(
            terminator=SimpleNamespace(address=("127.0.0.1", 1)), retrieve_port=12345
        )
        monkeypatch.setattr("headroom.cli.wrap._start_agy_servers", lambda *a, **k: fake_servers)
        monkeypatch.setattr("headroom.cli.wrap._stop_agy_servers", lambda servers: None)
        # Handshake succeeds (registered) — exposure alone decides WIRED.
        monkeypatch.setattr(
            "headroom.cli.wrap._setup_headroom_retrieve_mcp_agy",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(
            "headroom.cli.wrap._agy_exposes_retrieve_tool", lambda registrar: exposed
        )
        monkeypatch.setattr("headroom.cli.wrap._register_proxy_client", lambda *a, **k: None)

        seen: list[bool] = []

        def _spy(retrieve_wired: bool) -> None:
            seen.append(retrieve_wired)
            raise SystemExit(0)

        monkeypatch.setattr("headroom.cli.wrap._maybe_warn_agy_ccr_downgrade", _spy)
        monkeypatch.setattr("subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0))

        with pytest.raises(SystemExit):
            wrap_mod.agy.callback(
                port=8899,
                no_proxy=True,
                no_intercept=False,
                backend=None,
                no_mcp=False,
                no_serena=True,
                no_tokensave=True,
                code_graph=False,
                agy_args=(),
            )

        assert seen == [exposed]
        if exposed:
            assert os.environ.get("HEADROOM_AGY_RETRIEVE_WIRED") == "1"
        else:
            assert "HEADROOM_AGY_RETRIEVE_WIRED" not in os.environ
