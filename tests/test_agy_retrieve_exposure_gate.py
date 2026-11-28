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
            (None, True),  # unset -> default sqlite -> shared
            ("", True),
            ("sqlite", True),
            ("memory", False),  # per-process dict -> child sees empty store
            ("MEMORY", False),  # case-insensitive
            ("  memory  ", False),  # whitespace-insensitive
        ],
    )
    def test_only_memory_is_process_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
    ) -> None:
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
        monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
        if value is None:
            monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        else:
            monkeypatch.setenv("HEADROOM_CCR_BACKEND", value)
        assert _ccr_backend_is_cross_process() is expected

    def test_stateless_mode_is_not_cross_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
        monkeypatch.setenv("HEADROOM_STATELESS", "1")
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        monkeypatch.delenv("HEADROOM_COMPRESS_SESSION_STORE", raising=False)
        assert _ccr_backend_is_cross_process() is False

    def test_sqlite_init_fallback_is_not_cross_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
        monkeypatch.setattr(
            "headroom.cache.compression_store._create_default_ccr_backend", lambda: None
        )
        assert _ccr_backend_is_cross_process() is False

    def test_redis_without_url_falls_back_to_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
        monkeypatch.delenv("HEADROOM_REDIS_URL", raising=False)
        monkeypatch.delenv("HEADROOM_COMPRESS_SESSION_STORE", raising=False)
        assert _ccr_backend_is_cross_process() is False

    def test_primed_and_live_retrieve_tool_schema_match(self) -> None:
        pytest.importorskip("mcp")
        import asyncio

        from headroom.ccr.mcp_server import (
            CCR_RETRIEVE_TOOL_DESCRIPTION,
            CCR_RETRIEVE_TOOL_INPUT_SCHEMA,
            CCR_TOOL_NAME,
            create_ccr_mcp_server,
        )

        server = create_ccr_mcp_server()
        assert server is not None
        result = asyncio.run(server._on_list_tools(None, None))
        retrieve_tools = [t for t in result.tools if t.name == CCR_TOOL_NAME]
        assert len(retrieve_tools) == 1
        assert retrieve_tools[0].description == CCR_RETRIEVE_TOOL_DESCRIPTION
        assert retrieve_tools[0].input_schema == CCR_RETRIEVE_TOOL_INPUT_SCHEMA

    @pytest.mark.windows_newline
    def test_prime_cache_and_gemini_md_cleanup_use_lf_newlines(self, tmp_path: Path) -> None:
        from headroom.cli.wrap import (
            _AGY_GEMINI_BLOCK_END,
            _AGY_GEMINI_BLOCK_START,
            _prime_agy_retrieve_tool_cache,
            _remove_gemini_md_block,
        )

        reg = _registrar(tmp_path)
        _prime_agy_retrieve_tool_cache(reg)
        cache_path = reg.cache_dir / "headroom" / f"{CCR_TOOL_NAME}.json"
        assert b"\r\n" not in cache_path.read_bytes()

        gemini_md = tmp_path / "GEMINI.md"
        gemini_md.write_text(
            f"before\n{_AGY_GEMINI_BLOCK_START}\nmanaged\n{_AGY_GEMINI_BLOCK_END}\nafter\n",
            encoding="utf-8",
            newline="\n",
        )
        assert _remove_gemini_md_block(gemini_md) is True
        raw = gemini_md.read_bytes()
        assert b"\n" in raw
        assert b"\r\n" not in raw


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


def test_ccr_backend_probe_preserves_live_redis_health_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probing the CCR backend must not overwrite or unregister an active Redis guard."""
    from headroom.cache import redis_client
    from headroom.cli import wrap as wrap_mod

    class _DummyGuard:
        pass

    live_guard = _DummyGuard()
    monkeypatch.setitem(redis_client._GUARDS, "ccr", live_guard)

    class _FakeRedisBackend:
        def __init__(self) -> None:
            self._guard = _DummyGuard()
            redis_client._GUARDS["ccr"] = self._guard

        def close(self) -> None:
            redis_client._GUARDS.pop("ccr", None)

    monkeypatch.setattr(
        "headroom.cache.compression_store._create_default_ccr_backend",
        _FakeRedisBackend,
    )

    assert wrap_mod._ccr_backend_is_cross_process() is True
    assert redis_client._GUARDS.get("ccr") is live_guard


def test_exposure_gate_rejects_replacement_command_even_with_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the headroom command after confirmation must leave exposure False."""
    from headroom.cli import wrap as wrap_mod
    from headroom.mcp_registry.base import ServerSpec

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **k: True)

    reg1 = _registrar(tmp_path)
    reg2 = _registrar(tmp_path)
    assert wrap_mod._setup_headroom_retrieve_mcp_agy(reg1) is True
    assert _agy_exposes_retrieve_tool(reg1) is True

    # Replace the configured headroom command after confirmation while the
    # primed retrieve-tool cache already exists.
    replacement = ServerSpec(
        name="headroom",
        command="/other/unverified-headroom",
        args=("mcp", "serve"),
    )
    reg2.register_server(replacement, force=True)

    assert _agy_exposes_retrieve_tool(reg1) is False


def test_smoke_verify_mcp_handshake_rejects_jsonrpc_error_and_accepts_result() -> None:
    """Smoke probe must return False for JSON-RPC error responses and True for result objects."""
    import sys

    from headroom.cli import wrap as wrap_mod

    error_child = (
        "import json, sys; "
        "sys.stdin.readline(); "
        'print(json.dumps({"jsonrpc": "2.0", "id": 1, '
        '"error": {"code": -32603, "message": "init failed"}}), flush=True)'
    )
    ok_child = (
        "import json, sys; "
        "sys.stdin.readline(); "
        'print(json.dumps({"jsonrpc": "2.0", "id": 1, '
        '"result": {"protocolVersion": "2024-11-05", "capabilities": {}}}), flush=True)'
    )

    assert wrap_mod._smoke_verify_mcp_handshake(sys.executable, ["-c", error_child], {}) is False
    assert wrap_mod._smoke_verify_mcp_handshake(sys.executable, ["-c", ok_child], {}) is True


def test_exposure_gate_rechecks_verified_entry_after_backend_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the headroom command during _ccr_backend_is_cross_process must fail exposure."""
    from headroom.cli import wrap as wrap_mod
    from headroom.mcp_registry.base import ServerSpec

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **k: True)

    reg1 = _registrar(tmp_path)
    reg2 = _registrar(tmp_path)
    assert wrap_mod._setup_headroom_retrieve_mcp_agy(reg1) is True

    replacement = ServerSpec(
        name="headroom",
        command="/other/unverified-during-probe",
        args=("mcp", "serve"),
    )

    def _replace_during_backend_probe() -> bool:
        reg2.register_server(replacement, force=True)
        return True

    monkeypatch.setattr(wrap_mod, "_ccr_backend_is_cross_process", _replace_during_backend_probe)
    assert _agy_exposes_retrieve_tool(reg1) is False


def test_cache_priming_runs_under_config_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool cache priming runs inside confirm_server_if's config lock."""
    import threading

    from headroom.cli import wrap as wrap_mod
    from headroom.mcp_registry.base import RegisterStatus, ServerSpec

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **k: True)

    reg1 = _registrar(tmp_path)
    reg2 = _registrar(tmp_path)
    replacement = ServerSpec(
        name="headroom",
        command="/other/replacement",
        args=("mcp", "serve"),
    )

    entered_prime = threading.Event()
    release_prime = threading.Event()
    orig_prime = wrap_mod._prime_agy_retrieve_tool_cache

    def _paused_prime(registrar: Any) -> None:
        entered_prime.set()
        release_prime.wait(timeout=5.0)
        orig_prime(registrar)

    monkeypatch.setattr(wrap_mod, "_prime_agy_retrieve_tool_cache", _paused_prime)

    setup_results: list[bool] = []
    replace_results: list[Any] = []

    t1 = threading.Thread(
        target=lambda: setup_results.append(wrap_mod._setup_headroom_retrieve_mcp_agy(reg1))
    )
    t2 = threading.Thread(
        target=lambda: replace_results.append(reg2.register_server(replacement, force=True))
    )
    t1.start()
    assert entered_prime.wait(timeout=5.0)
    t2.start()
    t2.join(timeout=0.1)
    assert t2.is_alive(), "replacement did not block on config lock during cache priming"
    release_prime.set()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert setup_results == [True]
    assert replace_results[0].status == RegisterStatus.REGISTERED
