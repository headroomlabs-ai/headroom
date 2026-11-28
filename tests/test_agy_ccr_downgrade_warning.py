"""Tests for the loud ccr->lossless downgrade warning in ``headroom wrap agy``.

Originally written test-first (red) before ``_maybe_warn_agy_ccr_downgrade``
existed in ``headroom/cli/wrap.py``; the implementation has since landed.

Scope (headroom-svf; ccr-default per headroom-37g.32): when ``headroom wrap
agy`` runs in ``ccr`` mode (now the default -- unset/invalid resolve to ccr)
but the retrieve MCP could NOT be wired for the run, the Cloud Code Assist handler
(``headroom.proxy.handlers.gemini._resolve_agy_fr_mode``) silently downgrades
functionResponse compression to ``lossless`` (a no-op), so tool-output
savings collapse to ~0 with no user-visible warning. This must become loud
and actionable, with best-effort cause detection:

* ``mcp`` not importable in *this* (parent) interpreter -> ADVISORY hint to
  install ``headroom-ai[proxy]`` (the agy child is resolved via
  ``shutil.which("headroom")`` and need NOT share this venv, hence a
  likely-cause hint, not certainty).
* ``mcp`` importable -> the retrieve MCP failed to register or complete its
  handshake; point at the ``MCP retrieve tool:`` failure line already printed
  to the console (the agy path runs in-process servers and writes no
  ``proxy.log``).

The warning fires whenever the resolved mode is ccr (explicit, OR the
unset/invalid default) AND the retrieve MCP did not wire. It must stay silent
when retrieve DID wire, or when ``lossless`` was requested explicitly (no
downgrade occurred).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import headroom.cli.wrap as wrap_mod
from headroom.cli.wrap import _maybe_warn_agy_ccr_downgrade


class TestMaybeWarnAgyCcrDowngrade:
    # ------------------------------------------------------------------
    # Gating: fires only for ccr + not-wired.
    # ------------------------------------------------------------------

    def test_silent_when_retrieve_registered(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=True)
        out = capsys.readouterr().out
        assert out == ""

    def test_silent_when_lossless_requested(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "lossless")
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        assert out == ""

    def test_warns_when_unset_defaults_to_ccr_and_not_registered(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ccr is now the default (WU-CCRDEFAULT): unset + not-wired downgrades,
        # so the warning must fire (previously silent when lossless was default).
        monkeypatch.delenv("HEADROOM_AGY_FR_MODE", raising=False)
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out

    def test_warns_when_explicit_ccr_and_not_registered(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out

    def test_invalid_mode_value_treated_as_ccr_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mirrors _requested_agy_fr_mode's fallback-to-ccr for garbage values:
        # invalid -> ccr default -> not-wired -> warns.
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "bogus")
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        assert "DISABLED" in out

    # ------------------------------------------------------------------
    # Cause detection: in-parent `mcp` importability drives the branch.
    # The fakes are NAME-SENSITIVE (keyed on the probed module name), so the
    # branch tests also verify the probe asks about "mcp" specifically.
    # ------------------------------------------------------------------

    def test_mcp_missing_branch_recommends_proxy_extra(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ccr must be requested explicitly now that lossless is the default.
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        # False ONLY for "mcp": probing any other name would flip the branch.
        monkeypatch.setattr("headroom.cli.wrap._module_available", lambda name: name != "mcp")
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        assert "headroom-ai[proxy]" in out
        assert "pip install mcp" in out
        # Advisory caveat: parent-mcp-present/absent doesn't guarantee child state.
        assert "ADVISORY" in out or "likely cause" in out
        assert "proxy.log" not in out

    def test_mcp_present_branch_points_at_console_failure_line(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ccr must be requested explicitly now that lossless is the default.
        monkeypatch.setenv("HEADROOM_AGY_FR_MODE", "ccr")
        # True ONLY for "mcp": probing any other name would flip the branch.
        monkeypatch.setattr("headroom.cli.wrap._module_available", lambda name: name == "mcp")
        _maybe_warn_agy_ccr_downgrade(retrieve_wired=False)
        out = capsys.readouterr().out
        # The agy path runs in-process servers and writes NO proxy.log; the
        # handshake failure detail is the "MCP retrieve tool:" console line.
        assert "MCP retrieve tool:" in out
        # Cause text broadened for the exposure gate: handshake-OK-but-uncached
        # is now a distinct downgrade reason alongside register/handshake failure.
        assert "did not register/handshake" in out
        assert "exposed it as a callable tool" in out
        assert "proxy.log" not in out
        assert "headroom-ai[proxy]" not in out


class TestAgyCallSiteWiring:
    """Prove ``agy()`` actually invokes the warning on the downgrade path.

    All helper tests above exercise ``_maybe_warn_agy_ccr_downgrade`` in
    isolation; without this test, deleting the call site inside ``agy()``
    would leave the suite green — the exact silent-downgrade regression this
    feature exists to prevent. Behavioral spy: the call site raising through
    the spy aborts ``agy()`` before it would exec the agy binary.
    """

    def test_agy_invokes_downgrade_warning_when_retrieve_not_wired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        # -- Binary resolution: agy "installed", rtk absent. ---------------
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)

        # -- CA / child-env plumbing: no real crypto, no real env build. ----
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.ensure_root_ca", lambda: (None, None, None, None)
        )
        monkeypatch.setattr("headroom.proxy.agy_ca.build_combined_bundle", lambda: "/dev/null")
        monkeypatch.setattr("headroom.providers.agy.build_agy_env", lambda **kwargs: {})

        # -- Session stats / fail-open observability: inert fakes. ----------
        class _FakeStats:
            def snapshot_start(self) -> None:
                pass

            def print_summary(self, handler: Any) -> None:
                pass

        monkeypatch.setattr("headroom.providers.agy.stats.AgySessionStats", _FakeStats)
        monkeypatch.setattr("headroom.providers.agy.stats.install_fail_open_handler", lambda: None)
        monkeypatch.setattr(
            "headroom.providers.agy.stats.remove_fail_open_handler", lambda handler: None
        )

        # -- MCP registrar + tooling setup: inert fakes. --------------------
        class _FakeRegistrar:
            name = "agy"

            def register_server(self, spec: Any, force: bool = False) -> Any:
                raise AssertionError("register_server must not be reached in this test")

            def unregister_server(self, name: str) -> bool:
                return False

        monkeypatch.setattr("headroom.mcp_registry.agy.AgyRegistrar", _FakeRegistrar)
        monkeypatch.setattr(
            "headroom.cli.wrap._disable_tokensave_mcp", lambda *args, **kwargs: None
        )
        monkeypatch.setattr("headroom.cli.wrap._disable_serena_mcp", lambda *args, **kwargs: None)

        # -- In-process servers: fake handle with a live retrieve port. -----
        fake_servers = SimpleNamespace(
            terminator=SimpleNamespace(address=("127.0.0.1", 1)), retrieve_port=12345
        )
        monkeypatch.setattr(
            "headroom.cli.wrap._start_agy_servers", lambda *args, **kwargs: fake_servers
        )
        monkeypatch.setattr("headroom.cli.wrap._stop_agy_servers", lambda servers: None)

        # -- Downgrade scenario: retrieve MCP does not wire. -----------------
        monkeypatch.setattr(
            "headroom.cli.wrap._setup_headroom_retrieve_mcp_agy",
            lambda *args, **kwargs: False,
        )

        # -- Spy: record the call, abort agy() before it would exec agy. -----
        calls: list[bool] = []

        def _spy(retrieve_registered: bool) -> None:
            calls.append(retrieve_registered)
            raise SystemExit(0)

        monkeypatch.setattr("headroom.cli.wrap._maybe_warn_agy_ccr_downgrade", _spy)

        # -- Guard: if the call site is ever removed, never exec a binary. ---
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
        # _register_proxy_client writes a durable marker under workspace_dir()
        # (~/.headroom/proxy_clients/<port>/); stub it so this test never touches
        # the real client registry (conftest provides no HOME isolation).
        monkeypatch.setattr("headroom.cli.wrap._register_proxy_client", lambda *a, **k: None)

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

        assert calls == [False]
