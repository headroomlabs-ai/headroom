"""Regression test for `proxy --embedding-server` startup fallback.

The optional embedding-server sidecar module
(`headroom.memory.adapters.watchdog`) is not present on main, yet the
`--embedding-server` flag advertises a graceful fallback to the per-worker
embedder. A misplaced import made the flag raise ``ModuleNotFoundError`` at
startup and crash the proxy instead of falling back.
"""

import sys

from click.testing import CliRunner

from headroom.cli import main


def test_embedding_server_missing_sidecar_falls_back(monkeypatch):
    # Make the optional sidecar module unimportable regardless of whether it is
    # installed, so the fallback path is exercised deterministically.
    monkeypatch.setitem(sys.modules, "headroom.memory.adapters.watchdog", None)

    # Don't actually start a server.
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "run_server", lambda *args, **kwargs: None)

    result = CliRunner().invoke(main, ["proxy", "--embedding-server", "--port", "8799"])

    assert result.exit_code == 0, f"proxy crashed instead of falling back: {result.output}"
    assert result.exception is None
    assert "Falling back to per-worker embedder" in result.output
