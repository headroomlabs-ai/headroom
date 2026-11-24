"""Regression for headroom-r9k: `wrap agy -p X` must route `-p` to agy (print
mode), not be swallowed as the proxy ``--port``.

508.1 (b5814ffa) added ``@click.option("--port", "-p", ...)`` to the agy
subcommand, but agy's own ``-p`` is ``--print``. Click consumed ``-p`` as
``--port`` before it reached ``agy_args`` -> ``wrap agy -p PROMPT`` failed with
"Invalid value for --port/-p". Fix: the agy ``--port`` option no longer carries
the ``-p`` short alias (long ``--port`` only), so ``-p`` flows through
``ignore_unknown_options`` into ``agy_args`` and ``_agy_print_mode`` recognizes it.

These are parse-level tests via ``make_context`` -- it parses args WITHOUT
invoking the command callback, so no proxy is started.
"""

from __future__ import annotations

import pytest

from headroom.cli.wrap import _agy_print_mode, agy


def _port_option():
    return next(p for p in agy.params if getattr(p, "name", None) == "port")


class TestAgyPortAliasNoShadow:
    @pytest.mark.parametrize("args", [["--no-intercept"], ["--no-intercept", "--port", "9000"]])
    def test_passthrough_skips_proxy_discovery(self, monkeypatch, args):
        def forbidden(*args, **kwargs):
            raise AssertionError("passthrough must not discover or prompt for a proxy")

        monkeypatch.setattr("headroom.cli.port_discovery.reconcile_default_port", forbidden)
        ctx = agy.make_context("agy", args)
        assert ctx.params["no_intercept"]

    def test_dash_p_routes_to_agy_print_not_port(self) -> None:
        ctx = agy.make_context("agy", ["-p", "hello"])
        assert ctx.params["port"] == 8787  # -p did NOT set the proxy port
        assert ctx.params["agy_args"] == ("-p", "hello")
        # Ticket-mandated: proves routing to agy PRINT MODE, not just presence.
        assert _agy_print_mode(ctx.params["agy_args"]) is True

    def test_long_port_still_sets_port(self) -> None:
        ctx = agy.make_context("agy", ["--port", "9000", "foo"])
        assert ctx.params["port"] == 9000
        assert ctx.params["agy_args"] == ("foo",)

    def test_port_option_has_no_dash_p_alias(self) -> None:
        opt = _port_option()
        assert opt.opts == ["--port"]
        assert opt.secondary_opts == []
