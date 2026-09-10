"""Every upstream call must be bounded.

There was no timeout in this backend at all. Observed 2026-08-07 under load:
four agent workers blocked on ESTABLISHED connections for 36+ minutes while
the proxy answered /readyz in 0.11s. No error, no retry, no log line -- the
caller simply stops, forever, and that is indistinguishable from slow work.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from headroom.backends.litellm import (
    DEFAULT_UPSTREAM_TIMEOUT,
    UPSTREAM_TIMEOUT_ENV,
    _upstream_timeout,
)

_SRC = Path(__file__).resolve().parents[1] / "headroom" / "backends" / "litellm.py"


def test_every_acompletion_call_is_bounded():
    """A new dispatch path added without a timeout reintroduces the hang.

    Checked structurally rather than by mocking, because the failure mode is a
    call site someone ADDS later -- which no mock of the existing paths sees.
    """
    tree = ast.parse(_SRC.read_text())
    calls, guards = 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "acompletion":
            calls += 1
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "setdefault"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "timeout"
        ):
            guards += 1
    assert calls > 0, "no acompletion call sites found -- test is stale"
    assert guards >= calls, (
        f"{calls} acompletion call site(s) but only {guards} timeout guard(s); "
        "an unbounded upstream call blocks its caller forever"
    )


def test_a_junk_env_value_cannot_disable_the_timeout(monkeypatch):
    """`0` means 'no timeout' to httpx, i.e. exactly the bug. So does junk."""
    for bad in ("", "0", "-1", "nonsense", "None"):
        monkeypatch.setenv(UPSTREAM_TIMEOUT_ENV, bad)
        assert _upstream_timeout() == DEFAULT_UPSTREAM_TIMEOUT, bad


def test_an_operator_can_still_tune_it(monkeypatch):
    monkeypatch.setenv(UPSTREAM_TIMEOUT_ENV, "42.5")
    assert _upstream_timeout() == pytest.approx(42.5)


def test_the_default_is_generous_enough_for_real_work():
    """Streaming: litellm expands a float across all httpx phases, so this is
    the max gap BETWEEN CHUNKS, not a cap on total generation. A steady long
    answer is never cut off."""
    assert 60.0 <= DEFAULT_UPSTREAM_TIMEOUT <= 1800.0
