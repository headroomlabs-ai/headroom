"""The hot-path perf work only pays out if the Anthropic handler actually uses it.

Both optimisations in this area are opt-in at the call site: the pipeline accepts
``defer_waste_signals`` but defaults it off, and ``count_messages_memoized`` is a
free function nothing calls by construction. A previous revision shipped both
helpers with no production caller, so the advertised CPU reduction was inert
while every test still passed.

These are wiring tripwires: they assert the handler reaches for the helpers, so
"helper exists but is dead code" cannot regress silently again.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

HANDLER_SRC = inspect.getsource(AnthropicHandlerMixin.handle_anthropic_messages)


# ── Waste-signal deferral ────────────────────────────────────────────────


def test_response_path_apply_sites_defer_waste_signal_parsing() -> None:
    """Every response-path pipeline.apply must opt into deferral."""
    # token mode fast-pass, token mode main, non-cache mode, cache-delta mode.
    assert HANDLER_SRC.count("defer_waste_signals=True") == 4


def test_not_every_apply_site_defers() -> None:
    """The background-compression enqueue must keep inline detection.

    Its result feeds the compression cache, never ``emit_request_outcome``, so
    there is no funnel to collect a deferred provider from — deferring there
    would drop the signals rather than move them off the response path.
    """
    apply_sites = HANDLER_SRC.count("anthropic_pipeline.apply(")
    assert apply_sites > HANDLER_SRC.count("defer_waste_signals=True")


def test_deferred_signals_are_detected_off_the_response_path() -> None:
    """Deferral without the compensating task would silently lose telemetry."""
    assert "self._start_waste_signal_task(" in HANDLER_SRC
    # ...and the task has to reach the funnel that collects it.
    assert "waste_signals_task=waste_signals_task" in HANDLER_SRC


def test_inline_signals_still_win_when_present() -> None:
    """The deferred branch must not shadow an inline result."""
    inline = HANDLER_SRC.index("result.waste_signals.to_dict()")
    deferred = HANDLER_SRC.index("self._start_waste_signal_task(")
    assert inline < deferred, "inline capture must be tried before the deferred branch"


# ── Token-count memoisation ──────────────────────────────────────────────


def test_handler_recounts_route_through_the_memo() -> None:
    """No raw full-transcript recount may survive in the handler.

    The handler recounts the transcript at up to six points per request; each
    raw ``tokenizer.count_messages`` call re-tokenises messages that have not
    changed since the previous turn.
    """
    assert "tokenizer.count_messages(" not in HANDLER_SRC
    assert "self._get_token_count_memo(" in HANDLER_SRC
    # The memo helper is used, not just fetched.
    assert "count_messages_memoized" in HANDLER_SRC


def test_memo_is_scoped_per_session() -> None:
    """A memo shared across sessions would serve counts across conversations."""
    assert "self._get_token_count_memo(session_id)" in HANDLER_SRC


# ── _start_waste_signal_task behaviour ───────────────────────────────────


class _Harness:
    """Minimal stand-in for the proxy: just the background-runner contract."""

    def __init__(self) -> None:
        self.ran = False

    async def _run_compression_background(self, fn):  # noqa: ANN001, ANN202
        self.ran = True
        return fn()

    _start_waste_signal_task = AnthropicHandlerMixin._start_waste_signal_task


class _Result:
    def __init__(self, provider=None) -> None:  # noqa: ANN001
        self.waste_signals_provider = provider


@pytest.mark.asyncio
async def test_no_provider_means_no_task() -> None:
    """Nothing to defer -> no task, no bookkeeping."""
    harness = _Harness()
    task = harness._start_waste_signal_task(
        _Result(provider=None), model="claude-opus-4-8", provider_name="anthropic"
    )
    assert task is None


@pytest.mark.asyncio
async def test_provider_is_invoked_off_the_response_path() -> None:
    """The provider runs on the background runner and yields the signals dict."""

    class _Signals:
        @staticmethod
        def to_dict() -> dict[str, int]:
            return {"skipped_units": 3, "applied_units": 7}

    harness = _Harness()
    task = harness._start_waste_signal_task(
        _Result(provider=lambda: _Signals()),
        model="claude-opus-4-8",
        provider_name="anthropic",
    )
    assert task is not None
    assert await asyncio.wait_for(task, timeout=5) == {"skipped_units": 3, "applied_units": 7}
    assert harness.ran, "detection must go through the background runner"


@pytest.mark.asyncio
async def test_provider_failure_is_swallowed() -> None:
    """Telemetry is fail-open: a broken provider must not surface to the request."""

    def _boom():  # noqa: ANN202
        raise RuntimeError("detector exploded")

    harness = _Harness()
    task = harness._start_waste_signal_task(
        _Result(provider=_boom), model="claude-opus-4-8", provider_name="anthropic"
    )
    assert task is not None
    assert await asyncio.wait_for(task, timeout=5) is None
