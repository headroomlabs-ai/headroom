"""GitHub Copilot traffic must be labeled provider "copilot" in the outcome
funnel, even though it travels on the OpenAI/Anthropic wire.

``build_copilot_upstream_url`` is the single routing chokepoint for every
Copilot surface (OpenAI chat/responses and Anthropic messages all build their
upstream URL there), so it flags the request; ``emit_request_outcome`` reads
the flag and relabels the provider. The flag is a task-local ContextVar, so it
never bleeds across concurrent requests.
"""

import asyncio
import contextvars

from headroom import copilot_auth
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

COPILOT = "https://api.githubcopilot.com"


def _run_isolated(fn):
    """Run ``fn`` in a fresh context so the per-request
    ContextVar set by one test never leaks into the next."""
    return contextvars.Context().run(fn)


# --- chokepoint marking -----------------------------------------------------


def test_build_url_marks_request_routed_to_copilot() -> None:
    def scenario() -> bool:
        assert copilot_auth.request_routed_to_copilot() is False
        copilot_auth.build_copilot_upstream_url(COPILOT, "/v1/chat/completions")
        return copilot_auth.request_routed_to_copilot()

    assert _run_isolated(scenario) is True


def test_build_url_does_not_mark_non_copilot_hosts() -> None:
    def scenario() -> bool:
        copilot_auth.build_copilot_upstream_url("https://api.openai.com", "/v1/chat/completions")
        return copilot_auth.request_routed_to_copilot()

    assert _run_isolated(scenario) is False


def test_build_url_clears_stale_flag_for_non_copilot_hosts() -> None:
    def scenario() -> bool:
        copilot_auth.build_copilot_upstream_url(COPILOT, "/v1/messages")
        assert copilot_auth.request_routed_to_copilot() is True
        copilot_auth.build_copilot_upstream_url("https://api.openai.com", "/v1/chat/completions")
        return copilot_auth.request_routed_to_copilot()

    assert _run_isolated(scenario) is False


# --- outcome relabeling -----------------------------------------------------


class _Metrics:
    def __init__(self) -> None:
        self.failed: list[str] = []

    async def record_failed(self, provider: str) -> None:
        self.failed.append(provider)


class _Handler:
    # Exposes ONLY .metrics: a >=500 outcome must relabel and then hit the
    # failed-request guard without touching the success funnel.
    def __init__(self) -> None:
        self.metrics = _Metrics()


def _outcome(provider: str, status_code: int) -> RequestOutcome:
    return RequestOutcome(
        request_id="req-1",
        provider=provider,
        model="claude-opus-4-8",
        original_tokens=0,
        optimized_tokens=0,
        output_tokens=0,
        tokens_saved=0,
        attempted_input_tokens=0,
        status_code=status_code,
    )


def test_relabels_anthropic_to_copilot_when_routed() -> None:
    def scenario() -> _Handler:
        copilot_auth.build_copilot_upstream_url(COPILOT, "/v1/messages")
        handler = _Handler()
        asyncio.run(emit_request_outcome(handler, _outcome("anthropic", 503)))
        return handler

    handler = _run_isolated(scenario)
    # Relabeled before the 5xx guard, so even a failed Copilot request is
    # attributed to "copilot" rather than the wire provider.
    assert handler.metrics.failed == ["copilot"]


def test_relabels_openai_to_copilot_when_routed() -> None:
    def scenario() -> _Handler:
        copilot_auth.build_copilot_upstream_url(COPILOT, "/v1/chat/completions")
        handler = _Handler()
        asyncio.run(emit_request_outcome(handler, _outcome("openai", 503)))
        return handler

    handler = _run_isolated(scenario)
    assert handler.metrics.failed == ["copilot"]


def test_no_relabel_when_not_routed_to_copilot() -> None:
    def scenario() -> _Handler:
        handler = _Handler()
        asyncio.run(emit_request_outcome(handler, _outcome("anthropic", 503)))
        return handler

    handler = _run_isolated(scenario)
    assert handler.metrics.failed == ["anthropic"]


def test_flag_does_not_leak_to_a_later_outcome_in_the_same_context() -> None:
    # Regression: the Copilot flag is a ContextVar whose value persists until
    # overwritten. Within one execution context (e.g. successive messages on a
    # single long-lived WebSocket task), a Copilot request followed by a
    # non-Copilot one must NOT relabel the second. emit_request_outcome consumes
    # (reads AND clears) the flag, so only the first outcome is labeled copilot.
    async def scenario() -> _Handler:
        copilot_auth.build_copilot_upstream_url(COPILOT, "/v1/messages")
        handler = _Handler()
        await emit_request_outcome(handler, _outcome("anthropic", 503))  # routed
        await emit_request_outcome(handler, _outcome("openai", 503))  # not routed
        return handler

    handler = asyncio.run(scenario())
    assert handler.metrics.failed == ["copilot", "openai"]
