"""Token counting must run off the event loop (GH #1701): the Anthropic messages
handler resolved the tokenizer and counted the conversation inline in the async
handler. For HF-backed models (e.g. deepseek-*) first use triggers an unbounded
network download, freezing the whole server (610s request, then /livez, /readyz
and /health hang until kill). The fix routes resolution + counting through
HeadroomProxy._count_tokens_offloaded (compression executor, bounded by
COMPRESSION_TIMEOUT_SECONDS, fail-open to estimation) — shared by every provider
handler (Anthropic, OpenAI, Gemini), since the OpenAI passthrough endpoints
receive the same HF-backed models — and offloads the inline batch
pipeline.apply() calls the same way.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.batch import BatchHandlerMixin
from headroom.proxy.handlers.gemini import GeminiHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.server import (
    CompressionQuarantinedError,
    ProxyConfig,
    create_app,
)
from headroom.proxy.token_counting import (
    _count_offloaded,
    count_texts_offloaded,
    count_tokens_offloaded,
)
from headroom.tokenizers import EstimatingTokenCounter


def _make_proxy():  # noqa: ANN202 — returns the internal HeadroomProxy
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    return app.state.proxy


def test_handlers_offload_token_counting_and_batch_apply() -> None:
    """Wiring guard: the request paths must use the offloaded helpers, not inline
    get_tokenizer/count_messages or pipeline.apply on the event loop."""
    # Every provider handler that counts the original conversation must route
    # resolution + counting through the shared fail-open helper, never inline on
    # the loop. OpenAI /chat + /responses are multi-provider passthroughs, so an
    # HF-routed model (qwen, deepseek, llama, ...) can reach them and cold-load.
    for mixin, method in (
        (AnthropicHandlerMixin, "handle_anthropic_messages"),
        (OpenAIHandlerMixin, "handle_openai_chat"),
        (OpenAIHandlerMixin, "handle_openai_responses"),
        (GeminiHandlerMixin, "handle_gemini_generate_content"),
        (GeminiHandlerMixin, "handle_google_cloudcode_stream"),
        (GeminiHandlerMixin, "handle_gemini_count_tokens"),
    ):
        fn = getattr(mixin, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be async"
        src = inspect.getsource(fn)
        assert "_count_tokens_offloaded(" in src, f"{method}: token counting not offloaded"
        assert "tokenizer = get_tokenizer(" not in src, (
            f"{method}: tokenizer resolved inline on the loop"
        )

    fn = GeminiHandlerMixin.handle_gemini_stream_generate_content
    assert inspect.iscoroutinefunction(fn)
    src = inspect.getsource(fn)
    assert "_count_texts_offloaded(" in src, "streaming Gemini text counting not offloaded"
    assert "tokenizer = get_tokenizer(" not in src, "tokenizer resolved inline on the loop"
    assert "count_text(" not in src, "streaming Gemini count_text still runs on the loop"
    assert "_dict_parts(" in src, "streaming Gemini must reuse the shared _dict_parts coercion"
    assert 'isinstance(part.get("text"), str)' in src, (
        "streaming Gemini must skip non-str text so count_text can't 500"
    )

    for mixin, method in (
        (AnthropicHandlerMixin, "handle_anthropic_batch_create"),
        (BatchHandlerMixin, "handle_google_batch_create"),
        (BatchHandlerMixin, "_compress_batch_jsonl"),
    ):
        fn = getattr(mixin, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be async"
        src = inspect.getsource(fn)
        if "pipeline.apply(" in src:
            assert "_run_compression_in_executor(" in src, f"{method}: apply() not offloaded"
            assert "COMPRESSION_TIMEOUT_SECONDS" in src, f"{method}: offload missing timeout"

    helper_src = inspect.getsource(_count_offloaded)
    assert "COMPRESSION_TIMEOUT_SECONDS" in helper_src
    assert "EstimatingTokenCounter" in helper_src, "helper must fail open to estimation"


async def test_count_tokens_offloaded_runs_on_worker_thread(monkeypatch) -> None:  # noqa: ANN001
    proxy = _make_proxy()
    loop_thread = threading.current_thread().name
    seen: dict[str, str] = {}

    class _SpyTokenizer(EstimatingTokenCounter):
        def count_messages(self, messages):  # noqa: ANN001, ANN201
            seen["thread"] = threading.current_thread().name
            return super().count_messages(messages)

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda *a, **k: _SpyTokenizer())

    _, tokens = await proxy._count_tokens_offloaded("gpt-4", [{"role": "user", "content": "hi"}])

    assert tokens > 0
    assert seen["thread"].startswith("headroom-compress")
    assert seen["thread"] != loop_thread


async def test_count_tokens_offloaded_keeps_loop_responsive(monkeypatch) -> None:  # noqa: ANN001
    """A slow tokenizer (stand-in for an HF network load) must not starve the loop —
    the pre-fix inline call yielded ~0 ticks here."""
    proxy = _make_proxy()
    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    class _SlowTokenizer(EstimatingTokenCounter):
        def count_messages(self, messages):  # noqa: ANN001, ANN201
            time.sleep(0.3)
            return super().count_messages(messages)

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda *a, **k: _SlowTokenizer())

    tick_task = asyncio.create_task(_ticker())
    try:
        _, tokens = await proxy._count_tokens_offloaded("m", [{"role": "user", "content": "hi"}])
    finally:
        tick_task.cancel()

    assert tokens > 0
    assert ticks >= 5


async def test_count_tokens_offloaded_fails_open(monkeypatch) -> None:  # noqa: ANN001
    """Resolution errors and timeouts downgrade to estimation instead of raising."""
    proxy = _make_proxy()

    def _boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("tokenizer backend exploded")

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", _boom)

    tokenizer, tokens = await proxy._count_tokens_offloaded(
        "deepseek-chat", [{"role": "user", "content": "hello world"}]
    )

    assert isinstance(tokenizer, EstimatingTokenCounter)
    assert tokens > 0
    # Logged-once bookkeeping records the downgraded model.
    assert "deepseek-chat" in proxy._token_count_fallback_models


async def test_count_tokens_offloaded_fails_open_on_executor_quarantine() -> None:
    """Now that OpenAI/Gemini counting shares the compression executor, an
    unrelated request's compression timeout can quarantine it — the next
    ``_run_compression_in_executor`` call raises ``CompressionQuarantinedError``
    immediately (process-wide state). A request that is only counting tokens
    must not 500 on that; it fails open to estimation like any other error."""
    # The executor's ``except Exception`` fail-open only catches the quarantine
    # error because it subclasses Exception — pin that contract.
    assert issubclass(CompressionQuarantinedError, Exception)

    proxy = _make_proxy()
    # Record a concurrent compression as timed out so the real executor guard
    # quarantines the next call — no mock of the helper itself.
    proxy._compression_timed_out_in_flight = 1

    tokenizer, tokens = await proxy._count_tokens_offloaded(
        "qwen2.5-coder", [{"role": "user", "content": "hello world"}]
    )

    assert isinstance(tokenizer, EstimatingTokenCounter)
    assert tokens > 0
    assert "qwen2.5-coder" in proxy._token_count_fallback_models


async def test_count_tokens_offloaded_returns_count_text_capable_tokenizer() -> None:
    """The fail-open tokenizer should still support text counting for callers
    that need per-fragment accounting."""
    proxy = _make_proxy()
    # Quarantine forces the fail-open branch (an EstimatingTokenCounter).
    proxy._compression_timed_out_in_flight = 1

    # The empty-messages count is intentionally discarded by that handler
    # (it sums text parts itself), so only the tokenizer matters here.
    tokenizer, _ = await proxy._count_tokens_offloaded("qwen2.5-coder", [])

    assert isinstance(tokenizer, EstimatingTokenCounter)
    # The streaming handler's per-part loop must not raise on the fallback.
    assert tokenizer.count_text("hello world") > 0


async def test_count_texts_offloaded_runs_on_worker_thread(monkeypatch) -> None:  # noqa: ANN001
    proxy = _make_proxy()
    loop_thread = threading.current_thread().name
    seen: dict[str, str] = {}

    class _SpyTokenizer(EstimatingTokenCounter):
        def count_text(self, text):  # noqa: ANN001, ANN201
            seen["thread"] = threading.current_thread().name
            return super().count_text(text)

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda *a, **k: _SpyTokenizer())

    _, tokens = await proxy._count_texts_offloaded("gemini-pro", ["hello", "world"])

    assert tokens > 0
    assert seen["thread"].startswith("headroom-compress")
    assert seen["thread"] != loop_thread


async def test_count_texts_offloaded_fails_open(monkeypatch) -> None:  # noqa: ANN001
    """The texts variant downgrades to estimation on a resolution error, the same
    as the messages variant (its fail-open branch was previously uncovered)."""
    proxy = _make_proxy()

    def _boom(*a, **k):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("tokenizer backend exploded")

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", _boom)

    tokenizer, tokens = await proxy._count_texts_offloaded("deepseek-chat", ["hello", "world"])

    assert isinstance(tokenizer, EstimatingTokenCounter)
    assert tokens > 0
    assert "deepseek-chat" in proxy._token_count_fallback_models


async def test_count_offloaded_without_executor_estimates() -> None:
    """An owner with no compression executor (a lightweight caller or test double)
    fails open to estimation inline instead of crashing on the missing runner."""

    class _NoExecutorOwner:
        pass

    owner = _NoExecutorOwner()

    tok, n_msg = await count_tokens_offloaded(
        owner, "gpt-4", [{"role": "user", "content": "hello world"}]
    )
    assert isinstance(tok, EstimatingTokenCounter)
    assert n_msg > 0

    tok2, n_txt = await count_texts_offloaded(owner, "gemini-pro", ["hello", "world"])
    assert isinstance(tok2, EstimatingTokenCounter)
    assert n_txt > 0


async def test_count_texts_offloaded_sums_fragments(monkeypatch) -> None:  # noqa: ANN001
    """The streaming rewrite sums per-fragment counts, matching the old per-part
    count_text loop it replaced."""
    proxy = _make_proxy()
    monkeypatch.setattr(
        "headroom.tokenizers.get_tokenizer", lambda *a, **k: EstimatingTokenCounter()
    )
    fragments = ["hello", "world", "foo"]

    _, total = await proxy._count_texts_offloaded("gemini-pro", fragments)

    est = EstimatingTokenCounter()
    assert total == sum(est.count_text(f) for f in fragments)
    assert total > 0
