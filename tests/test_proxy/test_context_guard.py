"""Tests for the context-limit guard (headroom/proxy/context_guard.py)."""

from __future__ import annotations

import json

import pytest

from headroom.proxy.context_guard import (
    REPORT_FRACTION,
    StreamUsageGuard,
    believed_context_limit,
    context_guard_enabled,
    credential_scope,
    effective_context_limit,
    has_context_1m_beta,
    note_prompt_too_long,
    nudge_response_usage,
    reset_learned_limits,
)


@pytest.fixture(autouse=True)
def _clean_learned_limits():
    reset_learned_limits()
    yield
    reset_learned_limits()


def _message_start_event(
    input_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> bytes:
    payload = {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [],
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
                "output_tokens": 1,
            },
        },
    }
    return b"event: message_start\ndata: " + json.dumps(payload).encode() + b"\n\n"


def _parse_usage(event_bytes: bytes) -> dict:
    for line in event_bytes.split(b"\n"):
        if line.startswith(b"data:"):
            return json.loads(line[5:].strip())["message"]["usage"]
    raise AssertionError("no data line in event")


def _message_delta_event(
    input_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> bytes:
    payload = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
            "output_tokens": 42,
        },
    }
    return b"event: message_delta\ndata: " + json.dumps(payload).encode() + b"\n\n"


class TestBetaAndLimits:
    def test_has_context_1m_beta_detects_token(self):
        assert has_context_1m_beta("context-1m-2025-08-07")
        assert has_context_1m_beta("oauth-2025-04-20, context-1m-2025-08-07")
        assert not has_context_1m_beta("oauth-2025-04-20")
        assert not has_context_1m_beta(None)
        assert not has_context_1m_beta("")

    def test_believed_limit_raised_by_1m_beta(self):
        assert believed_context_limit(200_000, "context-1m-2025-08-07") == 1_000_000
        assert believed_context_limit(200_000, "oauth-2025-04-20") == 200_000

    def test_believed_limit_never_lowered(self):
        assert believed_context_limit(2_000_000, "context-1m-2025-08-07") == 2_000_000

    def test_effective_limit_optimistic_until_learned(self):
        beta = "context-1m-2025-08-07"
        assert effective_context_limit("claude-sonnet-4-5", 200_000, beta) == 1_000_000

    def test_learned_limit_caps_effective(self):
        beta = "context-1m-2025-08-07"
        learned = note_prompt_too_long(
            "claude-sonnet-4-5",
            beta,
            '{"error": {"message": "prompt is too long: 213021 tokens > 200000 maximum"}}',
        )
        assert learned == 200_000
        assert effective_context_limit("claude-sonnet-4-5", 200_000, beta) == 200_000

    def test_learned_limit_keyed_by_beta_presence(self):
        # A limit learned WITHOUT the 1m beta must not clamp sessions that
        # send it (their account may have real 1M access).
        note_prompt_too_long(
            "claude-sonnet-4-5",
            None,
            "prompt is too long: 201000 tokens > 200000 maximum",
        )
        assert (
            effective_context_limit("claude-sonnet-4-5", 200_000, "context-1m-2025-08-07")
            == 1_000_000
        )

    def test_note_prompt_too_long_ignores_other_errors(self):
        assert note_prompt_too_long("m", None, "rate limit exceeded") is None
        assert note_prompt_too_long("m", None, b"") is None

    def test_note_prompt_too_long_accepts_bytes(self):
        assert (
            note_prompt_too_long("m", None, b"prompt is too long: 250000 tokens > 200000 maximum")
            == 200_000
        )

    def test_enabled_by_default_and_kill_switch(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_CONTEXT_GUARD", raising=False)
        assert context_guard_enabled()
        monkeypatch.setenv("HEADROOM_CONTEXT_GUARD", "0")
        assert not context_guard_enabled()
        monkeypatch.setenv("HEADROOM_CONTEXT_GUARD", "false")
        assert not context_guard_enabled()


class TestLearnedLimitHygiene:
    """Regressions for what a learned limit is allowed to do."""

    def test_non_string_error_bodies_do_not_raise(self):
        # Upstreams and gateways put null, objects and lists in the message
        # slot. This runs while the client is still waiting for the real error
        # body, so it must return empty-handed rather than raise over it.
        for body in (None, {"message": {"nested": True}}, ["prompt is too long"], 42):
            assert note_prompt_too_long("claude-sonnet-4-5", None, body) is None
        assert effective_context_limit("claude-sonnet-4-5", 200_000, None) == 200_000

    def test_out_of_band_maximum_is_not_learned(self):
        # Same wire shape, different subject: gateways report per-request quota
        # and per-key throttles as "prompt is too long: N > M". A limit that is
        # not a context window must not clamp the budget for the process.
        assert (
            note_prompt_too_long(
                "claude-sonnet-4-5", None, "prompt is too long: 900 tokens > 1 maximum"
            )
            is None
        )
        assert (
            note_prompt_too_long(
                "claude-sonnet-4-5",
                None,
                "prompt is too long: 99000000 tokens > 50000000 maximum",
            )
            is None
        )
        assert effective_context_limit("claude-sonnet-4-5", 200_000, None) == 200_000

    def test_learned_limit_is_scoped_to_one_credential(self):
        # One tenant's capped key on a shared proxy must not clamp another's.
        note_prompt_too_long(
            "claude-sonnet-4-5",
            None,
            "prompt is too long: 213021 tokens > 200000 maximum",
            scope=credential_scope("sk-tenant-a"),
        )
        assert (
            effective_context_limit(
                "claude-sonnet-4-5", 400_000, None, scope=credential_scope("sk-tenant-a")
            )
            == 200_000
        )
        assert (
            effective_context_limit(
                "claude-sonnet-4-5", 400_000, None, scope=credential_scope("sk-tenant-b")
            )
            == 400_000
        )

    def test_credential_scope_never_echoes_the_credential(self):
        scope = credential_scope("sk-ant-secret-value")
        assert "secret" not in scope
        assert scope and scope != credential_scope("sk-ant-other-value")
        assert credential_scope(None) == credential_scope("") == ""

    def test_learned_limit_expires(self, monkeypatch):
        # Entitlements get provisioned and gateways get repointed; a lesson
        # from an hour ago must not hold the budget down forever.
        from headroom.proxy import context_guard

        clock = {"now": 1_000.0}
        monkeypatch.setattr(context_guard.time, "monotonic", lambda: clock["now"])
        note_prompt_too_long(
            "claude-sonnet-4-5", None, "prompt is too long: 213021 tokens > 200000 maximum"
        )
        assert effective_context_limit("claude-sonnet-4-5", 400_000, None) == 200_000
        clock["now"] += context_guard._LEARNED_LIMIT_TTL_SECONDS + 1
        assert effective_context_limit("claude-sonnet-4-5", 400_000, None) == 400_000


class TestNonStreamingNudge:
    """The buffered path raises the same budget, so it owes the same warning."""

    def test_near_limit_body_is_nudged(self):
        payload = {
            "type": "message",
            "usage": {
                "input_tokens": 185_000,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 7,
            },
        }
        assert nudge_response_usage(payload, believed_limit=200_000, effective_limit=200_000)
        assert payload["usage"]["input_tokens"] == int(200_000 * REPORT_FRACTION)

    def test_far_from_limit_body_is_untouched(self):
        payload = {"type": "message", "usage": {"input_tokens": 5_000, "output_tokens": 7}}
        assert not nudge_response_usage(payload, believed_limit=200_000, effective_limit=200_000)
        assert payload["usage"]["input_tokens"] == 5_000

    def test_bodies_without_usage_are_untouched(self):
        for payload in ({"type": "message"}, {"type": "error"}, "not a body", None):
            assert not nudge_response_usage(
                payload, believed_limit=200_000, effective_limit=200_000
            )


class TestStreamUsageGuard:
    def test_below_trigger_passes_through_byte_identical(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        event = _message_start_event(50_000, cache_read=100_000)
        assert guard.feed(event) == event
        # Inert afterwards: later chunks untouched even if they look like events.
        tail = b"event: content_block_delta\ndata: {}\n\n"
        assert guard.feed(tail) == tail

    def test_above_trigger_inflates_to_report_fraction_of_believed(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        # 185k total forwarded = 92.5% of the real 200k window.
        event = _message_start_event(5_000, cache_read=170_000, cache_creation=10_000)
        out = guard.feed(event)
        usage = _parse_usage(out)
        target_total = int(1_000_000 * REPORT_FRACTION)
        assert (
            usage["input_tokens"]
            + usage["cache_read_input_tokens"]
            + usage["cache_creation_input_tokens"]
            == target_total
        )
        # Cache components are never touched — only input_tokens absorbs the nudge.
        assert usage["cache_read_input_tokens"] == 170_000
        assert usage["cache_creation_input_tokens"] == 10_000
        assert usage["output_tokens"] == 1

    def test_same_limits_nudges_gauge_over_compact_threshold(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        event = _message_start_event(185_000)
        usage = _parse_usage(guard.feed(event))
        assert usage["input_tokens"] == int(200_000 * REPORT_FRACTION)

    def test_never_deflates(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        event = _message_start_event(199_000)  # above the 95% report target
        assert guard.feed(event) == event

    def test_split_chunks_across_event_boundary(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        event = _message_start_event(190_000)
        first, second = event[:40], event[40:]
        assert guard.feed(first) == b""  # held back: no complete event yet
        out = guard.feed(second)
        assert _parse_usage(out)["input_tokens"] == int(1_000_000 * REPORT_FRACTION)

    def test_ping_events_pass_through_before_message_start(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        ping = b'event: ping\ndata: {"type": "ping"}\n\n'
        assert guard.feed(ping) == ping
        out = guard.feed(_message_start_event(190_000))
        assert _parse_usage(out)["input_tokens"] == int(1_000_000 * REPORT_FRACTION)

    def test_non_message_start_first_event_disarms(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        error_event = b'event: error\ndata: {"type": "error"}\n\n'
        assert guard.feed(error_event) == error_event
        # Even a later message_start is untouched (protocol says it is first).
        event = _message_start_event(190_000)
        assert guard.feed(event) == event

    def test_malformed_json_passes_through(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        event = b"event: message_start\ndata: {not json\n\n"
        assert guard.feed(event) == event

    def test_buffer_cap_flushes_verbatim(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        blob = b"x" * (300 * 1024)  # no event boundary anywhere
        out = guard.feed(blob)
        assert out == blob
        assert guard.feed(b"more") == b"more"  # inert afterwards

    def test_flush_returns_held_bytes(self):
        guard = StreamUsageGuard(believed_limit=1_000_000, effective_limit=200_000)
        partial = b"event: message_start\ndata: {"
        assert guard.feed(partial) == b""
        assert guard.flush() == partial

    def test_unusable_limits_make_guard_inert(self):
        guard = StreamUsageGuard(believed_limit=0, effective_limit=200_000)
        event = _message_start_event(190_000)
        assert guard.feed(event) == event

    def test_rewritten_event_is_valid_sse(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        out = guard.feed(_message_start_event(185_000))
        assert out.startswith(b"event: message_start\n")
        assert out.endswith(b"\n\n")
        payload = _parse_usage(out)  # parses => data line is intact JSON
        assert payload["input_tokens"] > 185_000

    def test_armed_guard_rewrites_final_message_delta(self):
        # Clients merge the final cumulative-usage message_delta over
        # message_start (verified live with Claude Code 2026-08-12), so an
        # armed guard must rewrite both or the nudge loses the merge.
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        guard.feed(_message_start_event(180_000, cache_creation=5_000))
        content = b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
        assert guard.feed(content) == content  # armed but passthrough
        delta = _message_delta_event(180_000, cache_creation=5_000)
        out = guard.feed(delta)
        usage = json.loads(out.split(b"data: ")[1])["usage"]
        assert usage["input_tokens"] + usage["cache_creation_input_tokens"] == int(
            200_000 * REPORT_FRACTION
        )
        assert usage["output_tokens"] == 42
        # Inert afterwards.
        stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        assert guard.feed(stop) == stop

    def test_output_only_message_delta_untouched(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        guard.feed(_message_start_event(185_000))
        delta = (
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n'
        )
        assert guard.feed(delta) == delta

    def test_disarmed_guard_leaves_message_delta_alone(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        guard.feed(_message_start_event(50_000))  # below trigger -> inert
        delta = _message_delta_event(50_000)
        assert guard.feed(delta) == delta

    def test_message_delta_never_deflates(self):
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        guard.feed(_message_start_event(185_000))
        delta = _message_delta_event(199_000)  # already above the 190k target
        assert guard.feed(delta) == delta


class TestSseFraming:
    """The scan reads the SSE envelope, never the payload text."""

    def _armed_guard(self) -> StreamUsageGuard:
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        guard.feed(_message_start_event(190_000))
        return guard

    def test_assistant_text_naming_an_event_does_not_disarm(self):
        # The model writing about message_delta -- explaining SSE, quoting a
        # log line -- used to end the scan, so the final cumulative usage went
        # out un-nudged and the client's gauge snapped back.
        guard = self._armed_guard()
        chatter = (
            b"event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": "the final message_delta carries cumulative usage",
                    },
                }
            ).encode()
            + b"\n\n"
        )
        assert guard.feed(chatter) == chatter
        nudged = guard.feed(_message_delta_event(190_000))
        assert json.loads(nudged.split(b"data: ", 1)[1])["usage"]["input_tokens"] == int(
            200_000 * REPORT_FRACTION
        )

    def test_keepalive_frames_before_message_start_do_not_disarm(self):
        # An intermediary shim (nginx, a corporate egress proxy) injects SSE
        # comments and retry fields. They are not events, so they must not be
        # read as "a protocol we don't recognize" and switch the guard off.
        guard = StreamUsageGuard(believed_limit=200_000, effective_limit=200_000)
        for keepalive in (b": keepalive\n\n", b"retry: 3000\n\n", b":\n\n"):
            assert guard.feed(keepalive) == keepalive
        nudged = guard.feed(_message_start_event(190_000))
        assert _parse_usage(nudged)["input_tokens"] == int(200_000 * REPORT_FRACTION)


class TestStreamResponseIntegration:
    """The guard wired through _stream_response with a mocked upstream."""

    def _create_mock_proxy(self):
        from unittest.mock import AsyncMock, MagicMock

        import httpx

        from headroom.proxy.server import HeadroomProxy

        proxy = object.__new__(HeadroomProxy)
        proxy.http_client = MagicMock(spec=httpx.AsyncClient)
        proxy.metrics = MagicMock()
        proxy.metrics.record_request = AsyncMock(return_value=None)
        proxy.metrics.record_failed = AsyncMock(return_value=None)
        proxy.cost_tracker = MagicMock()
        proxy.cost_tracker.estimate_cost.return_value = 0.001
        proxy.cost_tracker.record_request.return_value = None
        proxy.stats = {
            "requests_total": 0,
            "requests_optimized": 0,
            "tokens": {"original": 0, "optimized": 0, "saved": 0},
            "cost": {"total_usd": 0, "savings_usd": 0},
            "errors": 0,
            "active_requests": 0,
            "requests_per_model": {},
        }
        proxy.memory_manager = None
        proxy._config = MagicMock()
        proxy._config.memory_enabled = False
        proxy._config.ccr_inject_tool = False
        proxy._config.retry_max_attempts = 3
        proxy._config.retry_base_delay_ms = 0
        proxy._config.retry_max_delay_ms = 0
        proxy.config = proxy._config
        proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
        proxy.memory_handler = None
        proxy.anthropic_provider = MagicMock()
        proxy.anthropic_provider.get_context_limit.return_value = 200_000
        return proxy

    def _mock_upstream(self, sse_bytes: bytes, status_code: int = 200):
        from unittest.mock import AsyncMock

        import httpx

        mock_response = AsyncMock()
        mock_response.headers = httpx.Headers({"content-type": "text/event-stream"})
        mock_response.status_code = status_code

        async def aiter_bytes():
            yield sse_bytes

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        mock_response.aread = AsyncMock(return_value=sse_bytes)
        return mock_response

    async def _run(self, proxy, mock_response, headers):
        from unittest.mock import AsyncMock, MagicMock

        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(return_value=mock_response)
        return await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers=headers,
            body={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-5",
            request_id="test-guard",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

    @pytest.mark.asyncio
    async def test_near_limit_message_start_is_nudged_on_the_wire(self):
        proxy = self._create_mock_proxy()
        # 185k = above the 90% trigger, below the 95% report target — the
        # assertions below can only pass if the rewrites actually happened.
        upstream = self._mock_upstream(
            _message_start_event(185_000)
            + b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
            + _message_delta_event(185_000)
            + b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        result = await self._run(proxy, upstream, {"x-api-key": "sk-test"})
        client_bytes = b"".join([chunk async for chunk in result.body_iterator])
        events = client_bytes.split(b"\n\n")
        usage = _parse_usage(events[0] + b"\n\n")
        assert usage["input_tokens"] == int(200_000 * REPORT_FRACTION)
        # The final cumulative-usage message_delta is nudged too — clients
        # merge it over message_start, so it must agree.
        delta_usage = json.loads(events[2].split(b"data: ")[1])["usage"]
        assert delta_usage["input_tokens"] == int(200_000 * REPORT_FRACTION)
        assert delta_usage["output_tokens"] == 42
        # Everything else is untouched.
        assert (
            b'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n' in client_bytes
        )
        assert b'event: message_stop\ndata: {"type":"message_stop"}\n\n' in client_bytes

    @pytest.mark.asyncio
    async def test_far_from_limit_stream_is_byte_identical(self):
        proxy = self._create_mock_proxy()
        sse = (
            _message_start_event(50_000) + b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        result = await self._run(proxy, self._mock_upstream(sse), {"x-api-key": "sk-test"})
        client_bytes = b"".join([chunk async for chunk in result.body_iterator])
        assert client_bytes == sse

    @pytest.mark.asyncio
    async def test_kill_switch_disables_nudge(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_CONTEXT_GUARD", "0")
        proxy = self._create_mock_proxy()
        sse = _message_start_event(190_000)
        result = await self._run(proxy, self._mock_upstream(sse), {"x-api-key": "sk-test"})
        client_bytes = b"".join([chunk async for chunk in result.body_iterator])
        assert client_bytes == sse

    @pytest.mark.asyncio
    async def test_streaming_400_learns_real_limit(self):
        proxy = self._create_mock_proxy()
        error_body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 213021 tokens > 200000 maximum",
                },
            }
        ).encode()
        upstream = self._mock_upstream(error_body, status_code=400)
        upstream.headers = __import__("httpx").Headers({"content-type": "application/json"})
        result = await self._run(
            proxy, upstream, {"x-api-key": "sk-test", "anthropic-beta": "context-1m-2025-08-07"}
        )
        assert result.status_code == 400
        # Learned against the credential that hit the wall, not process-wide.
        scope = credential_scope("sk-test")
        assert (
            effective_context_limit(
                "claude-sonnet-4-5", 200_000, "context-1m-2025-08-07", scope=scope
            )
            == 200_000
        )
        # A different key on the same proxy keeps its own (optimistic) budget.
        assert (
            effective_context_limit(
                "claude-sonnet-4-5",
                200_000,
                "context-1m-2025-08-07",
                scope=credential_scope("sk-other"),
            )
            == 1_000_000
        )
