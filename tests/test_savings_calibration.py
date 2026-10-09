"""Per-request savings in the provider's units (headroom.proxy.savings_calibration)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy.cost import CostTracker
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_calibration import (
    REMOVED_CONTENT_CORRECTION,
    SOURCE_MODEL_AVERAGE,
    SOURCE_NATIVE,
    SOURCE_REQUEST,
    SOURCE_UNCALIBRATED,
    SavingsCalibrator,
    local_request_overhead,
    removed_content_correction,
    reset_savings_calibrator,
    tokenizer_family,
)
from headroom.tokenizers import is_native_tokenizer


@pytest.fixture(autouse=True)
def _fresh_calibrator():
    reset_savings_calibrator()
    yield
    reset_savings_calibrator()


def _cal(calibrator: SavingsCalibrator, **overrides):  # noqa: ANN003, ANN202
    args = {
        "model": "claude-sonnet-5-5",
        "conversation_key": None,
        "billed_input_tokens": 160_000,
        "local_forwarded_tokens": 100_000,
        "local_covers_request": True,
        "tokens_saved": 10_000,
        "native_tokenizer": False,
    }
    args.update(overrides)
    return calibrator.calibrate(**args)


@pytest.mark.parametrize(
    "model,family",
    [
        ("claude-sonnet-4-6", "claude-4"),
        ("claude-3-5-sonnet-20241022", "claude-4"),
        ("anthropic.claude-sonnet-4-6-v1:0", "claude-4"),
        ("claude-opus-5-5", "claude-5"),
        ("claude-sonnet-5", "claude-5"),
        ("claude-haiku-4-5", "claude-4"),
        ("claude-opus-4-6", "claude-4"),
        ("claude-haiku-5-5", "claude-5"),
        ("claude-fable-5-1", "claude-5"),
        ("us.anthropic.claude-fable-5-1-v1:0", "claude-5"),
        ("gpt-5", None),
    ],
)
def test_tokenizer_family(model, family) -> None:  # noqa: ANN001
    assert tokenizer_family(model) == family


def test_native_tokenizer_is_left_exact() -> None:
    # OpenAI is counted with OpenAI's own tokenizer: nothing to convert.
    assert is_native_tokenizer("gpt-5-mini")
    assert not is_native_tokenizer("claude-sonnet-4-6")
    result = _cal(SavingsCalibrator(), model="gpt-5-mini", native_tokenizer=True)
    assert result.source == SOURCE_NATIVE
    assert result.factor == 1.0
    assert result.tokens_saved == 10_000
    assert result.baseline_input_tokens == 170_000


def test_request_ratio_times_family_correction() -> None:
    result = _cal(SavingsCalibrator())
    expected_factor = 1.6 * REMOVED_CONTENT_CORRECTION["claude-5"]
    assert result.source == SOURCE_REQUEST
    assert result.request_ratio == 1.6
    assert result.factor == pytest.approx(expected_factor, abs=1e-4)
    assert result.tokens_saved == round(10_000 * expected_factor)
    # Baseline is the provider's billed count plus the converted saving.
    assert result.baseline_input_tokens == 160_000 + result.tokens_saved
    assert result.reduction_percent == pytest.approx(
        result.tokens_saved / result.baseline_input_tokens * 100, abs=0.01
    )


def test_unmeasurable_request_uses_model_average_then_uncalibrated() -> None:
    calibrator = SavingsCalibrator()
    # Nothing seen yet for the model and no provider usage: stays local.
    first = _cal(calibrator, billed_input_tokens=0)
    assert first.source == SOURCE_UNCALIBRATED
    assert first.factor == 1.0
    assert first.baseline_estimated is True  # built from the local estimate
    assert first.baseline_input_tokens == 100_000 + 10_000

    _cal(calibrator)  # a measurable request sets the model's ratio (1.6)
    # Images make the local count rough: never measured, average used.
    with_media = _cal(calibrator, local_covers_request=False)
    assert with_media.source == SOURCE_MODEL_AVERAGE
    assert with_media.request_ratio == 1.6


def test_implausible_ratio_is_not_trusted() -> None:
    # A half-counted request (local far below billed) is an artefact, not a tokenizer.
    result = _cal(SavingsCalibrator(), local_forwarded_tokens=20_000)
    assert result.source == SOURCE_UNCALIBRATED


def test_carried_savings_keep_their_value_and_novel_uses_this_request() -> None:
    calibrator = SavingsCalibrator()
    turn1 = _cal(calibrator, conversation_key="c1", tokens_saved=1_000)
    assert turn1.carried_tokens_saved == 0
    assert turn1.novel_tokens_saved == turn1.tokens_saved

    # Next turn: the 1,000 removed earlier is still removed (carried) and
    # 500 more was removed from this turn's new content (novel).
    turn2 = _cal(
        calibrator,
        conversation_key="c1",
        tokens_saved=1_500,
        billed_input_tokens=170_000,
        local_forwarded_tokens=100_000,
    )
    assert turn2.carried_tokens_saved == turn1.tokens_saved
    assert turn2.novel_tokens_saved == round(500 * turn2.factor)
    assert turn2.tokens_saved == turn2.carried_tokens_saved + turn2.novel_tokens_saved


def test_local_request_overhead_counts_system_and_tools_and_flags_media() -> None:
    body = {
        "system": "You are helpful. " * 200,
        "tools": [{"name": "t", "description": "d" * 400, "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    tokens, covers = local_request_overhead("claude-sonnet-4-6", body)
    assert tokens > 500
    assert covers is True
    body["messages"] = [
        {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
        }
    ]
    assert local_request_overhead("claude-sonnet-4-6", body)[1] is False


def test_correction_defaults_to_one_for_unknown_family() -> None:
    assert removed_content_correction("gemini-2.5-pro") == 1.0
    assert removed_content_correction("claude-sonnet-4-6") == REMOVED_CONTENT_CORRECTION["claude-4"]


# ── one recorded Claude turn, end to end ─────────────────────────────────

FIXTURE = Path(__file__).parent / "fixtures" / "anthropic" / "stream_cached_turn.sse"
# The system prompt sent when the fixture was recorded (19 uncached + 8,857
# cache-read tokens billed by Anthropic).
RECORDED_SYSTEM = (
    "You are a terse assistant for a build system. Reference material follows.\n"
    + "\n".join(
        f"Rule {i}: when target t{i} fails with exit code {i % 7}, rerun job j{i} "
        f"with --retry and attach log l{i}.txt."
        for i in range(260)
    )
)


def test_recorded_claude_turn_reports_provider_unit_savings() -> None:
    from headroom.proxy.server import HeadroomProxy

    # Replay the real stream through the proxy's parser and finalizer.
    from tests.test_provider_billed_input import _replay_recorded_stream

    state = _replay_recorded_stream("anthropic", FIXTURE.read_bytes())
    handler = object.__new__(HeadroomProxy)
    handler.config = SimpleNamespace(log_full_messages=False)
    outcomes: list[RequestOutcome] = []

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    handler._record_request_outcome = record
    body = {
        "system": [{"type": "text", "text": RECORDED_SYSTEM}],
        "messages": [
            {"role": "user", "content": "Which job reruns target t17? Answer in five words."}
        ],
    }
    from headroom.tokenizers import get_tokenizer

    local_messages = get_tokenizer("claude-sonnet-4-6").count_messages(body["messages"])
    asyncio.run(
        handler._finalize_stream_response(
            body=body,
            provider="anthropic",
            model="claude-sonnet-4-6",
            request_id="req_cal",
            original_tokens=local_messages + 300,
            optimized_tokens=local_messages,
            tokens_saved=300,
            transforms_applied=[],
            optimization_latency=1.0,
            stream_state=state,
            start_time=0.0,
        )
    )
    (outcome,) = outcomes
    assert outcome.provider_input_tokens == 8_876
    assert outcome.local_counts_full_request is True
    assert outcome.local_forwarded_tokens > local_messages

    cost = CostTracker()
    logged = []
    sink = SimpleNamespace(
        metrics=PrometheusMetrics(cost_tracker=cost, stateless=True),
        cost_tracker=cost,
        logger=SimpleNamespace(log=logged.append),
    )
    asyncio.run(emit_request_outcome(sink, outcome))

    (log,) = logged
    local_full = outcome.local_forwarded_tokens
    ratio = 8_876 / local_full
    assert log.billed_input_tokens == 8_876
    assert log.input_tokens_source == "provider"
    assert log.calibration_source == SOURCE_REQUEST
    assert log.calibration_ratio == pytest.approx(ratio, abs=1e-3)
    assert log.calibration_factor == pytest.approx(
        ratio * REMOVED_CONTENT_CORRECTION["claude-4"], abs=1e-3
    )
    assert log.tokens_saved_provider == round(300 * log.calibration_factor)
    assert log.baseline_input_tokens == 8_876 + log.tokens_saved_provider
    assert log.savings_usd > 0

    m = sink.metrics
    assert m.tokens_saved_provider_total == log.tokens_saved_provider
    assert m.calibration_requests_by_source == {SOURCE_REQUEST: 1}
    assert cost._provider_tokens_saved_by_model == {"claude-sonnet-4-6": log.tokens_saved_provider}


def test_net_saving_can_be_negative_when_headroom_adds_more_than_it_removes() -> None:
    # Headroom added a 120-token tool definition and compressed nothing.
    result = _cal(SavingsCalibrator(), tokens_saved=-120)
    assert result.tokens_saved == round(-120 * result.factor)
    assert result.carried_tokens_saved == 0


def test_client_request_tag_is_internal() -> None:
    from headroom.proxy.savings_attribution import public_tags
    from headroom.proxy.savings_calibration import CLIENT_REQUEST_TOKENS_TAG

    assert CLIENT_REQUEST_TOKENS_TAG not in public_tags(
        {CLIENT_REQUEST_TOKENS_TAG: 5, "route": "x"}
    )


def test_deferred_tools_are_not_counted_as_sent() -> None:
    tool = {"name": "t", "description": "d " * 2_000, "input_schema": {"type": "object"}}
    full, _ = local_request_overhead("claude-sonnet-4-6", {"tools": [tool]})
    deferred, _ = local_request_overhead(
        "claude-sonnet-4-6", {"tools": [{**tool, "defer_loading": True}]}
    )
    assert full > 1_000
    assert deferred == 0


def test_full_request_count_nets_compression_deferral_and_additions() -> None:
    from headroom.proxy.savings_calibration import local_request_tokens

    model = "claude-sonnet-4-6"
    big_tool = {"name": "big", "description": "word " * 3_000, "input_schema": {"type": "object"}}
    client = {
        "system": "sys " * 100,
        "tools": [big_tool],
        "messages": [{"role": "user", "content": "data " * 2_000}],
    }
    forwarded = {
        "system": client["system"],
        # deferred (not billed) + one small tool Headroom added
        "tools": [
            {**big_tool, "defer_loading": True},
            {"name": "headroom_retrieve", "description": "r", "input_schema": {"type": "object"}},
        ],
        "messages": [{"role": "user", "content": "data " * 500}],
    }
    c, _ = local_request_tokens(model, client)
    f, _ = local_request_tokens(model, forwarded)
    # Messages shrank ~1,500 and the big tool (~3,000) is deferred; the small
    # added tool is subtracted, so the net is a bit under 4,500.
    assert 4_000 < c - f < 4_600
    # Per-message caching gives the same answer on a repeat.
    assert local_request_tokens(model, client)[0] == c


def test_removed_tool_definitions_use_the_tool_rate() -> None:
    from headroom.proxy.savings_calibration import TOOL_DEFINITION_RATE

    calibrator = SavingsCalibrator()
    # 10,000 net: 4,000 message compression + 6,000 deferred tool definitions.
    first = _cal(
        calibrator, conversation_key="t", tokens_saved=10_000, tool_definition_tokens_saved=6_000
    )
    tools = round(6_000 * TOOL_DEFINITION_RATE["claude-5"])
    assert first.tool_tokens_saved == tools
    assert first.tokens_saved == round(4_000 * first.factor) + tools
    assert first.carried_tokens_saved == 0  # first request: all new input

    # Next request: same deferral, nothing new compressed -> all carried.
    second = _cal(
        calibrator, conversation_key="t", tokens_saved=10_000, tool_definition_tokens_saved=6_000
    )
    assert second.novel_tokens_saved == 0
    assert second.carried_tokens_saved == second.tokens_saved
    assert second.tool_tokens_saved == tools


def test_native_tokenizer_leaves_tool_savings_exact() -> None:
    result = _cal(
        SavingsCalibrator(),
        model="gpt-5-mini",
        native_tokenizer=True,
        tokens_saved=10_000,
        tool_definition_tokens_saved=6_000,
    )
    assert result.tokens_saved == 10_000
    assert result.tool_tokens_saved == 6_000


# ── Devin review on #4055 ────────────────────────────────────────────────


def test_tool_compaction_uses_the_tool_rate_not_the_message_rate() -> None:
    from headroom.proxy.savings_calibration import TOOL_DEFINITION_RATE

    # 1,000 local tokens of tool schema compacted, nothing deferred, no
    # message compression: all of it is tool definitions.
    result = _cal(SavingsCalibrator(), tokens_saved=1_000, tool_definition_tokens_saved=1_000)
    assert result.tokens_saved == round(1_000 * TOOL_DEFINITION_RATE["claude-5"])
    assert result.tool_tokens_saved == result.tokens_saved


def test_tool_deferral_starting_mid_conversation_is_novel() -> None:
    calibrator = SavingsCalibrator()
    # Turn 1: tools forwarded normally, nothing saved.
    _cal(calibrator, conversation_key="c", tokens_saved=0, tool_definition_tokens_saved=0)
    # Turn 2: deferral starts. New this turn, so novel (priced as new input).
    turn2 = _cal(
        calibrator, conversation_key="c", tokens_saved=2_000, tool_definition_tokens_saved=2_000
    )
    assert turn2.carried_tokens_saved == 0
    assert turn2.novel_tokens_saved == turn2.tokens_saved > 0
    # Turn 3: the same deferral is now carried.
    turn3 = _cal(
        calibrator, conversation_key="c", tokens_saved=2_000, tool_definition_tokens_saved=2_000
    )
    assert turn3.novel_tokens_saved == 0
    assert turn3.carried_tokens_saved == turn2.tokens_saved
    # Turn 4: one more tool deferred: only the growth is novel.
    turn4 = _cal(
        calibrator, conversation_key="c", tokens_saved=2_500, tool_definition_tokens_saved=2_500
    )
    assert turn4.carried_tokens_saved == turn2.tokens_saved
    assert turn4.novel_tokens_saved == turn4.tokens_saved - turn2.tokens_saved > 0


def test_negative_savings_reach_the_licence_report() -> None:
    from headroom.telemetry.reporter import UsageReporter

    cost = CostTracker()
    cost.record_tokens(
        "claude-sonnet-4-6", 0, 1_000, provider_tokens_saved=500, provider_novel_tokens_saved=500
    )
    cost.record_tokens(
        "claude-sonnet-4-6", 0, 1_000, provider_tokens_saved=-120, provider_novel_tokens_saved=-120
    )
    reporter = object.__new__(UsageReporter)
    deltas = reporter._provider_counter_deltas(cost)
    assert deltas["tokens_saved_provider"] == 500
    assert deltas["tokens_added_provider"] == 120
    assert deltas["tokens_added_novel_provider"] == 120


def test_missing_provider_usage_gives_an_estimated_baseline() -> None:
    result = _cal(
        SavingsCalibrator(), billed_input_tokens=0, local_forwarded_tokens=1_000, tokens_saved=200
    )
    assert result.source == SOURCE_UNCALIBRATED
    assert result.baseline_estimated is True
    assert result.baseline_input_tokens == 1_200
    assert result.reduction_percent == pytest.approx(200 / 1_200 * 100, abs=0.01)
    measured = _cal(SavingsCalibrator())
    assert measured.baseline_estimated is False


def test_tool_count_follows_deferral_and_additions() -> None:
    from headroom.proxy.savings_calibration import local_request_counts

    model = "claude-sonnet-4-6"
    big = {"name": "big", "description": "word " * 2_000, "input_schema": {"type": "object"}}
    small = {"name": "headroom_retrieve", "description": "r", "input_schema": {"type": "object"}}
    msgs = [{"role": "user", "content": "hi"}]
    _, client_tools, _ = local_request_counts(model, {"tools": [big], "messages": msgs})
    _, fwd_tools, _ = local_request_counts(
        model, {"tools": [{**big, "defer_loading": True}, small], "messages": msgs}
    )
    assert client_tools > 1_900
    assert 0 < fwd_tools < 100  # only the added tool is billed


def test_extension_replaced_messages_are_the_starting_point(monkeypatch) -> None:  # noqa: ANN001
    """An INPUT_RECEIVED extension that rewrites the messages is not Headroom
    compression: the request's net saving must start from the rewritten
    messages, exactly like ``original_tokens``."""
    import dataclasses

    import httpx
    from fastapi.testclient import TestClient

    from headroom.config import TransformResult
    from headroom.pipeline import PipelineStage
    from headroom.proxy.savings_calibration import SavingsCalibrator as _Cal
    from headroom.proxy.server import ProxyConfig, create_app

    short = [{"role": "user", "content": "hello"}]

    class _Rewriter:
        def on_pipeline_event(self, event):  # noqa: ANN001, ANN202
            if event.stage is PipelineStage.INPUT_RECEIVED:
                return dataclasses.replace(event, messages=list(short))
            return None

    seen: list[dict] = []
    real = _Cal.calibrate

    def spy(self, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.append(kwargs)
        return real(self, **kwargs)

    monkeypatch.setattr(_Cal, "calibrate", spy)
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        pipeline_extensions=[_Rewriter()],
        discover_pipeline_extensions=False,
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy.anthropic_pipeline = SimpleNamespace(
            apply=lambda messages, model, **kwargs: TransformResult(
                messages=messages, tokens_before=5, tokens_after=5, transforms_applied=[]
            )
        )

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN202
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            )

        proxy._retry_request = _fake_retry
        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "long client message " * 200}],
            },
        )
    assert response.status_code == 200
    assert seen, "calibration did not run"
    assert seen[-1]["tokens_saved"] == 0


def test_estimated_baseline_stays_positive_when_headroom_adds_a_tool() -> None:
    # No usage, no prior ratio: a 1,000-local-token tool Headroom added to a
    # 100-token client request. The baseline is the client's own request.
    result = _cal(
        SavingsCalibrator(),
        billed_input_tokens=0,
        local_forwarded_tokens=1_100,
        local_forwarded_tool_tokens=1_000,
        tokens_saved=-1_000,
        tool_definition_tokens_saved=-1_000,
    )
    assert result.tokens_saved < 0
    assert result.baseline_estimated is True
    assert result.baseline_input_tokens == 100


def test_licence_report_provider_savings_are_net() -> None:
    from tests.test_provider_billed_input import _reporter_payload

    cost = CostTracker()
    cost.record_tokens(
        "claude-sonnet-4-6", 0, 1_000, provider_tokens_saved=500, provider_novel_tokens_saved=500
    )
    cost.record_tokens(
        "claude-sonnet-4-6", 0, 1_000, provider_tokens_saved=-120, provider_novel_tokens_saved=-120
    )
    payload = _reporter_payload(cost)
    # Net, the same meaning as /stats saved_provider; gross and added audit it.
    assert payload["tokens_saved_provider"] == 380
    assert payload["tokens_saved_provider_gross"] == 500
    assert payload["tokens_added_provider"] == 120
    assert payload["tokens_saved_novel_provider"] == 380
    assert payload["tokens_saved_novel_provider_gross"] == 500
    assert payload["tokens_added_novel_provider"] == 120


def _stats_after(*outcomes: RequestOutcome) -> dict:
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.savings_calibration import reset_savings_calibrator
    from headroom.proxy.server import ProxyConfig, create_app

    reset_savings_calibrator()
    app = create_app(
        ProxyConfig(
            optimize=False, cache_enabled=False, rate_limit_enabled=False, log_requests=False
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    for outcome in outcomes:
        asyncio.run(emit_request_outcome(app.state.proxy, outcome))
    with TestClient(app) as client:
        resp = client.get("/stats")
    reset_savings_calibrator()
    assert resp.status_code == 200, resp.text[:300]
    return resp.json()["tokens"]


def _mixed_outcome(**overrides: object) -> RequestOutcome:
    from headroom.proxy.savings_calibration import (
        CLIENT_REQUEST_TOKENS_TAG,
        CLIENT_TOOL_TOKENS_TAG,
    )

    fields: dict = {
        "request_id": "r",
        "provider": "openai",
        "model": "gpt-4o",
        "original_tokens": 1_000,
        "optimized_tokens": 1_000,
        "output_tokens": 5,
        "tokens_saved": 0,
        "attempted_input_tokens": 1_000,
        "local_forwarded_tokens": 1_000,
        "local_counts_full_request": True,
        "tags": {CLIENT_REQUEST_TOKENS_TAG: 1_000, CLIENT_TOOL_TOKENS_TAG: 0},
    }
    fields.update(overrides)
    return RequestOutcome(**fields)  # type: ignore[arg-type]


def test_stats_survive_negative_savings_on_estimated_input_requests() -> None:
    """Reviewer repro: one provider-reported request saving nothing, then an
    estimated-input request where Headroom added 1,000 tool tokens. The old
    ratio mixed populations (all savings over reported input only) and hit a
    zero denominator in GET /stats."""
    from headroom.proxy.savings_calibration import CLIENT_REQUEST_TOKENS_TAG, CLIENT_TOOL_TOKENS_TAG

    reported = _mixed_outcome(request_id="a", provider_input_tokens=1_000)
    estimated = _mixed_outcome(
        request_id="b",
        optimized_tokens=2_000,
        attempted_input_tokens=2_000,
        local_forwarded_tokens=2_000,
        local_forwarded_tool_tokens=1_000,
        tags={CLIENT_REQUEST_TOKENS_TAG: 1_000, CLIENT_TOOL_TOKENS_TAG: 0},
    )
    tokens = _stats_after(reported, estimated)
    assert tokens["input_provider_reported"] == 1_000
    assert tokens["input_estimated"] > 0
    assert tokens["saved_provider"] < 0  # the estimated request's addition
    # The percentage covers the reported request only: it saved nothing.
    assert tokens["saved_provider_reported"] == 0
    assert tokens["saved_provider_percent"] == 0


def test_saved_percent_pairs_reported_savings_with_reported_input() -> None:
    from headroom.proxy.savings_calibration import CLIENT_REQUEST_TOKENS_TAG, CLIENT_TOOL_TOKENS_TAG

    # Reported: client 1,500 -> forwarded 1,000 (500 saved, native tokenizer).
    reported = _mixed_outcome(
        request_id="a",
        provider_input_tokens=1_000,
        tags={CLIENT_REQUEST_TOKENS_TAG: 1_500, CLIENT_TOOL_TOKENS_TAG: 0},
    )
    # Estimated-input request that saved a lot must not move the percentage.
    estimated = _mixed_outcome(
        request_id="b",
        tags={CLIENT_REQUEST_TOKENS_TAG: 9_000, CLIENT_TOOL_TOKENS_TAG: 0},
    )
    tokens = _stats_after(reported, estimated)
    assert tokens["saved_provider_reported"] == 500
    assert tokens["saved_provider_percent"] == round(500 / 1_500 * 100, 2)


def test_provider_saved_percent_handles_a_nonpositive_baseline() -> None:
    from headroom.proxy.server import _provider_saved_percent

    assert _provider_saved_percent(-1_000, 1_000) == 0.0
    assert _provider_saved_percent(-1_500, 1_000) == 0.0
    assert _provider_saved_percent(0, 0) == 0.0
    assert _provider_saved_percent(-100, 1_000) == round(-100 / 900 * 100, 2)
