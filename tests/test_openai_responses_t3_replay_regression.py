from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from types import MethodType, SimpleNamespace

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
)


@dataclass(frozen=True)
class T3FailureCase:
    provider_log: str
    turn_id: str
    request_bytes: int
    unit_count: int


# T3 provider logs keep the Headroom 413 metadata, not the raw /v1/responses
# body. These cases recreate the failing byte scale and Responses item shape.
T3_FAILED_CASES = (
    T3FailureCase(
        provider_log="2b38b84f-b6b0-4d92-8ff0-42f83b59dd70.log",
        turn_id="019e8c3f-91d9-73b3-a6f8-4e6ae312f91b",
        request_bytes=674_436,
        unit_count=8,
    ),
    T3FailureCase(
        provider_log="cc084653-feba-4241-a8fd-6655c0dfa799.log",
        turn_id="019e8bdd-ffb3-7f31-9182-51b2bdb96f52",
        request_bytes=1_288_876,
        unit_count=12,
    ),
)


class TokenCounter:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)


def _handler_with_router(router: ContentRouter) -> OpenAIHandlerMixin:
    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = SimpleNamespace(
        get_token_counter=lambda _model: TokenCounter(),
    )
    return handler


def _tool_output(case: T3FailureCase, index: int, target_bytes: int) -> str:
    line = (
        f"{case.turn_id} {case.provider_log} "
        f"tool={index} path=/tmp/t3-live-output-{index}.txt status=ok "
        "alpha beta gamma delta epsilon zeta eta theta iota kappa\n"
    )
    return (line * ((target_bytes // len(line)) + 1))[:target_bytes]


def _payload_for_case(case: T3FailureCase) -> dict:
    envelope_budget = 2_500
    per_unit_bytes = max(2_048, (case.request_bytes - envelope_budget) // case.unit_count)
    return {
        "model": "gpt-5.4-mini",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "continue after tool output",
            },
            {
                "type": "function_call",
                "call_id": "call-shell",
                "name": "shell",
                "arguments": "{}",
            },
            *[
                {
                    "type": "function_call_output",
                    "call_id": f"call-shell-{index}",
                    "output": _tool_output(case, index, per_unit_bytes),
                }
                for index in range(case.unit_count)
            ],
        ],
    }


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))


def test_t3_failed_size_responses_payload_parallelizes_uncached_tool_outputs(monkeypatch):
    monkeypatch.setenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", "4")
    case = T3_FAILED_CASES[0]
    router = ContentRouter()
    lock = threading.Lock()
    release = threading.Event()
    active = {"count": 0, "max": 0, "calls": 0}

    def compress(self, content: str, **_kwargs):
        with lock:
            active["count"] += 1
            active["calls"] += 1
            active["max"] = max(active["max"], active["count"])
            if active["count"] >= 2:
                release.set()
        release.wait(0.05)
        try:
            marker = content.split(" tool=", 1)[1].split(" ", 1)[0]
            return RouterCompressionResult(
                compressed=f"summary for tool={marker}",
                original=content,
                strategy_used=CompressionStrategy.KOMPRESS,
            )
        finally:
            with lock:
                active["count"] -= 1

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    payload = _payload_for_case(case)

    new_payload, modified, saved, transforms, units_by_category, _strategy_chain, attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5.4-mini",
            request_id=f"t3_replay_{case.turn_id}",
        )
    )

    assert _json_bytes(payload) >= case.request_bytes * 0.95
    assert attempted > 0
    assert modified is True
    assert saved > 0
    assert active["calls"] == case.unit_count
    assert active["max"] >= 2
    assert units_by_category == {"applied": case.unit_count}
    assert "router:openai:responses:function_call_output:kompress" in transforms
    outputs = [
        item["output"]
        for item in new_payload["input"]
        if item.get("type") == "function_call_output"
    ]
    assert outputs == [f"summary for tool={index}" for index in range(case.unit_count)]


def test_t3_failed_size_exact_tool_output_cache_survives_history_changes():
    case = T3_FAILED_CASES[1]
    router = ContentRouter()
    calls = {"count": 0}

    def compress(self, content: str, **_kwargs):
        calls["count"] += 1
        marker = content.split(" tool=", 1)[1].split(" ", 1)[0]
        return RouterCompressionResult(
            compressed=f"cached summary for tool={marker}",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    first_payload = _payload_for_case(case)
    second_payload = {
        "model": "gpt-5.4-mini",
        "input": [
            # Simulate a harness that changed/trimmed the ancient envelope.
            {"type": "message", "role": "user", "content": "history compacted"},
            *first_payload["input"][2:],
            {
                "type": "function_call_output",
                "call_id": "call-shell-new",
                "output": _tool_output(case, case.unit_count, 32_000),
            },
        ],
    }

    first_new_payload, first_modified, first_saved, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            first_payload,
            model="gpt-5.4-mini",
            request_id=f"t3_replay_cache_first_{case.turn_id}",
        )
    )
    second_new_payload, second_modified, second_saved, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            second_payload,
            model="gpt-5.4-mini",
            request_id=f"t3_replay_cache_second_{case.turn_id}",
        )
    )

    assert first_modified is True
    assert second_modified is True
    assert first_saved > 0
    assert second_saved > 0
    assert calls["count"] == case.unit_count + 1
    assert [
        item["output"]
        for item in first_new_payload["input"]
        if item.get("type") == "function_call_output"
    ] == [f"cached summary for tool={index}" for index in range(case.unit_count)]
    assert [
        item["output"]
        for item in second_new_payload["input"]
        if item.get("type") == "function_call_output"
    ] == [
        *[f"cached summary for tool={index}" for index in range(case.unit_count)],
        f"cached summary for tool={case.unit_count}",
    ]
