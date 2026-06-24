from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

from headroom.proxy.handlers import openai as openai_handler
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.transforms.compression_units import UnitCompressionResult
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
)


class TokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


def _handler_with_router(router: ContentRouter) -> OpenAIHandlerMixin:
    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = SimpleNamespace(
        get_token_counter=lambda _model: TokenCounter(),
    )
    return handler


def test_openai_responses_unit_parallelism_env_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", raising=False)
    assert openai_handler._openai_responses_unit_parallelism() == 4

    monkeypatch.setenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", "bad")
    assert openai_handler._openai_responses_unit_parallelism() == 4

    monkeypatch.setenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", "0")
    assert openai_handler._openai_responses_unit_parallelism() == 1

    monkeypatch.setenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", "999")
    assert openai_handler._openai_responses_unit_parallelism() == 16


def test_openai_responses_cached_unit_handles_results_without_router_result():
    result = UnitCompressionResult(
        original="original",
        compressed="compressed",
        modified=True,
        tokens_before=2,
        tokens_after=1,
        tokens_saved=1,
        transforms_applied=[],
        strategy="none",
        router_result=None,
    )

    assert openai_handler._openai_responses_result_with_cache_hit(result) is result


def test_openai_responses_unit_cache_evicts_oldest_entry(monkeypatch):
    monkeypatch.setattr(openai_handler, "_OPENAI_RESPONSES_UNIT_CACHE_MAX_ENTRIES", 1)
    handler = OpenAIHandlerMixin()
    first = UnitCompressionResult(
        original="first",
        compressed="first compressed",
        modified=True,
        tokens_before=2,
        tokens_after=1,
        tokens_saved=1,
        transforms_applied=[],
        strategy="none",
        router_result=None,
    )
    second = UnitCompressionResult(
        original="second",
        compressed="second compressed",
        modified=True,
        tokens_before=2,
        tokens_after=1,
        tokens_saved=1,
        transforms_applied=[],
        strategy="none",
        router_result=None,
    )

    handler._store_openai_responses_cached_unit("first", first)
    handler._store_openai_responses_cached_unit("second", second)

    assert handler._get_openai_responses_cached_unit("first") is None
    assert handler._get_openai_responses_cached_unit("second") is second


def test_openai_responses_adapter_compresses_only_live_text_slots():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="kept words",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "reasoning", "encrypted_content": long_text},
            {"type": "function_call", "arguments": long_text},
            {"type": "local_shell_call_output", "call_id": "c1", "output": long_text},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": long_text}],
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["encrypted_content"] == long_text
    assert new_payload["input"][1]["arguments"] == long_text
    assert new_payload["input"][2]["output"] == "kept words"
    assert new_payload["input"][3]["content"][0]["text"] == long_text
    assert any(t.startswith("router:openai:responses:") for t in transforms)
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_compresses_custom_tool_call_output():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="custom output summary",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": long_text,
            }
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["output"] == "custom output summary"
    assert "router:openai:responses:custom_tool_call_output:kompress" in transforms
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_reuses_exact_tool_output_cache():
    router = ContentRouter()
    calls = {"count": 0}

    def compress(self, content: str, **_kwargs):
        calls["count"] += 1
        return RouterCompressionResult(
            compressed="cached output summary",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))

    payload_one = {
        "model": "gpt-5",
        "input": [
            {"type": "local_shell_call_output", "call_id": "c1", "output": long_text},
        ],
    }
    payload_two = {
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "changed envelope"},
            {"type": "local_shell_call_output", "call_id": "c2", "output": long_text},
        ],
    }

    new_payload_one, modified_one, saved_one, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload_one,
            model="gpt-5",
            request_id="req_cache_one",
        )
    )
    new_payload_two, modified_two, saved_two, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload_two,
            model="gpt-5",
            request_id="req_cache_two",
        )
    )

    assert calls["count"] == 1
    assert modified_one is True
    assert modified_two is True
    assert saved_one > 0
    assert saved_two == saved_one
    assert new_payload_one["input"][0]["output"] == "cached output summary"
    assert new_payload_two["input"][1]["output"] == "cached output summary"


def test_openai_responses_adapter_reuses_identical_tool_output_in_same_request():
    router = ContentRouter()
    calls = {"count": 0}

    def compress(self, content: str, **_kwargs):
        calls["count"] += 1
        return RouterCompressionResult(
            compressed="same request cached summary",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "function_call_output", "call_id": "c1", "output": long_text},
            {"type": "function_call_output", "call_id": "c2", "output": long_text},
        ],
    }

    new_payload, modified, saved, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_same_request_cache",
        )
    )

    assert calls["count"] == 1
    assert modified is True
    assert saved > 0
    assert [item["output"] for item in new_payload["input"]] == [
        "same request cached summary",
        "same request cached summary",
    ]


def test_openai_responses_adapter_parallelizes_cache_misses_preserving_order(monkeypatch):
    monkeypatch.setenv("HEADROOM_TOOL_OUTPUT_COMPRESSION_PARALLELISM", "4")
    router = ContentRouter()
    lock = threading.Lock()
    release = threading.Event()
    active = {"count": 0, "max": 0}

    def compress(self, content: str, **_kwargs):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            if active["count"] >= 2:
                release.set()
        release.wait(0.05)
        try:
            marker = content.rsplit(" marker", 1)[1]
            return RouterCompressionResult(
                compressed=f"summary marker{marker}",
                original=content,
                strategy_used=CompressionStrategy.KOMPRESS,
            )
        finally:
            with lock:
                active["count"] -= 1

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)

    def long_text(index: int) -> str:
        return " ".join(f"word{index}_{j}" for j in range(180)) + f" marker{index}"

    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": f"c{i}",
                "output": long_text(i),
            }
            for i in range(4)
        ],
    }

    new_payload, modified, saved, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_parallel",
        )
    )

    assert active["max"] >= 2
    assert modified is True
    assert saved > 0
    assert [item["output"] for item in new_payload["input"]] == [
        "summary marker0",
        "summary marker1",
        "summary marker2",
        "summary marker3",
    ]


def test_openai_responses_adapter_accepts_empty_input_list():
    router = ContentRouter()
    handler = _handler_with_router(router)
    payload = {"model": "gpt-5", "input": [], "tools": []}

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert new_payload == payload
    assert modified is False
    assert saved == 0
    assert transforms == []
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_preserves_headroom_retrieve_outputs():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="compressed retrieve output",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    retrieved = " ".join(f"retrieved{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_retrieve",
                "name": "mcp__headroom__headroom_retrieve",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve",
                "output": retrieved,
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_preserves_excluded_tool_outputs():
    """Regression for #940: outputs for HEADROOM_EXCLUDE_TOOLS tools stay raw.

    The Responses path carries the tool name on the ``function_call`` item and
    the originating ``call_id`` on the matching ``function_call_output``; the
    adapter must correlate them and skip compression for excluded tools.
    """
    router = ContentRouter()
    router.config.exclude_tools = {"serena.find_symbol", "find_symbol"}

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="should not be used",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    output = " ".join(f"sym{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "serena.find_symbol",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": output,
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_excludes_tool_case_insensitively_with_debug(monkeypatch):
    """Excluded match is case-insensitive, and the debug path stays exercised.

    The configured name is lowercase only; the call advertises a mixed-case
    name, so the protection must hit via the lowercased fallback. Debug logging
    is enabled so the protected-extraction debug record is also covered.
    """
    monkeypatch.setattr(openai_handler, "_log_codex_compression_debug", lambda *_a, **_k: None)
    router = ContentRouter()
    router.config.exclude_tools = {"serena.find_symbol"}

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="should not be used",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    output = " ".join(f"sym{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "Serena.Find_Symbol",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": output,
            },
        ],
    }

    new_payload, modified, saved, *_ = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert new_payload == payload


def test_openai_responses_adapter_compresses_non_excluded_tool_outputs():
    """Only excluded tools are protected; other tool outputs still compress."""
    router = ContentRouter()
    router.config.exclude_tools = {"serena.find_symbol"}

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="compressed tool output",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    output = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "some.other_tool",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": output,
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][1]["output"] == "compressed tool output"
    assert "router:openai:responses:function_call_output:kompress" in transforms
    assert units_by_category == {"applied": 1}


def test_openai_responses_adapter_keeps_small_and_opaque_items():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="short",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "local_shell_call_output", "call_id": "c1", "output": "too small"},
            {"type": "compaction", "encrypted_content": " ".join(["secret"] * 200)},
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {"size_floor": 1}
    assert strategy_chain == []


def test_openai_responses_payload_routes_through_content_router_without_rust(
    monkeypatch,
):
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="compressed fallback",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)

    import headroom._core as core

    def rust_must_not_run(*_args, **_kwargs):
        raise AssertionError("Responses payload compression should route through ContentRouter")

    monkeypatch.setattr(core, "compress_openai_responses_live_zone", rust_must_not_run)

    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "c1",
                "output": " ".join(f"word{i}" for i in range(180)),
            }
        ],
    }

    new_payload, modified, saved, transforms, reason, _, _, _ = (
        handler._compress_openai_responses_payload(
            payload,
            model="gpt-5",
            request_id="req_router",
        )
    )

    assert modified is True
    assert saved > 0
    assert reason is None
    assert new_payload["input"][0]["output"] == "compressed fallback"
    assert any(t.startswith("router:openai:responses:") for t in transforms)
