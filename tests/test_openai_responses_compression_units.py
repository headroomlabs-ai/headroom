from __future__ import annotations

from types import MethodType, SimpleNamespace

import anyio

from headroom.cache.compression_store import get_compression_store, reset_compression_store
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


def test_openai_responses_preflight_skips_executor_without_live_output():
    router = ContentRouter()
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "tools": [
            {
                "type": "function",
                "name": "sample",
                "description": "large schema only",
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"field_{idx}": {"type": "string", "description": "x" * 100}
                        for idx in range(100)
                    },
                },
            }
        ],
        "input": [{"type": "message", "role": "user", "content": "hello"}],
    }

    async def must_not_submit(*_args, **_kwargs):
        raise AssertionError("schema-only Responses payload should not enter executor")

    handler._run_compression_in_executor = must_not_submit

    async def run():
        return await handler._compress_openai_responses_payload_in_executor(
            payload,
            model="gpt-5",
            request_id="req_schema_only",
        )

    new_payload, modified, saved, transforms, reason, before, after, attempted, timing = anyio.run(
        run
    )

    assert new_payload == payload
    assert modified is False
    assert saved == 0
    assert transforms == []
    assert reason == "no_compressible_live_text"
    assert before == after
    assert attempted == 0
    assert "compression_preflight_live_text_scan" in timing


def test_openai_responses_preflight_skips_executor_for_below_floor_output():
    router = ContentRouter()
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-small",
                "output": "small output below the compression floor",
            }
        ],
    }

    async def must_not_submit(*_args, **_kwargs):
        raise AssertionError("below-floor Responses output should not enter executor")

    handler._run_compression_in_executor = must_not_submit

    async def run():
        return await handler._compress_openai_responses_payload_in_executor(
            payload,
            model="gpt-5",
            request_id="req_below_floor",
        )

    _new_payload, modified, saved, transforms, reason, _before, _after, attempted, _timing = (
        anyio.run(run)
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert reason == "no_compressible_live_text"
    assert attempted == 0


def test_openai_responses_preflight_compresses_oversized_output_without_executor(
    monkeypatch,
):
    reset_compression_store()
    router = ContentRouter()

    def router_must_not_run(self, content: str, **_kwargs):
        raise AssertionError("oversized output should use deterministic CCR inline")

    router.compress = MethodType(router_must_not_run, router)
    handler = _handler_with_router(router)
    monkeypatch.setattr(handler, "OPENAI_RESPONSES_ROUTER_MAX_BYTES", 128)
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-large-inline",
                "output": " ".join(f"row-{idx}" for idx in range(400)),
            }
        ],
    }

    async def must_not_submit(*_args, **_kwargs):
        raise AssertionError("oversized-only Responses payload should not enter executor")

    handler._run_compression_in_executor = must_not_submit

    async def run():
        return await handler._compress_openai_responses_payload_in_executor(
            payload,
            model="gpt-5",
            request_id="req_large_inline",
        )

    new_payload, modified, saved, transforms, reason, before, after, attempted, timing = anyio.run(
        run
    )

    compressed_output = new_payload["input"][0]["output"]

    assert modified is True
    assert saved > 0
    assert attempted == TokenCounter().count_text(payload["input"][0]["output"])
    assert reason is None
    assert before > after
    assert "Retrieve more: hash=" in compressed_output
    assert "openai:responses:large_text_ccr" in transforms
    assert "compression_inline_live_text" in timing


def test_openai_responses_preflight_does_not_call_stuck_router(monkeypatch):
    reset_compression_store()
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        raise AssertionError("Responses live output should not enter router")

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-stuck-router-proof",
                "output": " ".join(f"large-line-{idx}" for idx in range(1200)),
            },
        ],
    }

    async def must_not_submit(*_args, **_kwargs):
        raise AssertionError("Responses live output should not enter proxy executor")

    handler._run_compression_in_executor = must_not_submit

    async def run():
        with anyio.fail_after(1):
            return await handler._compress_openai_responses_payload_in_executor(
                payload,
                model="gpt-5",
                request_id="req_stuck_router_proof",
            )

    new_payload, modified, saved, transforms, reason, before, after, attempted, timing = anyio.run(
        run
    )

    compressed_output = new_payload["input"][0]["output"]

    assert modified is True
    assert saved > 0
    assert attempted == TokenCounter().count_text(payload["input"][0]["output"])
    assert reason is None
    assert before > after
    assert "Retrieve more: hash=" in compressed_output
    assert "openai:responses:large_text_ccr" in transforms
    assert "compression_inline_live_text" in timing


def test_openai_responses_preflight_compresses_medium_output_without_router():
    reset_compression_store()
    router = ContentRouter()

    def router_must_not_run(self, content: str, **_kwargs):
        raise AssertionError("medium Responses live output should not enter router")

    router.compress = MethodType(router_must_not_run, router)
    handler = _handler_with_router(router)
    original_output = " ".join(f"medium-{idx}" for idx in range(70))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-medium",
                "output": original_output,
            }
        ],
    }

    async def must_not_submit(*_args, **_kwargs):
        raise AssertionError("Responses live output should not enter proxy executor")

    handler._run_compression_in_executor = must_not_submit

    async def run():
        with anyio.fail_after(1):
            return await handler._compress_openai_responses_payload_in_executor(
                payload,
                model="gpt-5",
                request_id="req_medium_adaptive_ccr",
            )

    new_payload, modified, saved, transforms, reason, before, after, attempted, timing = anyio.run(
        run
    )

    compressed_output = new_payload["input"][0]["output"]
    marker = "Retrieve more: hash="
    hash_key = compressed_output.split(marker, 1)[1].split("]", 1)[0]
    entry = get_compression_store().retrieve(hash_key)

    assert modified is True
    assert saved > 0
    assert "openai:responses:large_text_ccr" in transforms
    assert reason is None
    assert before > after
    assert attempted == TokenCounter().count_text(original_output)
    assert marker in compressed_output
    assert len(compressed_output) < len(original_output)
    assert entry is not None
    assert entry.original_content == original_output
    assert entry.tool_call_id == "call-medium"
    assert "compression_inline_live_text" in timing


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


def test_openai_responses_adapter_compresses_cache_misses_preserving_order():
    router = ContentRouter()
    active = {"count": 0, "max": 0}
    routed_markers: list[str] = []

    def compress(self, content: str, **_kwargs):
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        try:
            marker = content.rsplit(" marker", 1)[1]
            routed_markers.append(marker)
            return RouterCompressionResult(
                compressed=f"summary marker{marker}",
                original=content,
                strategy_used=CompressionStrategy.KOMPRESS,
            )
        finally:
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
            request_id="req_ordered_cache_misses",
        )
    )

    assert active["max"] == 1
    assert routed_markers == ["0", "1", "2", "3"]
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


def test_openai_responses_oversized_tool_output_uses_ccr_without_router(monkeypatch):
    reset_compression_store()
    router = ContentRouter()

    def router_must_not_run(self, content: str, **_kwargs):
        raise AssertionError("oversized tool output should use bounded CCR fast path")

    router.compress = MethodType(router_must_not_run, router)
    handler = _handler_with_router(router)
    monkeypatch.setattr(handler, "OPENAI_RESPONSES_ROUTER_MAX_BYTES", 128)

    original_output = " ".join(f"row-{i}" for i in range(400))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-large",
                "output": original_output,
            }
        ],
    }

    new_payload, modified, saved, transforms, reason, _before, _after, attempted = (
        handler._compress_openai_responses_payload(
            payload,
            model="gpt-5",
            request_id="req_large_ccr",
        )
    )

    compressed_output = new_payload["input"][0]["output"]
    marker = "Retrieve more: hash="
    hash_key = compressed_output.split(marker, 1)[1].split("]", 1)[0]
    entry = get_compression_store().retrieve(hash_key)

    assert modified is True
    assert saved > 0
    assert attempted > 0
    assert reason is None
    assert "openai:responses:large_text_ccr" in transforms
    assert marker in compressed_output
    assert len(compressed_output) < len(original_output)
    assert entry is not None
    assert entry.original_content == original_output
    assert entry.tool_call_id == "call-large"


def test_openai_responses_oversized_tool_output_hashes_surrogates_safely(monkeypatch):
    reset_compression_store()
    router = ContentRouter()

    def router_must_not_run(self, content: str, **_kwargs):
        raise AssertionError("oversized tool output should use bounded CCR fast path")

    router.compress = MethodType(router_must_not_run, router)
    handler = _handler_with_router(router)
    monkeypatch.setattr(handler, "OPENAI_RESPONSES_ROUTER_MAX_BYTES", 128)

    original_output = ("row " * 800) + "\ud800"
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-surrogate",
                "output": original_output,
            }
        ],
    }

    new_payload, modified, saved, transforms, reason, _before, _after, _attempted = (
        handler._compress_openai_responses_payload(
            payload,
            model="gpt-5",
            request_id="req_large_surrogate_ccr",
        )
    )

    compressed_output = new_payload["input"][0]["output"]
    marker = "Retrieve more: hash="
    hash_key = compressed_output.split(marker, 1)[1].split("]", 1)[0]

    assert modified is True
    assert saved > 0
    assert reason is None
    assert "openai:responses:large_text_ccr" in transforms
    assert get_compression_store().retrieve(hash_key) is not None


def test_openai_responses_oversized_tool_output_preserves_existing_ccr_marker(
    monkeypatch,
):
    router = ContentRouter()

    def router_must_not_run(self, content: str, **_kwargs):
        raise AssertionError("already-compressed oversized output should remain idempotent")

    router.compress = MethodType(router_must_not_run, router)
    handler = _handler_with_router(router)
    monkeypatch.setattr(handler, "OPENAI_RESPONSES_ROUTER_MAX_BYTES", 128)

    original_output = (
        "[400 items compressed to 20. Retrieve more: hash=abc123abc123abc123abc123]\n"
        + ("row " * 200)
    )
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "call-existing-marker",
                "output": original_output,
            }
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, _strategy_chain, attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_existing_marker",
        )
    )

    assert modified is False
    assert saved == 0
    assert attempted == 0
    assert transforms == []
    assert units_by_category == {"already_compressed": 1}
    assert new_payload["input"][0]["output"] == original_output
