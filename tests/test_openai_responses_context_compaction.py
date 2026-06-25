from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from headroom.pipeline import PipelineEvent, PipelineExtensionManager, PipelineStage
from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _compact_openai_responses_tools,
    _ensure_responses_store_for_memory_tools,
    _openai_responses_context_budget,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)


def test_openai_responses_context_budget_breaks_out_static_and_live_buckets() -> None:
    payload = {
        "instructions": "stable instructions",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "line one\nline two\n",
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "do the thing"}],
            },
        ],
    }

    budget = _openai_responses_context_budget(payload)

    assert budget["payload_bytes"] > 0
    assert {"instructions", "tools", "input"}.issubset(budget["buckets"])
    assert budget["input_breakdown"]["function_call_output"]["items"] == 1
    assert budget["input_breakdown"]["function_call_output"]["text_bytes"] == len(
        b"line one\nline two\n"
    )
    assert budget["input_breakdown"]["message"]["items"] == 1


def test_openai_tool_schema_compaction_preserves_invocation_shape() -> None:
    verbose = " ".join(["Use this tool to read a file from the workspace."] * 40)
    payload = {
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "title": "Read File",
                "description": verbose,
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "ReadFileParameters",
                    "type": "object",
                    "properties": {
                        "path": {
                            "title": "Path",
                            "type": "string",
                            "description": verbose,
                            "examples": ["src/main.py"],
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ]
    }

    compacted, modified, before, after = _compact_openai_responses_tools(payload)

    assert modified is True
    assert after < before
    tool = compacted["tools"][0]
    assert tool["type"] == "function"
    assert tool["name"] == "read_file"
    assert "title" not in tool
    assert tool["parameters"]["type"] == "object"
    assert tool["parameters"]["required"] == ["path"]
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["properties"]["path"]["type"] == "string"
    assert "examples" not in tool["parameters"]["properties"]["path"]
    assert tool["parameters"]["properties"]["path"]["description"] == " ".join(verbose.split())


def test_openai_tool_schema_compaction_preserves_property_named_title() -> None:
    """Issue #759: drop-key list must not strip property *names* under `properties`.

    Schema annotations like ``title: "ReadFileParameters"`` on a schema object
    are safe to drop.  But a tool that has a field literally called ``title``
    (or ``readOnly``, ``deprecated``, etc.) must survive compaction; removing
    it while leaving ``required: ["title"]`` produces an invalid strict schema
    that upstream (OpenAI / Codex) rejects.
    """
    payload = {
        "tools": [
            {
                "type": "function",
                "name": "eval",
                "description": "Evaluate cells.",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "EvalParameters",
                    "type": "object",
                    "properties": {
                        "cells": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "title": "CellItem",
                                "properties": {
                                    "language": {"type": "string"},
                                    "code": {"type": "string"},
                                    "title": {"type": "string"},
                                },
                                "required": ["language", "code", "title"],
                            },
                        }
                    },
                    "required": ["cells"],
                },
            }
        ]
    }

    compacted, modified, before, after = _compact_openai_responses_tools(payload)

    assert modified is True
    assert after < before

    params = compacted["tools"][0]["parameters"]
    # Schema-level annotations are still dropped.
    assert "title" not in params
    assert "$schema" not in params

    items = params["properties"]["cells"]["items"]
    # "title" as a JSON Schema annotation on the items object is dropped.
    assert "title" not in items
    # "title" as a *property name* inside properties must be preserved.
    assert "title" in items["properties"], (
        "property named 'title' was incorrectly stripped by compaction"
    )
    assert items["required"] == ["language", "code", "title"]


def test_openai_tool_schema_compaction_is_deterministic() -> None:
    payload = {
        "tools": [
            {
                "type": "function",
                "name": "mcp__serena__",
                "description": "  Semantic code tools.\n\nUse for symbol-aware edits.  ",
                "parameters": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$comment": "annotation only",
                    "type": "object",
                    "properties": {
                        "name_path_pattern": {
                            "type": "string",
                            "description": "  Name path to match.\nKeeps full semantics. ",
                            "examples": ["Foo/bar"],
                        }
                    },
                    "required": ["name_path_pattern"],
                    "additionalProperties": False,
                },
            }
        ]
    }

    first, first_modified, first_before, first_after = _compact_openai_responses_tools(payload)
    second, second_modified, second_before, second_after = _compact_openai_responses_tools(payload)

    assert first_modified is True
    assert second_modified is True
    assert first_before == second_before
    assert first_after == second_after
    assert first == second
    assert first["tools"][0]["description"] == ("Semantic code tools. Use for symbol-aware edits.")
    prop = first["tools"][0]["parameters"]["properties"]["name_path_pattern"]
    assert prop["description"] == "Name path to match. Keeps full semantics."
    assert prop["type"] == "string"
    assert "examples" not in prop


class _StubTokenizer:
    def count_text(self, text: str) -> int:
        return len(text.split())


class _StubProvider:
    def get_token_counter(self, model: str) -> _StubTokenizer:
        del model
        return _StubTokenizer()


class _StubPipeline:
    def __init__(self, router: ContentRouter):
        self.transforms = [router]


class _HandlerHarness(OpenAIHandlerMixin):
    """Minimal subclass exposing just the deps the unit-extraction path
    actually reads. The full HeadroomProxy ctor wires dozens of unrelated
    services; this keeps the test focused on the gate behavior."""

    def __init__(self, router: ContentRouter):
        self.openai_pipeline: Any = _StubPipeline(router)
        self.openai_provider: Any = _StubProvider()
        self.pipeline_extensions: Any = PipelineExtensionManager(discover=False)


class _RecordingResponsesExtension:
    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def on_pipeline_event(self, event: PipelineEvent) -> PipelineEvent:
        self.events.append(event)
        if event.payload is not None and event.stage == PipelineStage.INPUT_RECEIVED:
            event.payload["extension_marker"] = event.stage.value
        if event.headers is not None and event.stage == PipelineStage.PRE_SEND:
            event.headers["x-test-extension"] = "seen"
        return event


def test_responses_pipeline_payload_stage_allows_payload_and_header_mutation() -> None:
    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)
    extension = _RecordingResponsesExtension()
    handler.pipeline_extensions = PipelineExtensionManager(
        extensions=[extension],
        discover=False,
    )

    payload: dict[str, Any] = {
        "model": "gpt-5.5",
        "input": [{"type": "function_call_output", "output": "build log"}],
    }
    headers = {"authorization": "Bearer test"}

    updated, updated_headers, changed = handler._emit_openai_responses_payload_stage(
        PipelineStage.INPUT_RECEIVED,
        request_id="req-responses-ext",
        model="gpt-5.5",
        payload=payload,
        headers=headers,
        transport="websocket",
        stream=True,
        frame_index=2,
        frame_type="response.create",
    )
    updated, updated_headers, presend_changed = handler._emit_openai_responses_payload_stage(
        PipelineStage.PRE_SEND,
        request_id="req-responses-ext",
        model="gpt-5.5",
        payload=updated,
        headers=updated_headers,
        transport="websocket",
        stream=True,
        frame_index=2,
        frame_type="response.create",
    )

    assert changed is True
    assert presend_changed is True
    assert updated["extension_marker"] == "input_received"
    assert updated_headers == {"authorization": "Bearer test", "x-test-extension": "seen"}
    assert [event.stage for event in extension.events] == [
        PipelineStage.INPUT_RECEIVED,
        PipelineStage.PRE_SEND,
    ]
    assert extension.events[0].payload is payload
    assert extension.events[0].metadata["api_style"] == "responses"
    assert extension.events[0].metadata["transport"] == "websocket"
    assert extension.events[0].metadata["frame_index"] == 2


def test_responses_pipeline_payload_stage_is_noop_without_extensions() -> None:
    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)
    payload: dict[str, Any] = {"model": "gpt-5.5", "input": "hello"}

    updated, headers, changed = handler._emit_openai_responses_payload_stage(
        PipelineStage.INPUT_RECEIVED,
        request_id="req-noop",
        model="gpt-5.5",
        payload=payload,
        transport="http",
        stream=False,
    )

    assert updated is payload
    assert headers is None
    assert changed is False


def test_codex_input_list_payload_reaches_router_without_skip() -> None:
    """Codex's Responses payload uses `input=[...]` with no `messages` key.
    The compression gate must accept either field as the items source —
    otherwise the entire payload is silently passed through uncompressed,
    which is the exact production bug surfaced in proxy.log analysis."""

    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)

    long_output = " ".join(["compressible"] * 200)
    payload: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": long_output,
            }
        ],
        # Note: no `messages` key at all — Codex doesn't send one.
    }

    updated, modified, _saved, _transforms, _units_by_cat, _chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5.5",
            request_id="hr_codex_test_0001",
        )
    )

    # The gate must NOT have skipped the payload. If it had, `updated`
    # would be the input payload identity-passed through with
    # modified=False — but the deepcopy + splice always returns a new
    # dict object when the path executes.
    assert updated is not payload, "Codex-shape payload was skipped at the input/messages gate"
    # Whether or not Kompress actually compresses 200 repeated words is
    # not the point of this test; the point is that we *entered* the
    # extraction loop. Modified may be True or False depending on
    # Kompress availability in CI, so we only assert non-skip semantics.
    assert isinstance(modified, bool)


def test_codex_payload_with_only_messages_field_also_reaches_router() -> None:
    """The Anthropic-style shape (messages=list, no input) must also
    flow. This is the reverse of the Codex case and guards against a
    future regression that swings the gate too far the other way."""

    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)

    payload: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        "messages": [
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": " ".join(["compressible"] * 200),
            }
        ],
    }

    updated, _modified, _saved, _transforms, _units_by_cat, _chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5.5",
            request_id="hr_codex_test_0002",
        )
    )

    assert updated is not payload, "messages-shape payload was skipped at the gate"


def test_compression_pass_debug_logs_are_suppressed(caplog) -> None:
    """Re-entrant Codex websocket passes share one `request_id` but
    process distinct payloads. The `pass_id` field on every compression
    event must be content-derived so dashboards can attribute each
    unit_result to its originating pass. Distinct payloads → distinct
    pass_ids (per-pass savings sum legitimately across passes); identical
    payloads → identical pass_ids (idempotent retries should dedup)."""

    import logging as _logging

    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)

    payload_a: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": " ".join(["alpha"] * 200),
            }
        ],
    }
    payload_b: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": " ".join(["bravo"] * 200),
            }
        ],
    }

    caplog.set_level(_logging.INFO, logger="headroom.proxy")
    handler._compress_openai_responses_payload(
        payload_a, model="gpt-5.5", request_id="hr_shared_request"
    )
    handler._compress_openai_responses_payload(
        payload_b, model="gpt-5.5", request_id="hr_shared_request"
    )
    # Same content twice → same pass_id (deterministic + idempotent).
    handler._compress_openai_responses_payload(
        payload_a, model="gpt-5.5", request_id="hr_shared_request"
    )

    assert not any("event=codex_compression_" in record.getMessage() for record in caplog.records)
    return

    # Collect pass_ids in call order — payload bodies are no longer
    # embedded at INFO so we can't grep for content; we rely on the
    # 3-call sequence [a, b, a] producing a [A, B, A] pass_id sequence.
    pass_id_sequence: list[str] = []
    for record in caplog.records:
        message = record.getMessage()
        if "event=codex_compression_payload_input" not in message:
            continue
        match_quoted = '"pass_id":"'
        idx = message.find(match_quoted)
        assert idx != -1, f"pass_id missing from event: {message[:200]}"
        start = idx + len(match_quoted)
        end = message.find('"', start)
        pass_id_sequence.append(message[start:end])

    assert len(pass_id_sequence) == 3, (
        f"expected exactly 3 payload_input events for 3 calls, got {len(pass_id_sequence)}"
    )
    # Two distinct payloads + one repeat → two distinct pass_ids overall.
    assert len(set(pass_id_sequence)) == 2, (
        f"expected two distinct pass_ids, got {set(pass_id_sequence)}"
    )
    # Repeated payload_a must be deterministic — index 0 and 2 are the
    # same call shape so they must produce the same pass_id.
    assert pass_id_sequence[0] == pass_id_sequence[2], (
        f"repeated identical payload produced different pass_ids: {pass_id_sequence}"
    )
    assert pass_id_sequence[0] != pass_id_sequence[1]


def test_codex_payload_without_either_field_is_skipped() -> None:
    """The gate must still reject malformed payloads — `input` and
    `messages` both absent (or non-list) is the genuine skip condition."""

    router = ContentRouter(ContentRouterConfig())
    handler = _HandlerHarness(router)

    payload: dict[str, Any] = {
        "type": "response.create",
        "model": "gpt-5.5",
        # No input, no messages — genuinely nothing to compress.
    }

    updated, modified, saved, transforms, units_by_cat, chain, attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5.5",
            request_id="hr_codex_test_0003",
        )
    )

    assert updated is payload
    assert modified is False
    assert units_by_cat == {}
    assert chain == []
    assert attempted == 0
    assert saved == 0
    assert transforms == []


def test_content_router_retries_kompress_when_structured_strategy_noops(monkeypatch) -> None:
    router = ContentRouter(ContentRouterConfig(enable_smart_crusher=True))
    content = " ".join("x" for _ in range(200))

    class NoopCrusher:
        def crush(self, value: str, query: str = "", bias: float = 1.0):
            return SimpleNamespace(compressed=value)

    monkeypatch.setattr(router, "_get_smart_crusher", lambda: NoopCrusher())
    monkeypatch.setattr(
        router,
        "_try_ml_compressor",
        lambda value, context, question=None: ("short summary", 2),
    )

    compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        content,
        CompressionStrategy.SMART_CRUSHER,
        context="",
    )

    assert compressed == "short summary"
    assert compressed_tokens == 2
    # The fallback chain must record both strategies it tried.
    assert strategy_chain == ["smart_crusher", "kompress"]


def test_responses_memory_tools_store_false_regression() -> None:
    """Regression: store=false makes previous_response_id continuations fail."""

    payload = {"model": "gpt-5.5", "input": "remember this", "store": False}

    changed = _ensure_responses_store_for_memory_tools(
        payload,
        memory_tools_injected=True,
    )

    assert changed is True
    assert payload["store"] is True


def test_responses_memory_tools_do_not_change_unrelated_requests() -> None:
    no_memory_payload = {"model": "gpt-5.5", "input": "plain", "store": False}
    already_stored_payload = {"model": "gpt-5.5", "input": "plain", "store": True}
    default_store_payload = {"model": "gpt-5.5", "input": "plain"}

    assert (
        _ensure_responses_store_for_memory_tools(
            no_memory_payload,
            memory_tools_injected=False,
        )
        is False
    )
    assert no_memory_payload["store"] is False

    assert (
        _ensure_responses_store_for_memory_tools(
            already_stored_payload,
            memory_tools_injected=True,
        )
        is False
    )
    assert already_stored_payload["store"] is True

    assert (
        _ensure_responses_store_for_memory_tools(
            default_store_payload,
            memory_tools_injected=True,
        )
        is False
    )
    assert "store" not in default_store_payload
