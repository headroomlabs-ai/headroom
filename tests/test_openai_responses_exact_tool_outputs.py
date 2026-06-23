from __future__ import annotations

from types import SimpleNamespace

from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _normalize_responses_tool_name,
    _responses_tool_name_by_call_id,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text.split()))


class _Handler(OpenAIHandlerMixin):
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.metrics = None
        self.openai_pipeline = object()
        self.openai_provider = SimpleNamespace(get_token_counter=lambda _model: _Tokenizer())


def _large(label: str) -> str:
    return f"{label} line with enough repeated content to pass the router floor\n" * 80


def test_responses_tool_name_map_normalizes_pi_and_mcp_namespaces() -> None:
    assert _normalize_responses_tool_name("functions.read") == "read"
    assert _normalize_responses_tool_name("mcp__Edit") == "edit"
    assert _normalize_responses_tool_name("headroom_retrieve") == "headroom_retrieve"
    assert _normalize_responses_tool_name("") is None

    assert _responses_tool_name_by_call_id(
        [
            {"type": "function_call", "call_id": "r", "name": "functions.read"},
            {"type": "function_call", "call_id": "s", "name": "shell"},
            {"type": "message", "role": "user", "content": "ignore me"},
        ]
    ) == {"r": "read", "s": "shell"}


def test_responses_compression_skips_exact_file_tool_outputs(monkeypatch) -> None:
    import headroom.transforms.compression_units as compression_units

    compressed_texts: list[str] = []

    def fake_compress_unit_with_router(unit, **_kwargs):
        compressed_texts.append(unit.text)
        return SimpleNamespace(
            original=unit.text,
            compressed="compressed tool output",
            modified=True,
            tokens_before=100,
            tokens_after=3,
            tokens_saved=97,
            transforms_applied=["fake"],
            strategy="fake",
            router_result=None,
            reason_category="applied",
            reason="fake compression",
            text_bytes=len(unit.text.encode("utf-8", errors="replace")),
            min_bytes=unit.min_bytes,
        )

    monkeypatch.setattr(compression_units, "find_content_router", lambda _pipeline: object())
    monkeypatch.setattr(
        compression_units,
        "compress_unit_with_router",
        fake_compress_unit_with_router,
    )

    read_output = _large("exact file content")
    shell_output = _large("shell build log")
    payload = {
        "model": "gpt-5.4",
        "input": [
            {
                "type": "function_call",
                "call_id": "call-read",
                "name": "functions.read",
                "arguments": '{"path":"headroom/proxy/handlers/openai.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-read",
                "output": read_output,
            },
            {
                "type": "function_call",
                "call_id": "call-shell",
                "name": "shell",
                "arguments": '{"command":"pytest"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-shell",
                "output": shell_output,
            },
        ],
    }

    updated, modified, tokens_saved, transforms, *_rest = (
        _Handler()._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5.4",
            request_id="test-exact-tool-output",
        )
    )

    assert modified is True
    assert tokens_saved == 97
    assert transforms
    assert compressed_texts == [shell_output]
    assert updated["input"][1]["output"] == read_output
    assert updated["input"][3]["output"] == "compressed tool output"
