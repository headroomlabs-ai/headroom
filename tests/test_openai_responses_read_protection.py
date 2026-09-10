"""Regression tests: file reads over the OpenAI Responses API path must stay verbatim.

Copilot CLI (and other Responses-native harnesses) read files two ways:

1. A first-class ``view`` tool (the Copilot equivalent of Claude Code's ``Read``)
   whose output is raw file content the model will byte-patch against.
2. Shell reads through ``bash`` (``cat``/``nl``/``sed -n`` …), which the
   chat/Anthropic path protects via ``HEADROOM_PROTECT_READS`` read-command
   detection in ``ContentRouter``.

The Responses compression-units path historically protected neither: only
``DEFAULT_EXCLUDE_TOOLS`` names were honored, and ``HEADROOM_PROTECT_READS``
was never consulted. Lossy (Kompress) compression of a fresh file read garbles
exactly the bytes the model needs for line-precise edits, forcing re-reads
(turn inflation) — the harm read protection exists to prevent.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
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


def _lossy_router() -> ContentRouter:
    """Router whose compress() always 'lossy-compresses' any candidate it sees."""

    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="kept words",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    return router


def _run(handler: OpenAIHandlerMixin, payload: dict):
    return handler._compress_openai_responses_live_text_units_with_router(
        payload,
        model="gpt-5",
        request_id="req_read_protection",
    )


_FILE_CONTENT = "\n".join(
    f"## Section {i}\nSome roadmap prose line {i} with enough words to matter" for i in range(90)
)

_NL_OUTPUT = "\n".join(
    f"{i}\tline {i} of the roadmap file with a handful of words in it" for i in range(1, 110)
)


def test_responses_view_tool_read_stays_verbatim():
    """Copilot's `view` tool returns raw file bytes: never lossy-compress them."""
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_view",
                "name": "view",
                "arguments": '{"path": "/repo/ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_view",
                "output": _FILE_CONTENT,
            },
        ],
    }

    new_payload, _modified, _saved, _t, _u, _s, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _FILE_CONTENT


def test_responses_bash_read_command_stays_verbatim_when_protect_reads(monkeypatch):
    """HEADROOM_PROTECT_READS=1 must cover bash file reads on the Responses path too."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_bash",
                "name": "bash",
                "arguments": ('{"command": "nl -ba .overlay/ROADMAP.md | sed -n \'1,75p\'"}'),
            },
            {
                "type": "function_call_output",
                "call_id": "call_bash",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, _modified, _saved, _t, _u, _s, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _NL_OUTPUT


def test_responses_excluded_read_tool_stays_verbatim_control():
    """Control: Claude-style `Read` outputs are already protected today."""
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": '{"file_path": "/repo/ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": _FILE_CONTENT,
            },
        ],
    }

    new_payload, _modified, _saved, _t, _u, _s, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _FILE_CONTENT


def test_responses_bash_read_compresses_when_protect_reads_disabled(monkeypatch):
    """Control: with HEADROOM_PROTECT_READS unset/0, bash reads stay compressible."""
    monkeypatch.delenv("HEADROOM_PROTECT_READS", raising=False)
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_bash",
                "name": "bash",
                "arguments": '{"command": "cat src/main.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_bash",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][1]["output"] == "kept words"


def test_responses_non_read_bash_command_still_compresses(monkeypatch):
    """Protection is type-specific: test/build/search output stays compressible."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_test",
                "name": "bash",
                "arguments": '{"command": "uv run pytest tests/ -q"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_test",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][1]["output"] == "kept words"


def test_responses_lockfile_read_stays_compressible(monkeypatch):
    """Lockfiles are tool-regenerated, never byte-patched: the command-level
    carve-out keeps `cat uv.lock` compressible even with protection on."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_lock",
                "name": "bash",
                "arguments": '{"command": "cat uv.lock"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_lock",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][1]["output"] == "kept words"


def test_responses_local_shell_call_read_stays_verbatim(monkeypatch):
    """Codex native shell: local_shell_call.action.command (argv) read protected."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call",
                "call_id": "call_lsc",
                "action": {"type": "exec", "command": ["nl", "-ba", "ROADMAP.md"]},
            },
            {
                "type": "local_shell_call_output",
                "call_id": "call_lsc",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _NL_OUTPUT


def test_responses_view_output_content_part_array_stays_verbatim():
    """`view` output shaped as a content-part array is protected byte-exactly,
    including non-text parts."""
    handler = _handler_with_router(_lossy_router())
    parts = [
        {"type": "output_text", "text": _FILE_CONTENT},
        {"type": "refusal", "refusal": "n/a"},
    ]
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_view",
                "name": "view",
                "arguments": '{"path": "/repo/ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_view",
                "output": parts,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == parts


def test_responses_view_json_shaped_output_stays_byte_exact():
    """Even JSON-shaped `view` output is verbatim: the byte-exact contract beats
    the lossless JSON minification other excluded tools accept."""
    handler = _handler_with_router(_lossy_router())
    pretty_json = "\n".join(
        ["{"] + [f'  "key_{i}": {i},' for i in range(120)] + ['  "end": true', "}"]
    )
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_view",
                "name": "view",
                "arguments": '{"path": "/repo/data.json"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_view",
                "output": pretty_json,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == pretty_json


def test_responses_malformed_arguments_do_not_break_extraction(monkeypatch):
    """Malformed function_call arguments yield no command -> normal compression."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_bad",
                "name": "bash",
                "arguments": "{not json at all",
            },
            {
                "type": "function_call_output",
                "call_id": "call_bad",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][1]["output"] == "kept words"


def test_responses_protected_read_survives_cross_turn_dedup(monkeypatch):
    """A repeated protected read must not be replaced by a [↑…] dedup pointer."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    router = _lossy_router()
    router._cross_turn_dedup_enabled = True
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_r1",
                "name": "bash",
                "arguments": '{"command": "nl -ba ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": _NL_OUTPUT,
            },
            {
                "type": "function_call",
                "call_id": "call_r2",
                "name": "bash",
                "arguments": '{"command": "nl -ba ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_r2",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _NL_OUTPUT
    assert new_payload["input"][3]["output"] == _NL_OUTPUT


def test_responses_debug_path_with_excluded_list_output(monkeypatch):
    """Regression: debug logging over an excluded tool's content-part output must
    not raise (latent unbound `fold` variable in the list branch)."""
    from headroom.proxy.handlers import openai as openai_handler

    monkeypatch.setattr(openai_handler, "_log_codex_compression_debug", lambda *a, **k: None)
    handler = _handler_with_router(_lossy_router())
    parts = [{"type": "output_text", "text": _FILE_CONTENT}]
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": '{"file_path": "/repo/ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": parts,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == parts


def test_responses_read_command_with_releasable_json_output_compresses(monkeypatch):
    """Content gate: a read command whose output is confidently DATA (JSON array)
    is released to compression even with HEADROOM_PROTECT_READS=1."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    json_output = "[" + ",".join(f'{{"line": {i}, "text": "value {i}"}}' for i in range(60)) + "]"
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_json",
                "name": "bash",
                "arguments": '{"command": "cat data.json"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_json",
                "output": json_output,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][1]["output"] == "kept words"


def test_responses_local_shell_call_string_command_read_stays_verbatim(monkeypatch):
    """local_shell_call with a string (not argv) command is also covered."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call",
                "call_id": "call_lsc_str",
                "action": {"type": "exec", "command": "cat src/app.py"},
            },
            {
                "type": "local_shell_call_output",
                "call_id": "call_lsc_str",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _NL_OUTPUT


def test_responses_debug_path_with_read_protected_output(monkeypatch):
    """Debug logging over a read-protected output records and does not raise."""
    from headroom.proxy.handlers import openai as openai_handler

    monkeypatch.setattr(openai_handler, "_log_codex_compression_debug", lambda *a, **k: None)
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_dbg",
                "name": "bash",
                "arguments": '{"command": "nl -ba ROADMAP.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_dbg",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, _modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert new_payload["input"][1]["output"] == _NL_OUTPUT


def test_responses_read_scan_tolerates_non_dict_and_missing_call_id(monkeypatch):
    """The producer scan must skip non-dict items and calls without a string
    call_id without breaking normal compression."""
    monkeypatch.setenv("HEADROOM_PROTECT_READS", "1")
    handler = _handler_with_router(_lossy_router())
    payload = {
        "model": "gpt-5",
        "input": [
            "a bare string item",
            {
                "type": "function_call",
                "name": "bash",
                "arguments": '{"command": "cat src/app.py"}',
            },
            {
                "type": "function_call",
                "call_id": 42,
                "name": "bash",
                "arguments": '{"command": "cat src/app.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_x",
                "output": _NL_OUTPUT,
            },
        ],
    }

    new_payload, modified, _s, _t, _u, _c, _a = _run(handler, payload)

    assert modified is True
    assert new_payload["input"][0] == "a bare string item"
    assert new_payload["input"][3]["output"] == "kept words"
