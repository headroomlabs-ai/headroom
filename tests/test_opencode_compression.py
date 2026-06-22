#!/usr/bin/env python3
"""End-to-end token-savings test for OpenCode (sst/opencode) + Headroom.

Simulates a real OpenCode coding session. OpenCode's tools emit large,
repetitive payloads — ``grep`` / ``glob`` hit lists, full file reads, test
runner output, and LSP diagnostics — which is exactly what Headroom's
SmartCrusher compresses natively (no ML model required).

No API key required. Compression runs fully local against the base install
(SmartCrusher only — the ``[ml]`` extra is not needed).

Usage:
    # Benchmark (pretty-printed report):
    cd headroom && uv run python tests/test_opencode_compression.py

    # Pytest (CI-friendly assertions):
    cd headroom && uv run --with pytest pytest tests/test_opencode_compression.py -v -s
"""

from __future__ import annotations

import json
import time

# OpenCode routes openai/anthropic model traffic through the proxy; the proxy
# pipeline is model-aware for tokenisation, so pin a concrete Claude model id.
MODEL = "claude-sonnet-4-5-20250929"


# ── Realistic OpenCode tool-output builders ──────────────────────────────────


def grep_results_json() -> str:
    """OpenCode ``grep`` tool output — many hits sharing one structure."""
    rows = [
        {
            "path": f"src/services/handler_{i:03d}.ts",
            "line_number": (i * 7) % 400 + 1,
            "line": "  logger.info('processing request', { id: requestId });",
            "match": "logger.info",
        }
        for i in range(1, 120)
    ]
    return json.dumps(rows, indent=2)


def file_read_payload() -> str:
    """OpenCode ``read`` tool output — a long source file with line numbers."""
    header = "import { createServer } from 'node:http';\nimport { Router } from './router';\n\n"
    body = "\n".join(
        f"{i:>5}\texport function handler_{i}(req: Request): Response {{ "
        f"return router.dispatch(req, {{ retries: {i % 5} }}); }}"
        for i in range(1, 220)
    )
    return header + body + "\n"


def jest_run_output() -> str:
    """OpenCode ``bash`` running a test suite — mostly passing noise, one FAIL."""
    lines = [
        f"PASS  src/services/handler_{i:03d}.test.ts ({(i % 30) + 4} ms)" for i in range(1, 90)
    ]
    lines.insert(
        61,
        "FAIL  src/services/handler_061.test.ts\n"
        "  ● handler_061 › retries on 503\n\n"
        "    expect(received).toBe(expected)\n\n"
        "    Expected: 200\n    Received: 503\n"
        "      at Object.<anonymous> (src/services/handler_061.test.ts:42:24)",
    )
    lines.append("\nTest Suites: 1 failed, 88 passed, 89 total")
    lines.append("Tests:       1 failed, 351 passed, 352 total")
    return "\n".join(lines)


def lsp_diagnostics_json() -> str:
    """OpenCode LSP diagnostics — nested, highly repetitive structure."""
    diags = [
        {
            "uri": f"file:///repo/src/services/handler_{i:03d}.ts",
            "range": {
                "start": {"line": i % 200, "character": 2},
                "end": {"line": i % 200, "character": 40},
            },
            "severity": 2,
            "code": "no-unused-vars",
            "source": "eslint",
            "message": "is assigned a value but never used.",
        }
        for i in range(1, 90)
    ]
    return json.dumps(diags, indent=2)


def build_opencode_session_messages() -> list[dict]:
    """A realistic multi-turn OpenCode session in OpenAI chat-completions shape."""
    return [
        {"role": "system", "content": "You are OpenCode, an autonomous coding agent."},
        {"role": "user", "content": "The 503 retry test is failing. Find and fix it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "grep",
                        "arguments": json.dumps({"pattern": "logger.info", "path": "src"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": grep_results_json()},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t2",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"filePath": "src/services/handler_061.ts"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t2", "content": file_read_payload()},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t3",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": "npm test"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t3", "content": jest_run_output()},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t4",
                    "type": "function",
                    "function": {"name": "diagnostics", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t4", "content": lsp_diagnostics_json()},
        # A follow-up turn so the large diagnostics result is no longer in the
        # protected "active conversation" window and becomes eligible for
        # compression (Headroom keeps the most-recent turns verbatim).
        {"role": "assistant", "content": "Fixing the unused vars and the 503 retry path now."},
        {"role": "user", "content": "Go ahead."},
    ]


def _table_row(label: str, before: int, after: int) -> str:
    saved = before - after
    pct = saved / max(before, 1) * 100
    bar = "█" * int(pct / 5)
    return f"  {label:<35} {before:>7,} → {after:>7,}   {pct:>5.1f}%  {bar}"


# ── Pytest tests ─────────────────────────────────────────────────────────────


def test_opencode_headroom_compression_saves_tokens() -> None:
    """Headroom must compress a realistic multi-turn OpenCode session."""
    from headroom import compress

    messages = build_opencode_session_messages()

    t0 = time.perf_counter()
    result = compress(messages, model=MODEL)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{_table_row('Full OpenCode session', result.tokens_before, result.tokens_after)}")
    print(f"  Latency: {latency_ms:.0f} ms   Transforms: {', '.join(result.transforms_applied)}")

    assert result.tokens_saved > 0, (
        f"Expected compression on the multi-turn OpenCode session. "
        f"before={result.tokens_before}, after={result.tokens_after}. "
        f"Transforms: {result.transforms_applied}"
    )
    assert len(result.messages) == len(messages), "Message count must not change"


def test_opencode_user_and_system_turns_are_verbatim() -> None:
    """OpenCode user/system turns must be byte-identical after compression."""
    from headroom import compress

    messages = build_opencode_session_messages()
    result = compress(messages, model=MODEL)

    for role in ("system", "user"):
        orig = [m["content"] for m in messages if m.get("role") == role]
        comp = [m["content"] for m in result.messages if m.get("role") == role]
        assert orig == comp, f"{role} turn(s) were mutated"


def test_opencode_lsp_diagnostics_compress() -> None:
    """A large OpenCode LSP diagnostics result (nested JSON) must compress."""
    from headroom import compress

    messages = [
        {"role": "user", "content": "Show me the lint diagnostics."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "d1",
                    "type": "function",
                    "function": {"name": "diagnostics", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "d1", "content": lsp_diagnostics_json()},
    ]

    result = compress(messages, model=MODEL)
    print(f"\n{_table_row('LSP diagnostics (89)', result.tokens_before, result.tokens_after)}")

    assert result.tokens_saved > 0, (
        f"LSP diagnostics JSON was not compressed. "
        f"before={result.tokens_before}, after={result.tokens_after}."
    )


def test_opencode_session_preserves_test_failure() -> None:
    """Compression must not drop the failing test the agent needs to act on."""
    from headroom import compress

    messages = build_opencode_session_messages()
    result = compress(messages, model=MODEL)

    blob = json.dumps(result.messages)
    # The single FAIL and its location must survive the round-trip.
    assert "FAIL" in blob
    assert "handler_061" in blob


# ── Standalone benchmark report ──────────────────────────────────────────────


def _main() -> None:
    from headroom import compress

    print("\n  Headroom × OpenCode — local compression benchmark\n")
    result = compress(build_opencode_session_messages(), model=MODEL)
    print(_table_row("Full OpenCode session", result.tokens_before, result.tokens_after))
    isolated = {
        "grep results (119 hits)": grep_results_json(),
        "file read (~220 lines)": file_read_payload(),
        "test output (352 tests)": jest_run_output(),
        "LSP diagnostics (89)": lsp_diagnostics_json(),
    }
    for label, content in isolated.items():
        # Place the payload outside the protected recent window so the isolated
        # numbers reflect what Headroom would compress mid-session.
        msgs = [
            {"role": "tool", "tool_call_id": "x", "content": content},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "continuing"},
            {"role": "user", "content": "."},
        ]
        r = compress(msgs, model=MODEL)
        print(_table_row(label, r.tokens_before, r.tokens_after))
    print()


if __name__ == "__main__":
    _main()
