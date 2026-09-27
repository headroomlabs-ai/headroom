"""Synthetic GitHub Copilot Chat session builder, matching the real
schema reverse-engineered from one user's actual session history
(extension v0.59.0-0.65.0). See census_copilot.py's module docstring for
the schema-drift notes this fixture is built to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLAUDE_SONNET_5_CATALOG_ENTRY: dict[str, Any] = {
    "id": "claude-sonnet-5",
    "vendor": "copilot",
    "name": "Claude Sonnet 5",
    "family": "claude-sonnet-5",
    "pricing": "In: 200 · Out: 1000 AICs/1M tokens",
    "inputCost": 200,
    "outputCost": 1000,
    "cacheCost": 20,
    "cacheWriteCost": 250,
    "maxInputTokens": 935793,
    "maxOutputTokens": 64000,
}


def file_read_request(
    request_id: str,
    *,
    model_id: str = "copilot/claude-sonnet-5",
    prompt_tokens: int = 45615,
    output_tokens: int = 2088,
    file_text: str = "# Blueprint\nSome file content.\n",
    thinking_text: str = "SECRET REASONING THAT MUST NEVER BE READ",
) -> dict[str, Any]:
    """New-schema shape: usage under result.metadata, tool content via
    toolCallResults keyed by toolCallId, a thinking block riding along in
    toolCallRounds (must never be touched by anything that reads this).
    """
    tool_call_id = f"tc_{request_id}"
    return {
        "requestId": request_id,
        "timestamp": 1,
        "agent": {"modelId": model_id},
        "message": {"text": "read the file"},
        "response": [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "copilot_readFile",
                "toolCallId": tool_call_id,
                "invocationMessage": {"value": "Read file"},
            }
        ],
        "result": {
            "metadata": {
                "promptTokens": prompt_tokens,
                "outputTokens": output_tokens,
                "toolCallRounds": [
                    {"modelId": model_id.split("/", 1)[-1], "thinking": {"text": thinking_text, "encrypted": "x"}}
                ],
                "toolCallResults": {
                    tool_call_id: {
                        "content": [{"value": {"node": {"children": [{"type": 2, "text": file_text}]}}}]
                    }
                },
            }
        },
    }


def old_schema_terminal_request(
    request_id: str,
    *,
    model_id: str = "copilot/claude-sonnet-5",
    prompt_tokens: int = 5000,
    completion_tokens: int = 300,
    terminal_output: str = "5 passed in 1.2s\n",
) -> dict[str, Any]:
    """Old-schema shape: usage at top level (`promptTokens`/`completionTokens`,
    not under result.metadata), terminal output as a standalone response
    part rather than folded into toolSpecificData.
    """
    return {
        "requestId": request_id,
        "timestamp": 2,
        "agent": {"modelId": model_id},
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "message": {"text": "run tests"},
        "response": [
            {
                "kind": "terminal",
                "terminalCommandId": f"term_{request_id}",
                "commandLine": {"original": "pytest -q"},
                "terminalCommandOutput": {"text": terminal_output, "lineCount": terminal_output.count("\n")},
            }
        ],
    }


def grep_request(
    request_id: str,
    *,
    model_id: str = "ollama-models/glm4:latest",
    prompt_tokens: int = 1000,
    output_tokens: int = 50,
    grep_output: str = "foo.py:1:def foo(): pass\n",
) -> dict[str, Any]:
    tool_call_id = f"tc_{request_id}"
    return {
        "requestId": request_id,
        "timestamp": 3,
        "agent": {"modelId": model_id},
        "message": {"text": "search for foo"},
        "response": [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "copilot_findTextInFiles",
                "toolCallId": tool_call_id,
                "invocationMessage": {"value": "Searched"},
            }
        ],
        "result": {
            "metadata": {
                "promptTokens": prompt_tokens,
                "outputTokens": output_tokens,
                "toolCallResults": {tool_call_id: {"content": [{"text": grep_output}]}},
            }
        },
    }


def unpriced_model_request(request_id: str, *, model_id: str = "gpt-unknown-model") -> dict[str, Any]:
    return {
        "requestId": request_id,
        "timestamp": 4,
        "agent": {"modelId": model_id},
        "message": {"text": "hi"},
        "response": [],
        "result": {"metadata": {"promptTokens": 100, "outputTokens": 10}},
    }


def write_session(
    path: Path,
    *,
    lines: list[list[dict[str, Any]]],
    catalog_entries: list[dict[str, Any]] | None = None,
) -> Path:
    """Writes a multi-line chatSessions .jsonl file, one line per snapshot,
    each carrying the given list of requests -- mirrors how VS Code
    re-saves the whole session state on every update.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    input_state = {"selectedModel": {"metadata": {"extension": (catalog_entries or [None])[0]}}}
    raw_lines = []
    for requests in lines:
        raw_lines.append(json.dumps({"kind": 0, "v": {"requests": requests, "inputState": input_state}}))
    path.write_text("\n".join(raw_lines) + "\n")
    return path


def build_basic_session(tmp_path: Path, *, workspace_hash: str = "hash1", session_id: str = "session1") -> Path:
    """One session, two snapshot lines (partial then full -- exercises
    "best line wins"), covering: new-schema file read (with a thinking
    block that must never be read), old-schema terminal run, a local-Ollama
    grep call ($0 real cost), and one unpriced model.
    """
    req_new = file_read_request("req1")
    req_old = old_schema_terminal_request("req2")
    req_ollama = grep_request("req3")
    req_unpriced = unpriced_model_request("req4")

    path = Path(tmp_path) / workspace_hash / "chatSessions" / f"{session_id}.jsonl"
    return write_session(
        path,
        lines=[[req_new], [req_new, req_old, req_ollama, req_unpriced]],
        catalog_entries=[CLAUDE_SONNET_5_CATALOG_ENTRY],
    )
