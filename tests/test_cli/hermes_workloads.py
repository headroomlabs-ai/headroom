"""Shared Hermes benchmark workloads for live tests and operator scripts.

Headroom compresses large ``role: tool`` outputs (SmartCrusher), not plain
user/system prose. These fixtures mirror bot/agent traffic on spot-tech-ci.
"""

from __future__ import annotations

import json


def large_tool_output(items: int = 150) -> str:
    return json.dumps(
        [
            {
                "id": i,
                "severity": "error",
                "service": "api-gateway",
                "message": "connection reset by peer",
                "stack": f"at Handler.process(line:{100 + i})\n at Router.dispatch(line:{200 + i})",
                "payload": "x" * 180,
            }
            for i in range(items)
        ]
    )


def agent_tool_messages(
    *,
    tool_items: int = 150,
    tool_call_id: str = "call_bench_1",
    user_prompt: str = "Run fleet diagnostics and report error count.",
    follow_up: str = "How many error rows in the tool output? Reply COUNT_<number> only.",
) -> list[dict]:
    return [
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "run_diagnostics", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": large_tool_output(tool_items),
        },
        {"role": "user", "content": follow_up},
    ]


def multi_turn_agent_messages(turns: int = 3, tool_items: int = 100) -> list[dict]:
    messages: list[dict] = [
        {
            "role": "system",
            "content": "Short answers only. Summarize tool output briefly.",
        }
    ]
    for turn in range(1, turns + 1):
        call_id = f"call_turn_{turn}"
        messages.extend(
            [
                {"role": "user", "content": f"Turn {turn}: run diagnostics batch {turn}."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "run_diagnostics", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": large_tool_output(tool_items + turn * 10),
                },
                {
                    "role": "user",
                    "content": f"Turn {turn}: reply TURN_{turn} only.",
                },
            ]
        )
    return messages