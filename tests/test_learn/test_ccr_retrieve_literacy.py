from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from headroom.ccr.retrieve_policy import LEARN_SECTION
from headroom.learn.analyzer import SessionAnalyzer
from headroom.learn.models import ProjectInfo, SessionData, SessionEvent, ToolCall
from headroom.learn.plugins.codex import CodexPlugin


def _project(tmp_path: Path) -> ProjectInfo:
    project_path = tmp_path / "project"
    project_path.mkdir()
    data_path = tmp_path / "data"
    data_path.mkdir()
    return ProjectInfo(name="demo", project_path=project_path, data_path=data_path)


def _retrieve_call(
    msg_index: int,
    query: str | None = None,
    *,
    name: str = "headroom_retrieve",
) -> ToolCall:
    input_data = {"hash": "abc123"}
    if query is not None:
        input_data["query"] = query
    return ToolCall(
        name=name,
        tool_call_id=f"tc-{msg_index}",
        input_data=input_data,
        output='{"rows": 1}',
        is_error=False,
        msg_index=msg_index,
        output_bytes=11,
    )


@patch(
    "headroom.learn.analyzer._call_llm",
    return_value={"context_file_rules": [], "memory_file_rules": []},
)
def test_analyzer_emits_ccr_retrieve_literacy_rule_for_redundant_retrieval(
    _mock_call_llm, tmp_path: Path
) -> None:
    tool_call = _retrieve_call(2)
    session = SessionData(
        session_id="s1",
        tool_calls=[tool_call],
        events=[
            SessionEvent(
                type="user_message",
                msg_index=1,
                text="Be sure the kept summary did not miss anything.",
            ),
            SessionEvent(type="tool_call", msg_index=2, tool_call=tool_call),
        ],
    )

    result = SessionAnalyzer(model="test-model").analyze(_project(tmp_path), [session])

    assert any(rec.section == LEARN_SECTION for rec in result.recommendations)
    rule = next(rec for rec in result.recommendations if rec.section == LEARN_SECTION)
    assert "Trust kept rows unless you have a concrete gap." in rule.content
    assert rule.evidence_count == 1


@patch(
    "headroom.learn.analyzer._call_llm",
    return_value={"context_file_rules": [], "memory_file_rules": []},
)
def test_analyzer_skips_targeted_retrieve_requests(_mock_call_llm, tmp_path: Path) -> None:
    tool_call = _retrieve_call(2, query="auth middleware")
    session = SessionData(
        session_id="s1",
        tool_calls=[tool_call],
        events=[
            SessionEvent(
                type="user_message", msg_index=1, text="Check the auth middleware entries."
            ),
            SessionEvent(type="tool_call", msg_index=2, tool_call=tool_call),
        ],
    )

    result = SessionAnalyzer(model="test-model").analyze(_project(tmp_path), [session])

    assert all(rec.section != LEARN_SECTION for rec in result.recommendations)


@patch(
    "headroom.learn.analyzer._call_llm",
    return_value={"context_file_rules": [], "memory_file_rules": []},
)
def test_analyzer_emits_rule_for_thoroughness_retrieve_even_with_query(
    _mock_call_llm, tmp_path: Path
) -> None:
    tool_call = _retrieve_call(2, query="auth middleware")
    session = SessionData(
        session_id="s1",
        tool_calls=[tool_call],
        events=[
            SessionEvent(
                type="user_message",
                msg_index=1,
                text="Be sure the kept summary did not miss anything.",
            ),
            SessionEvent(type="tool_call", msg_index=2, tool_call=tool_call),
        ],
    )

    result = SessionAnalyzer(model="test-model").analyze(_project(tmp_path), [session])

    assert any(rec.section == LEARN_SECTION for rec in result.recommendations)


@patch(
    "headroom.learn.analyzer._call_llm",
    return_value={"context_file_rules": [], "memory_file_rules": []},
)
def test_analyzer_counts_namespaced_mcp_retrieve_calls(_mock_call_llm, tmp_path: Path) -> None:
    tool_call = _retrieve_call(2, name="mcp__headroom__headroom_retrieve")
    session = SessionData(
        session_id="s1",
        tool_calls=[tool_call],
        events=[
            SessionEvent(
                type="user_message",
                msg_index=1,
                text="Be sure the kept summary did not miss anything.",
            ),
            SessionEvent(type="tool_call", msg_index=2, tool_call=tool_call),
        ],
    )

    result = SessionAnalyzer(model="test-model").analyze(_project(tmp_path), [session])

    assert any(rec.section == LEARN_SECTION for rec in result.recommendations)


@patch("headroom.learn.analyzer._detect_default_model", side_effect=RuntimeError("no backend"))
def test_analyzer_returns_deterministic_recommendation_when_model_detection_fails(
    _mock_detect_default_model, tmp_path: Path
) -> None:
    tool_call = _retrieve_call(2)
    session = SessionData(
        session_id="s1",
        tool_calls=[tool_call],
        events=[
            SessionEvent(
                type="user_message",
                msg_index=1,
                text="Be sure the kept summary did not miss anything.",
            ),
            SessionEvent(type="tool_call", msg_index=2, tool_call=tool_call),
        ],
    )

    result = SessionAnalyzer().analyze(_project(tmp_path), [session])

    assert any(rec.section == LEARN_SECTION for rec in result.recommendations)


def test_codex_rollout_scanner_preserves_user_messages(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions" / "2026" / "06"
    sessions_dir.mkdir(parents=True)
    session_file = sessions_dir / "rollout-1.jsonl"
    session_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "rollout-1"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Be sure the summary is enough."}
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "headroom_retrieve",
                            "call_id": "call-1",
                            "arguments": json.dumps({"hash": "abc123"}),
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-1",
                            "output": '{"rows": 1}',
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    plugin = CodexPlugin(codex_dir=tmp_path)
    session = plugin._scan_jsonl_session(session_file)

    assert session is not None
    assert any(event.type == "user_message" for event in session.events)
    assert any(event.type == "tool_call" for event in session.events)
    user_message = next(event for event in session.events if event.type == "user_message")
    assert "Be sure" in user_message.text
