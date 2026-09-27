from __future__ import annotations

from pathlib import Path

import census_copilot as cc
import pytest

from .fixtures_copilot import (
    build_basic_session,
    file_read_request,
    grep_request,
    old_schema_terminal_request,
)


def test_load_best_snapshot_picks_the_fullest_line(tmp_path: Path) -> None:
    session_file = build_basic_session(tmp_path)
    v = cc.load_best_snapshot(session_file)
    assert v is not None
    assert len(v["requests"]) == 4  # the fuller second line, not the 1-request first line


def test_find_model_catalog_locates_embedded_entry(tmp_path: Path) -> None:
    session_file = build_basic_session(tmp_path)
    v = cc.load_best_snapshot(session_file)
    catalog = cc.find_model_catalog(v)
    assert catalog["claude-sonnet-5"] == {
        "input": 200,
        "output": 1000,
        "cache_read": 20,
        "cache_write": 250,
    }


def test_normalize_model_id_strips_copilot_prefix() -> None:
    assert cc.normalize_model_id("copilot/claude-sonnet-5") == "claude-sonnet-5"
    assert cc.normalize_model_id("claude-sonnet-5") == "claude-sonnet-5"
    assert cc.normalize_model_id("ollama-models/glm4:latest") == "ollama-models/glm4:latest"


def test_estimate_request_cost_catalog_path() -> None:
    catalog = {"claude-sonnet-5": {"input": 200, "output": 1000, "cache_read": 20, "cache_write": 250}}
    estimate = cc.estimate_request_cost(
        "copilot/claude-sonnet-5", prompt_tokens=45615, output_tokens=2088, catalog=catalog
    )
    assert estimate.source == "catalog"
    assert estimate.usd == pytest.approx(0.11211, rel=1e-4)


def test_estimate_request_cost_local_ollama_is_free() -> None:
    estimate = cc.estimate_request_cost(
        "ollama-models/glm4:latest", prompt_tokens=1000, output_tokens=50, catalog={}
    )
    assert estimate.source == "local_free"
    assert estimate.usd == 0.0


def test_estimate_request_cost_unpriced_is_never_silent_zero() -> None:
    estimate = cc.estimate_request_cost("gpt-unknown-model", prompt_tokens=100, output_tokens=10, catalog={})
    assert estimate.source == "unpriced"
    assert estimate.usd is None


def test_extract_text_from_value_walks_the_rich_text_ast() -> None:
    value = {
        "content": [
            {"value": {"node": {"children": [{"type": 2, "text": "first leaf"}, {"type": 2, "text": "second leaf"}]}}}
        ]
    }
    text = cc.extract_text_from_value(value)
    assert "first leaf" in text
    assert "second leaf" in text


def test_extract_request_usage_new_schema() -> None:
    req = file_read_request("req1")
    prompt_tokens, output_tokens, model_id = cc.extract_request_usage(req)
    assert (prompt_tokens, output_tokens) == (45615, 2088)
    assert model_id == "claude-sonnet-5"  # toolCallRounds[-1].modelId wins over agent.modelId


def test_extract_request_usage_old_schema_top_level_fields() -> None:
    req = old_schema_terminal_request("req2", prompt_tokens=5000, completion_tokens=300)
    prompt_tokens, output_tokens, model_id = cc.extract_request_usage(req)
    assert (prompt_tokens, output_tokens) == (5000, 300)
    assert model_id == "copilot/claude-sonnet-5"  # no toolCallRounds here -> falls back to agent.modelId


def test_extract_tool_calls_never_reads_thinking_block() -> None:
    huge_thinking = "LEAKED " * 20_000  # if this ever counted, the token total would be enormous
    req = file_read_request("req1", file_text="short file content", thinking_text=huge_thinking)
    observations = cc.extract_tool_calls(req)
    assert len(observations) == 1
    assert observations[0].bucket == "file_read"
    assert observations[0].tokens < 50  # bounded by "short file content", nowhere near huge_thinking's size


def test_extract_tool_calls_handles_standalone_terminal_kind() -> None:
    req = old_schema_terminal_request("req2", terminal_output="line one\nline two\n")
    observations = cc.extract_tool_calls(req)
    assert len(observations) == 1
    assert observations[0].bucket == "shell_log"
    assert observations[0].tokens > 0


def test_extract_tool_calls_grep_bucket() -> None:
    req = grep_request("req3")
    observations = cc.extract_tool_calls(req)
    assert len(observations) == 1
    assert observations[0].bucket == "grep"


def test_extract_tool_calls_terminal_dedupes_repeated_terminal_command_id() -> None:
    req = old_schema_terminal_request("req2")
    req["response"] = req["response"] * 2  # same terminalCommandId appearing twice
    observations = cc.extract_tool_calls(req)
    assert len(observations) == 1  # deduped, not double-counted


def test_census_session_end_to_end(tmp_path: Path) -> None:
    session_file = build_basic_session(tmp_path)
    v = cc.load_best_snapshot(session_file)
    result = cc.CensusResult()
    cc.census_session(v, result)

    assert result.total_requests == 4
    assert result.total_real_cost_usd == pytest.approx(0.11211 + 0.013, rel=1e-4)
    assert result.local_free_requests == 1
    assert result.unpriced_requests == 1
    assert set(result.bucket_stats) == {"file_read", "shell_log", "grep"}
    for stats in result.bucket_stats.values():
        assert stats.occurrences == 1
        assert stats.tokens > 0


def test_main_runs_against_a_real_directory_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_basic_session(tmp_path, workspace_hash="hash1")
    build_basic_session(tmp_path, workspace_hash="hash2", session_id="session2")

    # main() reads sys.argv directly; exercise it through the real CLI path.
    import sys

    old_argv = sys.argv
    sys.argv = ["census_copilot.py", "--root", str(tmp_path)]
    try:
        exit_code = cc.main()
    finally:
        sys.argv = old_argv

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Sessions scanned: 2" in captured.out
    assert "Total estimated real cost (USD)" in captured.out
