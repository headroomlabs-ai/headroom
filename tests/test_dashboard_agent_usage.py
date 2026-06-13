from headroom.proxy.server import (
    _agent_label,
    _build_agent_usage_summary,
    _classify_agent_from_log,
    _normalize_agent_key,
)


def test_agent_usage_groups_exact_logged_requests_by_client() -> None:
    summary = _build_agent_usage_summary(
        [
            {
                "provider": "openai",
                "model": "gpt-5.2-codex",
                "tags": {"client": "codex"},
                "input_tokens_original": 1000,
                "input_tokens_optimized": 650,
                "output_tokens": 100,
                "tokens_saved": 350,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "tags": {"client": "claude-code"},
                "input_tokens_original": 800,
                "input_tokens_optimized": 500,
                "output_tokens": 80,
                "tokens_saved": 300,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "tags": {"client": "cursor"},
                "input_tokens_original": 500,
                "input_tokens_optimized": 400,
                "output_tokens": 60,
                "tokens_saved": 100,
            },
        ],
        requests_by_provider={},
        requests_by_model={},
        global_before_tokens=2300,
        global_after_tokens=1550,
        global_tokens_saved=750,
        global_output_tokens=240,
    )

    rows = {row["agent"]: row for row in summary["agents"]}

    assert rows["codex"]["label"] == "Codex"
    assert rows["codex"]["before_tokens"] == 1000
    assert rows["codex"]["after_tokens"] == 650
    assert rows["codex"]["tokens_saved"] == 350
    assert rows["codex"]["savings_percent"] == 35.0

    assert rows["claude-code"]["label"] == "Claude"
    assert rows["claude-code"]["savings_percent"] == 37.5

    assert rows["cursor"]["label"] == "Cursor"
    assert rows["cursor"]["share_of_saved_percent"] == 13.33

    assert summary["coverage"] == {
        "logged_requests": 3,
        "exact_token_rows": 3,
        "mode": "request_logs",
    }


def test_agent_usage_falls_back_to_inferred_model_counts_when_complete() -> None:
    summary = _build_agent_usage_summary(
        [],
        requests_by_provider={"anthropic": 2, "openai": 3},
        requests_by_model={"claude-sonnet-4-6": 2, "gpt-5.2-codex": 3},
        global_before_tokens=1000,
        global_after_tokens=700,
        global_tokens_saved=300,
        global_output_tokens=90,
    )

    rows = {row["agent"]: row for row in summary["agents"]}

    assert set(rows) == {"claude-code", "codex"}
    assert rows["claude-code"]["label"] == "Claude"
    assert rows["claude-code"]["source"] == "model"
    assert rows["claude-code"]["requests"] == 2
    assert rows["claude-code"]["models"] == {"claude-sonnet-4-6": 2}
    assert rows["codex"]["label"] == "Codex"
    assert rows["codex"]["requests"] == 3
    assert rows["codex"]["models"] == {"gpt-5.2-codex": 3}
    assert summary["totals"]["savings_percent"] == 30.0
    assert summary["coverage"]["mode"] == "aggregate_fallback"


def test_agent_usage_fallback_does_not_duplicate_provider_and_model_rows() -> None:
    summary = _build_agent_usage_summary(
        [],
        requests_by_provider={"anthropic": 2, "openai": 3},
        requests_by_model={"claude-sonnet-4-6": 2, "gpt-5.2-codex": 3},
        global_before_tokens=1000,
        global_after_tokens=700,
        global_tokens_saved=300,
        global_output_tokens=90,
    )

    rows = {row["agent"]: row for row in summary["agents"]}

    assert set(rows) == {"claude-code", "codex"}
    assert all(row["requests"] > 0 for row in rows.values())
    assert summary["totals"]["requests"] == 5


def test_agent_usage_skips_partial_model_fallback_counts() -> None:
    summary = _build_agent_usage_summary(
        [],
        requests_by_provider={"anthropic": 2, "openai": 3},
        requests_by_model={"claude-sonnet-4-6": 2},
        global_before_tokens=1000,
        global_after_tokens=700,
        global_tokens_saved=300,
        global_output_tokens=90,
    )

    rows = {row["agent"]: row for row in summary["agents"]}

    assert set(rows) == {"anthropic", "openai"}
    assert rows["anthropic"]["label"] == "Claude"
    assert rows["anthropic"]["requests"] == 2
    assert rows["openai"]["label"] == "OpenAI"
    assert rows["openai"]["requests"] == 3


def test_agent_classifier_uses_model_before_generic_provider() -> None:
    agent, label, source = _classify_agent_from_log(
        {
            "provider": "openai",
            "model": "gpt-5.2-codex",
            "tags": {},
        }
    )

    assert (agent, label, source) == ("codex", "Codex", "model")


def test_agent_usage_upgrades_source_when_stronger_evidence_arrives() -> None:
    summary = _build_agent_usage_summary(
        [
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "tags": {},
                "input_tokens_original": 10,
                "input_tokens_optimized": 8,
                "tokens_saved": 2,
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "tags": {"client": "claude-code"},
                "input_tokens_original": 20,
                "input_tokens_optimized": 12,
                "tokens_saved": 8,
            },
        ],
        requests_by_provider={},
        requests_by_model={},
        global_before_tokens=30,
        global_after_tokens=20,
        global_tokens_saved=10,
        global_output_tokens=0,
    )

    row = summary["agents"][0]

    assert row["agent"] == "claude-code"
    assert row["source"] == "client"
    assert row["requests"] == 2


def test_agent_key_normalizes_wrapped_underscore_clients() -> None:
    assert _normalize_agent_key("wrap_claude_cli") == "claude-code"


def test_agent_key_normalizes_claude_code_cli_alias() -> None:
    assert _normalize_agent_key("claude-code-cli") == "claude-code"


def test_agent_label_title_cases_unknown_agent_key() -> None:
    assert _agent_label("custom-agent") == "Custom Agent"


def test_agent_classifier_uses_stack_tag_before_model() -> None:
    agent, label, source = _classify_agent_from_log(
        {
            "provider": "openai",
            "model": "gpt-5.2-codex",
            "tags": {"headroom-stack": "openclaw"},
        }
    )

    assert (agent, label, source) == ("openclaw", "OpenClaw", "stack")


def test_agent_classifier_falls_back_to_unknown() -> None:
    agent, label, source = _classify_agent_from_log(
        {
            "provider": "",
            "model": "",
            "tags": [],
        }
    )

    assert (agent, label, source) == ("unknown", "Unidentified", "unknown")


def test_agent_usage_recovers_before_tokens_from_after_and_saved() -> None:
    summary = _build_agent_usage_summary(
        [
            {
                "provider": "openai",
                "model": "custom-model",
                "tags": {"client": "custom-agent"},
                "input_tokens_original": 0,
                "input_tokens_optimized": 70,
                "output_tokens": 5,
                "tokens_saved": 30,
            }
        ],
        requests_by_provider={},
        requests_by_model={},
        global_before_tokens=100,
        global_after_tokens=70,
        global_tokens_saved=0,
        global_output_tokens=5,
    )

    row = summary["agents"][0]

    assert row["agent"] == "custom-agent"
    assert row["label"] == "Custom Agent"
    assert row["before_tokens"] == 100
    assert row["savings_percent"] == 30.0
    assert row["after_percent"] == 70.0
    assert row["share_of_saved_percent"] == 0.0
    assert summary["totals"]["savings_percent"] == 0.0


def test_agent_usage_clamps_negative_token_values() -> None:
    summary = _build_agent_usage_summary(
        [
            {
                "provider": None,
                "model": None,
                "tags": {},
                "input_tokens_original": -100,
                "input_tokens_optimized": -50,
                "output_tokens": -5,
                "tokens_saved": -25,
            }
        ],
        requests_by_provider={},
        requests_by_model={},
        global_before_tokens=0,
        global_after_tokens=0,
        global_tokens_saved=0,
        global_output_tokens=0,
    )

    row = summary["agents"][0]

    assert row["agent"] == "unknown"
    assert row["requests"] == 1
    assert row["before_tokens"] == 0
    assert row["after_tokens"] == 0
    assert row["tokens_saved"] == 0
    assert row["output_tokens"] == 0
    assert row["has_exact_tokens"] is False
