"""Tests for Claude session mode simulation benchmark."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from benchmarks.claude_session_mode_benchmark import (
    PROXY_MODE_CACHE,
    PROXY_MODE_TOKEN,
    ModeSummary,
    ReplayTurn,
    SessionReplay,
    _extract_cache_stable_last_message_suffix,
    _merge_appended_message_delta,
    _rewrite_scope,
    _write_checkpoint_by_session_id,
    build_dataset_and_observed_from_files,
    classify_metric_impact,
    decode_project_key,
    determine_winners,
    load_session_replay,
    resolve_checkpoint_dir,
    simulate_replays,
    summarize_mode_impact_vs_baseline,
    summarize_observed_usage,
    trim_replay_to_recent_turns,
)


def test_decode_project_key_windows_path() -> None:
    assert decode_project_key("C--git-BetBlocker") == r"C:\git\BetBlocker"


def test_load_session_replay_groups_assistant_request_events(tmp_path: Path) -> None:
    project_dir = tmp_path / "C--git-BetBlocker"
    project_dir.mkdir()
    session_file = project_dir / "sess-1.jsonl"
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2026-03-13T01:00:00Z",
        },
        {
            "type": "assistant",
            "requestId": "req-1",
            "timestamp": "2026-03-13T01:00:01Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "thinking", "thinking": "..."}],
                "usage": {"output_tokens": 2},
            },
        },
        {
            "type": "assistant",
            "requestId": "req-1",
            "timestamp": "2026-03-13T01:00:02Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {"output_tokens": 5},
            },
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "Next"},
            "timestamp": "2026-03-13T01:01:00Z",
        },
        {
            "type": "assistant",
            "requestId": "req-2",
            "timestamp": "2026-03-13T01:01:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Done"}],
                "usage": {"output_tokens": 3},
            },
        },
    ]
    session_file.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    replay = load_session_replay(session_file)

    assert replay is not None
    assert len(replay.turns) == 2
    assert replay.turns[0].request_id == "req-1"
    assert replay.turns[0].output_tokens == 5
    assert replay.turns[0].input_messages == [{"role": "user", "content": "Hello"}]
    assert replay.turns[1].input_messages == [{"role": "user", "content": "Next"}]
    assert replay.turns[1].assistant_message["content"] == [{"type": "text", "text": "Done"}]


def test_simulation_and_winner_logic() -> None:
    tool_blob = '{"rows":[1,2,3,4]}' * 80
    turn1 = ReplayTurn(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        request_id="r1",
        model="claude-sonnet-4-6",
        timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
        input_messages=[
            {"role": "user", "content": "Summarize this JSON"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": tool_blob,
                    }
                ],
            },
        ],
        assistant_message={"role": "assistant", "content": "ok"},
        output_tokens=20,
    )
    turn2 = ReplayTurn(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        request_id="r2",
        model="claude-sonnet-4-6",
        timestamp=datetime.fromisoformat("2026-03-13T01:03:00+00:00"),
        input_messages=[
            {"role": "user", "content": "Now tell me the anomalies again"},
        ],
        assistant_message={"role": "assistant", "content": "ok2"},
        output_tokens=25,
    )
    replay = SessionReplay(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        turns=[turn1, turn2],
    )

    dataset, summaries = simulate_replays([replay], cache_ttl_minutes=5)

    assert dataset.requests == 2
    assert summaries["baseline"].raw_input_tokens > 0
    assert (
        summaries[PROXY_MODE_TOKEN].forwarded_input_tokens
        <= summaries["baseline"].forwarded_input_tokens
    )
    assert summaries[PROXY_MODE_CACHE].cache_read_tokens >= 0
    assert summaries["baseline"].cache_bust_turns == 0
    assert summaries[PROXY_MODE_CACHE].cache_bust_turns == 0
    assert summaries[PROXY_MODE_TOKEN].cache_bust_turns >= 0
    assert summaries[PROXY_MODE_TOKEN].rewrite_turns >= 0
    assert summaries[PROXY_MODE_CACHE].rewrite_turns >= 0

    winners = determine_winners(summaries)
    assert winners["total_cost"] in {"baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE}
    assert winners["window_with_cache"] in {"baseline", PROXY_MODE_TOKEN, PROXY_MODE_CACHE}


def test_observed_usage_summary_tracks_cache_patterns() -> None:
    turns = [
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r1",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
            input_messages=[{"role": "user", "content": "a"}],
            assistant_message={"role": "assistant", "content": "x"},
            output_tokens=5,
            observed_input_tokens=10,
            observed_cache_read_tokens=0,
            observed_cache_write_tokens=100,
        ),
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r2",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:01:00+00:00"),
            input_messages=[{"role": "user", "content": "b"}],
            assistant_message={"role": "assistant", "content": "y"},
            output_tokens=6,
            observed_input_tokens=9,
            observed_cache_read_tokens=80,
            observed_cache_write_tokens=90,
        ),
        ReplayTurn(
            session_id="s1",
            project_key="C--git-demo",
            decoded_project_path=r"C:\git\demo",
            request_id="r3",
            model="claude-sonnet-4-6",
            timestamp=datetime.fromisoformat("2026-03-13T01:02:00+00:00"),
            input_messages=[{"role": "user", "content": "c"}],
            assistant_message={"role": "assistant", "content": "z"},
            output_tokens=7,
            observed_input_tokens=9,
            observed_cache_read_tokens=80,
            observed_cache_write_tokens=120,
        ),
    ]
    replay = SessionReplay(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        turns=turns,
    )

    observed = summarize_observed_usage([replay])

    assert observed.requests == 3
    assert observed.cache_read_tokens == 160
    assert observed.cache_write_tokens == 310
    assert observed.healthy_growth_turns == 1
    assert observed.broken_prefix_turns == 2


def test_checkpoint_write_omits_per_turn_payload(tmp_path: Path) -> None:
    summary = ModeSummary(
        mode=PROXY_MODE_TOKEN,
        sessions=1,
        requests=1,
        turns=[],
    )

    _write_checkpoint_by_session_id(tmp_path, PROXY_MODE_TOKEN, "session-1", summary)

    payload = json.loads((tmp_path / f"{PROXY_MODE_TOKEN}--session-1.json").read_text())
    assert payload["turns"] == []


def test_trim_replay_to_recent_turns_keeps_latest_slice() -> None:
    replay = SessionReplay(
        session_id="s1",
        project_key="C--git-demo",
        decoded_project_path=r"C:\git\demo",
        turns=[
            ReplayTurn(
                session_id="s1",
                project_key="C--git-demo",
                decoded_project_path=r"C:\git\demo",
                request_id=f"r{i}",
                model="claude-sonnet-4-6",
                timestamp=datetime.fromisoformat(f"2026-03-13T01:0{i}:00+00:00"),
                input_messages=[{"role": "user", "content": str(i)}],
                assistant_message={"role": "assistant", "content": str(i)},
                output_tokens=i,
            )
            for i in range(4)
        ],
    )

    trimmed = trim_replay_to_recent_turns(replay, 2)

    assert [turn.request_id for turn in trimmed.turns] == ["r2", "r3"]


def test_build_dataset_and_observed_from_files_applies_recent_turn_sampling(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "C--git-BetBlocker"
    project_dir.mkdir()
    session_file = project_dir / "sess-1.jsonl"
    lines = []
    for i in range(3):
        lines.append(
            {
                "type": "user",
                "message": {"role": "user", "content": f"Hello {i}"},
                "timestamp": f"2026-03-13T01:0{i}:00Z",
            }
        )
        lines.append(
            {
                "type": "assistant",
                "requestId": f"req-{i}",
                "timestamp": f"2026-03-13T01:0{i}:01Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": f"Hi {i}"}],
                    "usage": {
                        "output_tokens": 3,
                        "input_tokens": 10,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 5,
                    },
                },
            }
        )
    session_file.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    dataset, observed = build_dataset_and_observed_from_files(
        [session_file],
        recent_turns_per_session=2,
    )

    assert dataset.requests == 2
    assert dataset.sampled_requests == 2
    assert dataset.sampling_note == "Most recent 2 turns per session"
    assert observed.requests == 2


def test_determine_winners_includes_no_cache_counterfactual() -> None:
    summaries = {
        "baseline": ModeSummary(
            mode="baseline",
            paid_input_cost_usd=1.0,
            cache_read_cost_usd=0.2,
            paid_output_cost_usd=0.5,
        ),
        PROXY_MODE_TOKEN: ModeSummary(
            mode=PROXY_MODE_TOKEN,
            paid_input_cost_usd=0.8,
            cache_read_cost_usd=0.1,
            paid_output_cost_usd=0.5,
        ),
        PROXY_MODE_CACHE: ModeSummary(
            mode=PROXY_MODE_CACHE,
            paid_input_cost_usd=0.7,
            cache_read_cost_usd=0.3,
            paid_output_cost_usd=0.5,
        ),
    }

    winners = determine_winners(summaries)

    assert winners["no_cache_total_cost"] == PROXY_MODE_TOKEN


def test_resolve_checkpoint_dir_namespaces_sampling_mode() -> None:
    base = Path("benchmark_results") / "checkpoints"

    assert resolve_checkpoint_dir(base).name == "v5__ttl_5m__full"
    assert (
        resolve_checkpoint_dir(base, recent_turns_per_session=200).name == "v5__ttl_5m__recent_200"
    )


def test_cache_suffix_helpers_support_append_only_text_growth() -> None:
    suffix_delta = _extract_cache_stable_last_message_suffix(
        [{"role": "user", "content": "prefix + raw suffix"}],
        [{"role": "user", "content": "prefix"}],
        [{"role": "user", "content": "COMPRESSED_PREFIX"}],
    )

    assert suffix_delta is not None
    stable_prefix, stable_last_message, delta_messages = suffix_delta
    assert stable_prefix == []
    assert stable_last_message == {"role": "user", "content": "COMPRESSED_PREFIX"}
    assert delta_messages == [{"role": "user", "content": " + raw suffix"}]

    merged = _merge_appended_message_delta(
        stable_last_message,
        {"role": "user", "content": " + COMPRESSED_SUFFIX"},
    )
    assert merged == {"role": "user", "content": "COMPRESSED_PREFIX + COMPRESSED_SUFFIX"}


def test_mode_impact_classification_marks_assist_harm_and_no_change() -> None:
    baseline = ModeSummary(
        mode="baseline",
        forwarded_input_tokens=100,
        cache_read_tokens=50,
        cache_write_tokens=10,
        regular_input_tokens=40,
        output_tokens=5,
        total_cost_usd=1.0,
    )
    token = ModeSummary(
        mode=PROXY_MODE_TOKEN,
        forwarded_input_tokens=80,
        cache_read_tokens=70,
        cache_write_tokens=8,
        regular_input_tokens=30,
        output_tokens=5,
        total_cost_usd=0.8,
    )
    cache = ModeSummary(
        mode=PROXY_MODE_CACHE,
        forwarded_input_tokens=120,
        cache_read_tokens=45,
        cache_write_tokens=15,
        regular_input_tokens=60,
        output_tokens=5,
        total_cost_usd=1.2,
    )

    assert classify_metric_impact(baseline, token, "forwarded_input_tokens")["impact"] == "assist"
    assert classify_metric_impact(baseline, token, "cache_read_tokens")["impact"] == "assist"
    assert classify_metric_impact(baseline, cache, "total_cost_usd")["impact"] == "harm"
    assert classify_metric_impact(baseline, token, "output_tokens")["impact"] == "no_change"

    impacts = summarize_mode_impact_vs_baseline(
        {"baseline": baseline, PROXY_MODE_TOKEN: token, PROXY_MODE_CACHE: cache}
    )
    assert impacts[PROXY_MODE_TOKEN]["total_cost_usd"]["impact"] == "assist"
    assert impacts[PROXY_MODE_CACHE]["cache_write_tokens"]["impact"] == "harm"


def test_rewrite_scope_distinguishes_retroactive_from_latest_turn_only() -> None:
    rewrite, retroactive = _rewrite_scope(
        [{"role": "user", "content": "prefix"}, {"role": "user", "content": "new raw"}],
        [{"role": "user", "content": "prefix"}, {"role": "user", "content": "new compressed"}],
        stable_prefix_message_count=1,
    )
    assert rewrite is True
    assert retroactive is False

    rewrite, retroactive = _rewrite_scope(
        [{"role": "user", "content": "prefix"}, {"role": "user", "content": "new raw"}],
        [
            {"role": "user", "content": "compressed prefix"},
            {"role": "user", "content": "new compressed"},
        ],
        stable_prefix_message_count=1,
    )
    assert rewrite is True
    assert retroactive is True


def test_synthetic_token_mode_busts_cache_while_cache_mode_stays_stable(monkeypatch) -> None:
    class _FakeProvider:
        @staticmethod
        def get_context_limit(model: str) -> int:
            return 200_000

    class _FakePipeline:
        @staticmethod
        def apply(messages, **kwargs):  # noqa: ANN001
            rewritten = []
            should_rewrite_history = len(messages) > 2
            for message in messages:
                content = message.get("content")
                if (
                    should_rewrite_history
                    and isinstance(content, list)
                    and any(
                        isinstance(block, dict) and block.get("type") == "tool_result"
                        for block in content
                    )
                ):
                    new_blocks = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            new_blocks.append({**block, "content": "[compressed-tool-result]"})
                        else:
                            new_blocks.append(block)
                    rewritten.append({**message, "content": new_blocks})
                else:
                    rewritten.append(message)
            return SimpleNamespace(messages=rewritten)

    class _FakeProxy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(image_optimize=False)
            self.anthropic_provider = _FakeProvider()
            self.anthropic_pipeline = _FakePipeline()

    monkeypatch.setattr(
        "benchmarks.claude_session_mode_benchmark._make_proxy",
        lambda mode: _FakeProxy(),
    )

    tool_blob = "X" * 800
    replay = SessionReplay(
        session_id="synth-bust",
        project_key="C--git-synth",
        decoded_project_path=r"C:\git\synth",
        turns=[
            ReplayTurn(
                session_id="synth-bust",
                project_key="C--git-synth",
                decoded_project_path=r"C:\git\synth",
                request_id="r1",
                model="claude-sonnet-4-6",
                timestamp=datetime.fromisoformat("2026-03-13T01:00:00+00:00"),
                input_messages=[
                    {"role": "user", "content": "Summarize this tool output"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": tool_blob,
                            }
                        ],
                    },
                ],
                assistant_message={"role": "assistant", "content": "ok"},
                output_tokens=10,
            ),
            ReplayTurn(
                session_id="synth-bust",
                project_key="C--git-synth",
                decoded_project_path=r"C:\git\synth",
                request_id="r2",
                model="claude-sonnet-4-6",
                timestamp=datetime.fromisoformat("2026-03-13T01:02:00+00:00"),
                input_messages=[{"role": "user", "content": "What changed?"}],
                assistant_message={"role": "assistant", "content": "done"},
                output_tokens=12,
            ),
        ],
    )

    _, summaries = simulate_replays([replay], cache_ttl_minutes=5)

    token = summaries[PROXY_MODE_TOKEN]
    cache = summaries[PROXY_MODE_CACHE]

    assert token.cache_bust_turns == 1
    assert token.rewrite_turns >= 1
    assert token.busting_rewrite_turns >= 1
    assert token.non_cache_eligible_rewrite_turns == 0
    assert token.stable_replay_rewrite_turns == 0
    assert token.retroactive_rewrite_turns >= 1
    assert cache.cache_bust_turns == 0
    assert cache.busting_rewrite_turns == 0
    assert cache.non_cache_eligible_rewrite_turns == 0
    assert cache.retroactive_rewrite_turns == 0
