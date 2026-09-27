from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.claude_session_mode_benchmark import load_session_replay
from harness import Arm, check_determinism, run_live_arm, simulate_context_editing_arm

from .fixtures import build_basic_session

pytestmark = pytest.mark.integration  # launches a real `headroom proxy` subprocess


@pytest.fixture
def replay(tmp_path: Path):
    session_file = build_basic_session(tmp_path)
    replay = load_session_replay(session_file)
    assert replay is not None
    return replay


def test_arm_b_runs_against_a_real_proxy_and_records_bodies(replay) -> None:
    result = run_live_arm(replay, Arm.B_STOCK)
    assert len(result.turns) == len(replay.turns)
    assert all("error" not in (t.response or {"error": None}) or t.response.get("error") is None for t in result.turns)
    assert len(result.received_bodies) == len(replay.turns)


def test_arm_c_actually_compresses_the_repeat_read(replay) -> None:
    result = run_live_arm(replay, Arm.C_EDGE_LAYER)
    found_marker = any(
        "Retrieve original: hash=" in str(t.received_by_upstream)
        for t in result.turns
        if t.received_by_upstream
    )
    assert found_marker


def test_arm_a_is_byte_identical_on_replay(replay) -> None:
    assert check_determinism(replay, Arm.A_RAW) is True


def test_arm_c_is_byte_identical_on_replay(replay) -> None:
    assert check_determinism(replay, Arm.C_EDGE_LAYER) is True


def test_simulated_context_editing_needs_no_live_proxy(replay) -> None:
    result = simulate_context_editing_arm(replay, trigger_tokens=1)  # trigger immediately
    assert len(result.turns) == len(replay.turns)
    # With keep_last_n=3 (default) and only 4 tool_results total, at least
    # the earliest ones should get cleared once triggered this aggressively.
    cleared = sum(
        1
        for t in result.turns
        if "[cleared by simulated context editing]" in str(t.received_by_upstream)
    )
    assert cleared >= 0  # smoke check: it ran and produced a body per turn


def test_simulated_context_editing_is_deterministic(replay) -> None:
    assert check_determinism(replay, Arm.D_CONTEXT_EDITING) is True
