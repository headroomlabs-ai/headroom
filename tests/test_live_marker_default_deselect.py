"""The default pytest run must deselect tests marked ``live`` (they call real
provider APIs with the developer's keys) and ``-m live`` must still select them.

Collection only: nothing here contacts a provider.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_FILE = "tests/test_thinking_signature_scope_live.py"
MIXED_FILE = "tests/test_realignment_live_multi_turn.py"
OFFLINE_TEST = "test_classify_auth_mode_routes_payg_vs_oauth"


def _collect(*args: str) -> str:
    env = {**os.environ, "ANTHROPIC_API_KEY": "sk-ant-dummy-collect-only"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.stdout


def test_default_run_deselects_live_tests() -> None:
    out = _collect(LIVE_FILE, MIXED_FILE)
    assert "test_exact_replay_is_accepted" not in out
    assert "test_anthropic_cache_hit_across_two_turns" not in out


def test_default_run_keeps_the_offline_test_in_a_live_file() -> None:
    assert OFFLINE_TEST in _collect(MIXED_FILE)


def test_dash_m_live_selects_live_tests_again() -> None:
    out = _collect("-m", "live", LIVE_FILE, MIXED_FILE)
    assert "test_exact_replay_is_accepted" in out
    assert "test_anthropic_cache_hit_across_two_turns" in out
    assert OFFLINE_TEST not in out
