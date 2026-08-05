"""SQLiteMemoryStore._build_query_conditions scope filtering.

`_build_query_conditions` only reads the filter, so it is exercised directly via
``object.__new__`` (no DB, no embedder).
"""

from __future__ import annotations

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.ports import MemoryFilter


def _conditions(**kwargs) -> tuple[list[str], list]:
    store = object.__new__(SQLiteMemoryStore)
    return store._build_query_conditions(MemoryFilter(**kwargs))


def test_turn_id_is_applied_without_agent_id():
    """A (user, session, turn) filter without agent_id must still narrow to the
    turn — previously the turn_id condition was nested inside the agent_id block
    and silently dropped, returning the whole session."""
    conditions, params = _conditions(user_id="u", session_id="s", turn_id="t")

    assert "turn_id = ?" in conditions
    assert "t" in params


def test_agent_id_and_turn_id_both_applied():
    conditions, params = _conditions(user_id="u", session_id="s", agent_id="a", turn_id="t")

    assert "agent_id = ?" in conditions
    assert "turn_id = ?" in conditions
    assert "a" in params and "t" in params


def test_agent_id_only_still_applied():
    conditions, _ = _conditions(user_id="u", session_id="s", agent_id="a")

    assert "agent_id = ?" in conditions
    assert "turn_id = ?" not in conditions
