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


def test_metadata_scalar_filters_bind_native_values():
    """A metadata filter on a numeric/boolean value must bind the NATIVE value.

    ``json_extract`` returns a native SQLite value, so binding ``json.dumps(value)``
    ("5", "true") compared the column against text and matched nothing (SQLite
    never equates ``5 = '5'``). Scalars must be bound as-is; only non-scalar
    (dict/list) values keep the JSON-text form.
    """
    conditions, params = _conditions(
        user_id="u",
        metadata_filters={"priority": 5, "archived": True, "score": 1.5, "tag": "x"},
    )

    assert any("json_extract(metadata, '$.priority')" in c for c in conditions)
    # Native scalars, not their json.dumps text forms.
    assert 5 in params and "5" not in params
    assert 1.5 in params
    assert "x" in params
    # bool binds as its native value (a JSON ``true`` extracts to 1).
    assert True in params
    assert "true" not in params


def test_metadata_non_scalar_filter_keeps_json_text():
    """A dict/list metadata value is not a bindable SQLite type, so it keeps the
    JSON-text comparison rather than raising when the query runs."""
    import json

    conditions, params = _conditions(user_id="u", metadata_filters={"tags": ["a", "b"]})

    assert any("json_extract(metadata, '$.tags')" in c for c in conditions)
    assert json.dumps(["a", "b"]) in params
