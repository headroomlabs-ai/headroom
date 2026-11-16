"""Exercise the dependency's real recent-conversation storage boundary."""

import os
import tempfile
from importlib.metadata import version

os.environ["MEM0_TELEMETRY"] = "false"
from mem0.memory.main import _build_session_scope
from mem0.memory.storage import SQLiteManager

first = {"agent_id": "alpha&user_id=beta", "user_id": "gamma"}
second = {"agent_id": "alpha", "user_id": "beta&user_id=gamma"}
first_key = _build_session_scope(first)
second_key = _build_session_scope(second)
assert (
    _build_session_scope({"agent_id": "agent", "user_id": "plain"})
    == "agent_id=agent&user_id=plain"
)
with tempfile.TemporaryDirectory(prefix="headroom-mem0-tenant-") as directory:
    storage = SQLiteManager(directory + "/history.db")
    try:
        storage.save_messages(
            [{"role": "user", "content": "first tenant private conversation"}], first_key
        )
        other_messages = storage.get_last_messages(second_key)
        print("mem0ai=" + version("mem0ai"), flush=True)
        print(
            "distinct identity tuples share scope key=" + str(first_key == second_key), flush=True
        )
        print("other tenant retrieved messages=" + str(len(other_messages)), flush=True)
        assert first_key != second_key
        assert other_messages == []
        assert len(storage.get_last_messages(first_key)) == 1
        print("PASS: separate tenants cannot retrieve the same recent-conversation buffer")
    finally:
        storage.connection.close()
