"""Session-aware /v1/compress served by several replicas sharing Redis.

A gateway without sticky sessions sends each turn to any replica. With
HEADROOM_COMPRESS_SESSION_STORE set, a replica that did not serve the last turn
takes the session's state from Redis, freezes the prefix the caller already
forwarded, and returns it byte for byte.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache import shared_sessions  # noqa: E402
from headroom.cache.backends.redis import RedisBackend  # noqa: E402
from headroom.cache.compression_store import _create_default_ccr_backend  # noqa: E402
from headroom.exceptions import ConfigurationError  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from tests.test_compress_session_mode import _big_tool_history, _compress  # noqa: E402


@pytest.fixture
def redis_server(monkeypatch):
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setenv(shared_sessions.STORE_ENV, "redis://127.0.0.1:6399/0")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")  # CCR sharing is tested on its own
    monkeypatch.setattr(
        redis.Redis,
        "from_url",
        classmethod(lambda cls, *a, **k: fakeredis.FakeRedis(server=server)),
    )
    return fakeredis.FakeRedis(server=server)


def _replica() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
    )
    return TestClient(create_app(config), base_url="http://127.0.0.1", client=("127.0.0.1", 1))


def _second_tool_result() -> list[dict]:
    rows = [
        {"id": i, "path": f"src/mod_{i}.py", "line": i * 3, "hit": "x" * 200} for i in range(150)
    ]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "grep", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": json.dumps(rows)},
    ]


def test_any_replica_serves_the_next_turn_byte_for_byte(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        turn1 = _compress(a, history, session_id="conv")
        assert turn1["messages"][2]["content"] != history[2]["content"]  # compressed

        # Turn 2 on B adds a second tool result, compressed by B.
        history2 = history + _second_tool_result()
        turn2 = _compress(b, history2, session_id="conv")
        assert turn2["messages"][: len(turn1["messages"])] == turn1["messages"]
        assert turn2["session"]["frozen_message_count"] >= len(turn1["messages"])
        assert turn2["messages"][4]["content"] != history2[4]["content"]

        # Back on A, which last saw turn 1: only B's state has B's bytes.
        history3 = history2 + [{"role": "user", "content": "again"}]
        turn3 = _compress(a, history3, session_id="conv")
        assert turn3["messages"][: len(turn2["messages"])] == turn2["messages"]
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_without_the_store_a_new_replica_starts_cold(monkeypatch):
    # What the store fixes: B never saw the compressed tool result, so it does
    # not freeze it and compresses it again.
    monkeypatch.delenv(shared_sessions.STORE_ENV, raising=False)
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        turn1 = _compress(a, history, session_id="conv")
        turn2 = _compress(b, history + [{"role": "user", "content": "x"}], session_id="conv")
        assert turn2["session"]["frozen_message_count"] < len(turn1["messages"])


def test_a_turn_writes_only_the_messages_redis_lacks(redis_server):
    history = _big_tool_history()
    with _replica() as a:
        _compress(a, history, session_id="conv-delta")
        (msgs_key,) = redis_server.keys("headroom:session:*:msgs")
        before = set(redis_server.hkeys(msgs_key))

        import redis

        written: list[int] = []
        real_hset = redis.client.Pipeline.hset

        def counting_hset(self, name, *args, mapping=None, **kwargs):
            written.append(len(mapping or {}))
            return real_hset(self, name, *args, mapping=mapping, **kwargs)

        redis.client.Pipeline.hset = counting_hset
        try:
            _compress(
                a, history + [{"role": "user", "content": "one more"}], session_id="conv-delta"
            )
        finally:
            redis.client.Pipeline.hset = real_hset
        after = set(redis_server.hkeys(msgs_key))
        # The new user message, and at most the few whose cache marker moved.
        assert 1 <= sum(written) <= 3 < len(before)
        assert before <= after


def test_a_lost_key_is_stored_whole_again(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        _compress(a, history, session_id="conv-lost")
        redis_server.flushall()
        turn2 = _compress(a, history + [{"role": "user", "content": "y"}], session_id="conv-lost")
        assert redis_server.keys("headroom:session:*:meta")
        # And another replica is warm from it.
        turn3 = _compress(
            b,
            history + [{"role": "user", "content": "y"}, {"role": "user", "content": "z"}],
            session_id="conv-lost",
        )
        assert turn3["messages"][: len(turn2["messages"])] == turn2["messages"]
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_the_usage_relay_reaches_and_publishes_a_session_served_elsewhere(redis_server):
    with _replica() as a, _replica() as b:
        _compress(a, _big_tool_history(), session_id="conv-usage")
        (meta_key,) = redis_server.keys("headroom:session:*:meta")
        token = redis_server.hget(meta_key, "t")
        resp = b.post(
            "/v1/usage",
            json={
                "session_id": "conv-usage",
                "usage": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 50_000},
            },
        )
        assert resp.status_code == 200, resp.text
        assert redis_server.hget(meta_key, "t") != token  # B published its update


def test_a_redis_outage_or_corrupt_state_never_fails_a_turn(redis_server, monkeypatch):
    import redis

    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        turn1 = _compress(a, history, session_id="conv-down")
        (meta_key,) = redis_server.keys("headroom:session:*:meta")
        redis_server.hset(meta_key, "m", "{not json")
        turn2 = _compress(b, history + [{"role": "user", "content": "y"}], session_id="conv-down")
        assert turn2["messages"][0] == turn1["messages"][0]

        def down(*args, **kwargs):
            raise redis.ConnectionError("connection refused")

        monkeypatch.setattr(fakeredis.FakeRedis, "execute_command", down)
        turn3 = _compress(a, history + [{"role": "user", "content": "z"}], session_id="conv-down")
        assert turn3["messages"][: len(turn1["messages"])] == turn1["messages"]


def test_sharing_sessions_shares_the_ccr_store_too(monkeypatch):
    monkeypatch.setenv(shared_sessions.STORE_ENV, "redis://127.0.0.1:6399/0")
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)
    assert isinstance(_create_default_ccr_backend(), RedisBackend)


def test_a_store_url_that_is_not_redis_stops_startup(monkeypatch):
    monkeypatch.setenv(shared_sessions.STORE_ENV, "memcached://cache:11211")
    with pytest.raises(ConfigurationError, match="redis://"):
        shared_sessions.open_shared_sessions()


def _meta_key(redis_server):
    (meta_key,) = redis_server.keys("headroom:session:*:meta")
    return meta_key


def test_messages_evicted_on_their_own_are_stored_whole_again(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        _compress(a, history, session_id="conv-evict")
        meta_key = _meta_key(redis_server)
        redis_server.delete(meta_key.replace(b":meta", b":msgs"))  # maxmemory took one key
        history2 = history + [{"role": "user", "content": "y"}]
        turn2 = _compress(a, history2, session_id="conv-evict")
        turn3 = _compress(b, history2 + [{"role": "user", "content": "z"}], session_id="conv-evict")
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_corrupt_state_is_rewritten_by_the_next_replica_to_write(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b, _replica() as c:
        _compress(a, history, session_id="conv-corrupt")
        redis_server.hset(_meta_key(redis_server), "m", "{not json")
        history2 = history + [{"role": "user", "content": "y"}]
        turn2 = _compress(b, history2, session_id="conv-corrupt")  # repairs it
        turn3 = _compress(
            c, history2 + [{"role": "user", "content": "z"}], session_id="conv-corrupt"
        )
        assert turn3["messages"][: len(turn2["messages"])] == turn2["messages"]
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_messages_no_transcript_references_are_pruned(redis_server):
    history = _big_tool_history()
    with _replica() as a:
        _compress(a, history, session_id="conv-prune")
        msgs_key = _meta_key(redis_server).replace(b":meta", b":msgs")
        redis_server.hset(msgs_key, mapping={f"stale{i}": "{}" for i in range(50)})
        _compress(a, history + [{"role": "user", "content": "y"}], session_id="conv-prune")
        assert not any(k.startswith(b"stale") for k in redis_server.hkeys(msgs_key))


def test_a_writer_whose_clock_runs_ahead_is_still_trusted(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        turn1 = _compress(a, history, session_id="conv-skew")
        meta_key = _meta_key(redis_server)
        meta = json.loads(redis_server.hget(meta_key, "m"))
        meta["last_activity"] += 30  # A's clock is 30 s ahead of B's
        redis_server.hset(meta_key, "m", json.dumps(meta))
        turn2 = _compress(b, history + [{"role": "user", "content": "y"}], session_id="conv-skew")
        assert turn2["session"]["frozen_message_count"] >= len(turn1["messages"])


def test_a_lost_message_is_noticed_even_beside_unreferenced_ones(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        _compress(a, history, session_id="conv-masked")
        msgs_key = _meta_key(redis_server).replace(b":meta", b":msgs")
        # Old variants left behind outnumber the one referenced message lost.
        redis_server.hset(msgs_key, mapping={f"old{i}": "{}" for i in range(5)})
        redis_server.hdel(msgs_key, redis_server.hkeys(msgs_key)[0])
        history2 = history + [{"role": "user", "content": "y"}]
        turn2 = _compress(a, history2, session_id="conv-masked")
        turn3 = _compress(
            b, history2 + [{"role": "user", "content": "z"}], session_id="conv-masked"
        )
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_a_corrupt_token_never_fails_a_turn_and_is_replaced(redis_server):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        _compress(a, history, session_id="conv-token")
        redis_server.hset(_meta_key(redis_server), "t", b"\xff")
        history2 = history + [{"role": "user", "content": "y"}]
        turn2 = _compress(b, history2, session_id="conv-token")
        turn3 = _compress(a, history2 + [{"role": "user", "content": "z"}], session_id="conv-token")
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


@pytest.fixture
def evict_before_cas(monkeypatch):
    """Delete the messages hash just before the next ``times`` compare-and-set
    scripts run, as maxmemory might between the message writes and the script."""
    budget = {"times": 0}
    real_init = shared_sessions.SharedSessions.__init__

    def init(self, client):
        real_init(self, client)
        cas = self._cas

        def evicting_cas(keys, args):
            if budget["times"]:
                budget["times"] -= 1
                client.delete(keys[1])
            return cas(keys=keys, args=args)

        self._cas = evicting_cas

    monkeypatch.setattr(shared_sessions.SharedSessions, "__init__", init)
    return budget


def test_eviction_before_the_first_publish_is_never_acknowledged(redis_server, evict_before_cas):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        evict_before_cas["times"] = 1
        turn1 = _compress(a, history, session_id="conv-evict-first")
        turn2 = _compress(
            b, history + [{"role": "user", "content": "y"}], session_id="conv-evict-first"
        )
        assert turn2["messages"][: len(turn1["messages"])] == turn1["messages"]
        assert turn2["session"]["frozen_message_count"] >= len(turn1["messages"])


@pytest.mark.parametrize("corrupt", [False, True], ids=["delta", "whole-rewrite"])
def test_eviction_before_a_later_publish_is_never_acknowledged(
    redis_server, evict_before_cas, corrupt
):
    history = _big_tool_history()
    with _replica() as a, _replica() as b, _replica() as c:
        _compress(a, history, session_id="conv-evict-later")
        if corrupt:  # B then rewrites the session whole, assuming no message stored
            redis_server.hset(_meta_key(redis_server), "m", "{not json")
        history2 = history + [{"role": "user", "content": "y"}]
        evict_before_cas["times"] = 1
        turn2 = _compress(b, history2, session_id="conv-evict-later")
        turn3 = _compress(
            c, history2 + [{"role": "user", "content": "z"}], session_id="conv-evict-later"
        )
        assert turn3["messages"][: len(turn2["messages"])] == turn2["messages"]
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])


def test_a_session_whose_messages_keep_vanishing_is_not_published(redis_server, evict_before_cas):
    history = _big_tool_history()
    with _replica() as a, _replica() as b:
        evict_before_cas["times"] = 2  # the write and its whole rewrite
        _compress(a, history, session_id="conv-evict-twice")
        assert not redis_server.keys("headroom:session:*:meta")
        # Once Redis keeps the messages again, the next turn publishes it whole.
        history2 = history + [{"role": "user", "content": "y"}]
        turn2 = _compress(a, history2, session_id="conv-evict-twice")
        turn3 = _compress(
            b, history2 + [{"role": "user", "content": "z"}], session_id="conv-evict-twice"
        )
        assert turn3["session"]["frozen_message_count"] >= len(turn2["messages"])
