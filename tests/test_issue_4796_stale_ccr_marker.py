"""A cached result whose CCR original is gone must not be served (#4796).

The result cache outlives the CCR entries its markers point at: the store
evicts by capacity and expires on its own TTL, while a re-compression of the
same content hashes to the SAME cache key. So a cached marker could point at a
deleted original for the rest of the process, and every turn handed the model a
pointer that 404s on retrieve. Serving it is worse than a cache miss.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import (
    CompressionStore,
    clear_request_compression_store,
    set_request_compression_store,
)
from headroom.exceptions import StorageError
from headroom.transforms.content_router import (
    CompressionCache,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _ccr_marker_hashes,
)

MARKER_RE = re.compile(r"<<ccr:([0-9a-f]{24})>>")


@pytest.fixture
def ccr_store():
    """A small request-scoped store, so capacity eviction is easy to trigger."""
    store = CompressionStore(max_entries=2, backend=InMemoryBackend())
    set_request_compression_store(store)
    try:
        yield store
    finally:
        clear_request_compression_store()


@pytest.mark.parametrize(
    "marker",
    [
        "<<ccr:{hash}>>",
        "<<ccr:{hash},base64,2.0KB>>",
        "<<ccr:{hash} 12_rows_offloaded>>",
        "[80 lines compressed to 8. Retrieve more: hash={hash}]",
        "[128 tokens compressed. Retrieve original: hash={hash}]",
    ],
)
def test_a_cached_marker_is_a_miss_once_its_original_is_gone(ccr_store, marker):
    hash_key = ccr_store.store("the original tool output", "the compressed view")
    cache = CompressionCache()
    cache.put("k", marker.format(hash=hash_key), 0.2, "x")
    assert cache.get("k") is not None, "a live marker is still served"

    ccr_store.store("an unrelated tool output", "its compressed view")
    ccr_store.store("a third tool output", "its compressed view")  # over capacity
    assert ccr_store.exists(hash_key) is False
    assert cache.get("k") is None
    assert cache.stats["cache_evictions"] == 1


def test_a_marker_free_result_is_never_checked_against_the_store(ccr_store, monkeypatch):
    cache = CompressionCache()
    cache.put("k", "no pointer here", 0.5, "kompress")

    def explode():
        raise AssertionError("marker-free results must not touch the CCR store")

    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", explode)
    assert cache.get("k") == ("no pointer here", 0.5, "kompress")


def test_a_check_that_cannot_run_keeps_serving_the_cached_result(monkeypatch):
    # Fail open: a store that is absent or failing is not evidence the original
    # is gone, and a miss would only cost another compression.
    cache = CompressionCache()
    cache.put("k", "head <<ccr:0123456789abcdef>>", 0.2, "x")

    def explode():
        raise RuntimeError("ccr store unavailable")

    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", explode)
    assert cache.get("k") == ("head <<ccr:0123456789abcdef>>", 0.2, "x")


def test_an_outage_is_not_an_eviction(ccr_store, monkeypatch):
    """A store that cannot answer must not cost us a still-valid result.

    Both shared backends report a failed read as "not found", so a liveness
    check on the return value alone would evict a cached result during a
    Redis or SQLite outage — and the original is still there afterwards.
    """
    hash_key = ccr_store.store("the original tool output", "the compressed view")
    cache = CompressionCache()
    cache.put("k", f"head <<ccr:{hash_key}>>", 0.2, "x")
    assert cache.get("k") is not None

    monkeypatch.setattr(ccr_store, "all_live", lambda hash_keys: None)
    assert cache.get("k") is not None, "an unanswered check is not a miss"
    assert cache.stats["cache_evictions"] == 0
    assert ccr_store.exists(hash_key), "the original is untouched"

    # The store recovers and the entry is genuinely gone: now it is a miss.
    monkeypatch.undo()
    ccr_store.store("an unrelated tool output", "its compressed view")
    ccr_store.store("a third tool output", "its compressed view")
    assert cache.get("k") is None
    assert cache.stats["cache_evictions"] == 1


def test_all_live_reports_a_backend_that_cannot_answer():
    class Unavailable(InMemoryBackend):
        def get(self, hash_key):
            return None

        def available(self):
            return False

    store = CompressionStore(max_entries=4, backend=Unavailable())
    assert store.all_live({"abc"}) is None

    class Reachable(InMemoryBackend):
        def available(self):
            return True

    live = CompressionStore(max_entries=4, backend=Reachable())
    assert live.all_live({"abc"}) is False
    hash_key = live.store("an original", "a compressed view")
    assert live.all_live({hash_key}) is True


def test_all_live_carries_a_probe_that_cannot_answer():
    """A key-only probe that fails answers unknown, not a miss.

    A backend may keep a usable handle while the probe itself fails, so
    ``available()`` cannot be the only signal: the probe's own ``None`` has to
    survive ``all_live`` even when the backend still looks reachable.
    """

    class FlakyProbe(InMemoryBackend):
        def live(self, hash_key):
            return None

        def available(self):
            return True

    store = CompressionStore(max_entries=4, backend=FlakyProbe())
    hash_key = store.store("an original", "a compressed view")
    assert store.all_live({hash_key}) is None


def test_sqlite_probe_failure_reads_as_unknown_not_evicted(tmp_path):
    """A failed SQLite read must not cost the router its cached result.

    Regression for the reproduced read failure: the authorizer denies the
    probe's SELECT while the connection stays open, so ``available()`` is
    still True. Reading that as an eviction dropped a result whose original was
    readable again once the denial lifted.
    """
    sqlite3 = pytest.importorskip("sqlite3")
    from headroom.cache.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "store.db")
    store = CompressionStore(max_entries=4, backend=backend)
    hash_key = store.store("the original tool output", "the compressed view")

    assert store.all_live([hash_key]) is True
    assert backend.available() is True

    backend._conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
    try:
        result = store.all_live([hash_key])
        still_open = backend.available()
    finally:
        backend._conn.set_authorizer(None)

    assert still_open is True, "the handle survives a transient read failure"
    assert store.exists(hash_key), "the original survived the failed probe"
    assert result is None, "a failed probe is unknown, not an eviction"
    assert store.all_live([hash_key]) is True, "and the store works again after recovery"


def test_sqlite_probe_failure_keeps_a_cached_result(tmp_path):
    """The router therefore still serves the cached result during the outage."""
    sqlite3 = pytest.importorskip("sqlite3")
    from headroom.cache.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "store.db")
    store = CompressionStore(max_entries=4, backend=backend)
    cache = CompressionCache()
    hash_key = store.store("the original tool output", "the compressed view")
    cache.put("k", f"head <<ccr:{hash_key}>>", 0.2, "x")
    assert cache.get("k", store) is not None

    backend._conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
    try:
        result = cache.get("k", store)
    finally:
        backend._conn.set_authorizer(None)

    assert result is not None, "an unanswered check is not a miss"
    assert cache.stats["cache_evictions"] == 0
    assert store.exists(hash_key), "the original is untouched"


def test_router_cache_hit_survives_unknown_sqlite_probe_in_both_paths(tmp_path, monkeypatch):
    """The post-get retrievability guard must preserve an unknown probe too."""
    sqlite3 = pytest.importorskip("sqlite3")
    from headroom.cache.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(tmp_path / "router-store.db")
    store = CompressionStore(max_entries=4, backend=backend)
    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    calls: list[str] = []

    def compress(content, context="", bias=1.0, **kwargs):
        calls.append(content)
        hash_key = store.store(content, "compressed view")
        return SimpleNamespace(
            original=content,
            compressed=f"compressed copy <<ccr:{hash_key}>>",
            compression_ratio=0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
            strategy_chain=[],
        )

    monkeypatch.setattr(router, "compress", compress)
    string_args = (payload("sqlite-string"), Tokenizer())
    first = tool_output(router.apply(*string_args, compression_store=store))
    backend._conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
    try:
        assert tool_output(router.apply(*string_args, compression_store=store)) == first
    finally:
        backend._conn.set_authorizer(None)
    assert len(calls) == 1, "string path must preserve cache on unknown liveness"

    block_message = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "sqlite-block",
                "content": tool_log("sqlite-block"),
            }
        ],
    }
    block_args = ([block_message], Tokenizer())
    first_blocks = router.apply(*block_args, compression_store=store)
    original_block_text = first_blocks.messages[-1]["content"][0]["content"]
    backend._conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
    try:
        second_blocks = router.apply(*block_args, compression_store=store)
    finally:
        backend._conn.set_authorizer(None)
    assert second_blocks.messages[-1]["content"][0]["content"] == original_block_text
    assert len(calls) == 2, "block path must preserve cache on unknown liveness"


def test_same_text_replacement_survives_stale_probe(monkeypatch):
    """An identical-text replacement is a new entry and must not be dropped."""
    cache = CompressionCache()
    text = "head <<ccr:0123456789abcdef>>"
    cache.put("k", text, 0.2, "x")

    class Store:
        def all_live(self, hash_keys):
            cache.put("k", text, 0.2, "x")
            return False

    store = Store()
    cache._drop_if_a_marker_died("k", store)
    assert cache.get("k", store) == (text, 0.2, "x")
    assert cache.stats["cache_evictions"] == 0


def test_a_successful_zero_exists_is_a_definite_miss_despite_a_failing_write():
    """The liveness answer is the probe's, not the guard's overall state.

    A Redis that refuses writes (``maxmemory`` + ``noeviction``) still answers
    ``EXISTS``: a successful 0 is a real miss and must not be reported as
    unknown just because the write kind is failing.
    """
    fakeredis = pytest.importorskip("fakeredis")
    redis = pytest.importorskip("redis")
    from headroom.cache.backends.redis import RedisBackend

    class RefusesWrites(fakeredis.FakeRedis):
        def set(self, *args, **kwargs):
            raise redis.ResponseError("OOM command not allowed")

    backend = RedisBackend("redis://localhost", client=RefusesWrites())
    store = CompressionStore(max_entries=4, backend=backend)
    with pytest.raises(StorageError):
        store.store("an original " * 40, "a compressed view")

    assert store.all_live(["abc123def456"]) is False
    assert backend.available() is False, "the failed write is still recorded"


def test_recompression_restores_the_original_in_the_explicit_store():
    """A request's explicit store is the one its compressors must write to.

    The reviewer's reproduction: warm a log marker with
    ``compression_store=explicit`` while the request-scoped store is a
    different, empty store, clear the original from the explicit store, and
    repeat twice. Recompression used to persist only to the request-scoped
    store, so the explicit store never healed and every repeat recompressed.
    """
    explicit = CompressionStore(max_entries=4, backend=InMemoryBackend())
    ambient = CompressionStore(max_entries=4, backend=InMemoryBackend())
    set_request_compression_store(ambient)
    try:
        router = ContentRouter(ContentRouterConfig(enable_kompress=False))
        args = (payload("explicit-store"), Tokenizer())

        first = tool_output(router.apply(*args, compression_store=explicit))
        (hash_key,) = _ccr_marker_hashes(first)
        assert explicit.exists(hash_key), "the request store holds the original"
        assert ambient.exists(hash_key) is False, "the process store never saw it"

        explicit._backend.delete(hash_key)  # the original is gone, the result is not
        again = tool_output(router.apply(*args, compression_store=explicit))
        assert explicit.exists(hash_key), "recompression restores it in the request store"
        assert ambient.exists(hash_key) is False, "and not in the process store"

        assert tool_output(router.apply(*args, compression_store=explicit)) == again
        assert router._cache.stats["cache_hits"] == 1, "the restored original is a hit"
    finally:
        clear_request_compression_store()


def test_a_permission_denied_exists_falls_back_to_get():
    """A Redis user may be allowed GET but not EXISTS (ACL ``-exists``).

    The probe used to answer unknown for that denial, so a cached marker whose
    original had been deleted was served as live. A permission error says
    nothing about whether the key is there: read it instead. An operator that
    denies EXISTS still leaves GET, which is all the probe needs.
    """
    fakeredis = pytest.importorskip("fakeredis")
    redis = pytest.importorskip("redis")
    from headroom.cache.backends.redis import RedisBackend

    class DeniesExists(fakeredis.FakeRedis):
        def exists(self, *args, **kwargs):
            raise redis.exceptions.NoPermissionError(
                "NOPERM this user has no permissions to run the 'exists' command"
            )

    backend = RedisBackend("redis://localhost", client=DeniesExists())
    store = CompressionStore(max_entries=4, backend=backend)
    hash_key = store.store("the original tool output", "the compressed view")
    cache = CompressionCache()
    cache.put("k", f"head <<ccr:{hash_key}>>", 0.2, "x")

    assert backend.live(hash_key) is True
    assert cache.get("k", store) is not None

    backend._client.delete(backend._key(hash_key))
    assert backend.live(hash_key) is False, "GET answered what EXISTS was not allowed to"
    assert cache.get("k", store) is None, "a deleted original must not be served"
    assert cache.stats["cache_evictions"] == 1


def test_a_permission_denied_exists_whose_get_also_fails_is_unknown():
    """Only a permission error earns the GET fallback; other failures stay unknown."""
    fakeredis = pytest.importorskip("fakeredis")
    redis = pytest.importorskip("redis")
    from headroom.cache.backends.redis import RedisBackend

    class DeniesExistsAndGet(fakeredis.FakeRedis):
        def exists(self, *args, **kwargs):
            raise redis.exceptions.NoPermissionError("NOPERM exists")

        def get(self, *args, **kwargs):
            raise redis.ConnectionError("connection lost")

    backend = RedisBackend("redis://localhost", client=DeniesExistsAndGet())
    store = CompressionStore(max_entries=4, backend=backend)
    assert store.all_live(["52a7090074ed6e50f0601afd"]) is None, (
        "a failed read is unknown, not a miss"
    )


def test_a_live_probe_that_raises_is_unknown_not_a_miss():
    """A backend whose optional ``live()`` raises must not evict, in either path.

    Reviewer's reproduction: the first cache check tolerates the exception, but
    the router's second (post-get) check aborted the request even though the
    original was still there.
    """

    class TimesOut(InMemoryBackend):
        armed = False

        def live(self, hash_key):
            if self.armed:
                raise TimeoutError("probe budget exceeded")
            return self.exists(hash_key)

    backend = TimesOut()
    store = CompressionStore(max_entries=4, backend=backend)
    hash_key = store.store("the original tool output", "the compressed view")
    assert store.all_live([hash_key]) is True

    backend.armed = True
    assert store.all_live([hash_key]) is None, "a raising probe cannot answer"
    assert store.exists(hash_key), "the original is still there"

    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    calls: list[str] = []

    def compress(content, context="", bias=1.0, **kwargs):
        calls.append(content)
        key = store.store(content, "compressed view")
        return SimpleNamespace(
            original=content,
            compressed=f"compressed copy <<ccr:{key}>>",
            compression_ratio=0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
            strategy_chain=[],
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(router, "compress", compress)
        string_args = (payload("timeout-string"), Tokenizer())
        first = tool_output(router.apply(*string_args, compression_store=store))
        assert tool_output(router.apply(*string_args, compression_store=store)) == first
        assert len(calls) == 1, "string path must keep the hit when the probe raises"

        block_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "timeout-block",
                    "content": tool_log("timeout-block"),
                }
            ],
        }
        block_args = ([block_message], Tokenizer())
        first_blocks = router.apply(*block_args, compression_store=store)
        cached_block = first_blocks.messages[-1]["content"][0]["content"]
        backend.armed = True
        second_blocks = router.apply(*block_args, compression_store=store)
        assert second_blocks.messages[-1]["content"][0]["content"] == cached_block
        assert len(calls) == 2, "block path must keep the hit when the probe raises"


def test_every_marker_shape_is_read_for_its_hash():
    assert _ccr_marker_hashes(
        "a <<ccr:abcdef012345>> b <<ccr:1234567890ab,base64,2.0KB>> "
        "c [80 lines compressed to 8. Retrieve more: hash=fedcba987654]"
    ) == {"abcdef012345", "1234567890ab", "fedcba987654"}
    assert _ccr_marker_hashes("Retrieve original: hash=ABCDEF0123456789") == {"abcdef0123456789"}
    # Shorter than 12 hex chars is not a hash this store ever issues.
    assert _ccr_marker_hashes("hash=abcdef01234") == set()


class Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(str(text)) // 4)


def tool_log(case: str) -> str:
    return "\n".join(
        f"2026-01-01 12:00:{i % 60:02d} INFO case{case} processed record {i}" for i in range(251)
    )


def payload(case: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{case}",
                    "type": "function",
                    "function": {"name": "run", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{case}", "content": tool_log(case)},
    ]


def tool_output(result) -> str:
    return result.messages[-1]["content"]


def test_an_evicted_original_makes_the_router_recompress(ccr_store, monkeypatch):
    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    calls: list[str] = []

    def compress(content, context="", bias=1.0, **kwargs):
        calls.append(content)
        compressed = f"compressed copy {len(calls)}"
        hash_key = ccr_store.store(content, compressed)
        return SimpleNamespace(
            original=content,
            compressed=f"{compressed} <<ccr:{hash_key}>>",
            compression_ratio=0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
            strategy_chain=[],
        )

    monkeypatch.setattr(router, "compress", compress)

    first = tool_output(router.apply(payload("a"), Tokenizer()))
    first_hash = MARKER_RE.search(first).group(1)
    assert ccr_store.exists(first_hash)

    # Two distinct logs fill the store; "a" and then "b" are evicted, exactly
    # the capacity pressure that leaves a stale marker in the cache.
    router.apply(payload("b"), Tokenizer())
    router.apply(payload("c"), Tokenizer())
    assert ccr_store.exists(first_hash) is False
    assert len(calls) == 3

    again = tool_output(router.apply(payload("a"), Tokenizer()))
    assert len(calls) == 4, "a stale marker must be a cache miss, not a hit"
    # Recompressing the same content hashes to the same key, which is exactly
    # why the stale marker could never heal on its own: the cache has to miss
    # so the store gets the original back under that same hash.
    assert MARKER_RE.search(again).group(1) == first_hash
    assert ccr_store.exists(first_hash), "the recompression must restore the original"
    assert router._cache.stats["cache_hits"] == 0

    # The fresh result is cached in turn, so the next apply reuses it.
    assert tool_output(router.apply(payload("a"), Tokenizer())) == again
    assert len(calls) == 4


def test_markers_are_checked_against_the_router_effective_store():
    """An explicit store's markers must not be looked up in the empty default.

    A request served by ``compression_store=explicit`` writes its originals
    there; the process default can be empty. Checking the default would evict
    a result whose marker is live in the store that actually backs the request.
    """
    explicit = CompressionStore(max_entries=4, backend=InMemoryBackend())
    fallback = CompressionStore(max_entries=4, backend=InMemoryBackend())
    set_request_compression_store(fallback)
    try:
        router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
        calls: list[str] = []

        def compress(content, context="", bias=1.0, **kwargs):
            calls.append(content)
            compressed = f"compressed copy {len(calls)}"
            hash_key = explicit.store(content, compressed)
            return SimpleNamespace(
                original=content,
                compressed=f"{compressed} <<ccr:{hash_key}>>",
                compression_ratio=0.1,
                strategy_used=CompressionStrategy.KOMPRESS,
                strategy_chain=[],
            )

        router.compress = compress

        first = tool_output(router.apply(payload("e"), Tokenizer(), compression_store=explicit))
        hash_key = MARKER_RE.search(first).group(1)
        assert explicit.exists(hash_key)
        assert fallback.exists(hash_key) is False, "the default store never saw it"

        again = tool_output(router.apply(payload("e"), Tokenizer(), compression_store=explicit))
        assert again == first
        assert len(calls) == 1, "the effective store still resolves the marker"
        assert router._cache.stats["cache_hits"] == 1
    finally:
        clear_request_compression_store()
