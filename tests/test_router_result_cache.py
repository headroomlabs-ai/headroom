"""The content router's compression cache stays within its memory budget."""

from __future__ import annotations

import subprocess
import sys
import threading

from headroom.cache.result_store import ResultStore, open_result_store
from headroom.transforms.content_router import (
    CompressionCache,
    ContentRouter,
    ContentRouterConfig,
    content_cache_key,
)


def test_results_stay_within_the_memory_budget_least_recently_used_first():
    cache = CompressionCache(max_bytes=10_000)
    for i in range(100):
        cache.put(f"k{i}", "x" * 1000, 0.5, "kompress")
        cache.get("k0")  # k0 stays recently used
    assert cache.stats["cache_bytes"] <= 10_000
    assert cache.get("k0") is not None
    assert cache.get("k1") is None  # evicted, not kept forever


def test_an_evicted_result_comes_back_from_disk_unchanged(tmp_path):
    cache = CompressionCache(max_bytes=3_000, store=ResultStore(tmp_path / "c.db", 1800))
    cache.put("first", "compressed once", 0.4, "kompress")
    for i in range(10):
        cache.put(f"k{i}", "y" * 1000, 0.5, "kompress")
    # Out of memory, but read back: no second compression, same bytes.
    assert cache.get("first") == ("compressed once", 0.4, "kompress")
    assert cache.stats["cache_disk_hits"] == 1


def test_the_skip_set_is_bounded_too():
    cache = CompressionCache(max_bytes=1024 * 1024)
    for i in range(50_000):
        cache.mark_skip(f"s{i}")
    assert cache.skip_size <= 2000
    assert cache.is_skipped("s49999")


def test_keys_are_the_same_in_every_process():
    code = (
        "from headroom.transforms.content_router import content_cache_key;"
        "print(content_cache_key('same text', 0.5, gated=True))"
    )
    other = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert other == content_cache_key("same text", 0.5, gated=True)
    assert content_cache_key("same text", 0.5) != other  # gated tool output kept apart


def test_stateless_mode_writes_no_cache_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    assert open_result_store(1800) is None
    assert not (tmp_path / "compression_cache.db").exists()


def test_a_result_in_use_is_kept_and_an_idle_one_is_dropped(tmp_path, monkeypatch):
    # Nothing signals a session's end: a result expires once unused for the
    # ttl, from memory and from disk, while a session still reading it keeps it.
    import time as time_module

    clock = [1000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time_module, "time", lambda: clock[0])
    store = ResultStore(tmp_path / "c.db", ttl_seconds=100)
    cache = CompressionCache(ttl_seconds=100, store=store)
    cache.put("live", "kept", 0.5, "kompress")
    cache.put("idle", "dropped", 0.5, "kompress")
    for _ in range(5):  # 'live' is read every 60 s, for 300 s
        clock[0] += 60
        assert cache.get("live") is not None
    assert cache.get("idle") is None
    assert store.get(cache._namespace + "idle") is None
    assert store.get(cache._namespace + "live") == ("kept", 0.5, "kompress")


def test_routers_configured_differently_never_share_output(tmp_path, monkeypatch):
    # The proxy can run two routers (per-provider Kompress overrides), and a
    # restart can change settings; both read the same file.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_PATH", str(tmp_path / "c.db"))
    lossy = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    lossless = ContentRouter(ContentRouterConfig(result_cache_disk=True, lossless=True))
    lossy._cache.put("k", "lossy output", 0.3, "kompress")
    assert lossless._cache.get("k") is None
    same = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    assert same._cache.get("k") == ("lossy output", 0.3, "kompress")


def test_the_namespace_is_the_same_in_every_process():
    code = (
        "from headroom.transforms.content_router import ContentRouterConfig, _config_namespace;"
        "print(_config_namespace(ContentRouterConfig()))"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(seen) == 1


def test_a_broken_database_is_a_miss_not_a_failed_request(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store)
    store._db.close()
    cache.put("k", "text", 0.5, "kompress")  # no raise
    cache.clear()
    assert cache.get("k") is None


def test_a_result_pointing_at_a_ccr_original_is_not_kept_on_disk(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store)
    cache.put("k", "first lines <<ccr:0123456789abcdef>>", 0.2, "x")
    assert store.get(cache._namespace + "k") is None


def test_a_nonfinite_budget_falls_back_instead_of_stopping_startup(monkeypatch):
    for raw in ("nan", "inf", "-5"):
        monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_MB", raw)
        assert CompressionCache().stats["cache_max_bytes"] == 64 * 1024 * 1024


def test_a_changed_compressor_setting_does_not_reuse_old_results(tmp_path, monkeypatch):
    # Settings the router reads from the environment, outside its config.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_PATH", str(tmp_path / "c.db"))
    before = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    before._cache.put("k", "made under the old setting", 0.3, "kompress")
    monkeypatch.setenv("HEADROOM_LOSSLESS_THEN_LOSSY", "1")
    after = ContentRouter(ContentRouterConfig(result_cache_disk=True))
    assert after._cache.get("k") is None


def test_a_marker_result_replacing_a_plain_one_removes_it_from_disk(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=4_000, store=store)
    cache.put("k", "plain result", 0.5, "kompress")
    cache.put("k", "first lines <<ccr:0123456789abcdef>>", 0.2, "x")
    for i in range(10):  # push "k" out of memory
        cache.put(f"filler{i}", "f" * 1000, 0.5, "kompress")
    assert cache.get("k") is None


def test_clear_also_clears_the_routers_rows_on_disk(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store, namespace="ns1:")
    other = CompressionCache(store=store, namespace="ns2:")
    cache.put("k", "value", 0.5, "kompress")
    other.put("k", "theirs", 0.5, "kompress")
    cache.clear()
    assert cache.get("k") is None
    assert store.get("ns2:k") == ("theirs", 0.5, "kompress")


def test_an_older_put_never_undoes_a_newer_ones_disk_row(tmp_path):
    # A marker put admitted first but reaching the disk after a newer plain
    # put must not delete the plain result's row.
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(store=store)

    class NewerPutRunsFirst:
        def __init__(self) -> None:
            self._lock, self._pending = threading.Lock(), True

        def __enter__(self):
            if self._pending:  # between the marker's admit and its disk step
                self._pending = False
                cache.put("k", "plain output", 0.5, "text")
            self._lock.acquire()

        def __exit__(self, *exc):
            self._lock.release()

    cache._disk_lock = NewerPutRunsFirst()
    cache.put("k", "<<ccr:0123456789abcdef 3_rows_offloaded>>", 0.5, "text")
    assert store.get(cache._namespace + "k")[0] == "plain output"


def test_a_cache_without_a_namespace_clears_only_its_own_rows(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    mine = CompressionCache(store=store)
    other = CompressionCache(store=store, namespace="router-b:")
    mine.put("a", "result a", 0.5, "text")
    other.put("b", "result b", 0.5, "text")
    mine.clear()
    assert store.get("router-b:b") is not None


def test_a_disk_read_in_flight_never_replaces_a_newer_put(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=600, store=store)
    cache.put("k", "old", 0.5, "text")
    cache.put("other", "z" * 300, 0.5, "text")  # evicts k from memory
    read_done, release = threading.Event(), threading.Event()
    real_get = store.get

    def paused_get(key):
        found = real_get(key)
        store.get = real_get  # pause only the first read
        read_done.set()
        assert release.wait(5)
        return found

    store.get = paused_get
    results = []
    reader = threading.Thread(target=lambda: results.append(cache.get("k")))
    reader.start()
    assert read_done.wait(5)  # the read has "old" in hand
    cache.put("k", "new", 0.5, "text")
    release.set()
    reader.join(5)
    assert results == [("new", 0.5, "text")]
    assert cache.get("k") == ("new", 0.5, "text")
    assert store.get(cache._namespace + "k")[0] == "new"


def test_a_disk_read_during_clear_never_restores_a_cleared_result(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=600, store=store)
    cache.put("k", "old", 0.5, "text")
    cache.put("other", "z" * 300, 0.5, "text")  # evicts k from memory
    deleting, release = threading.Event(), threading.Event()
    real_clear_prefix = store.clear_prefix

    def paused_clear_prefix(prefix):
        deleting.set()
        assert release.wait(5)
        real_clear_prefix(prefix)

    store.clear_prefix = paused_clear_prefix
    clearer = threading.Thread(target=cache.clear)
    clearer.start()
    assert deleting.wait(5)  # memory cleared, disk row not yet deleted
    reader = threading.Thread(target=cache.get, args=("k",))
    reader.start()
    reader.join(0.5)  # finishes here unless clear() still holds it off
    release.set()
    clearer.join(5)
    reader.join(5)
    assert store.get(cache._namespace + "k") is None
    assert cache.get("k") is None


class PauseAfterFetch:
    """Stands in for a cache's ``_disk_read_lock``: the first ``pauses`` reads
    stop after their fetch, outside the lock and before admission."""

    def __init__(self, pauses: int) -> None:
        self._lock = threading.Lock()
        self.fetched = [threading.Event() for _ in range(pauses)]
        self.release = [threading.Event() for _ in range(pauses)]
        self._exits = 0

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *exc):
        i, self._exits = self._exits, self._exits + 1
        self._lock.release()
        if i < len(self.fetched):
            self.fetched[i].set()
            assert self.release[i].wait(5)


def test_concurrent_disk_reads_of_one_key_each_return_the_result(tmp_path):
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=600, store=store)
    cache.put("k", "kept", 0.5, "text")
    cache.put("other", "z" * 300, 0.5, "text")  # evicts k from memory
    reads = cache._disk_read_lock = PauseAfterFetch(2)
    results: dict[str, object] = {}
    first = threading.Thread(target=lambda: results.update(first=cache.get("k")))
    second = threading.Thread(target=lambda: results.update(second=cache.get("k")))
    first.start()
    assert reads.fetched[0].wait(5)
    second.start()
    assert reads.fetched[1].wait(5)  # both reads in flight; the first finishes first
    reads.release[0].set()
    first.join(5)
    reads.release[1].set()
    second.join(5)
    assert results == {"first": ("kept", 0.5, "text"), "second": ("kept", 0.5, "text")}


def test_an_older_disk_read_never_replaces_a_later_reads_result(tmp_path):
    # Two workers share one store: a read that fetched earlier holds an older
    # row than one that fetched after the other worker's put.
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=600, store=store)
    other_worker = CompressionCache(store=store)
    cache.put("k", "old", 0.5, "text")
    cache.put("other", "z" * 300, 0.5, "text")  # evicts k from memory
    reads = cache._disk_read_lock = PauseAfterFetch(1)
    results = []
    reader = threading.Thread(target=lambda: results.append(cache.get("k")))
    reader.start()
    assert reads.fetched[0].wait(5)  # the first read has "old" in hand
    other_worker.put("k", "new", 0.5, "text")
    assert cache.get("k") == ("new", 0.5, "text")  # a later read admits "new"
    reads.release[0].set()
    reader.join(5)
    assert results == [("new", 0.5, "text")]
    assert cache.get("k") == ("new", 0.5, "text")


def test_a_read_that_fetches_later_is_never_outranked_by_an_older_row(tmp_path):
    # Read A is numbered before read B but fetches after another worker's
    # put: what A fetched is the newest row and must be what memory keeps.
    store = ResultStore(tmp_path / "c.db", 1800)
    cache = CompressionCache(max_bytes=600, store=store)
    other_worker = CompressionCache(store=store)
    cache.put("k", "old", 0.5, "text")
    cache.put("other", "z" * 300, 0.5, "text")  # evicts k from memory
    fetching, release = threading.Event(), threading.Event()
    real_get = store.get

    def paused_get(key):
        store.get = real_get  # pause only the first read, before it fetches
        fetching.set()
        assert release.wait(5)
        return real_get(key)

    store.get = paused_get
    results = []
    first = threading.Thread(target=lambda: results.append(cache.get("k")))
    first.start()
    assert fetching.wait(5)  # A is numbered; its fetch has not run
    second = threading.Thread(target=cache.get, args=("k",))
    second.start()
    second.join(0.5)  # B reads "old" here unless A's read holds it off
    other_worker.put("k", "new", 0.5, "text")
    release.set()
    first.join(5)
    second.join(5)
    assert results == [("new", 0.5, "text")]
    assert cache.get("k") == ("new", 0.5, "text")
    assert store.get(cache._namespace + "k")[0] == "new"


def test_a_result_that_expires_while_a_read_waits_is_not_returned(tmp_path, monkeypatch):
    # get() misses memory, then waits for the disk-read lock; a marker result
    # admitted meanwhile expires before the wait ends. Its second look at
    # memory must apply the same lifetime as the first, not return it.
    import time as time_module

    clock = [1000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: clock[0])
    cache = CompressionCache(store=ResultStore(tmp_path / "c.db", 1800), marker_ttl_seconds=1)
    waiting, real = threading.Event(), cache._disk_read_lock

    class SignalsWait:
        def __enter__(self):
            waiting.set()
            return real.__enter__()

        def __exit__(self, *exc):
            return real.__exit__(*exc)

    cache._disk_read_lock = SignalsWait()
    results = []
    with real:
        reader = threading.Thread(target=lambda: results.append(cache.get("k")))
        reader.start()
        assert waiting.wait(5)
        cache.put("k", "head <<ccr:0123456789abcdef>>", 0.2, "x")
        clock[0] += 2
    reader.join(5)
    assert results == [None]
    assert cache.get("k") is None
