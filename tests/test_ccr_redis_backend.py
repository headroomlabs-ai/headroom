"""One CCR store for every replica: HEADROOM_CCR_BACKEND=redis.

Behind a gateway that spreads requests across replicas, the retrieval of a
<<ccr:hash>> marker can land on a replica other than the one that compressed.
"""

from __future__ import annotations

import json

import pytest

fakeredis = pytest.importorskip("fakeredis")

from headroom.cache.backends.redis import RedisBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    CompressionStore,
    _create_default_ccr_backend,
)
from headroom.exceptions import StorageError  # noqa: E402


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _replica(server) -> CompressionStore:
    backend = RedisBackend("redis://localhost", client=fakeredis.FakeRedis(server=server))
    return CompressionStore(backend=backend)


def test_a_marker_made_on_one_replica_is_retrieved_on_another(server):
    hash_key = _replica(server).store("the full original output " * 50, "short")
    entry = _replica(server).retrieve(hash_key)
    assert entry is not None and entry.original_content.startswith("the full original output")


def test_an_entry_expires_in_redis_when_it_expires(server):
    store = _replica(server)
    hash_key = store.store("original " * 40, "short", ttl=60)
    ttl_ms = fakeredis.FakeRedis(server=server).pttl(f"headroom:ccr:{hash_key}")
    assert 0 < ttl_ms <= 60_000


def test_deployments_with_different_prefixes_do_not_share(server, monkeypatch):
    hash_key = _replica(server).store("original " * 40, "short")
    monkeypatch.setenv("HEADROOM_REDIS_KEY_PREFIX", "staging")
    assert _replica(server).retrieve(hash_key) is None


def test_redis_down_is_a_miss_and_issues_no_hash(monkeypatch):
    import redis

    class Down(fakeredis.FakeRedis):
        def execute_command(self, *args, **kwargs):
            raise redis.ConnectionError("connection refused")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Down()))
    with pytest.raises(StorageError):
        store.store("original " * 40, "short")
    assert store.retrieve("abc123def456") is None


def test_a_corrupt_entry_is_a_miss(server):
    store = _replica(server)
    fakeredis.FakeRedis(server=server).set("headroom:ccr:abc123def456", "not json")
    assert store.retrieve("abc123def456") is None


def test_the_env_selects_the_redis_backend(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setenv("HEADROOM_REDIS_URL", "redis://127.0.0.1:6399/0")
    assert isinstance(_create_default_ccr_backend(), RedisBackend)


def test_a_url_that_is_not_redis_falls_back_to_the_local_store(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setenv("HEADROOM_REDIS_URL", "memcached://cache:11211")
    assert _create_default_ccr_backend() is None


def test_redis_is_skipped_for_a_while_after_a_connection_error():
    import redis

    calls = []

    class Down(fakeredis.FakeRedis):
        def execute_command(self, *args, **kwargs):
            calls.append(args[0])
            raise redis.ConnectionError("connection refused")

    backend = RedisBackend("redis://localhost", client=Down())
    for _ in range(5):
        backend.get("abc123def456")
    assert len(calls) == 1  # one failure, then back-off


def test_storing_never_scans_the_shared_keyspace(server, monkeypatch):
    # A per-replica cap would count (SCAN) the whole store on every write and
    # evict other replicas' live entries; Redis expires entries by TTL.
    store = _replica(server)
    store._max_entries = 2
    monkeypatch.setattr(RedisBackend, "_scan", lambda self: pytest.fail("scanned"))
    hashes = [store.store(f"original {i} " * 40, f"short {i}") for i in range(5)]
    # Re-stores (a marker met again every turn) and retrievals neither.
    for _ in range(300):
        store.store("original 0 " * 40, "short 0")
    assert all(store.retrieve(h) is not None for h in hashes)
    assert not store._eviction_heap and not store._expiration_heap  # nothing grows


def test_a_redis_that_reads_but_refuses_writes_logs_once(caplog):
    import redis

    class Full(fakeredis.FakeRedis):
        def set(self, *args, **kwargs):
            raise redis.ResponseError("OOM command not allowed")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Full()))
    with caplog.at_level("INFO", logger="headroom.cache.redis_client"):
        for i in range(5):
            with pytest.raises(StorageError):
                store.store(f"original {i} " * 40, "short")
            store.retrieve("abc123def456")
    messages = [r.getMessage() for r in caplog.records]
    assert sum("failed" in m for m in messages) == 1
    assert not any("works again" in m for m in messages)


def test_a_prefix_with_glob_characters_matches_only_itself(server, monkeypatch):
    _replica(server).store("ours " * 40, "short")
    monkeypatch.setenv("HEADROOM_REDIS_KEY_PREFIX", "head*")
    assert RedisBackend("redis://localhost", client=fakeredis.FakeRedis(server=server)).count() == 0


def test_stats_and_retrieval_never_read_or_rewrite_whole_entries(server, monkeypatch):
    store = _replica(server)
    hashes = [
        store.store(f"original {i} " * 40, f"short {i}", original_tokens=80, compressed_tokens=2)
        for i in range(5)
    ]
    reads, writes = [], []
    monkeypatch.setattr(fakeredis.FakeRedis, "get", lambda self, k: reads.append(k))
    stats = store.get_stats()
    assert stats["entry_count"] == 5 and not reads
    # Numeric totals (/stats contract): what this replica stored and retrieved.
    assert (stats["total_original_tokens"], stats["total_retrievals"]) == (400, 0)
    monkeypatch.undo()
    monkeypatch.setattr(fakeredis.FakeRedis, "set", lambda self, *a, **kw: writes.append(a))
    assert store.retrieve(hashes[0]) is not None
    assert not writes
    assert store.get_stats()["total_retrievals"] == 1


def test_credentials_in_a_redis_error_are_not_logged(caplog):
    import redis

    class Refusing(fakeredis.FakeRedis):
        def get(self, *args, **kwargs):
            raise redis.ConnectionError("cannot reach redis://app:s3cret@cache:6379/0")

    backend = RedisBackend("redis://localhost", client=Refusing())
    with caplog.at_level("WARNING", logger="headroom.cache.redis_client"):
        assert backend.get("abc") is None
    assert "s3cret" not in caplog.text and "//***@cache" in caplog.text


def test_a_write_redis_refuses_adds_no_tokens(caplog):
    import redis

    class Full(fakeredis.FakeRedis):
        def set(self, *args, **kwargs):
            raise redis.ResponseError("OOM command not allowed")

    store = CompressionStore(backend=RedisBackend("redis://localhost", client=Full()))
    with pytest.raises(StorageError):
        store.store("original " * 40, "short", original_tokens=80, compressed_tokens=2)
    stats = store.get_stats()
    assert (stats["total_original_tokens"], stats["totals_scope"]) == (0, "this_replica")


# --- A producer never advertises an original Redis did not take -------------

_CONFIG = "[server]\n" + "".join(
    f"# comment line {i} describing port_{i}\nport_{i} = {8000 + i}\n" for i in range(30)
)


class _Flaky(fakeredis.FakeRedis):
    """A Redis that refuses writes (``oom``) or cannot be reached (``down``)."""

    outage: str | None = None

    def execute_command(self, *args, **kwargs):
        import redis

        if self.outage == "down":
            raise redis.ConnectionError("connection refused")
        if self.outage == "oom" and args[0] == "SET":
            raise redis.ResponseError("OOM command not allowed")
        return super().execute_command(*args, **kwargs)


@pytest.fixture
def flaky_store(server, monkeypatch):
    from headroom.cache import compression_store, redis_client

    monkeypatch.setattr(redis_client, "BACKOFF_SECONDS", 0.0)  # recovery is immediate
    client = _Flaky(server=server)
    store = CompressionStore(backend=RedisBackend("redis://localhost", client=client))
    # Global too: executor threads (the MCP tools) and startup do not see the
    # request-scoped store.
    monkeypatch.setattr(compression_store, "_compression_store", store)
    compression_store.set_request_compression_store(store)
    yield store, client
    compression_store.clear_request_compression_store()


def _marker_hash(text: str) -> str | None:
    import re

    found = re.search(r"hash=([0-9a-f]{24})", text)
    return found.group(1) if found else None


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_config_elision_keeps_the_original_while_redis_fails_then_resumes(flaky_store, outage):
    from headroom.transforms.config_compressor import ConfigCompressor

    store, client = flaky_store
    client.outage = outage
    failed = ConfigCompressor().compress(_CONFIG)
    assert failed.compressed == _CONFIG and failed.ccr_hash is None and failed.lines_elided == 0

    client.outage = None
    resumed = ConfigCompressor().compress(_CONFIG)
    assert resumed.lines_elided == 30
    entry = store.retrieve(_marker_hash(resumed.compressed))
    assert entry is not None and entry.original_content == _CONFIG


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_read_maturation_keeps_the_read_while_redis_fails_then_resumes(flaky_store, outage):
    from headroom.config import ReadMaturationConfig
    from headroom.transforms.read_maturation import ReadMaturationManager

    store, client = flaky_store
    content = "     1\tdef foo():\n     2\t    return 42\n" * 60
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/x.py"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r1", "content": content}],
        },
        *[{"role": "assistant", "content": [{"type": "text", "text": "t"}]} for _ in range(5)],
    ]
    manager = ReadMaturationManager(
        ReadMaturationConfig(enabled=True, quiesce_turns=5), compression_store=store
    )
    client.outage = outage
    assert manager.apply(messages).messages[2]["content"][0]["content"] == content

    client.outage = None
    matured = manager.apply(messages).messages[2]["content"][0]["content"]
    entry = store.retrieve(_marker_hash(matured))
    assert entry is not None and entry.original_content == content


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_read_lifecycle_keeps_a_stale_read_while_redis_fails(flaky_store, outage):
    from headroom.config import ReadLifecycleConfig
    from headroom.transforms.read_lifecycle import ReadLifecycleManager

    store, client = flaky_store
    content = "x = 1\n" * 200

    def call(tc_id, name, args):
        fn = {"name": name, "arguments": json.dumps(args)}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": tc_id, "function": fn}]}

    messages = [
        call("r1", "Read", {"file_path": "/a.py"}),
        {"role": "tool", "tool_call_id": "r1", "content": content},
        call("e1", "Edit", {"file_path": "/a.py", "old_string": "x", "new_string": "y"}),
        {"role": "tool", "tool_call_id": "e1", "content": "ok"},
    ]
    manager = ReadLifecycleManager(ReadLifecycleConfig(enabled=True), compression_store=store)
    client.outage = outage
    assert manager.apply(messages).messages[1]["content"] == content

    client.outage = None
    entry = store.retrieve(_marker_hash(manager.apply(messages).messages[1]["content"]))
    assert entry is not None and entry.original_content == content


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_rust_log_compression_keeps_the_log_while_redis_fails(flaky_store, outage):
    pytest.importorskip("headroom._core")
    from headroom.transforms.log_compressor import LogCompressor, LogCompressorConfig

    store, client = flaky_store
    log = "\n".join(["npm WARN deprecated x"] * 30 + ["npm ERR! something broke"] * 5)
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=True, min_lines_for_ccr=3))
    client.outage = outage
    failed = compressor.compress(log)
    assert failed.compressed == log and failed.cache_key is None

    client.outage = None
    resumed = compressor.compress(log)
    assert resumed.cache_key is not None
    assert store.retrieve(resumed.cache_key).original_content == log


def test_mcp_read_returns_the_file_and_caches_no_hash_while_redis_is_down(flaky_store, tmp_path):
    import asyncio

    from tests._mcp_stub import import_module_with_mcp_stub

    mcp_server = import_module_with_mcp_stub("headroom.ccr.mcp_server")
    store, client = flaky_store
    path = tmp_path / "a.py"
    source = "x = 1\n" * 50
    path.write_bytes(source.encode())  # exact bytes: write_text would emit CRLF on Windows
    server = mcp_server.HeadroomMCPServer(check_proxy=False)
    client.outage = "down"
    response = asyncio.run(server._handle_read({"file_path": str(path)}))
    assert "x = 1" in response[0].kwargs["text"] and not server._file_cache

    client.outage = None
    asyncio.run(server._handle_read({"file_path": str(path)}))
    (_, ccr_hash, _, _) = next(iter(server._file_cache.values()))
    assert store.retrieve(ccr_hash).original_content == source


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_mcp_compress_returns_the_original_while_redis_fails_then_resumes(flaky_store, outage):
    import asyncio

    from tests._mcp_stub import import_module_with_mcp_stub

    mcp_server = import_module_with_mcp_stub("headroom.ccr.mcp_server")
    store, client = flaky_store
    content = json.dumps([{"id": i, "status": "ok", "tag": "alpha"} for i in range(60)])
    server = mcp_server.HeadroomMCPServer(check_proxy=False)

    def compress() -> dict:
        response = asyncio.run(server._handle_compress({"content": content}))
        return json.loads(response[0].kwargs["text"])

    client.outage = outage
    failed = compress()
    assert (failed["compressed"], failed["hash"], failed["tokens_saved"]) == (content, None, 0)

    client.outage = None
    resumed = compress()
    assert resumed["compressed"] != content
    assert store.retrieve(resumed["hash"]).original_content == content


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_proxy_starts_when_redis_fails_restoring_prefix_state(
    server, flaky_store, outage, monkeypatch, tmp_path
):
    pytest.importorskip("fastapi")
    import contextvars

    from fastapi.testclient import TestClient

    from headroom.cache.prefix_tracker import PREFIX_STATE_FILE_ENV, SessionTrackerStore
    from headroom.proxy.server import ProxyConfig, create_app
    from tests.test_anthropic_fork_lineage_cache import _send

    store, client = flaky_store
    hash_key = store.store(json.dumps(list(range(400))), "[0, 1]", ttl=900)
    saved = SessionTrackerStore()
    text = f"[400 items compressed to 2. Retrieve more: hash={hash_key}]"
    contextvars.Context().run(_send, saved, [{"role": "user", "content": text}])
    path = tmp_path / "prefix-state.json"
    assert saved.save_state(str(path)) == 1
    fakeredis.FakeRedis(server=server).flushall()  # Redis lost it; only the state file has it
    monkeypatch.setenv(PREFIX_STATE_FILE_ENV, str(path))

    def start() -> int:
        config = ProxyConfig(optimize=False, rate_limit_enabled=False, cost_tracking_enabled=False)
        with TestClient(create_app(config)) as app_client:
            return app_client.app.state.proxy.session_tracker_store.active_sessions

    client.outage = outage
    # load_state treats the failed import like a bad file: the proxy starts and
    # installs no tracker, so none replays a marker whose original is gone.
    assert start() == 0


def _router(**config):
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    return ContentRouter(ContentRouterConfig(enable_kompress=False, **config))


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_smart_crusher_row_drop_keeps_the_rows_while_redis_fails_then_resumes(
    flaky_store, outage, monkeypatch
):
    pytest.importorskip("headroom._core")
    import re

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    store, client = flaky_store
    rows = [{"id": i, "status": "ok", "tag": "alpha"} for i in range(60)]
    router = _router(smart_crusher_with_compaction=False)  # the row-drop path
    client.outage = outage
    assert "<<ccr:" not in router.compress(json.dumps(rows)).compressed

    client.outage = None
    resumed = router.compress(json.dumps(rows)).compressed
    entry = store.retrieve(re.search(r"<<ccr:([0-9a-f]+)", resumed).group(1))
    assert entry is not None and json.loads(entry.original_content) == rows


@pytest.mark.parametrize("outage", ["oom", "down"])
def test_external_compressor_output_is_dropped_while_redis_fails_then_resumes(
    flaky_store, outage, monkeypatch
):
    import hashlib

    from headroom.transforms.compressor_registry import CompressorDescriptor, CompressOutput

    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    store, client = flaky_store

    class External:
        descriptor = CompressorDescriptor(
            name="ext_json",
            content_types=["application/json"],
            lossless=False,
            cost_tier="fast",
            recoverable=True,
        )

        def compress(self, inp):
            digest = hashlib.sha256(inp.content.encode()).hexdigest()[:24]
            return CompressOutput(
                content=f"[external <<ccr:{digest}>>]",
                tokens_before=len(inp.content.split()),
                tokens_after=2,
                lossless=False,
                markers=[],
                recoverable={digest: inp.content},
                warnings=[],
            )

    content = json.dumps([{"id": i, "note": f"descriptive field {i} " * 3} for i in range(40)])
    router = _router(active_external_compressors=["ext_json"])
    router.compressor_registry.register(External(), replace=True)
    router._active_external_compressors = router._resolve_active_external_compressors()
    client.outage = outage
    failed = router.compress(content)
    assert "external:ext_json" not in failed.strategy_chain
    assert "<<ccr:" not in failed.compressed

    client.outage = None
    resumed = router.compress(content)
    assert "external:ext_json" in resumed.strategy_chain
    digest = hashlib.sha256(content.encode()).hexdigest()[:24]
    assert store.retrieve(digest).original_content == content
