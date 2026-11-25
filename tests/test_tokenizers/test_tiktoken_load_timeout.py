"""tiktoken vocab loading must be bounded (GH #956).

tiktoken downloads its BPE vocab via ``requests.get`` with no timeout, so a
stalled/firewalled connection blocks indefinitely. The proxy calls this lazily
inside a request worker, so the only bound was the 30s compression timeout —
yielding "every request times out, 0 compression". The bounded loader caps the
wait and falls back to estimation instead.

The load runs in a dedicated ``python -m headroom.tokenizers._tiktoken_worker``
subprocess (never ``multiprocessing`` spawn, which re-imports the caller's
script), transfers plain encoding data (never an ``Encoding``, which pickles as
its registry name and re-downloads in the parent), and backs off per encoding
after a timeout.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
from unittest.mock import patch

import pytest

from headroom.tokenizers import tiktoken_counter as tc
from headroom.tokenizers._tiktoken_worker import _encoding_payload
from headroom.tokenizers.estimator import EstimatingTokenCounter
from headroom.tokenizers.registry import TokenizerRegistry


@pytest.fixture(autouse=True)
def _reset_encoding_state():
    import tiktoken

    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_permanently_failed.clear()
    tc._cooldown_until.clear()
    tiktoken.registry.ENCODINGS.clear()
    yield
    tc._get_encoding.cache_clear()
    tc._download_failed.clear()
    tc._disk_load_permanently_failed.clear()
    tc._cooldown_until.clear()
    tiktoken.registry.ENCODINGS.clear()


def _synthetic_payload(encoding_name: str) -> dict:
    """Payload for a real, tiny encoding - no network, no disk."""
    import tiktoken

    enc = tiktoken.Encoding(
        name=encoding_name,
        pat_str=r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        mergeable_ranks={bytes([b]): b for b in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    return {"status": "ok", **_encoding_payload(enc)}


def _stub_worker(monkeypatch: pytest.MonkeyPatch, payload: dict, delay: float = 0.0) -> None:
    """Point the loader at a subprocess that writes ``payload`` after ``delay``."""
    script = (
        "import pickle,sys,time;"
        f"time.sleep({delay!r});"
        "pickle.dump(pickle.loads(bytes.fromhex(sys.argv[1])), open(sys.argv[2],'wb'))"
    )
    blob = pickle.dumps(payload).hex()
    monkeypatch.setattr(
        tc, "_worker_cmd", lambda mode, name, out: [sys.executable, "-c", script, blob, out]
    )


def _sleep_cmd(mode: str, name: str, out: str) -> list[str]:
    return [sys.executable, "-c", "import time;time.sleep(30)"]


def _stub_stalled_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the loader at a child that never finishes."""
    monkeypatch.setattr(tc, "_worker_cmd", _sleep_cmd)


def _stub_worker_exiting(monkeypatch: pytest.MonkeyPatch, code: int = 3) -> None:
    monkeypatch.setattr(
        tc, "_worker_cmd", lambda mode, name, out: [sys.executable, "-c", f"sys.exit({code})"]
    )


def test_load_encoding_is_bounded_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled child is killed at the deadline, and the encoding backs off."""
    _stub_stalled_worker(monkeypatch)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("stall-enc")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.5, f"load was not bounded (took {elapsed:.2f}s vs the 30s stall)"
    assert tc._cooldown_until["stall-enc"] > time.monotonic()


def test_dead_worker_is_reported_without_waiting_out_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that exits immediately must not cost the full 10s budget."""
    _stub_worker_exiting(monkeypatch)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "5")

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError, match="failed to load from local cache"):
        tc.load_encoding("dead-enc")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"dead child cost {elapsed:.2f}s of the 5s deadline"


def test_repeated_stall_does_not_wait_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second model sharing the stalled vocab fails fast instead of re-waiting."""
    _stub_stalled_worker(monkeypatch)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("shared-enc")

    start = time.perf_counter()
    with pytest.raises(tc.TiktokenLoadError, match="retrying after cooldown"):
        tc.load_encoding("shared-enc")
    assert time.perf_counter() - start < 0.1


def test_failed_encoding_allows_retry_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient timeout must not block the encoding forever."""
    import tiktoken

    monkeypatch.setattr(tc, "_COOLDOWN_SECONDS", 0.05)
    monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")
    _stub_stalled_worker(monkeypatch)
    with pytest.raises(tc.TiktokenLoadError):
        tc.load_encoding("retry-enc")

    time.sleep(0.1)  # let the cooldown expire
    _stub_worker(monkeypatch, _synthetic_payload("retry-enc"))

    enc = tc.load_encoding("retry-enc")
    assert isinstance(enc, tiktoken.Encoding)
    assert "retry-enc" not in tc._cooldown_until


def test_worker_deadline_bounds_real_disk_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real worker's file read is inside the deadline (not just its cache check)."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("needs FIFOs")

    import hashlib

    with tempfile.TemporaryDirectory() as cache_dir:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        # A FIFO opens but never yields bytes: a genuine stalled disk read.
        os.mkfifo(os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest()))

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.3")

        start = time.perf_counter()
        with pytest.raises(tc.TiktokenLoadError, match="cache timed out"):
            tc.load_encoding("cl100k_base")
        assert time.perf_counter() - start < 2.0


def test_unguarded_client_script_is_not_reexecuted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading must not re-run the caller's script (spawn did)."""
    marker = tmp_path / "runs.txt"
    script = tmp_path / "client.py"
    script.write_text(
        f"open({str(marker)!r}, 'a').write('run\\n')\n"
        "from headroom.tokenizers.tiktoken_counter import TiktokenLoadError, load_encoding\n"
        "try:\n"
        "    load_encoding('cl100k_base')\n"
        "except TiktokenLoadError:\n"
        "    pass\n"
    )
    env = tc._worker_env()
    env.update(
        {
            "HEADROOM_OFFLINE": "1",
            "HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS": "5",
            "TIKTOKEN_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    import subprocess

    subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path, timeout=60, check=True)

    assert marker.read_text().count("run") == 1, "loader re-executed the client script"


def test_payload_rebuild_does_not_touch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker result must be rebuilt from data - never reloaded by name.

    tiktoken pickles a *registered* Encoding as its name, and unpickling calls
    ``tiktoken.registry.get_encoding`` in the parent: an unbounded, un-deadlined
    vocabulary download after the child already succeeded.
    """
    import tiktoken

    _stub_worker(monkeypatch, _synthetic_payload("synthetic-enc"))

    def _must_not_load(*args, **kwargs):
        raise AssertionError("parent re-loaded the vocabulary instead of using the payload")

    monkeypatch.setattr(tiktoken, "get_encoding", _must_not_load)
    monkeypatch.setattr(tiktoken.registry, "get_encoding", _must_not_load)

    enc = tc.load_encoding("synthetic-enc")
    assert enc.encode("hi") == [ord("h"), ord("i")]


def test_fast_load_returns_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    sentinel = object()
    tiktoken.registry.ENCODINGS["fast-enc"] = sentinel
    tc._get_encoding.cache_clear()

    assert tc.load_encoding("fast-enc") is sentinel


def test_registry_falls_back_to_estimator_on_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        _stub_stalled_worker(monkeypatch)
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")

        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert isinstance(counter, EstimatingTokenCounter)
        assert counter._provisional is True


def test_estimator_is_not_cached_so_model_can_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provisional estimator must be re-resolved, not pinned in the registry."""
    calls: list[str] = []

    def _fails(name: str):
        calls.append(name)
        raise tc.TiktokenLoadError("vocab unavailable")

    monkeypatch.setattr(tc, "load_encoding", _fails)

    first = TokenizerRegistry.get("gpt-4o")
    second = TokenizerRegistry.get("gpt-4o")

    assert isinstance(first, EstimatingTokenCounter)
    assert isinstance(second, EstimatingTokenCounter)
    assert len(calls) == 2, "estimator was cached; the model can never recover"


def test_offline_warm_cache_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache: encoding already in memory should work offline."""
    import tiktoken

    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    counter = TokenizerRegistry()._create_tiktoken("gpt-4")
    assert not isinstance(counter, EstimatingTokenCounter)
    assert counter.encoding is not None


def test_offline_warm_disk_cache_works_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache on disk: the real worker loads it offline with no network."""
    import tiktoken

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        tiktoken.get_encoding("cl100k_base")

        tc._get_encoding.cache_clear()
        tiktoken.registry.ENCODINGS.clear()
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert not isinstance(counter, EstimatingTokenCounter)
        assert counter.encoding.encode("hello world") == [15339, 1917]


def test_offline_cold_cache_refuses_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cold cache + offline: no download, fall back to estimation."""
    import tiktoken

    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        counter = TokenizerRegistry()._create_tiktoken("gpt-4")
        assert isinstance(counter, EstimatingTokenCounter)


def test_offline_invalid_cache_entry_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid cache entry (hash mismatch) fails fast without any network call."""
    import hashlib

    import tiktoken

    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        with open(os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest()), "wb") as f:
            f.write(b"invalid tiktoken data")

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        with patch("requests.get") as mock_get:
            with pytest.raises(tc.TiktokenLoadError):
                tc.load_encoding("cl100k_base")
            assert not mock_get.called, "requests.get should not be called for invalid cache"


def test_offline_warm_cache_payload_tokens_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm cache + offline: payload_tokens works without network."""
    import tiktoken

    from headroom.transforms.kompress_compressor import payload_tokens

    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    tc._get_encoding.cache_clear()

    with patch("requests.get") as mock_get:
        assert payload_tokens("hello world") == 2
        assert not mock_get.called


def test_offline_warm_cache_evaluate_token_counting_accuracy_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm cache + offline: the accuracy benchmark works without network."""
    import tiktoken

    from headroom.evals.batch_compression_eval import evaluate_token_counting_accuracy

    tiktoken.get_encoding("cl100k_base")
    assert "cl100k_base" in tiktoken.registry.ENCODINGS
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    tc._get_encoding.cache_clear()

    with patch("requests.get") as mock_get:
        result = evaluate_token_counting_accuracy()
        assert result.test_cases == 6
        assert result.reference_available is True
        assert result.ground_truth_counts is not None
        assert not mock_get.called


def test_online_corrupt_cache_repairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Online mode: corrupt cache entry is repaired by downloading a fresh copy."""
    import hashlib

    import tiktoken

    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        cache_path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest())
        with open(cache_path, "wb") as f:
            f.write(b"corrupt tiktoken data")

        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "0")
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "60")

        enc = tc.load_encoding("cl100k_base")
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == "cl100k_base"
        assert os.path.getsize(cache_path) > 100000  # valid .tiktoken file


def test_disk_load_miss_retryable_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline: a cache miss is retryable once the vocabulary is on disk."""
    import hashlib

    import tiktoken

    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        tiktoken.get_encoding("cl100k_base")  # populate cache online
        tiktoken.registry.ENCODINGS.clear()
        tc._get_encoding.cache_clear()

        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        monkeypatch.setenv("HEADROOM_TIKTOKEN_LOAD_TIMEOUT_SECONDS", "2.0")

        enc = tc.load_encoding("cl100k_base")
        assert isinstance(enc, tiktoken.Encoding)
        assert enc.name == "cl100k_base"

        tc._get_encoding.cache_clear()
        tiktoken.registry.ENCODINGS.clear()

        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        cache_path = os.path.join(cache_dir, hashlib.sha1(url.encode()).hexdigest())
        os.remove(cache_path)

        with pytest.raises(tc.TiktokenLoadError) as exc_info:
            tc.load_encoding("cl100k_base")
        assert "not in local cache and HEADROOM_OFFLINE=1" in str(exc_info.value)
        assert "cl100k_base" not in tc._disk_load_permanently_failed
        assert "cl100k_base" not in tc._download_failed

        monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
        tiktoken.get_encoding("cl100k_base")
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")
        tiktoken.registry.ENCODINGS.clear()
        tc._get_encoding.cache_clear()

        enc2 = tc.load_encoding("cl100k_base")
        assert enc2.name == "cl100k_base"
        assert enc2.encode("hello world") == [15339, 1917]


def test_offline_accuracy_benchmark_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline with cold cache: the benchmark reports no reference, no accuracy."""
    import tiktoken

    from headroom.evals.batch_compression_eval import evaluate_token_counting_accuracy

    tc._get_encoding.cache_clear()
    tiktoken.registry.ENCODINGS.clear()

    with tempfile.TemporaryDirectory() as cache_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", cache_dir)
        monkeypatch.setenv("HEADROOM_OFFLINE", "1")

        result = evaluate_token_counting_accuracy()

        assert result.reference_available is False
        assert result.ground_truth_counts is None
        assert result.results == []
        assert result.exact_matches == 0
        assert "unavailable" in result.summary()
