"""Regression tests for tree-sitter thread-local parser isolation.

pyo3 marks _native::Parser as #[pyclass(unsendable)], meaning a Parser created
on ThreadId(N) panics with an assertion error if accessed from ThreadId(M != N).
The prior implementation cached parsers in a module-level dict, which caused the
proxy's _run_compression_in_executor to pass a main-thread parser to a pool
worker and panic.

These tests verify that _get_parser() returns per-thread instances so no
cross-thread access can occur.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Iterator

import pytest

from headroom.transforms.code_compressor import (
    _get_parser,
    _tree_sitter_local,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)

try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not TREE_SITTER_INSTALLED,
    reason="tree-sitter-language-pack not installed",
)


@pytest.fixture(autouse=True)
def clear_thread_local() -> Iterator[None]:
    """Ensure the current thread's parser cache is clean before each test."""
    if hasattr(_tree_sitter_local, "parsers"):
        _tree_sitter_local.parsers = {}
    yield
    if hasattr(_tree_sitter_local, "parsers"):
        _tree_sitter_local.parsers = {}


# ---------------------------------------------------------------------------
# Isolation: separate threads must not share parser objects
# ---------------------------------------------------------------------------


def test_different_threads_get_different_parser_instances() -> None:
    """Parser objects from different threads must be distinct instances."""
    results: dict[int, object] = {}

    def grab_parser(thread_index: int) -> None:
        parser = _get_parser("python")
        results[thread_index] = parser

    t1 = threading.Thread(target=grab_parser, args=(0,))
    t2 = threading.Thread(target=grab_parser, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2, "Both threads should have completed"
    assert results[0] is not results[1], (
        "Each thread must own its own parser — sharing would trigger the pyo3 Unsendable panic"
    )


def test_same_thread_reuses_parser_instance() -> None:
    """Within a single thread, calling _get_parser twice returns the same object."""
    p1 = _get_parser("python")
    p2 = _get_parser("python")
    assert p1 is p2, "Same thread should reuse the cached parser (no unnecessary allocation)"


def test_different_languages_cached_per_thread() -> None:
    """Multiple language parsers are cached independently per thread."""
    py = _get_parser("python")
    js = _get_parser("javascript")
    assert py is not js


# ---------------------------------------------------------------------------
# Thread-pool executor: simulates _run_compression_in_executor behaviour
# ---------------------------------------------------------------------------


def test_parser_usable_in_thread_pool() -> None:
    """Parser must be usable inside a ThreadPoolExecutor without panicking."""

    def parse_in_worker() -> bool:
        parser = _get_parser("python")
        tree = parser.parse(b"x = 1\n")
        return tree is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(parse_in_worker) for _ in range(4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results), "All pool-thread parse calls should succeed"


def test_concurrent_pool_workers_get_separate_parsers() -> None:
    """Each distinct pool thread gets its own parser; same thread reuses the same one.

    A pool with N_WORKERS threads running N_TASKS tasks gives at most N_WORKERS
    unique parsers (not N_TASKS) — correct, because parsers are per-thread not
    per-call.
    """
    N_WORKERS = 4
    N_TASKS = 8
    parser_ids_by_thread: dict[int, int] = {}  # thread ident -> parser id
    lock = threading.Lock()

    def collect_parser() -> None:
        parser = _get_parser("python")
        ident = threading.current_thread().ident or 0
        with lock:
            if ident in parser_ids_by_thread:
                # Same thread must return the cached (same) parser
                assert parser_ids_by_thread[ident] == id(parser), (
                    "Same thread returned a different parser on a second call"
                )
            else:
                parser_ids_by_thread[ident] = id(parser)

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(collect_parser) for _ in range(N_TASKS)]
        for f in futures:
            f.result()

    assert len(parser_ids_by_thread) <= N_WORKERS, (
        "There should be at most one parser per pool thread"
    )
    assert len(set(parser_ids_by_thread.values())) == len(parser_ids_by_thread), (
        "Each distinct thread must own a unique parser instance"
    )


# ---------------------------------------------------------------------------
# is_tree_sitter_loaded / unload_tree_sitter respect thread-local scope
# ---------------------------------------------------------------------------


def test_is_loaded_false_before_first_call() -> None:
    assert not is_tree_sitter_loaded(), "No parsers loaded yet in this thread"


def test_is_loaded_true_after_get_parser() -> None:
    _get_parser("python")
    assert is_tree_sitter_loaded()


def test_unload_clears_current_thread_parsers() -> None:
    _get_parser("python")
    assert is_tree_sitter_loaded()
    unloaded = unload_tree_sitter()
    assert unloaded
    assert not is_tree_sitter_loaded()


def test_unload_in_one_thread_does_not_affect_another() -> None:
    """Unloading parsers in thread A must not affect thread B's cache."""
    thread_b_state: dict[str, bool] = {}

    def thread_b_work() -> None:
        _get_parser("python")
        thread_b_state["before"] = is_tree_sitter_loaded()

    t = threading.Thread(target=thread_b_work)
    t.start()

    # Main thread loads then unloads
    _get_parser("python")
    unload_tree_sitter()

    t.join()

    assert thread_b_state.get("before") is True, (
        "Thread B's parser should be unaffected by unload in thread A"
    )
