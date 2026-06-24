"""Tests for tree-sitter Parser thread safety (issue #562).

Verifies that code-aware compression works correctly when parsers are
used from ThreadPoolExecutor workers, which is how the proxy runs
compression in production.
"""

import concurrent.futures
import threading

import pytest

from headroom.transforms.code_compressor import (
    _check_tree_sitter_available,
    _get_parser,
    is_tree_sitter_loaded,
    unload_tree_sitter,
)

pytestmark = pytest.mark.skipif(
    not _check_tree_sitter_available(),
    reason="tree-sitter not installed (pip install headroom-ai[code])",
)

PYTHON_CODE = "def hello():\n    return 42\n"
JS_CODE = "function hello() {\n    return 42;\n}\n"


class TestParserThreadLocal:
    """Verify that _get_parser returns thread-local instances."""

    def test_same_thread_returns_same_parser(self):
        """Calling _get_parser twice on the same thread returns the same object."""
        p1 = _get_parser("python")
        p2 = _get_parser("python")
        assert p1 is p2

    def test_different_languages_return_different_parsers(self):
        """Different languages get distinct parser instances."""
        py = _get_parser("python")
        js = _get_parser("javascript")
        assert py is not js

    def test_different_threads_get_different_parsers(self):
        """Each thread must get its own Parser to avoid the unsendable panic."""
        main_parser = _get_parser("python")
        worker_parser = [None]

        def grab_parser():
            worker_parser[0] = _get_parser("python")

        t = threading.Thread(target=grab_parser)
        t.start()
        t.join()

        assert worker_parser[0] is not None
        assert worker_parser[0] is not main_parser

    def test_is_tree_sitter_loaded_per_thread(self):
        """is_tree_sitter_loaded reflects the current thread's state."""
        # Ensure current thread has a parser
        _get_parser("python")
        assert is_tree_sitter_loaded() is True

        # A fresh thread should report not loaded
        result = [None]

        def check():
            result[0] = is_tree_sitter_loaded()

        t = threading.Thread(target=check)
        t.start()
        t.join()
        assert result[0] is False

    def test_unload_tree_sitter_per_thread(self):
        """unload_tree_sitter only affects the calling thread."""
        _get_parser("python")
        assert is_tree_sitter_loaded() is True

        # Load a parser on a worker, then unload on main — worker unaffected
        worker_loaded_after = [None]

        def worker():
            _get_parser("python")
            # Wait for main thread to unload
            event.wait()
            worker_loaded_after[0] = is_tree_sitter_loaded()

        event = threading.Event()
        t = threading.Thread(target=worker)
        t.start()

        unload_tree_sitter()
        assert is_tree_sitter_loaded() is False

        event.set()
        t.join()
        assert worker_loaded_after[0] is True


class TestParserCrossThreadParsing:
    """Verify that parsing works from ThreadPoolExecutor workers."""

    def test_parse_from_single_worker(self):
        """A parser created and used on the same worker thread works."""

        def parse_on_worker():
            parser = _get_parser("python")
            tree = parser.parse(bytes(PYTHON_CODE, "utf-8"))
            return tree.root_node.child_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(parse_on_worker).result()
        assert result > 0

    def test_parse_from_multiple_workers(self):
        """Multiple workers can parse concurrently without panics."""
        results = []
        errors = []

        def parse_on_worker(code: str, lang: str):
            parser = _get_parser(lang)
            tree = parser.parse(bytes(code, "utf-8"))
            return tree.root_node.child_count

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for _ in range(8):
                futures.append(pool.submit(parse_on_worker, PYTHON_CODE, "python"))
                futures.append(pool.submit(parse_on_worker, JS_CODE, "javascript"))

            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except BaseException as e:
                    # PyO3's PanicException is a BaseException, not an Exception.
                    errors.append(e)

        assert not errors, f"Cross-thread parsing errors: {errors}"
        assert len(results) == 16
        assert all(r > 0 for r in results)

    def test_repeated_parse_same_worker(self):
        """The same worker can parse repeatedly (parser reuse works)."""

        def parse_many():
            counts = []
            for _ in range(10):
                parser = _get_parser("python")
                tree = parser.parse(bytes(PYTHON_CODE, "utf-8"))
                counts.append(tree.root_node.child_count)
            return counts

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            counts = pool.submit(parse_many).result()

        assert len(counts) == 10
        assert all(c > 0 for c in counts)
        # All parses of the same code should give the same result
        assert len(set(counts)) == 1
