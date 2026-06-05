#!/usr/bin/env python3
"""Reproducer for tree-sitter Parser unsendable panic (issue #562).

Demonstrates that tree-sitter ≥ 0.23 marks Parser as PyO3
#[pyclass(unsendable)] — it hard-panics if accessed from a different
thread than its creator.

The OLD pattern (shared dict + lock) triggers the panic.
The NEW pattern (thread-local storage) works correctly.

Usage:
    python repro_unsendable_panic.py
"""

import concurrent.futures
import threading

CODE = b"def hello():\n    return 42\n"


def test_old_pattern_shared_dict():
    """OLD pattern: shared parser dict with a lock — PANICS.

    The bug is triggered by tree_sitter_language_pack.get_parser(), which
    returns the Rust/PyO3 #[pyclass(unsendable)] parser (module `_native`).
    NOT tree_sitter.Parser — that is a C extension with no thread affinity,
    so sharing it across threads works fine and never reproduces the panic.
    """
    print("=== OLD pattern: shared dict + lock ===")
    from tree_sitter_language_pack import get_parser

    lock = threading.Lock()
    shared_parsers: dict = {}

    def get_shared_parser(lang: str):
        with lock:
            if lang not in shared_parsers:
                shared_parsers[lang] = get_parser(lang)
            return shared_parsers[lang]

    # Create the parser on the main thread
    parser = get_shared_parser("python")
    print(f"  Created parser on {threading.current_thread().name}")

    # Access it from a pool thread — this triggers the panic
    def use_parser():
        thread = threading.current_thread().name
        try:
            tree = parser.parse(CODE)
            print(f"  Parsed on {thread}: {tree.root_node.child_count} children")
        except BaseException as e:
            # PyO3's PanicException is a BaseException, not an Exception.
            print(f"  {type(e).__name__} on {thread}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(use_parser) for _ in range(4)]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except BaseException as e:
                print(f"  Future {type(e).__name__}: {e}")


def test_new_pattern_thread_local():
    """NEW pattern: thread-local parsers — works correctly."""
    print("\n=== NEW pattern: thread-local storage ===")
    _local = threading.local()

    def get_thread_local_parser(lang: str):
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language

        parsers = getattr(_local, "parsers", None)
        if parsers is None:
            parsers = {}
            _local.parsers = parsers
        if lang not in parsers:
            p = Parser()
            p.language = get_language(lang)
            parsers[lang] = p
            print(f"  Created parser on {threading.current_thread().name}")
        return parsers[lang]

    def use_parser():
        thread = threading.current_thread().name
        parser = get_thread_local_parser("python")
        tree = parser.parse(CODE)
        print(f"  Parsed on {thread}: {tree.root_node.child_count} children")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(use_parser) for _ in range(4)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    print("  All tasks completed successfully!")


if __name__ == "__main__":
    print("Reproducer for tree-sitter Parser unsendable panic\n")

    test_new_pattern_thread_local()

    print("\nAbout to run the OLD pattern.")
    print("PyO3 raises a PanicException on the worker thread (surfaced via")
    print("future.result()) rather than killing the process. Seeing")
    print("'PanicException: _native::Parser is unsendable, but sent to")
    print("another thread' confirms the bug.\n")

    try:
        test_old_pattern_shared_dict()
    except BaseException as e:
        print(f"\nCaught {type(e).__name__}: {e}")
