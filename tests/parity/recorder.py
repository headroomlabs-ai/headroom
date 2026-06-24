"""Fixture recorder for the Rust-vs-Python parity harness.

This module is new (no modifications to existing Python under `headroom/`).
It provides a decorator that captures `(input, config, output)` triples and
writes them as JSON fixtures under
`tests/parity/fixtures/<transform_name>/<hash>.json`.

`record_all()` monkey-patches the Phase-1 transform classes in-process so
that every call made during a workload produces a fixture. The
`scripts/record_fixtures.py` entry point drives a minimal synthetic workload
and does not require network access or real LLM calls.

Schema of each fixture file:

```json
{
  "transform":      "log_compressor",
  "input":          "<original input>",
  "config":         { "max_total_lines": 100, ... },
  "output":         "<serialized output>",
  "recorded_at":    "2026-04-23T00:00:00Z",
  "input_sha256":   "<hex digest of canonicalized input>"
}
```
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("headroom.parity.recorder")

# tests/parity/recorder.py -> repo root -> tests/parity/fixtures
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES_ROOT = _REPO_ROOT / "tests" / "parity" / "fixtures"


# ---------------------------------------------------------------------------
# JSON-safe serialization helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Best-effort JSON fallback for dataclasses/enums/bytes."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # enum.Enum
        return obj.value
    if isinstance(obj, set | frozenset):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return repr(obj)


def _to_jsonable(obj: Any) -> Any:
    """Round-trip obj through json to guarantee it's JSON-safe."""
    return json.loads(json.dumps(obj, default=_json_default, sort_keys=True))


def _canonical_digest(payload: Any) -> str:
    blob = json.dumps(payload, default=_json_default, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Core `@record` decorator
# ---------------------------------------------------------------------------


def record(
    transform_name: str,
    *,
    root: Path | None = None,
    input_arg: int = 0,
    input_kw: str | None = None,
    config_attr: str = "config",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a `(input, ...) -> output` callable so every call writes a fixture.

    Args:
        transform_name: Name of the transform (matches Rust comparator name
            and the fixture subdirectory).
        root: Override the fixtures root (mostly for tests).
        input_arg: Positional index of the primary input. Default 0; use
            `input_kw` for keyword-only callables.
        input_kw: Keyword name of the primary input. Takes precedence over
            `input_arg` when the kwarg is present.
        config_attr: Attribute on `self` whose value should be serialized as
            the fixture config. Defaults to `"config"`; methods that have no
            such attribute record an empty config.
    """
    fixtures_root = Path(root) if root is not None else _FIXTURES_ROOT
    out_dir = fixtures_root / transform_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            try:
                _write_fixture(
                    fn=fn,
                    transform_name=transform_name,
                    out_dir=out_dir,
                    args=args,
                    kwargs=kwargs,
                    input_arg=input_arg,
                    input_kw=input_kw,
                    config_attr=config_attr,
                    result=result,
                )
            except Exception as e:  # pragma: no cover - best effort
                LOG.warning("recorder: failed to write fixture for %s: %s", transform_name, e)
            return result

        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _write_fixture(
    *,
    fn: Callable[..., Any],
    transform_name: str,
    out_dir: Path,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_arg: int,
    input_kw: str | None,
    config_attr: str,
    result: Any,
) -> None:
    # Resolve the primary input. If this is a bound method, args[0] is self
    # and the actual input lives at args[input_arg + 1].
    is_method = bool(args) and not isinstance(
        args[0], str | bytes | list | dict | int | float | bool | type(None)
    )
    self_obj = args[0] if is_method else None
    positional_inputs = args[1:] if is_method else args

    if input_kw and input_kw in kwargs:
        primary_input = kwargs[input_kw]
    elif len(positional_inputs) > input_arg:
        primary_input = positional_inputs[input_arg]
    else:
        primary_input = None

    config_obj: Any = {}
    if self_obj is not None and hasattr(self_obj, config_attr):
        config_obj = getattr(self_obj, config_attr)

    payload_input = _to_jsonable(primary_input)
    payload_config = _to_jsonable(config_obj)
    payload_output = _to_jsonable(result)

    digest_source = {
        "transform": transform_name,
        "input": payload_input,
        "config": payload_config,
        "fn": f"{fn.__module__}.{fn.__qualname__}",
    }
    digest = _canonical_digest(digest_source)

    fixture = {
        "transform": transform_name,
        "input": payload_input,
        "config": payload_config,
        "output": payload_output,
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "input_sha256": digest,
    }

    target = out_dir / f"{digest[:16]}.json"
    target.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# record_all(): monkey-patch the Phase-1 transforms
# ---------------------------------------------------------------------------


def record_all(root: Path | None = None) -> dict[str, str]:
    """Monkey-patch the Phase-1 transform classes so every call writes a
    fixture. Returns a dict mapping transform name -> status ("patched",
    "blocked:<reason>").

    Safe to call repeatedly; wrappers are idempotent (they tag themselves).
    """
    statuses: dict[str, str] = {}

    # --- log_compressor ----------------------------------------------------
    try:
        from headroom.transforms.log_compressor import LogCompressor

        _wrap_method(LogCompressor, "compress", "log_compressor", root=root)
        statuses["log_compressor"] = "patched"
    except Exception as e:
        statuses["log_compressor"] = f"blocked:{e.__class__.__name__}:{e}"

    # --- diff_compressor ---------------------------------------------------
    try:
        from headroom.transforms.diff_compressor import DiffCompressor

        _wrap_method(DiffCompressor, "compress", "diff_compressor", root=root)
        statuses["diff_compressor"] = "patched"
    except Exception as e:
        statuses["diff_compressor"] = f"blocked:{e.__class__.__name__}:{e}"

    # --- tokenizer ---------------------------------------------------------
    try:
        from headroom.tokenizer import Tokenizer

        _wrap_method(Tokenizer, "count_text", "tokenizer", root=root)
        statuses["tokenizer"] = "patched"
    except Exception as e:
        statuses["tokenizer"] = f"blocked:{e.__class__.__name__}:{e}"

    # --- cache_aligner -----------------------------------------------------
    # CacheAligner.apply() takes a Tokenizer argument — recording its output
    # requires building a tokenizer. We do that in the workload driver, but
    # still install the patch here so calls made elsewhere are captured.
    try:
        from headroom.transforms.cache_aligner import CacheAligner

        _wrap_method(
            CacheAligner,
            "apply",
            "cache_aligner",
            root=root,
            input_arg=0,  # first non-self positional is `messages`
        )
        statuses["cache_aligner"] = "patched"
    except Exception as e:
        statuses["cache_aligner"] = f"blocked:{e.__class__.__name__}:{e}"

    # --- ccr ---------------------------------------------------------------
    # Phase 0: the Python CCR implementation is split across a tool injector
    # (encoder-side) and a response handler (decoder-side). We record the
    # deterministic synchronous entry point on the injector. The
    # response-handler decode path is async + requires a batch store, so it
    # is recorded only from the workload driver with a real injected call.
    try:
        from headroom.ccr.tool_injection import CCRToolInjector

        _wrap_method(
            CCRToolInjector,
            "inject_tool_definition",
            "ccr",
            root=root,
        )
        statuses["ccr"] = "patched"
    except Exception as e:
        statuses["ccr"] = f"blocked:{e.__class__.__name__}:{e}"

    # --- content_detector --------------------------------------------------
    # `detect_content_type` is a module-level function, so we monkey-patch
    # the module attribute rather than a class method.
    try:
        from headroom.transforms import content_detector as _cd_mod

        _wrap_function(
            _cd_mod,
            "detect_content_type",
            "content_detector",
            root=root,
        )
        statuses["content_detector"] = "patched"
    except Exception as e:
        statuses["content_detector"] = f"blocked:{e.__class__.__name__}:{e}"

    return statuses


def _wrap_method(
    cls: type,
    method_name: str,
    transform_name: str,
    *,
    root: Path | None = None,
    input_arg: int = 0,
    input_kw: str | None = None,
) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_parity_recorder_wrapped", False):
        return  # idempotent

    decorator = record(
        transform_name,
        root=root,
        input_arg=input_arg,
        input_kw=input_kw,
    )
    wrapped = decorator(original)
    wrapped._parity_recorder_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapped)


def _wrap_function(
    module: Any,
    fn_name: str,
    transform_name: str,
    *,
    root: Path | None = None,
    input_arg: int = 0,
    input_kw: str | None = None,
) -> None:
    """Monkey-patch a module-level free function the same way `_wrap_method`
    handles class methods. Idempotent; safe to call repeatedly."""
    original = getattr(module, fn_name)
    if getattr(original, "_parity_recorder_wrapped", False):
        return  # idempotent

    decorator = record(
        transform_name,
        root=root,
        input_arg=input_arg,
        input_kw=input_kw,
    )
    wrapped = decorator(original)
    wrapped._parity_recorder_wrapped = True  # type: ignore[attr-defined]
    setattr(module, fn_name, wrapped)


# ---------------------------------------------------------------------------
# Minimal workload helpers — callable independently of the scripts entry.
# ---------------------------------------------------------------------------


def _varied_log_inputs() -> list[str]:
    """20 varied log-compressor inputs: short, medium, long; pytest/npm/cargo/etc."""
    base_short = [
        "INFO starting\nERROR database connection failed\nINFO shutting down",
        "PASSED test_foo\nPASSED test_bar\nFAILED test_baz\nassert 1 == 2",
    ]
    pytest_output = "\n".join(
        [
            "============================= test session starts ==============================",
            "collected 42 items",
            *[f"tests/test_mod_{i}.py::test_case PASSED [{i * 2}%]" for i in range(25)],
            "tests/test_mod_25.py::test_bad FAILED",
            "=================================== FAILURES ===================================",
            "___________________________________ test_bad ___________________________________",
            "    def test_bad():",
            ">       assert compute(1, 2) == 4",
            "E       assert 3 == 4",
            "tests/test_mod_25.py:17: AssertionError",
            "=========================== short test summary info ============================",
            "FAILED tests/test_mod_25.py::test_bad",
            "1 failed, 25 passed in 0.42s",
        ]
    )
    npm_output = "\n".join(
        [
            "npm WARN deprecated foo@1.0.0: use bar",
            *[f"added {i} packages in 3s" for i in range(5)],
            "npm ERR! code ERESOLVE",
            "npm ERR! ERESOLVE unable to resolve dependency tree",
            "npm ERR! While resolving: project@1.0.0",
            "npm ERR! Found: react@17.0.2",
        ]
    )
    cargo_output = "\n".join(
        [
            *[f"   Compiling crate_{i} v0.1.{i}" for i in range(10)],
            "error[E0308]: mismatched types",
            "  --> src/lib.rs:42:9",
            "   |",
            "42 |         return x;",
            "   |         ^^^^^^^^^ expected `i32`, found `u64`",
            "error: aborting due to previous error",
        ]
    )
    make_output = "make: *** [Makefile:12: all] Error 2\ngcc -c foo.c -o foo.o\nfoo.c:5:3: error: 'undeclared' undeclared"
    big = "\n".join(
        [f"line {i}: INFO processing request" for i in range(300)]
        + [
            "ERROR something broke at step 42",
            "Traceback (most recent call last):",
            '  File "a.py", line 5',
            "RuntimeError: boom",
        ]
    )

    out = list(base_short)
    out.extend([pytest_output] * 3)
    out.extend([npm_output] * 3)
    out.extend([cargo_output] * 4)
    out.append(make_output)
    out.append(big)
    # pad variants so we're at >= 20 unique inputs
    for i in range(20 - len(out)):
        out.append(f"INFO iteration {i}\nERROR error {i}\nWARN warn {i}\nINFO done {i}")
    # tweak each to guarantee uniqueness
    return [f"{s}\n# variant {i}" for i, s in enumerate(out)]


def _varied_diff_inputs() -> list[str]:
    tiny = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
 y = 3
 z = 4
"""
    medium = "\n".join(
        [
            "diff --git a/src/main.rs b/src/main.rs",
            "--- a/src/main.rs",
            "+++ b/src/main.rs",
            "@@ -10,7 +10,7 @@",
            *[f" unchanged_line_{i}" for i in range(5)],
            "-    let x = 1;",
            "+    let x = 2;",
            *[f" unchanged_after_{i}" for i in range(5)],
        ]
    )
    big = []
    for f in range(8):
        big.append(f"diff --git a/file_{f}.py b/file_{f}.py")
        big.append(f"--- a/file_{f}.py")
        big.append(f"+++ b/file_{f}.py")
        big.append("@@ -1,10 +1,12 @@")
        big.extend([f" context_{i}_{f}" for i in range(5)])
        big.extend([f"-removed_{i}_{f}" for i in range(3)])
        big.extend([f"+added_{i}_{f}" for i in range(5)])
        big.extend([f" tail_{i}_{f}" for i in range(5)])
    big_diff = "\n".join(big)

    new_file = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,4 @@
+def hello():
+    return 'world'
+
+x = hello()
"""
    # Bug-fix coverage: each of these exercises a path that was silently
    # dropping information before the 2026-04-25 fix. They produce *new*
    # fixtures (different SHA256), so existing 20 fixtures stay unchanged.
    rename_diff = """diff --git a/auth/old_handler.py b/auth/new_handler.py
similarity index 92%
rename from auth/old_handler.py
rename to auth/new_handler.py
--- a/auth/old_handler.py
+++ b/auth/new_handler.py
@@ -1,8 +1,8 @@
 import os
 import sys
-from auth import legacy
+from auth import modern

 def authenticate(user):
     return user.is_valid()
"""
    combined_diff = """diff --git a/merge_target.py b/merge_target.py
--- a/merge_target.py
+++ b/merge_target.py
@@@ -1,5 -1,5 +1,6 @@@
  unchanged_a
  unchanged_b
- old_from_branch_1
 -old_from_branch_2
++new_in_merge
 +new_added_too
  unchanged_c
"""
    no_newline_diff = """diff --git a/last.txt b/last.txt
--- a/last.txt
+++ b/last.txt
@@ -1,8 +1,8 @@
-old_first
+new_first
 ctx_a
 ctx_b
 ctx_c
 ctx_d
 ctx_e
 ctx_f
\\ No newline at end of file
"""
    pre_diff_content = (
        """commit abc1234567890abcdef
Author: Test <t@example.com>
Date:   Mon Apr 25 12:00:00 2026

    Refactor: rename and modify auth module

"""
        + rename_diff
    )

    out: list[str] = []
    for i in range(7):
        out.append(f"{tiny}# variant {i}")
    for i in range(6):
        out.append(f"{medium}\n# variant {i}")
    for i in range(4):
        out.append(f"{big_diff}\n# variant {i}")
    for i in range(3):
        out.append(f"{new_file}# variant {i}")
    # Bug-fix path coverage. Padded with the same `# variant N` trick used
    # above so each input is unique (avoids fixture-hash collisions).
    out.append(f"{rename_diff}\n# bugfix:rename")
    out.append(f"{combined_diff}\n# bugfix:combined-diff-3way")
    out.append(f"{no_newline_diff}\n# bugfix:no-newline-marker")
    out.append(f"{pre_diff_content}\n# bugfix:pre-diff-content")

    # Routing-gap path coverage (2026-04-25 follow-up). These exercise the
    # ContentRouter→DiffCompressor pipeline gaps:
    #   1. `diff --combined <path>` merge-commit header (parser had hardcoded
    #      `diff --git`, so the whole input fell into pre-diff blob).
    #   2. `diff --cc <path>` (alternate merge-commit form).
    #   3. Long pre-diff content (>50 lines) that previously slipped past
    #      the detector's first-50-lines scan window.
    merge_combined_diff = """diff --combined merge_target.py
index abc..def..ghi 100644
--- a/merge_target.py
+++ b/merge_target.py
@@@ -1,4 -1,4 +1,5 @@@
  unchanged_a
- old_branch_1
 -old_branch_2
++new_in_merge
 +new_added
  unchanged_b
"""
    merge_cc_diff = """diff --cc cc_target.py
index abc..def..ghi
--- a/cc_target.py
+++ b/cc_target.py
@@@ -1,3 -1,3 +1,4 @@@
  ctx
- removed_p1
 -removed_p2
++added_in_merge
  more_ctx
"""
    long_pre_diff = (
        "commit 0123456789abcdef\n"
        "Author: Tester <t@example.com>\n"
        "Date:   Mon Apr 25 12:00:00 2026\n"
        "\n" + "\n".join(f"    msg line {i}" for i in range(60)) + "\n\n" + rename_diff
    )
    out.append(f"{merge_combined_diff}\n# bugfix:diff-combined")
    out.append(f"{merge_cc_diff}\n# bugfix:diff-cc")
    out.append(f"{long_pre_diff}\n# bugfix:long-pre-diff")
    return out


def _varied_text_inputs() -> list[str]:
    base = [
        "",
        "hello",
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "def foo(x):\n    return x + 1\n\nclass Bar:\n    pass",
        "SELECT id, name FROM users WHERE email LIKE '%@example.com'",
        "a" * 1000,
        "a\nb\nc\n" * 100,
        '{"role": "user", "content": "hi"}',
        "🚀 unicode ✨ emoji 中文 русский",
    ]
    out: list[str] = []
    for i in range(2):
        for b in base:
            out.append(f"{b}\n[[variant {i}]]")
    return out[:20]


def _varied_content_detector_inputs() -> list[str]:
    """Hit every dispatch branch in `detect_content_type`. Each entry below
    targets a specific path so parity diffs surface the right branch."""
    json_array_dicts = '[{"id":1,"name":"a"},{"id":2,"name":"b"},{"id":3,"name":"c"}]'
    json_array_scalars = "[1, 2, 3, 4, 5]"
    json_empty_array = "[]"
    json_object = '{"foo": "bar"}'

    git_diff = (
        "diff --git a/foo.py b/foo.py\n"
        "index abc..def 100644\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old line\n"
        "+new line\n"
        " unchanged\n"
    )
    merge_diff = (
        "diff --combined merged.py\n"
        "index aaa..bbb..ccc 100644\n"
        "--- a/merged.py\n"
        "+++ b/merged.py\n"
        "@@@ -1,4 -1,4 +1,5 @@@\n"
        "  unchanged\n"
        "- branch_a_only\n"
        " -branch_b_only\n"
        "++merge_added\n"
    )

    html_doctype = (
        "<!DOCTYPE html>\n<html>\n<head><title>x</title></head>\n"
        "<body><div><span>hi</span></div></body>\n</html>"
    )
    html_structural = "<div>a</div>\n<span>b</span>\n<script>x()</script>\n<style>y</style>"
    html_below_threshold = "<p>just one tag</p>"

    search_results = (
        "src/main.py:42:def process():\n"
        "src/util.py:13:    return None\n"
        "lib/x.py:7:class X:\n"
        "tests/test_a.py:3:    assert True"
    )

    build_log = (
        "INFO starting build\n"
        "WARN deprecated API used\n"
        "ERROR compilation failed\n"
        "FAILED test_x\n"
        "PASSED test_y\n"
        "============================================\n"
    )
    pytest_output = (
        "============================= test session starts ==============================\n"
        + "\n".join(f"tests/test_{i}.py::test_case PASSED" for i in range(5))
        + "\nFAILED tests/test_5.py::test_bad\n"
        + "Traceback (most recent call last)\n"
    )

    python_code = (
        "import os\n"
        "from typing import Any\n\n"
        "@dataclass\n"
        "class Foo:\n"
        '    """Docstring."""\n'
        "    def bar(self):\n"
        "        return 42\n\n"
        "def baz():\n"
        "    pass\n\n"
        "if __name__ == '__main__':\n"
        "    baz()\n"
    )
    js_code = (
        "import x from 'y';\n"
        "export const foo = 1;\n"
        "function bar() { return 42; }\n"
        "const f = async function() {};\n"
        "module.exports = { foo, bar };\n"
    )
    ts_code = (
        "interface User { id: number; name: string; }\n"
        "type Maybe<T> = T | null;\n"
        "enum Color { Red, Green, Blue }\n"
        "function f(x: number): boolean { return x > 0; }\n"
    )
    go_code = (
        "package main\n\n"
        'import "fmt"\n\n'
        "type Foo struct { ID int }\n\n"
        "func (f *Foo) Bar() int { return f.ID }\n\n"
        'func main() { fmt.Println("hi") }\n'
    )
    rust_code = (
        "use std::collections::HashMap;\n\n"
        "pub struct Foo { id: u32 }\n\n"
        "impl Foo {\n"
        "    pub fn new() -> Self { Self { id: 0 } }\n"
        "}\n\n"
        "fn main() {}\n"
        "#[derive(Debug)]\n"
        "enum Color { Red, Green }\n"
    )
    java_code = (
        "package com.example;\n\n"
        "public class Foo {\n"
        "    @Override\n"
        '    public String toString() { return "foo"; }\n'
        "}\n\n"
        "private interface Bar {}\n"
        "protected enum Baz { A, B }\n"
    )

    plain_text = "Just some prose text without any structure or special markers."
    empty = ""
    whitespace = "   \n\t  \n"

    return [
        json_array_dicts,
        json_array_scalars,
        json_empty_array,
        json_object,
        git_diff,
        merge_diff,
        html_doctype,
        html_structural,
        html_below_threshold,
        search_results,
        build_log,
        pytest_output,
        python_code,
        js_code,
        ts_code,
        go_code,
        rust_code,
        java_code,
        plain_text,
        empty,
        whitespace,
    ]


def _varied_message_batches() -> list[list[dict[str, Any]]]:
    today = _dt.date.today().isoformat()
    out: list[list[dict[str, Any]]] = []
    for i in range(20):
        out.append(
            [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. The date is {today}. Request id {i}.",
                },
                {"role": "user", "content": f"Question number {i}: what is 2+2?"},
            ]
        )
    return out


def run_default_workload(root: Path | None = None) -> dict[str, int]:
    """Drive synthetic inputs through every patched transform. Returns a
    dict mapping transform name to count of fixtures produced."""
    counts: dict[str, int] = {
        "log_compressor": 0,
        "diff_compressor": 0,
        "tokenizer": 0,
        "cache_aligner": 0,
        "ccr": 0,
        "content_detector": 0,
    }

    # log_compressor
    try:
        from headroom.transforms.log_compressor import LogCompressor

        lc = LogCompressor()
        for s in _varied_log_inputs():
            lc.compress(s)
            counts["log_compressor"] += 1
    except Exception as e:
        LOG.warning("log_compressor workload failed: %s", e)

    # diff_compressor
    try:
        from headroom.transforms.diff_compressor import DiffCompressor

        dc = DiffCompressor()
        for s in _varied_diff_inputs():
            dc.compress(s)
            counts["diff_compressor"] += 1
    except Exception as e:
        LOG.warning("diff_compressor workload failed: %s", e)

    # tokenizer
    try:
        from headroom.providers.openai import OpenAITokenCounter
        from headroom.tokenizer import Tokenizer

        tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")
        for s in _varied_text_inputs():
            tok.count_text(s)
            counts["tokenizer"] += 1
    except Exception as e:
        LOG.warning("tokenizer workload failed: %s", e)

    # cache_aligner — needs Tokenizer; reuse the one above
    try:
        from headroom.providers.openai import OpenAITokenCounter
        from headroom.tokenizer import Tokenizer
        from headroom.transforms.cache_aligner import CacheAligner

        tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")
        aligner = CacheAligner()
        for batch in _varied_message_batches():
            aligner.apply(batch, tok)
            counts["cache_aligner"] += 1
    except Exception as e:
        LOG.warning("cache_aligner workload failed: %s", e)

    # ccr — CCRToolInjector is a dataclass whose `inject_tool_definition`
    # takes `tools: list[dict] | None` and returns `(tools, was_injected)`.
    # It only mutates state when it has already scanned messages with
    # compression markers, so we force `has_compressed_content` by planting
    # a hash in the detected set directly.
    try:
        from headroom.ccr.tool_injection import CCRToolInjector

        for i in range(25):
            injector = CCRToolInjector(provider="anthropic" if i % 2 == 0 else "openai")
            # Plant a unique 24-hex-char hash per iteration so the injector
            # treats each call as having compressed content.
            planted_hash = hashlib.sha256(f"planted-{i}".encode()).hexdigest()[:24]
            injector._detected_hashes.append(planted_hash)  # noqa: SLF001
            # Always include a unique marker tool in the list so input
            # hashes never collide across iterations.
            existing_tools: list[dict[str, Any]] | None = [
                {"name": f"other_tool_{i}", "description": f"desc {i}"}
            ]
            try:
                injector.inject_tool_definition(existing_tools)
                counts["ccr"] += 1
            except Exception as e:
                LOG.debug("ccr inject failed on input %d: %s", i, e)
    except Exception as e:
        LOG.warning("ccr workload failed: %s", e)

    # content_detector — drive a wide mix of content types so every dispatch
    # branch (json_array, diff, html, search, log, code-by-language,
    # plain-text fallback) is exercised at least once.
    try:
        from headroom.transforms import content_detector as _cd_mod

        for s in _varied_content_detector_inputs():
            _cd_mod.detect_content_type(s)
            counts["content_detector"] += 1
    except Exception as e:
        LOG.warning("content_detector workload failed: %s", e)

    return counts


__all__ = [
    "record",
    "record_all",
    "run_default_workload",
]
