"""Language-aware stack-trace detection and frame collapse.

Covers the new trace flavors (Go panics/goroutine dumps, Rust panics with
backtraces, .NET inner-exception chains, Java `Caused by:` continuation) and
the runtime-frame collapse that replaces blind tail truncation for oversized
traces. The Rust unit tests pin the state-machine behavior; these tests pin
the Python-visible surface: content detection, the shim's mirrored patterns,
and end-to-end compression through `LogCompressor`.
"""

from __future__ import annotations

import pytest

from headroom.transforms.content_detector import ContentType, detect_content_type
from headroom.transforms.log_compressor import LogCompressor, LogCompressorConfig

# Fixtures --------------------------------------------------------------------


def go_panic_dump(goroutines: int = 24) -> str:
    lines = [
        "panic: runtime error: invalid memory address or nil pointer dereference",
        "",
        "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x4a2b3c]",
        "",
        "goroutine 1 [running]:",
        "main.handler(0xc000010000)",
        "\t/app/cmd/server/main.go:42 +0x1d",
    ]
    for g in range(2, goroutines + 2):
        lines += [
            f"goroutine {g} [chan receive]:",
            "runtime.gopark(0x0, 0x0, 0x0, 0x0, 0x0)",
            "\t/usr/local/go/src/runtime/proc.go:381 +0xd6",
            "runtime.chanrecv(0xc00006e0c0, 0x0, 0x1, 0x0)",
            "\t/usr/local/go/src/runtime/chan.go:583 +0x49d",
            "",
        ]
    return "\n".join(lines)


def rust_panic_backtrace(frames: int = 20) -> str:
    lines = [
        "thread 'main' panicked at src/main.rs:5:5:",
        "index out of bounds: the len is 3 but the index is 99",
        "stack backtrace:",
    ]
    for i in range(frames):
        lines += [
            f"  {i}: core::panicking::panic_fmt",
            f"            at /rustc/abc123/library/core/src/panicking.rs:{i + 1}:5",
        ]
    lines += ["  20: app::run", "            at ./src/main.rs:5:5"]
    lines.append("note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace")
    return "\n".join(lines)


def java_chained_trace(runtime_frames: int = 30) -> str:
    lines = ['Exception in thread "main" java.lang.IllegalStateException: boom']
    lines.append("at com.example.App.handle(App.java:10)")
    lines.append("at com.example.App.dispatch(App.java:20)")
    for i in range(runtime_frames):
        lines.append(f"at java.base/java.util.stream.Op{i}.eval(Op{i}.java:{i + 1})")
    lines.append("Caused by: java.io.IOException: disk gone")
    lines.append("at com.example.Disk.read(Disk.java:77)")
    for i in range(runtime_frames):
        lines.append(f"at java.base/java.lang.Thread{i}.run(Thread.java:{i + 1})")
    lines.append("... 17 more")
    return "\n".join(lines)


def dotnet_trace() -> str:
    return "\n".join(
        [
            "Unhandled exception. System.InvalidOperationException: outer failed",
            " ---> System.ArgumentNullException: inner value was null",
            "   at App.Data.Load(String path) in /src/App/Data.cs:line 42",
            "   --- End of inner exception stack trace ---",
            "   at App.Program.Main(String[] args) in /src/App/Program.cs:line 12",
        ]
    )


# Detection -------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        go_panic_dump(),
        rust_panic_backtrace(),
        java_chained_trace(),
        # .NET trace padded with neutral build lines to clear line-count floors
        dotnet_trace() + "\n" + "\n".join(f"Restore complete {i}" for i in range(6)),
    ],
    ids=["go", "rust", "java", "dotnet"],
)
def test_traces_detected_as_build_output(content: str) -> None:
    result = detect_content_type(content)
    assert result.content_type is ContentType.BUILD_OUTPUT


def test_node_async_frames_count_toward_log_detection() -> None:
    content = "\n".join(
        [
            "Error: connect ECONNREFUSED 127.0.0.1:5432",
            "    at async Database.connect (/app/src/db.js:14:3)",
            "    at async Pool.acquire (/app/src/pool.js:88:9)",
            "    at async handleRequest (/app/src/routes.js:31:5)",
            "    at async main (/app/src/index.js:5:1)",
        ]
        + [f"request {i} handled" for i in range(10)]
    )
    result = detect_content_type(content)
    assert result.content_type is ContentType.BUILD_OUTPUT


# Shim pattern mirror ----------------------------------------------------------


def test_parse_lines_marks_new_flavor_openers() -> None:
    compressor = LogCompressor()
    lines = [
        "panic: boom",
        "goroutine 7 [select]:",
        "\t/app/main.go:10 +0x20",
        "thread 'main' panicked at src/lib.rs:1:1:",
        "stack backtrace:",
        "   0: rust_begin_unwind",
        "Unhandled exception. System.Exception: x",
        "   at App.Main(String[] a) in /src/P.cs:line 3",
        "Caused by: java.io.IOException: y",
        "   ... 3 more",
        "",  # blank resets the shim's legacy in-trace latch
        "plain line",
    ]
    parsed = compressor._parse_lines(lines)
    flags = [ln.is_stack_trace for ln in parsed]
    assert all(flags[:-2]), f"unmarked opener among {flags}"
    assert not flags[-1]


# End-to-end compression --------------------------------------------------------


def test_go_dump_collapses_runtime_frames_keeps_panic_and_app_frame() -> None:
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=False))
    content = go_panic_dump()
    result = compressor.compress(content)
    assert result.compressed_line_count < result.original_line_count
    assert "panic: runtime error" in result.compressed
    assert "main.handler" in result.compressed  # the app frame
    assert "frames collapsed]" in result.compressed
    # The runtime scheduler noise does not dominate the output.
    assert result.compressed.count("runtime.gopark") <= 2


def test_java_chain_heads_survive_collapse() -> None:
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=False, min_lines_for_ccr=10))
    result = compressor.compress(java_chained_trace())
    assert "Caused by: java.io.IOException" in result.compressed
    assert "com.example.Disk.read" in result.compressed
    assert "... 17 more" in result.compressed
    assert "frames collapsed]" in result.compressed


def test_collapse_can_be_disabled() -> None:
    compressor = LogCompressor(LogCompressorConfig(enable_ccr=False, collapse_runtime_frames=False))
    result = compressor.compress(go_panic_dump())
    assert "frames collapsed]" not in result.compressed


def test_collapse_config_plumbs_through() -> None:
    # Constructor accepts the new knobs and forwards them to Rust without error.
    compressor = LogCompressor(
        LogCompressorConfig(trace_head_frames=1, trace_app_frames=2, enable_ccr=False)
    )
    result = compressor.compress(go_panic_dump())
    assert result.compressed  # smoke: no TypeError from the PyO3 signature
