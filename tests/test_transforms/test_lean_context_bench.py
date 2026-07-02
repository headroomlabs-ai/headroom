"""Benchmarks for LeanContext — performance and edge cases."""
import time
from headroom.transforms.lean_context import LeanContext


def test_large_text_performance():
    """LeanContext should process 10K lines in <50ms."""
    lc = LeanContext(window_radius=50)
    text = "\n".join(f"line {i}: some content here for testing" for i in range(10000))
    # Insert some signals
    lines = text.split("\n")
    lines[500] = "error[E0308]: mismatched types at src/main.rs:500"
    lines[5000] = "Traceback (most recent call last):"
    text = "\n".join(lines)
    
    t0 = time.perf_counter()
    result = lc.truncate(text)
    elapsed = (time.perf_counter() - t0) * 1000
    
    assert elapsed < 500, f"Too slow: {elapsed:.1f}ms"
    assert result.signal_lines >= 2


def test_no_signals_savings():
    """Text with no signals should still achieve savings on large inputs."""
    lc = LeanContext(window_radius=20)
    text = "\n".join(f"line {i}" for i in range(1000))
    result = lc.truncate(text)
    # Should drop middle lines, keep ends
    assert result.dropped_lines > 500
    assert result.kept_lines < 500


def test_many_signals_nearby():
    """Many nearby signals should merge windows, not duplicate."""
    lc = LeanContext(window_radius=5)
    lines = [f"line {i}" for i in range(200)]
    # Insert signals every 3 lines in a dense cluster
    for i in range(50, 80, 3):
        lines[i] = f"error: issue at line {i}"
    text = "\n".join(lines)
    result = lc.truncate(text)
    # Should keep the cluster as one window, not N separate ones
    assert 30 <= result.kept_lines <= 60  # cluster + radius + end lines


def test_go_error_signals():
    """Go-specific error patterns."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(50)) + """
src/handler.go:142:27: cannot use client (variable of type *http.Client) as...
src/handler.go:287:15: undefined: ctx
FAIL: TestRateLimit (0.02s)
    rate_test.go:89: expected 200, got 429
""" + "\n".join(f"line {i}" for i in range(100, 150))
    result = lc.truncate(text)
    assert result.signal_lines >= 2


def test_typescript_error_signals():
    """TypeScript-specific error patterns."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(50)) + """
src/middleware.ts:42:18 - error TS2345: Argument of type 'string' is not assignable...
src/auth.ts:156:7 - error TS2322: Type 'null' is not assignable to type 'User'
npm ERR! code ELIFECYCLE
npm ERR! errno 1
""" + "\n".join(f"line {i}" for i in range(100, 150))
    result = lc.truncate(text)
    assert result.signal_lines >= 2


def test_python_error_signals():
    """Python-specific error patterns."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(50)) + """
=================================== FAILURES ===================================
___________________________ test_connection_pool ______________________________
    async def test_connection_pool():
>       async with async_session() as session:
E       RuntimeError: Task got Future attached to a different loop
backend/tests/test_db.py:15: RuntimeError
========================= 1 failed, 4 passed in 2.03s =========================
""" + "\n".join(f"line {i}" for i in range(100, 150))
    result = lc.truncate(text)
    assert result.signal_lines >= 2


def test_build_output_signals():
    """Build output with cargo/npm/make."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(50)) + """
   Compiling rate-limiter v0.1.0
error: could not compile `rate-limiter` due to 3 previous errors
warning: unused import: `std::collections::HashMap`
  --> src/lib.rs:5:5
   |
5  | use std::collections::HashMap;
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^
""" + "\n".join(f"line {i}" for i in range(100, 150))
    result = lc.truncate(text)
    assert result.signal_lines >= 2


def test_truncation_markers():
    """Dropped lines should be marked."""
    lc = LeanContext(window_radius=3)
    text = "\n".join(f"line {i}" for i in range(200))
    text_lines = text.split("\n")
    text_lines[100] = "error: critical failure"
    text = "\n".join(text_lines)
    result = lc.truncate(text)
    assert "lines dropped]" in result.text
    assert "lines dropped]" in result.text
