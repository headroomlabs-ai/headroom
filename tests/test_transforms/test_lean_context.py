"""Tests for LeanContext window-based truncation."""

from headroom.transforms.lean_context import LeanContext


def test_no_signals_keeps_ends():
    """Text with no signals should keep first and last N lines."""
    lc = LeanContext(window_radius=10)
    text = "\n".join(f"line {i}" for i in range(100))
    result = lc.truncate(text)
    assert result.dropped_lines > 0
    assert result.kept_lines < 100
    assert result.signal_lines == 0


def test_short_text_kept_entirely():
    """Short text should not be truncated."""
    lc = LeanContext(window_radius=50)
    text = "\n".join(f"line {i}" for i in range(30))
    result = lc.truncate(text)
    assert result.dropped_lines == 0
    assert result.kept_lines == 30


def test_error_signal_keeps_context():
    """Lines near an error should be kept."""
    lc = LeanContext(window_radius=5)
    lines = [f"line {i}" for i in range(100)]
    lines[50] = "error: something went wrong"
    text = "\n".join(lines)
    result = lc.truncate(text)
    # Lines 45-55 should be kept (50 - 5 to 50 + 5)
    assert result.kept_lines >= 10
    assert result.signal_lines >= 1


def test_traceback_signal():
    """Python traceback should be detected as signal."""
    lc = LeanContext(window_radius=5)
    text = """line 0
line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10
line 11
line 12
line 13
line 14
line 15
line 16
line 17
line 18
line 19
line 20
line 21
line 22
line 23
line 24
line 25
line 26
line 27
line 28
line 29
Traceback (most recent call last):
  File "app.py", line 42, in process
    result = data["key"]
KeyError: 'key'
line 100
line 101
line 102
line 103
line 104
line 105
line 106
line 107
line 108
line 109
line 110
line 111
line 112
line 113
line 114
line 115
line 116
line 117
line 118
line 119
line 120
line 121
line 122
line 123
line 124
line 125
line 126
line 127
line 128
line 129"""
    result = lc.truncate(text)
    assert result.signal_lines >= 1
    assert result.kept_lines < len(text.split("\n"))


def test_rustc_error_signal():
    """Rust compiler error should be detected."""
    lc = LeanContext(window_radius=5)
    text = """line 0
line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10
line 11
line 12
line 13
line 14
line 15
line 16
line 17
line 18
line 19
line 20
line 21
line 22
line 23
line 24
line 25
line 26
line 27
line 28
line 29
error[E0308]: mismatched types
  --> src/main.rs:13:74
   |
13 |     let x: i32 = "hello";
   |                  ^^^^^^^ expected i32, found &str
line 100
line 101
line 102
line 103
line 104
line 105
line 106
line 107
line 108
line 109
line 110
line 111
line 112
line 113
line 114
line 115
line 116
line 117
line 118
line 119
line 120
line 121
line 122
line 123
line 124
line 125
line 126
line 127
line 128
line 129"""
    result = lc.truncate(text)
    assert result.signal_lines >= 1
    assert result.kept_lines < len(text.split("\n"))


def test_diff_signal():
    """Git diff should be detected as signal."""
    lc = LeanContext(window_radius=5)
    text = """line 0
line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10
line 11
line 12
line 13
line 14
line 15
line 16
line 17
line 18
line 19
line 20
line 21
line 22
line 23
line 24
line 25
line 26
line 27
line 28
line 29
--- a/src/main.rs
+++ b/src/main.rs
@@ -10,6 +10,8 @@
 unchanged line
+added line
-removed line
line 100
line 101
line 102
line 103
line 104
line 105
line 106
line 107
line 108
line 109
line 110
line 111
line 112
line 113
line 114
line 115
line 116
line 117
line 118
line 119
line 120
line 121
line 122
line 123
line 124
line 125
line 126
line 127
line 128
line 129"""
    result = lc.truncate(text)
    assert result.signal_lines >= 1


def test_empty_text():
    """Empty text should not crash."""
    lc = LeanContext()
    result = lc.truncate("")
    assert result.original_lines == 0


def test_savings_pct():
    """Savings percentage should be correct."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(100))
    result = lc.truncate(text)
    assert 0 < result.savings_pct < 100


def test_no_spurious_leading_drop_marker():
    """Long output where line 0 is retained must not insert a leading drop marker."""
    lc = LeanContext(window_radius=5)
    text = "\n".join(f"line {i}" for i in range(100))
    result = lc.truncate(text)
    output_lines = result.text.split("\n")
    assert output_lines[0] == "line 0"
    assert not output_lines[0].startswith("... [")


def test_standalone_helper_does_not_inflate_proxy_stats():
    """Standalone apply_leanctx_python calls must not add to proxy request totals."""
    from headroom.proxy.helpers import (
        apply_leanctx_python,
        get_leanctx_python_stats,
        reset_leanctx_python_stats,
    )

    reset_leanctx_python_stats()
    text = "\n".join(f"line {i}" for i in range(200))
    out = apply_leanctx_python(text, window_radius=10)
    assert "... [" in out
    stats = get_leanctx_python_stats()
    assert stats["calls"] == 1
    assert stats["tokens_saved"] > 0
    assert stats["proxy_calls"] == 0
    assert stats["proxy_tokens_saved"] == 0
    assert stats["proxy_attempted_tokens"] == 0
    reset_leanctx_python_stats()


def test_apply_leanctx_to_messages_proxy_accounting():
    """Proxy message truncation updates proxy_tokens_saved and proxy_attempted_tokens."""
    from headroom.proxy.helpers import (
        apply_leanctx_to_messages,
        get_leanctx_python_stats,
        reset_leanctx_python_stats,
    )

    reset_leanctx_python_stats()
    long_tool_output = "\n".join(
        ["error: compilation failed at line 50" if i == 50 else f"log line {i}" for i in range(250)]
    )
    messages = [
        {"role": "user", "content": "Run the build"},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": long_tool_output,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_456", "content": long_tool_output},
    ]

    updated, dropped = apply_leanctx_to_messages(
        messages, window_radius=10, record_proxy_stats=True
    )
    assert dropped > 0
    assert updated is not messages
    assert "error: compilation failed at line 50" in updated[1]["content"][0]["content"]
    assert "... [" in updated[1]["content"][0]["content"]
    assert "... [" in updated[2]["content"]

    stats = get_leanctx_python_stats()
    assert stats["proxy_calls"] == 2
    assert stats["proxy_lines_dropped"] == dropped
    assert 0 < stats["proxy_tokens_saved"] <= stats["proxy_attempted_tokens"]
    reset_leanctx_python_stats()
