"""Tests for LeanContext window-based truncation."""
from headroom.transforms.lean_context import LeanContext, TruncationResult


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
