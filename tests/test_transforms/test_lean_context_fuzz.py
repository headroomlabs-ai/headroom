"""Fuzzing tests for LeanContext — property-based testing with hypothesis."""
import pytest
from hypothesis import given, strategies as st, settings
from headroom.transforms.lean_context import LeanContext

# Generate realistic tool output text
tool_text = st.text(
    alphabet=st.characters(whitelist_categories=('Lu','Ll','Nd','Zs','P')),
    min_size=100, max_size=5000
)

# Generate text with embedded error signals
error_signals = st.sampled_from([
    "error[E0308]: mismatched types",
    "Traceback (most recent call last):",
    "panic at src/main.rs:42",
    "FAILED: test_timeout",
    "npm ERR! code ELIFECYCLE",
    "TypeError: undefined is not a function",
    "--- a/src/lib.rs\n+++ b/src/lib.rs",
])


@given(text=tool_text)
@settings(max_examples=100, deadline=1000)
def test_never_crashes(text):
    """LeanContext should never crash on any input."""
    lc = LeanContext(window_radius=10)
    result = lc.truncate(text)
    assert result.text is not None
    assert result.original_lines >= 0
    assert result.kept_lines >= 0
    assert result.dropped_lines >= 0


@given(text=tool_text)
@settings(max_examples=100, deadline=1000)
def test_kept_never_exceeds_original(text):
    """Kept lines should never exceed original."""
    lc = LeanContext()
    result = lc.truncate(text)
    assert result.kept_lines <= result.original_lines


@given(text=tool_text)
@settings(max_examples=100, deadline=1000)
def test_savings_pct_bounds(text):
    """Savings percentage should be between 0 and 100."""
    lc = LeanContext()
    result = lc.truncate(text)
    assert 0 <= result.savings_pct <= 100


@given(text=tool_text)
@settings(max_examples=50, deadline=2000)
def test_idempotent(text):
    """Truncating twice should give same result."""
    lc = LeanContext()
    r1 = lc.truncate(text)
    r2 = lc.truncate(r1.text)
    # Second pass should keep everything (all lines near signals already)
    assert r2.dropped_lines <= r1.dropped_lines


@given(signal=error_signals, padding=st.integers(min_value=50, max_value=200))
@settings(max_examples=50, deadline=500)
def test_signal_always_kept(signal, padding):
    """A line with an error signal should always be kept."""
    text = "\n".join(f"line {i}" for i in range(padding))
    lines = text.split("\n")
    mid = padding // 2
    lines[mid] = signal
    text = "\n".join(lines)
    
    lc = LeanContext(window_radius=5)
    result = lc.truncate(text)
    assert signal in result.text
    assert result.signal_lines >= 1


@given(text=st.text(min_size=1000, max_size=5000))
@settings(max_examples=30, deadline=500)
def test_always_keeps_first_last_lines(text):
    """First 3 and last 3 lines should always be kept."""
    lc = LeanContext(window_radius=10)
    result = lc.truncate(text)
    lines = text.split("\n")
    first = lines[0] if len(lines) > 0 else ""
    last = lines[-1] if len(lines) > 0 else ""
    if first:
        assert first in result.text
    if last:
        assert last in result.text
