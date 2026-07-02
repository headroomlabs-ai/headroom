# LeanContext — Window-Based Context Truncation

Pure-Python, zero-dependency alternative to RTK. Keeps lines near errors, drops the rest.

## Quick Start

```python
from headroom.transforms.lean_context import truncate_text

# Truncate tool output to lines near errors/diffs/tracebacks
result = truncate_text(tool_output, window_radius=50)
```

## How It Works

1. Scan each line for **signal patterns** (16 total): compiler errors, tracebacks, diffs, test failures, build output
2. Keep all lines within `window_radius` of any signal
3. Always keep first 3 and last 3 lines (context boundaries)
4. Mark dropped sections with `... [N lines dropped] ...`

## Signal Patterns

| Category | Examples |
|---|---|
| Compiler errors | `error[E0308]`, `error: Could not compile` |
| Tracebacks | `Traceback (most recent call last)`, `File "x", line N` |
| Build output | Error pointers (`--> file:line:col`), line-numbered code |
| Diffs | `+++/---`, `@@ -N,N +N,N @@` |
| Search results | `file:line:` grep/ripgrep output |
| General | fix, critical, fatal, panic, deadlock, timeout, OOM |

## Configuration

```python
lc = LeanContext(window_radius=50)  # Lines to keep around each signal
result = lc.truncate(text)
print(f"Dropped {result.dropped_lines} of {result.original_lines} lines ({result.savings_pct:.1f}%)")
print(f"Found {result.signal_lines} signal lines")
```

## Performance

| Text Size | Time |
|---|---|
| 1K lines | <10ms |
| 10K lines | <500ms |
| 100K lines | <5s |

## vs RTK

| | RTK | LeanContext |
|---|---|---|
| Dependencies | Rust binary (~20MB) | Pure Python (0 deps) |
| Install | Download + hook registration | Import only |
| Method | ML relevance scoring | Window around signals |
| Transparency | Black box | Visible markers |
| Speed | ML inference | O(n) single pass |

## Testing

```bash
pytest tests/test_transforms/test_lean_context.py -v         # 8 unit tests
pytest tests/test_transforms/test_lean_context_bench.py -v   # 8 benchmarks
pytest tests/test_transforms/test_lean_context_fuzz.py -v    # 6 fuzzing tests
```
