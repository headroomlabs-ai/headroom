# Text Compression Utilities

For coding tasks, Headroom provides **standalone text compression utilities** that applications can use explicitly. These are **opt-in** — they're not applied automatically, giving you full control over when and how to compress text content.

> **Design Philosophy**: SmartCrusher compresses JSON automatically because it's structure-preserving and safe. Text compression is lossy and context-dependent, so applications should decide when to use it.

## Available Utilities

| Utility | Input Type | Use Case |
|---------|------------|----------|
| `SearchCompressor` | grep/ripgrep output | Search results with `file:line:content` format |
| `LogCompressor` | Build/test logs | pytest, npm, cargo, make output |
| `TextCompressor` | Generic text | Any plain text with anchor preservation |
| `detect_content_type` | Any content | Detect content type for routing decisions |

## SearchCompressor

Compresses search results (grep, ripgrep, ag) while preserving relevant matches.

```python
from headroom.transforms import SearchCompressor

# Your grep/ripgrep output (could be 1000s of lines)
search_results = """
src/utils.py:42:def process_data(items):
src/utils.py:43:    \"\"\"Process items.\"\"\"
src/models.py:15:class DataProcessor:
src/models.py:89:    def process(self, items):
... hundreds more matches ...
"""

# Explicitly compress when you decide it's appropriate
compressor = SearchCompressor()
result = compressor.compress(search_results, context="find process")

print(f"Compressed {result.original_match_count} matches to {result.compressed_match_count}")
print(result.compressed)
```

### What Gets Preserved

- **Exact query matches**: Lines containing the search term
- **High-relevance matches**: Scored by BM25 similarity to context
- **File diversity**: Ensures results from different files are kept
- **First/last matches**: Context from start and end of results

## LogCompressor

Compresses build and test output while preserving errors, warnings, and summaries.

```python
from headroom.transforms import LogCompressor

# pytest output with 1000s of lines
build_output = """
===== test session starts =====
collected 500 items
tests/test_foo.py::test_1 PASSED
... hundreds of passed tests ...
tests/test_bar.py::test_fail FAILED
AssertionError: expected 5, got 3
===== 1 failed, 499 passed =====
"""

# Compress logs, preserving errors and stack traces
compressor = LogCompressor()
result = compressor.compress(build_output)

# Errors, stack traces, and summary are preserved
print(result.compressed)
print(f"Compression ratio: {result.compression_ratio:.1%}")
```

### What Gets Preserved

- **Errors and failures**: Any line with ERROR, FAILED, Exception, etc.
- **Warnings**: Warning messages that might be important
- **Stack traces**: Full tracebacks for debugging
- **Summaries**: Test/build summary lines
- **Section headers**: Structural markers like `=====`

## TextCompressor

General-purpose text compression with anchor preservation.

```python
from headroom.transforms import TextCompressor

long_text = """
... thousands of lines of documentation ...
"""

compressor = TextCompressor()
result = compressor.compress(long_text, context="authentication")

print(result.compressed)
```

### What Gets Preserved

- **Relevant paragraphs**: Scored by similarity to context
- **Anchors**: Headers, section markers, important keywords
- **Structure**: Document organization is maintained

## Content Type Detection

Automatically detect content type to route to the right compressor.

```python
from headroom.transforms import detect_content_type, ContentType

content = "src/main.py:42:def process():"

detection = detect_content_type(content)
if detection.content_type == ContentType.SEARCH_RESULTS:
    # Route to SearchCompressor
    pass
elif detection.content_type == ContentType.BUILD_OUTPUT:
    # Route to LogCompressor
    pass
elif detection.content_type == ContentType.PLAIN_TEXT:
    # Route to TextCompressor
    pass
```

### Content Types

| Type | Detection Pattern |
|------|-------------------|
| `SEARCH_RESULTS` | `file:line:content` format |
| `BUILD_OUTPUT` | pytest, npm, cargo markers |
| `JSON` | Valid JSON structure |
| `PLAIN_TEXT` | Default fallback |

## Integration Pattern

```python
from headroom.transforms import (
    detect_content_type, ContentType,
    SearchCompressor, LogCompressor, TextCompressor
)

def compress_tool_output(content: str, context: str = "") -> str:
    """Application-level compression with explicit control."""
    detection = detect_content_type(content)

    if detection.content_type == ContentType.SEARCH_RESULTS:
        result = SearchCompressor().compress(content, context)
        return result.compressed
    elif detection.content_type == ContentType.BUILD_OUTPUT:
        result = LogCompressor().compress(content)
        return result.compressed
    elif detection.content_type == ContentType.PLAIN_TEXT:
        result = TextCompressor().compress(content, context)
        return result.compressed
    else:
        # JSON or other - let SmartCrusher handle it automatically
        return content
```

## Configuration

Each compressor accepts configuration options:

```python
from headroom.transforms import SearchCompressor, SearchCompressorConfig

config = SearchCompressorConfig(
    max_results=50,           # Keep up to 50 matches
    preserve_file_diversity=True,  # Ensure different files represented
    relevance_threshold=0.3,  # Minimum relevance score to keep
)

compressor = SearchCompressor(config)
```

## Performance

| Compressor | Typical Input | Output | Speed |
|------------|---------------|--------|-------|
| SearchCompressor | 1000 matches | 30-50 matches | ~2ms |
| LogCompressor | 5000 lines | 100-200 lines | ~3ms |
| TextCompressor | 10000 chars | 2000 chars | ~2ms |

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| JSON tool output | Let SmartCrusher handle automatically |
| grep/ripgrep results | Use SearchCompressor |
| pytest/npm/cargo output | Use LogCompressor |
| Documentation/README | Use TextCompressor |
| Unknown content | Use detect_content_type to route |
