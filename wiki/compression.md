# Universal Compression

Headroom's Universal Compression module provides intelligent, automatic compression with ML-based content detection and structure preservation.

## Overview

Universal Compression combines several techniques:

1. **ML-based Detection** - Automatically detects content type (JSON, code, logs, text) using Magika
2. **Structure Preservation** - Keeps keys, signatures, and templates intact via structure masks
3. **Intelligent Compression** - Compresses content while preserving meaning with the optional ML compressor (Kompress)
4. **Reversible via CCR** - Stores originals for retrieval when LLM needs full context

## Quick Start

### One-Liner

```python
from headroom.compression import compress

result = compress(content)
print(result.compressed)
print(f"Saved {result.savings_percentage:.0f}% tokens")
```

### With Configuration

```python
from headroom.compression import UniversalCompressor, UniversalCompressorConfig

config = UniversalCompressorConfig(
    compression_ratio_target=0.5,  # Keep 50% of content
    use_entropy_preservation=True,  # Preserve UUIDs, hashes
)

compressor = UniversalCompressor(config=config)
result = compressor.compress(content)
```

---

## How It Works

### Detection Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Content   │───>│   Detect    │───>│   Extract   │───>│  Compress   │
│   Input     │    │   Type      │    │   Structure │    │  Content    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │                   │
                         ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │   Magika    │    │   Handler   │    │  Kompress   │
                   │   (ML)      │    │   (JSON,    │    │  (ML, opt-  │
                   │             │    │   Code...)  │    │  in [ml])   │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

### Structure Masks

Structure masks identify what to preserve:

| Content Type | What's Preserved | What's Compressed |
|--------------|------------------|-------------------|
| **JSON** | Keys, brackets, booleans, nulls, short values, UUIDs | Long string values, whitespace |
| **Code** | Imports, function signatures, class definitions, types | Function bodies, comments |
| **Logs** | Timestamps, log levels, error messages | Repeated patterns, verbose details |
| **Text** | High-entropy tokens (IDs, hashes) | Low-information content |

---

## Configuration

### UniversalCompressorConfig

```python
from headroom.compression import UniversalCompressorConfig

config = UniversalCompressorConfig(
    # Detection
    use_magika=True,               # Use ML-based detection (requires magika)

    # Compression
    # (Note: the legacy `use_llmlingua` flag was retired with the
    # LLMLingua-2 integration. The optional ML compressor is now Kompress,
    # installed via `headroom-ai[ml]` and configured separately.)
    compression_ratio_target=0.3,  # Keep 30% of content (70% reduction)
    min_content_length=100,        # Skip content shorter than this

    # Structure preservation
    use_entropy_preservation=True, # Preserve high-entropy tokens
    entropy_threshold=0.85,        # Entropy threshold for preservation

    # CCR
    ccr_enabled=True,              # Store originals for retrieval
)
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_magika` | `True` | Use ML-based content detection |
| `use_llmlingua` | `True` | Use LLMLingua for compression |
| `compression_ratio_target` | `0.3` | Target ratio (0.3 = keep 30%) |
| `min_content_length` | `100` | Minimum chars to compress |
| `use_entropy_preservation` | `True` | Preserve high-entropy tokens |
| `entropy_threshold` | `0.85` | Entropy threshold (0.0-1.0) |
| `ccr_enabled` | `True` | Enable CCR storage |

---

## Content Handlers

### JSON Handler

Preserves JSON structure while compressing values:

```python
from headroom.compression.handlers.json_handler import JSONStructureHandler

handler = JSONStructureHandler(
    preserve_short_values=True,     # Keep values < 20 chars
    short_value_threshold=20,       # Threshold for "short"
    preserve_high_entropy=True,     # Keep UUIDs, hashes
    entropy_threshold=0.85,         # Entropy threshold
    max_array_items_full=3,         # Keep first N array items full
    max_number_digits=10,           # Preserve numbers up to N digits
)
```

**What's Preserved:**
- All keys (navigational - LLM sees schema)
- Structural syntax (`{`, `}`, `[`, `]`, `:`, `,`)
- Booleans and nulls (semantically important)
- High-entropy strings (UUIDs, hashes - identifiers)
- Short numbers (often IDs)

**Example:**

```python
# Before
{
    "id": "usr_abc123",
    "name": "Alice Johnson",
    "bio": "A long description that goes on and on..."
}

# After (structure preserved, long values compressed)
{
    "id": "usr_abc123",
    "name": "Alice Johnson",
    "bio": "A long...[compressed]..."
}
```

### Code Handler

Preserves code structure using AST parsing (tree-sitter) or regex fallback:

```python
from headroom.compression.handlers.code_handler import CodeStructureHandler

handler = CodeStructureHandler(
    preserve_comments=False,        # Preserve comments as structural
    use_tree_sitter=True,           # Use tree-sitter for parsing
    default_language="python",      # Default when detection fails
)
```

**What's Preserved:**
- Import statements
- Function/method signatures
- Class definitions
- Type annotations
- Decorators

**What's Compressed:**
- Function bodies (implementations)
- Comments (unless `preserve_comments=True`)

**Example:**

```python
# Before
def process_data(items: List[str]) -> Dict[str, int]:
    """Process items and count occurrences."""
    result = {}
    for item in items:
        item = item.strip().lower()
        if item in result:
            result[item] += 1
        else:
            result[item] = 1
    return result

# After (signature preserved, body compressed)
def process_data(items: List[str]) -> Dict[str, int]:
    """Process items and count occurrences."""
    result = {}
    for item in items:
    ...[compressed]...
```

### Supported Languages

| Language | Parser | Support Level |
|----------|--------|---------------|
| Python | tree-sitter | Full AST |
| JavaScript | tree-sitter | Full AST |
| TypeScript | tree-sitter | Full AST |
| Go | tree-sitter | Full AST |
| Rust | tree-sitter | Full AST |
| Java | tree-sitter | Full AST |
| C | tree-sitter | Full AST |
| C++ | tree-sitter | Full AST |

---

## Compression Result

```python
from headroom.compression import compress

result = compress(content)

# Access result fields
print(result.compressed)           # Compressed content
print(result.original)             # Original content
print(result.compression_ratio)    # e.g., 0.35 (35% of original size)
print(result.tokens_before)        # Estimated tokens before
print(result.tokens_after)         # Estimated tokens after
print(result.tokens_saved)         # tokens_before - tokens_after
print(result.savings_percentage)   # e.g., 65.0 (65% savings)

# Detection info
print(result.content_type)         # ContentType.JSON, CODE, etc.
print(result.detection_confidence) # 0.0-1.0

# Structure info
print(result.handler_used)         # "json", "code", etc.
print(result.preservation_ratio)   # Fraction preserved as structure

# CCR info
print(result.ccr_key)              # Key for retrieval (if CCR enabled)
```

---

## Batch Compression

For multiple contents, batch compression is more efficient:

```python
from headroom.compression import UniversalCompressor

compressor = UniversalCompressor()

contents = [
    '{"users": [...]}',
    'def hello(): pass',
    'Plain text content',
]

results = compressor.compress_batch(contents)

for result in results:
    print(f"{result.content_type}: {result.savings_percentage:.0f}% saved")
```

---

## Custom Handlers

Register custom handlers for specific content types:

```python
from headroom.compression import UniversalCompressor
from headroom.compression.detector import ContentType
from headroom.compression.handlers.base import BaseStructureHandler, HandlerResult
from headroom.compression.masks import StructureMask


class LogStructureHandler(BaseStructureHandler):
    """Custom handler for log content."""

    def __init__(self):
        super().__init__(name="log")

    def can_handle(self, content: str) -> bool:
        return "[INFO]" in content or "[ERROR]" in content

    def _extract_mask(self, content, tokens, **kwargs):
        # Mark timestamps and log levels as structural
        mask = [False] * len(content)
        # ... (custom logic)
        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.9,
        )


# Register the custom handler
compressor = UniversalCompressor()
compressor.register_handler(ContentType.TEXT, LogStructureHandler())
```

---

## CCR Integration

Universal Compression integrates with CCR (Compress-Cache-Retrieve) for reversible compression:

```python
from headroom.compression import UniversalCompressor, UniversalCompressorConfig

config = UniversalCompressorConfig(ccr_enabled=True)
compressor = UniversalCompressor(config=config)

result = compressor.compress(large_content)

# CCR key for retrieval
if result.ccr_key:
    print(f"Original stored with key: {result.ccr_key}")
    # LLM can request original via CCR when needed
```

See [CCR Guide](ccr.md) for full CCR documentation.

---

## Performance

| Content Type | Compression | Speed | Accuracy |
|--------------|-------------|-------|----------|
| JSON (large arrays) | 70-90% | ~1ms | Keys preserved |
| Code (Python) | 50-70% | ~10ms | Signatures preserved |
| Plain text | 60-80% | ~5ms | High-entropy preserved |

**Overhead:** ~1-10ms per compression depending on content size and type.

---

## Installation

```bash
# Basic compression (fallback to simple compression)
pip install headroom-ai

# With ML detection (recommended)
pip install "headroom-ai[magika]"

# With LLMLingua compression
pip install "headroom-ai[llmlingua]"

# With AST-based code handling
pip install "headroom-ai[code]"

# Everything
pip install "headroom-ai[all]"
```

---

## Example: Full Pipeline

```python
from headroom.compression import UniversalCompressor, UniversalCompressorConfig

# Configure for aggressive compression
config = UniversalCompressorConfig(
    compression_ratio_target=0.25,  # Keep 25%
    use_magika=True,
    use_llmlingua=True,
    ccr_enabled=True,
)

compressor = UniversalCompressor(config=config)

# Compress JSON API response
json_content = """
{
    "users": [
        {"id": "usr_123", "name": "Alice", "bio": "Software engineer..."},
        {"id": "usr_456", "name": "Bob", "bio": "Product manager..."}
    ],
    "total": 2,
    "page": 1
}
"""

result = compressor.compress(json_content)

print(f"Type: {result.content_type}")          # ContentType.JSON
print(f"Handler: {result.handler_used}")        # json
print(f"Saved: {result.savings_percentage:.0f}%")  # ~60%
print(f"Structure: {result.preservation_ratio:.0%} preserved")  # ~40%
print(f"CCR Key: {result.ccr_key}")             # For retrieval
```

---

## See Also

- [Transforms Reference](transforms.md) - Other compression transforms
- [CCR Guide](ccr.md) - Reversible compression architecture
- [Text Compression](text-compression.md) - Opt-in utilities for search/logs
