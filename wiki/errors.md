# Error Handling

Headroom provides explicit exceptions for debugging, with a safety guarantee that compression failures never break your LLM calls.

## Exception Hierarchy

```python
from headroom import (
    HeadroomError,        # Base class - catch all Headroom errors
    ConfigurationError,   # Invalid configuration
    ProviderError,        # Provider issues (unknown model, etc.)
    StorageError,         # Database/storage failures
    CompressionError,     # Compression failures (rare)
    ValidationError,      # Setup validation failures
)
```

## Usage

```python
from headroom import (
    HeadroomClient,
    HeadroomError,
    ConfigurationError,
    StorageError,
)

try:
    client = HeadroomClient(...)
    response = client.chat.completions.create(...)

except ConfigurationError as e:
    print(f"Config issue: {e}")
    print(f"Details: {e.details}")  # Additional context

except StorageError as e:
    print(f"Storage issue: {e}")
    # Headroom continues to work, just without metrics persistence

except HeadroomError as e:
    print(f"Headroom error: {e}")
```

## Exception Types

### ConfigurationError

Raised when configuration is invalid.

```python
# Examples:
# - Invalid mode value
# - Missing required provider
# - Invalid model context limit

try:
    client = HeadroomClient(
        original_client=OpenAI(),
        provider=OpenAIProvider(),
        default_mode="invalid_mode",  # Will raise ConfigurationError
    )
except ConfigurationError as e:
    print(f"Config error: {e}")
    print(f"Field: {e.details.get('field')}")
```

### ProviderError

Raised for provider-specific issues.

```python
# Examples:
# - Unknown model name
# - Provider API error
# - Token counting failure

try:
    response = client.chat.completions.create(
        model="unknown-model-xyz",
        messages=[...]
    )
except ProviderError as e:
    print(f"Provider error: {e}")
    print(f"Provider: {e.details.get('provider')}")
```

### StorageError

Raised when database operations fail.

```python
# Examples:
# - Database connection failure
# - Write permission denied
# - Disk full

try:
    metrics = client.get_metrics()
except StorageError as e:
    print(f"Storage error: {e}")
    # Application can continue - just won't have metrics
```

### CompressionError

Raised when compression fails (rare).

```python
# Examples:
# - Malformed JSON in tool output
# - Unexpected data structure

# Note: In practice, compression errors are caught internally
# and the original content passes through unchanged.
# This exception is only raised if you explicitly enable strict mode.
```

### ValidationError

Raised when setup validation fails.

```python
result = client.validate_setup()
if not result["valid"]:
    raise ValidationError(
        "Setup validation failed",
        details={"issues": result["issues"]}
    )
```

## Safety Guarantee

**If compression fails, the original content passes through unchanged.**

This is a core design principle. Your LLM calls never fail due to Headroom:

```python
# Even if SmartCrusher encounters unexpected data:
messages = [
    {"role": "tool", "content": "malformed json {{{"}
]

# This will NOT raise an exception
# Instead, the malformed content passes through unchanged
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
```

## Logging Errors

Enable logging to see error details:

```python
import logging
logging.basicConfig(level=logging.WARNING)

# Now you'll see warnings when compression is skipped:
# WARNING:headroom.transforms.smart_crusher:Skipping compression: invalid JSON
```

## Error Details

All Headroom exceptions include a `details` dict with context:

```python
try:
    client = HeadroomClient(...)
except HeadroomError as e:
    print(f"Error: {e}")
    print(f"Type: {type(e).__name__}")
    print(f"Details: {e.details}")

    # Details might include:
    # - field: which config field caused the error
    # - provider: which provider was involved
    # - model: which model was requested
    # - original_error: underlying exception
```

## Best Practices

### 1. Catch Specific Exceptions

```python
# Good: catch specific exceptions
try:
    response = client.chat.completions.create(...)
except ConfigurationError:
    # Handle config issues
    pass
except ProviderError:
    # Handle provider issues
    pass

# Avoid: catching all exceptions
try:
    response = client.chat.completions.create(...)
except Exception:
    # Too broad - might hide real bugs
    pass
```

### 2. Let StorageError Pass

```python
# Storage errors don't affect core functionality
try:
    metrics = client.get_metrics()
except StorageError:
    metrics = []  # Continue without historical metrics
```

### 3. Validate on Startup

```python
client = HeadroomClient(...)

# Validate once at startup
result = client.validate_setup()
if not result["valid"]:
    raise SystemExit(f"Headroom setup invalid: {result['issues']}")

# Then use client normally
response = client.chat.completions.create(...)
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Shows detailed transform decisions
# DEBUG:headroom.transforms.smart_crusher:Analyzing 1000 items...
# DEBUG:headroom.transforms.smart_crusher:Kept 15 items (errors: 2, anomalies: 3)
```

### Check Stats After Error

```python
try:
    response = client.chat.completions.create(...)
except HeadroomError:
    # Check what happened
    stats = client.get_stats()
    print(f"Last request stats: {stats}")
```
