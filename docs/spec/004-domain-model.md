# 004. Domain Model

**Status:** done

## Core Entities

### Session

A session represents a conversation context between a user and an AI agent.

**Fields:**
- `session_id: UUID` — Unique identifier
- `created_at: datetime` — Creation time
- `updated_at: datetime` — Last modification
- `agent_type: str` — "claude", "codex", "gemini", etc.
- `messages: list[Message]` — Conversation history
- `metadata: dict` — Agent-specific metadata

```python
@dataclass
class Session:
    session_id: UUID
    created_at: datetime
    updated_at: datetime
    agent_type: str
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

---

### Message

A single message in a conversation.

**Fields:**
- `message_id: UUID` — Unique identifier
- `session_id: UUID` — Associated session
- `role: str` — "user", "assistant", "system", "tool"
- `content: str` — Message content
- `tokens: int | None` — Token count (if known)
- `created_at: datetime` — Creation time
- `attachments: list[Attachment] | None` — Optional attachments

```python
@dataclass
class Message:
    message_id: UUID
    session_id: UUID
    role: str
    content: str
    tokens: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    attachments: list[Attachment] | None = None
```

---

### Request

A single API request to an AI provider.

**Fields:**
- `request_id: UUID` — Unique identifier
- `session_id: UUID` — Associated session
- `provider: str` — "anthropic", "openai", "google", etc.
- `model: str` — Model identifier
- `input_tokens: int` — Tokens in request
- `output_tokens: int` — Tokens in response
- `compressed: bool` — Whether compression was applied
- `savings_percentage: float` — Savings as decimal
- `timestamp: datetime` — Request time
- `duration_ms: int` — Request duration
- `error: str | None` — Error message if failed

```python
@dataclass
class Request:
    request_id: UUID
    session_id: UUID
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    compressed: bool = False
    savings_percentage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: int = 0
    error: str | None = None
```

---

### Savings

A record of token savings from a compressed request.

**Fields:**
- `savings_id: UUID` — Unique identifier
- `request_id: UUID` — Associated request
- `original_tokens: int` — Tokens before compression
- `compressed_tokens: int` — Tokens after compression
- `savings_percentage: float` — Savings as decimal
- `savings_amount: float` — Absolute token savings
- `provider: str` — Provider name
- `model: str` — Model used
- `window_start: datetime | None` — Billing window start

```python
@dataclass
class Savings:
    savings_id: UUID
    request_id: UUID
    original_tokens: int
    compressed_tokens: int
    savings_percentage: float
    savings_amount: float
    provider: str
    model: str
    window_start: datetime | None = None
```

---

### CacheEntry

A cached compression result.

**Fields:**
- `cache_key: str` — Hash of input
- `input_hash: str` — Hash of original content
- `output: str` — Compressed output
- `compression_type: str` — "semantic", "summary", "ccr"
- `tokens_before: int` — Tokens before compression
- `tokens_after: int` — Tokens after compression
- `created_at: datetime` — Creation time
- `ttl: int` — Time to live in seconds
- `hit_count: int` — Number of cache hits
- `last_accessed: datetime` — Last access time

```python
@dataclass
class CacheEntry:
    cache_key: str
    input_hash: str
    output: str
    compression_type: str
    tokens_before: int
    tokens_after: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 3600
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
```

---

### ToolDefinition

A tool/function definition available to an AI agent.

**Fields:**
- `tool_id: UUID` — Unique identifier
- `name: str` — Tool name
- `description: str` — Tool description
- `parameters: dict` — JSON Schema for parameters
- `provider: str` — Provider that defined this tool
- `created_at: datetime` — Creation time

```python
@dataclass
class ToolDefinition:
    tool_id: UUID
    name: str
    description: str
    parameters: dict
    provider: str
    created_at: datetime = field(default_factory=datetime.utcnow)
```

---

### PluginInterface

A learn plugin for agent-specific compression.

**Fields:**
- `plugin_id: UUID` — Unique identifier
- `name: str` — Plugin name
- `agent_type: str` — Supported agent type
- `version: str` — Plugin version
- `entry_point: str` — Import path or file path
- `config: dict` — Plugin configuration
- `enabled: bool` — Whether plugin is active

```python
@dataclass
class PluginInterface:
    plugin_id: UUID
    name: str
    agent_type: str
    version: str
    entry_point: str
    config: dict = field(default_factory=dict)
    enabled: bool = True
```

---

### TOINTenant

A tenant configuration for TOIN (Tenant-specific ONNX).

**Fields:**
- `tenant_id: UUID` — Unique identifier
- `name: str` — Tenant name
- `model_path: str | Path` — Path to ONNX model
- `model_version: str` — Model version
- `config: dict` — Tenant-specific configuration
- `created_at: datetime` — Creation time
- `updated_at: datetime` — Last modification

```python
@dataclass
class TOINTenant:
    tenant_id: UUID
    name: str
    model_path: str | Path
    model_version: str
    config: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
```

---

### CCRContext

Claude Code Relay context tracking.

**Fields:**
- `context_id: UUID` — Unique identifier
- `session_id: UUID` — Associated session
- `agent_type: str` — "claude", "claude-desktop", etc.
- `window_start: datetime` — Context window start
- `window_end: datetime` — Context window end
- `messages_tracked: int` — Number of messages
- `compressed_content: str | None` — Compressed context

```python
@dataclass
class CCRContext:
    context_id: UUID
    session_id: UUID
    agent_type: str
    window_start: datetime
    window_end: datetime
    messages_tracked: int = 0
    compressed_content: str | None = None
```

---

### SubscriptionState

Subscription/quota tracking state.

**Fields:**
- `subscription_id: UUID` — Unique identifier
- `provider: str` — Provider name
- `plan_name: str` — Plan name
- `window_start: datetime` — Billing window start
- `window_end: datetime` — Billing window end
- `max_tokens: int | None` — Maximum tokens in window
- `tokens_used: int` — Tokens used in window
- `tokens_remaining: int | None` — Tokens remaining
- `is_active: bool` — Whether subscription is active
- `last_updated: datetime` — Last update time

```python
@dataclass
class SubscriptionState:
    subscription_id: UUID
    provider: str
    plan_name: str
    window_start: datetime
    window_end: datetime
    max_tokens: int | None = None
    tokens_used: int = 0
    tokens_remaining: int | None = None
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.utcnow)
```

---

## Relationships

```
Session 1───N Message
Session 1───N Request
Request 1───1 Savings
Request 1───1 CacheEntry
Session 1───N CCRContext
TOINTenant 1───N Request
ToolDefinition N───N Session (via tool_use)
```

---

## Value Objects

### CompressResult

```python
@dataclass
class CompressResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    compression_ratio: float
    transforms_applied: list[str]
    content_type: ContentType
    model: str | None
    ccr_hash: str | None
    cached: bool
```

### ContentType (Enum)

```python
class ContentType(Enum):
    PLAINTEXT = "text/plain"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    HTML = "text/html"
    XML = "text/xml"
    PYTHON = "text/x-python"
    JAVASCRIPT = "text/javascript"
    TYPESCRIPT = "text/typescript"
    YAML = "text/yaml"
    MARKDOWN_SNAPSHOT = "text/markdown-snapshot"
    UNKNOWN = "application/octet-stream"
```

### ProxyMode (Enum)

```python
class ProxyMode(Enum):
    PASSTHROUGH = "passthrough"
    COMPRESS = "compress"
    LEARN = "learn"
    DETACHED = "detached"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial domain model |
