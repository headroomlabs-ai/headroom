# 002. Architecture

**Status:** done

## System Overview

Headroom is a context compression proxy for LLM applications, featuring intelligent transforms, semantic caching, and CCR (Compress-Cache-Retrieve) architecture.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Headroom                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │  Proxy   │   │   SDK    │   │   Wrap   │   │  CCR MCP    │  │
│  │ Server   │   │ (Python) │   │   CLI    │   │   Server    │  │
│  │ (8787)   │   │          │   │          │   │             │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬──────┘  │
│       │              │              │                 │          │
│  ┌────┴──────────────┴──────────────┴─────────────────┴──────┐  │
│  │                    Compression Layer                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │  │
│  │  │SmartCrusher│  │CacheAligner│  │  RollingWindow     │   │  │
│  │  │(JSON array │  │(Prefix     │  │  (Token cap)       │   │  │
│  │  │ crush)     │  │ stabilization)│  │                    │   │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Learn System                            │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────┐   │  │
│  │  │Claude  │  │Codex   │  │Gemini  │  │   Generic     │   │  │
│  │  │Scanner │  │Scanner │  │Scanner │  │   Writer      │   │  │
│  │  └────────┘  └────────┘  └────────┘  └────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Dashboard   │  │  CCR         │  │   TOIN               │  │
│  │  (HTML)      │  │  (Compress-  │  │   (Telemetry-based    │  │
│  │              │  │   Cache-     │  │    Intelligence)      │  │
│  │              │  │   Retrieve)   │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### Proxy Server (`headroom/proxy/server.py`)

**`HeadroomProxy` class** — Main proxy server (default port 8787):

```python
class HeadroomProxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self.compression_cache = CompressionCache(...)
        self.transform_pipeline = [...]
        self.ccr_store = CompressionStore(...)
        self.prefix_freeze = PrefixFreeze(...)

    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def handle_request(request: Request) -> Response: ...
```

**`ProxyConfig` dataclass** (from `headroom/models/config.py`):
```python
@dataclass
class ProxyConfig:
    store_url: str = "sqlite:///headroom.db"
    default_mode: HeadroomMode = HeadroomMode.AUDIT
    tool_crusher: ToolCrusherConfig = field(default_factory=ToolCrusherConfig)
    smart_crusher: SmartCrusherConfig = field(default_factory=SmartCrusherConfig)
    cache_aligner: CacheAlignerConfig = field(default_factory=CacheAlignerConfig)
    rolling_window: RollingWindowConfig = field(default_factory=RollingWindowConfig)
    cache_optimizer: CacheOptimizerConfig = field(default_factory=CacheOptimizerConfig)
    ccr: CCRConfig = field(default_factory=CCRConfig)
    prefix_freeze: PrefixFreezeConfig = field(default_factory=PrefixFreezeConfig)
```

**`HeadroomMode` enum** (actual modes):
```python
class HeadroomMode(str, Enum):
    AUDIT = "audit"       # Observe only, no modifications
    OPTIMIZE = "optimize" # Apply deterministic transforms
    SIMULATE = "simulate" # Return transform plan without API call
```

**HTTP Endpoints (actual):**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Proxy chat completions |
| `/v1/embeddings` | POST | Proxy embeddings |
| `/health` | GET | Basic health check |
| `/livez` | GET | Liveness check |
| `/readyz` | GET | Readiness check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/compress` | POST | Direct compression |
| `/v1/retrieve` | POST | CCR retrieval |
| `/stats` | GET | Compression statistics |

**Default Port:** 8787 (not 8765)

---

### Python SDK (`headroom/client.py`)

**`HeadroomClient` class:**
```python
class HeadroomClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8787",
        timeout: float = 60.0,
        fallback: bool = True,
        retries: int = 3,
    ) -> None:
        pass

    async def compress(
        self,
        messages: list[dict],
        options: CompressOptions | None = None,
    ) -> CompressResult: ...

    async def get_stats(self) -> Stats: ...

    async def close(self) -> None: ...
```

---

### Wrap CLI (`headroom/cli/wrap.py`)

**Commands (all use default port 8787):**
- `claude` — Wrap Claude Code
- `copilot` — Wrap GitHub Copilot
- `codex` — Wrap OpenAI Codex
- `aider` — Wrap Aider
- `cursor` — Wrap Cursor
- `openclaw` — Wrap OpenClaw

```python
@click.command()
@click.option("--port", "-p", default=8787, help="Proxy port")
@click.argument("command", nargs=-1, required=True)
def wrap(command: tuple, port: int) -> None:
    """Wrap a command with Headroom proxy."""
    ...
```

---

### CCR MCP Server (`headroom/ccr/mcp_server.py`)

Model Context Protocol server for Claude Desktop integration.

```python
class CCRMcpServer:
    async def compress(messages: list[dict]) -> CompressResult: ...
    async def retrieve(hash: str, query: str) -> RetrieveResult: ...
    async def get_stats() -> Stats: ...
```

**MCP Tools:**
- `headroom_compress` — Compress messages
- `headroom_retrieve` — Retrieve cached content
- `headroom_stats` — Get statistics

---

### ASGI Middleware (`headroom/integrations/asgi.py`)

ASGI-compatible middleware for Python web frameworks.

```python
class HeadroomMiddleware:
    def __init__(
        self,
        app: ASGIApplication,
        headroom_url: str = "http://localhost:8787",
    ) -> None: ...
```

---

### LiteLLM Callback (`headroom/integrations/litellm_callback.py`)

Callback for LiteLLM proxy integration.

```python
class LiteLLMCallback:
    def on_completion(self, completion_response: dict) -> dict: ...
    def on_error(self, error: Exception) -> None: ...
```

---

## Compression Layer (`headroom/transforms/`)

### SmartCrusher (`headroom/transforms/smart_crusher.py`)

Statistical JSON array compression preserving schema.

```python
class SmartCrusher:
    def __init__(self, config: SmartCrusherConfig) -> None: ...
    def crush(self, content: str, context: TransformContext) -> TransformResult: ...
```

### CacheAligner (`headroom/transforms/cache_aligner.py`)

Prefix stabilization for provider cache optimization.

```python
class CacheAligner:
    def __init__(self, config: CacheAlignerConfig) -> None: ...
    def align(self, messages: list[dict]) -> TransformResult: ...
```

### RollingWindow (`headroom/transforms/rolling_window.py`)

Rolling window token cap.

```python
class RollingWindow:
    def __init__(self, config: RollingWindowConfig) -> None: ...
    def apply(self, messages: list[dict]) -> TransformResult: ...
```

### ContentRouter (`headroom/transforms/content_router.py`)

Routes content to appropriate compressor based on type.

```python
class ContentRouter:
    def route(self, messages: list[dict]) -> list[dict]: ...
```

---

## Learn System (`headroom/learn/`)

**`LearnPlugin` interface** (actual):

```python
class LearnPlugin(ConversationScanner):
    @property
    def name(self) -> str: ...
    @property
    def display_name(self) -> str: ...
    @abstractmethod
    def detect(self) -> bool: ...
    @abstractmethod
    def discover_projects(self) -> list[ProjectInfo]: ...
    @abstractmethod
    def scan_project(self, project: ProjectInfo, max_workers: int = 1) -> list[SessionData]: ...
    @abstractmethod
    def create_writer(self) -> ContextWriter: ...
```

**Scanner implementations:**
- `ClaudeScanner` — Claude Code session parsing
- `CodexScanner` — Codex session parsing
- `CursorScanner` — Cursor session parsing

---

## CCR System (`headroom/ccr/`)

CCR (Compress-Cache-Retrieve) makes compression reversible.

```python
class CompressionStore:
    def store(self, hash: str, original: str, metadata: dict) -> None: ...
    def retrieve(self, hash: str) -> str | None: ...

class ContextTracker:
    def track(self, session_id: str, messages: list[dict]) -> CCRContext: ...
    def get_context(self, session_id: str) -> CCRContext | None: ...
```

**CCRConfig fields:**
- `enabled: bool = True`
- `store_max_entries: int = 1000`
- `store_ttl_seconds: int = 300`
- `inject_retrieval_marker: bool = True`
- `feedback_enabled: bool = True`

---

## TOIN (`headroom/telemetry/toin.py`)

Tool Output Intelligence Network — telemetry-based compression hints.

```python
class TOINCollector:
    def record_retrieval(self, tool_name: str, field: str, query: str) -> None: ...
    def get_hints(self, tool_name: str) -> dict[str, float]: ...
```

---

## Dashboard (`headroom/dashboard/`)

Simple HTML dashboard served by the proxy.

```python
def get_dashboard_html() -> str:
    """Load the dashboard HTML template."""
    return (TEMPLATES_DIR / "dashboard.html").read_text()
```

---

## Data Flow

```
Client → Headroom Proxy → [ContentRouter] → [SmartCrusher/CacheAligner/RollingWindow]
              │                    │              │
              │              [CCR Store] ←───────────────────────┘
              │
              │         [Telemetry/Metrics]
              │
              ▼
       Provider API
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial architecture document |
