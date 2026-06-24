# 003. Architecture Decision Records

**Status:** done

## ADR-001: Why a Proxy Instead of SDK Injection

**Context:** Headroom could be implemented as an SDK that users import, or as a network proxy.

**Decision:** Network proxy with SDK for complex use cases.

**Rationale:**
- Works with any HTTP client without code changes
- No need to modify existing applications
- Centralized configuration
- Can intercept all traffic, not just explicit SDK calls

**Consequences:**
- Requires running a separate service (or using wrap mode)
- Network latency added (~5-10ms)
- Need to handle connection pooling

---

## ADR-002: Why SQLite for Local Storage

**Context:** Headroom needs to store compression cache, savings history, memory graphs.

**Decision:** SQLite with optional external stores (Redis, PostgreSQL, etc.).

**Rationale:**
- Zero configuration — works out of the box
- Single file, easy backup
- ACID compliant transactions
- Good performance for single-node deployments
- sqlite-vec for vector similarity search
- FTS5 for full-text search

**Consequences:**
- Not distributed by default
- Must configure external stores for multi-node deployments
- Cloud-hosted databases require additional setup

---

## ADR-003: Why CCR (Compress-Cache-Retrieve) Pattern

**Context:** Compression alone doesn't leverage repeated context patterns.

**Decision:** CCR pattern for semantic caching and retrieval.

**Rationale:**
- Semantic similarity enables cache hits across different phrasings
- Retrieves relevant compressed context for new requests
- Reduces provider API calls for similar patterns
- Enables cross-session learning

**Consequences:**
- Requires storing compressed content
- Semantic hashing adds latency
- Cache invalidation is complex

---

## ADR-004: Why Per-Agent Plugins for Learn System

**Context:** Different AI agents (Claude, Codex, Gemini) have different context patterns.

**Decision:** Plugin architecture with agent-specific analyzers.

**Rationale:**
- Tailored compression per agent type
- Extensible for new agents
- Clear interface contract
- Independent versioning

**Consequences:**
- Plugin API must be stable
- Multiple plugins may conflict
- Testing complexity increases

---

## ADR-005: Why ONNX for TOIN

**Context:** TOIN (Tenant-specific ONNX) requires ML inference.

**Decision:** Use ONNX Runtime for portable ML inference.

**Rationale:**
- Hardware acceleration (CPU/GPU)
- Cross-platform (Windows, Linux, macOS)
- Model interchange format
- Single model file deployment

**Consequences:**
- ONNX model files must be hosted
- Version compatibility issues
- Larger package size

---

## ADR-006: Why Python for Core Implementation

**Context:** Language choice for the main implementation.

**Decision:** Python as the primary language.

**Rationale:**
- Primary language for AI/ML ecosystem
- Easy integration with provider APIs
- Rich async ecosystem (asyncio, httpx)
- Strong type annotation support (mypy)
- Good testing infrastructure

**Consequences:**
- GIL limitations for threading
- Slower than compiled languages
- Type checking adds build time

---

## ADR-007: Why TypeScript SDK for npm

**Context:** JavaScript/TypeScript ecosystem for frontend integrations.

**Decision:** Official TypeScript SDK published to npm.

**Rationale:**
- Node.js compatibility
- TypeScript type safety
- Wide adoption in AI tooling
- ESM and CommonJS support

**Consequences:**
- Dual language maintenance
- Must keep SDK in sync with Python core
- Additional CI/CD pipeline needed

---

## ADR-008: Why HierarchicalMemory with SQLite + vec + FTS5

**Context:** Memory system needs to store, search, and reason over context.

**Decision:** Hierarchical memory using SQLite + sqlite-vec + FTS5.

**Rationale:**
- SQLite: ACID transactions, single file
- sqlite-vec: Vector similarity for semantic search
- FTS5: Full-text search for keyword matching
- Hierarchical: Session < Conversation < Message structure

**Consequences:**
- Memory hierarchy adds complexity
- SQLite limitations for concurrent writes
- Vector search accuracy depends on embedding quality

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial ADRs |
