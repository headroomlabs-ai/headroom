# 001. Vision

**Status:** done

## What Headroom Is

Headroom is a **context compression proxy** for AI provider APIs. It sits between AI coding tools (Claude Code, Copilot, Codex, etc.) and provider APIs (OpenAI, Anthropic, Google, Cohere), compressing context before it reaches the provider to reduce token usage and costs.

### Core Value Proposition

1. **Token Savings** — 30-90% reduction in tokens sent to providers through semantic compression
2. **Cost Reduction** — Lower API costs via compression before provider transmission
3. **Context Window Extension** — Effective 2-10x larger context windows via compression
4. **Privacy** — Prompts never logged by default; all processing stays local
5. **Compatibility** — Works with existing AI coding tools via proxy or SDK

### What Headroom Is Not

- A model provider — Headroom does not host or run AI models
- A data store — No prompt storage by default (local SQLite is optional)
- A logging service — No prompt logging by default
- A billing service — Usage tracking is internal only

---

## Design Principles

### 1. Local-First Privacy

**Principle:** Prompt data never leaves the proxy unless explicitly exported.

**Implications:**
- All compression happens locally or through the proxy
- No third-party data sharing
- Optional SQLite storage with user control
- Export must be explicitly configured

### 2. Transparent Compression

**Principle:** Users see exactly what is being compressed and why.

**Implications:**
- Full observability into compression decisions
- Metrics and logs show savings
- Transform audit trail available
- Dashboard visualizes all compression activity

### 3. Composable Integration

**Principle:** Headroom works alongside existing tools without requiring workflow changes.

**Implications:**
- Proxy mode: route traffic through Headroom
- SDK mode: integrate into custom applications
- CLI mode: wrap existing AI commands
- Agent mode: MCP/LiteLLM/ASGI integrations

### 4. Production-Ready Defaults

**Principle:** Safe defaults that work out of the box.

**Implications:**
- Compression enabled by default
- No logging by default
- Cache enabled by default
- Learning disabled by default

---

## Core Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Never logs prompts** | No prompt data in logs unless exporter configured |
| **Never leaves proxy** | All data stays local unless explicitly exported |
| **Composable** | Works alongside Claude Code, Copilot, Codex, etc. |
| **Transparent** | Full observability into compression decisions |
| **Type-safe** | Full type annotations, mypy compliance |
| **Test-covered** | Unit, integration, and E2E test coverage |

---

## Target Users

| User | Use Case |
|------|----------|
| **Individual Developers** | Reduce API costs for personal AI coding |
| **Development Teams** | Shared compression with learn plugins |
| **Enterprises** | Self-hosted deployment with security guarantees |
| **Plugin Authors** | Extend Headroom via plugin ABI |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token savings | >30% | (tokens_before - tokens_after) / tokens_before |
| Compression latency | <50ms | Per-request proxy overhead |
| Cache hit rate | >60% | cache_hits / total_requests |
| Zero data exfiltration | 100% | No prompts in logs by default |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial vision document |
