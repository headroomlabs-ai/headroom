# Headroom Living Specification

**Version:** 1.0.0-draft
**Date:** 2026-04-16
**Status:** Draft — In Progress
**Related Issue:** GitHub #183

---

## Constitution

This specification is the **canonical source of truth** for how Headroom is designed to behave. It serves:

1. **New contributors** — One place to understand how Headroom works
2. **Enterprises evaluating adoption** — Clear guarantees about behavior, privacy, security
3. **Operators running managed deployments** — Operational guidance for all surfaces
4. **Plugin authors** — Clear contracts for extension points
5. **PR review** — A checklist target: "does the code match the spec?"

### Spec Governance

| Rule | Description |
|------|-------------|
| **Canonical** | When code and spec diverge, the spec is the target; the code needs updating |
| **Living** | Spec updates are required for behavior-changing changes (PR checklist) |
| **Comprehensive** | Spec covers every user-visible surface, behavior, and guarantee |
| **Language-agnostic** | Spec enables complete rewrite in any language with parity |
| **Versioned** | Changes increment version; breaking changes require major version bump |

### Spec Sections

| # | Section | Status | Description |
|---|---------|:------:|-------------|
| 001 | [Vision](001-vision.md) | done | What Headroom is, what it is not |
| 002 | [Architecture](002-architecture.md) | done | Component diagram + descriptions |
| 003 | [ADRs](003-adrs.md) | done | Architecture Decision Records |
| 004 | [Domain Model](004-domain-model.md) | done | Core entities |
| 005 | [Integrations](005-integrations.md) | done | Agent contracts |
| 006 | [Actors](006-actors.md) | done | User types + interactions |
| 007 | [Behavior](007-behavior.md) | done | Mode-by-mode specification |
| 008 | [Capabilities](008-capabilities.md) | done | Feature matrix |
| 009 | [Compliance](009-compliance.md) | done | Data guarantees, privacy |
| 010 | [Data](010-data.md) | done | Storage, retention, env vars |
| 011 | [Deployment](011-deployment.md) | done | Profiles, presets, runtimes |
| 012 | [Diagrams](012-diagrams.md) | done | Component, sequence, data-flow |
| 013 | [Disaster Recovery](013-disaster-recovery.md) | done | Failure modes + recovery |
| 014 | [Governance](014-governance.md) | done | Decision-making, releases |
| 015 | [Interfaces](015-interfaces.md) | done | CLI, HTTP, env var, plugin ABI |
| 016 | [Observability](016-observability.md) | done | Telemetry, metrics, logs |
| 017 | [Operations](017-operations.md) | done | Health, logs, upgrades |
| 018 | [Policies](018-policies.md) | done | Defaults + overrides |
| 019 | [Quality](019-quality.md) | done | Test pyramid coverage |
| 020 | [Security](020-security.md) | done | Threat model, supply-chain |
| 021 | [Testing](021-testing.md) | done | Test strategy per surface |

---

## Quick Reference

### What Headroom Is

- A **context compression proxy** for AI provider APIs
- A **Python package** (`headroom-ai`) with proxy, SDK, and CLI
- A **TypeScript SDK** (`@headroom/sdk`) for Node.js
- A **dashboard** for visualizing savings
- A **learn system** with per-agent plugins

### What Headroom Is Not

- A model provider
- A data store for prompts (by default)
- A logging service (by default)
- A billing service

### Core Guarantees

1. **Never logs prompts by default** — No prompt data leaves the proxy unless an exporter is configured
2. **Never leaves the proxy by default** — All data stays local unless explicitly exported
3. **Composable** — Works alongside existing tools (Claude Code, Copilot, etc.)
4. **Transparent** — Full observability into what's being compressed and why

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial draft — 21 sections outlined |
