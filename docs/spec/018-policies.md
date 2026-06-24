# 018. Policies

**Status:** done

## Default Behaviors

### Compression

| Setting | Default | Description |
|---------|---------|-------------|
| Compression enabled | `true` | Apply compression by default |
| Cache enabled | `true` | Use semantic cache |
| Summary enabled | `true` | Use summary compression |
| Token budget enforced | `false` | No budget limit by default |

### Telemetry

| Setting | Default | Description |
|---------|---------|-------------|
| Metrics enabled | `true` | Emit Prometheus metrics |
| Tracing enabled | `false` | No OTEL tracing |
| Dashboard enabled | `false` | No built-in dashboard |

### Learn

| Setting | Default | Description |
|---------|---------|-------------|
| Learn enabled | `false` | Learning disabled |
| Plugin auto-detect | `true` | Auto-load plugins |
| Feedback collection | `false` | No CCR feedback |

---

## Override Mechanisms

### Environment Variables

All settings can be overridden via environment variables:
```bash
HEADROOM_MODE=token
headroom proxy --no-cache
```

### Config File

```yaml
# ~/.headroom/config.yaml
proxy:
  host: 0.0.0.0
  port: 8787

compression:
  enabled: true
  cache:
    enabled: true
    ttl: 3600

telemetry:
  metrics:
    enabled: true
  tracing:
    enabled: false

learn:
  enabled: false
```

### Runtime API

```bash
# Disable compression for single request
curl -X POST http://localhost:8787/v1/messages \
  -H "X-Headroom-Compress: false"
```

### Runtime Headers

| Header | Description |
|--------|-------------|
| `X-Headroom-Compress` | Override compression (true/false) |
| `X-Headroom-Mode` | Override mode (passthrough/compress/learn) |
| `X-Headroom-Cache` | Override cache (true/false) |

---

## Policy Hierarchy

Settings are evaluated in this order (highest wins):

1. **Runtime headers** — Per-request overrides
2. **Environment variables** — Process-level
3. **Config file** — Persistent settings
4. **Defaults** — Built-in defaults

---

## Per-Tenant Policies (TOIN)

TOIN tenants can have custom policies:

```yaml
tenant:
  id: "acme-corp"
  policies:
    compression:
      enabled: true
      threshold: 3000
    budget:
      daily_limit: 1000000
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial policies document |
