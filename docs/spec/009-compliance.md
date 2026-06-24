# 009. Compliance

**Status:** done

## Data Handling Guarantees

### Default Behavior (No Export)

| Data Type | Stored | Exported | Logged |
|-----------|:------:|:--------:|:------:|
| Prompts | Optional (cache) | Never | Never |
| Responses | Optional (cache) | Never | Never |
| Savings metrics | Yes | Never | Never |
| Session metadata | Yes | Never | Never |
| Telemetry | Aggregated | Never | Never |

---

### With Exporters Configured

**Warning:** Enabling exporters may send data outside your infrastructure.

| Exporter | Data Sent | Destination |
|----------|-----------|-------------|
| Prometheus | Metrics only | Prometheus server |
| OpenTelemetry | Traces/spans | OTLP endpoint |
| Custom webhook | Configurable | HTTP endpoint |

---

## Privacy Commitments

1. **No prompt logging by default** — Headroom never logs prompt content unless explicitly configured
2. **No data leaves proxy by default** — All processing happens locally
3. **User control** — All data handling is configurable via environment variables
4. **Transparency** — Response headers indicate compression was applied
5. **No telemetry to Headroom project** — No data sent to external servers without explicit opt-in

---

## Compliance Considerations

### SOC 2

*To be documented if applicable.*

### GDPR

| Requirement | Headroom Support |
|-------------|:----------------:|
| Data minimization | ✓ Default no-logging |
| Right to deletion | ✓ Cache can be cleared |
| Data portability | Export available via API |
| Breach notification | N/A (no external data) |

### HIPAA

*To be documented if applicable.*

---

## Configuration for Maximum Privacy

```bash
HEADROOM_TELEMETRY=off
HEADROOM_STATELESS=true
headroom proxy --no-cache --no-optimize
```

This configuration results in:
- No prompt data stored
- No data exported
- No analytics collected
- Headroom acts as a passthrough proxy

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial compliance document |
