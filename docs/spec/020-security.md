# 020. Security

**Status:** done

## Threat Model

### Threats

| Threat | Impact | Mitigation |
|--------|--------|------------|
| Prompt data exfiltration | High | No logging, local-only |
| API key theft | High | Key rotation, secrets management |
| Cache poisoning | Medium | Input validation |
| DoS via large prompts | Medium | Token limits |
| SSRF via redirects | Medium | URL validation |

---

### Trust Boundaries

```
┌──────────────────┐     ┌──────────────────┐
│   User's App     │────▶│  Headroom Proxy  │
└──────────────────┘     └────────┬─────────┘
                                  │
                         ┌────────┴─────────┐
                         │                   │
                   ┌─────▼─────┐     ┌──────▼──────┐
                   │ Provider  │     │  Database   │
                   │   APIs    │     │  (SQLite)   │
                   └───────────┘     └─────────────┘
```

---

## Security Controls

### Authentication

| Surface | Auth Method |
|---------|-------------|
| Proxy | API key (optional) |
| Dashboard | None by default |
| Health endpoints | None |
| Metrics | None |

### Authorization

- **No multi-user support** — Single-tenant by design
- **API keys** — For proxy authentication
- **CORS** — Configurable origins

---

## Data Protection

| Data | At Rest | In Transit |
|------|---------|------------|
| Prompts | Encrypted (if DB encrypted) | TLS |
| Responses | Encrypted (if DB encrypted) | TLS |
| API keys | Encrypted | TLS |
| Metrics | Plain text | TLS |

---

## Input Validation

- **Prompt length** — Enforced via token budget
- **URL validation** — For provider redirects
- **Schema validation** — For all API inputs

---

## Secrets Management

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

### Secrets Management Systems

Headroom supports:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault

---

## Supply Chain Security

### Dependencies

- **Pinned versions** — All dependencies pinned
- **Audit** — Regular `pip audit`
- **SBOM** — Software Bill of Materials generated

### Build

- **Reproducible** — Docker builds are reproducible
- **Signed releases** — Code signed

---

## Vulnerability Reporting

See `SECURITY.md` for:
- Reporting process
- Response timeline
- Disclosure policy

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial security document |
