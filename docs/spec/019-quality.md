# 019. Quality

**Status:** done

## Test Pyramid

```
           ┌─────────────┐
           │     E2E     │  ← Few, slow, comprehensive
           │   Tests     │
           └──────┬──────┘
                  │
           ┌──────┴──────┐
           │  Integration │  ← Medium, moderate
           │    Tests    │
           └──────┬──────┘
                  │
     ┌────────────┴────────────┐
     │                         │
┌────┴────┐              ┌────┴────┐
│  Unit   │              │  Unit   │
│ Tests   │              │ Tests   │
└─────────┘              └─────────┘
Many, fast, isolated
```

---

## Test Coverage

| Surface | Unit | Integration | E2E |
|---------|:----:|:------------:|:---:|
| Proxy Server | ✓ | ✓ | ✓ |
| SDK | ✓ | ✓ | - |
| Compression | ✓ | ✓ | ✓ |
| Cache | ✓ | ✓ | - |
| Learn | ✓ | ✓ | - |
| CCR | ✓ | ✓ | ✓ |
| TOIN | ✓ | ✓ | - |
| Dashboard | ✓ | - | ✓ |

---

## Coverage Targets

| Metric | Target | Threshold |
|--------|--------|-----------|
| Line coverage | 80% | 70% |
| Branch coverage | 70% | 60% |
| Critical path | 100% | 100% |

---

## Critical Paths

These must always pass:

1. **Compression pipeline** — Input → Compress → Output
2. **Cache hit path** — Input → Cache check → Return
3. **Provider proxy** — Request → Proxy → Provider → Response
4. **Learn feedback** — Session → Analyze → Compress → Store

---

## Performance Benchmarks

| Operation | Target | Threshold |
|-----------|--------|-----------|
| Compression | < 50ms | < 200ms |
| Cache lookup | < 5ms | < 20ms |
| Proxy latency | +10ms | +50ms |

---

## CI/CD

### Required Checks

| Check | Command | Timeout |
|-------|---------|---------|
| Lint | `ruff check` | 2m |
| Type check | `mypy` | 5m |
| Unit tests | `pytest tests/unit/` | 10m |
| Integration | `pytest tests/ -k integration` | 15m |
| E2E | `pytest e2e/` | 30m |

### Workflow (`.github/workflows/`)

1. **Lint** — `ruff check` + `ruff format --check`
2. **Type check** — `mypy src/`
3. **Unit tests** — `pytest tests/unit/ --cov`
4. **Integration** — `pytest tests/ -k integration`
5. **E2E** — `pytest e2e/ --api-key=$ANTHROPIC_API_KEY`

---

## Quality Gates

PRs must pass:
- All tests green
- Type checking passes (`mypy`)
- Lint passes (`ruff`)
- Coverage maintained or improved

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial quality document |
