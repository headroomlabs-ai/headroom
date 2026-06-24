# 021. Testing

**Status:** done

## Test Strategy by Surface

### Proxy Server

**Unit Tests:**
- Request routing
- Response header injection
- Mode switching
- Error handling

**Integration Tests:**
- Full request/response cycle
- Provider mocking
- Cache integration

**E2E Tests:**
- Against real provider APIs (with API keys)

---

### Compression

**Unit Tests:**
- Token counting
- Semantic hashing
- Summary compression
- Budget enforcement

**Integration Tests:**
- Cache round-trip
- Multi-stage compression

**Benchmarks:**
- Latency at scale
- Memory usage

---

### Learn System

**Unit Tests:**
- Plugin interface compliance
- Error classification
- Session analysis

**Integration Tests:**
- Plugin discovery
- Cross-plugin interaction

**E2E Tests:**
- Real session analysis

---

### CCR (Claude Code Relay)

**Unit Tests:**
- Context tracking
- Feedback collection
- Batch processing

**Integration Tests:**
- Tool injection
- Response handling

**E2E Tests:**
- Full CCR workflow

---

## Test Utilities

### Fixtures

Located in `tests/conftest.py`:
- `mock_provider` — Mock AI provider
- `sample_session` — Sample conversation
- `temp_db` — Temporary database

### Mock Libraries

- `responses` — HTTP mocking
- `pytest-mock` — Function mocking
- `aioresponses` — Async HTTP mocking

---

## Running Tests

### All Tests

```bash
pytest
```

### By Surface

```bash
pytest tests/test_proxy/
pytest tests/test_compression/
pytest tests/test_learn/
```

### With Coverage

```bash
pytest --cov=headroom --cov-report=html
```

### E2E Tests

```bash
# Requires API keys
pytest e2e/ --api-key=$ANTHROPIC_API_KEY
```

---

## Test Data

### Sample Sessions

Stored in `tests/fixtures/sessions/`:
- `claude_code_short.json` — Short Claude Code session
- `claude_code_long.json` — Long session with tool use
- `codex_completion.json` — Codex completion
- `gemini_multimodal.json` — Gemini with images

---

## Continuous Integration

### Required Checks

| Check | Command | Timeout |
|-------|---------|---------|
| Lint | `ruff check` | 2m |
| Type check | `mypy` | 5m |
| Unit tests | `pytest tests/` | 10m |
| Integration | `pytest tests/ -k integration` | 15m |
| E2E | `pytest e2e/` | 30m |

---

## Test Maintenance

- **Fixtures** — Keep realistic but small
- **Mocks** — Don't over-mock providers
- **Flaky tests** — Mark with `@pytest.mark.flaky`
- **Coverage drops** — PR blocked if coverage drops

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial testing document |
