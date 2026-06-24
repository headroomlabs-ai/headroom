# 014. Governance

**Status:** done

## Project Structure

- **Repository:** `github.com/JerrettDavis/headroom`
- **License:** Apache 2.0
- **Main Languages:** Python (core), TypeScript (SDK/dashboard)

---

## Release Cadence

| Channel | Frequency | Stability |
|---------|-----------|-----------|
| Stable | Monthly | Production-ready |
| Beta | As needed | May have issues |
| Nightly | Daily | Unstable |

---

## Semantic Versioning

Headroom follows semver:

- **MAJOR:** Breaking changes to API, CLI, or core behavior
- **MINOR:** New features, backwards-compatible
- **PATCH:** Bug fixes, backwards-compatible

---

## Decision Making

### RFC Process

1. Open GitHub issue with `[RFC]` prefix
2. Gather community feedback (2 weeks minimum)
3. Core team reviews
4. Decision documented in ADRs
5. Implementation proceeds

### Criteria for Core Team Approval

- Consistent with project vision
- Does not break existing guarantees
- Implementation is feasible
- Tests can be written

---

## Spec-Driven Development

This specification is the **canonical source of truth** for Headroom behavior:

| Rule | Description |
|------|-------------|
| **Canonical** | When code and spec diverge, the spec is the target |
| **Living** | Spec updates required for behavior-changing changes |
| **Comprehensive** | Spec covers every user-visible surface |
| **Language-agnostic** | Enables complete rewrite in any language |

### Spec Change Process

1. Propose change in GitHub issue
2. Discuss in RFC format
3. Update relevant spec section
4. Update SPEC.md version and change log
5. Implement code change
6. Verify implementation matches spec

---

## Supply Chain Posture

| Component | Policy |
|-----------|--------|
| Dependencies | Pin versions, audit regularly |
| Build artifacts | Reproducible builds |
| Signing | Code signed for releases |
| Vulnerability reporting | See SECURITY.md |

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/JerrettDavis/headroom.git
cd headroom

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Style

- Format: `ruff format`
- Lint: `ruff check`
- Type check: `mypy`

---

## Code of Conduct

See `CODE_OF_CONDUCT.md`.

---

## Security

See `SECURITY.md` for vulnerability reporting and response timeline.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial governance document |
