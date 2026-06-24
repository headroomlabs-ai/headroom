## Description

Implement unified CI/CD release automation with semantic versioning across all three packages:
- **Python (headroom-ai)** — pip package on PyPI
- **TypeScript SDK (headroom-ai)** — npm package on npmjs.org
- **OpenClaw plugin (headroom-openclaw)** — npm package on npmjs.org and GitHub Package Registry

Currently the three packages are independently versioned (0.5.25 / 0.1.0 / 0.1.0). This PR introduces a single-source-of-truth version in `pyproject.toml` that propagates to all packages on every release, driven by conventional commit messages.

Fixes #(issue number)

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [x] Documentation update
- [ ] Performance improvement
- [x] Code refactoring (no functional changes)

## Changes Made

### New Files

**Scripts:**
- `scripts/version-sync.py` — Reads version from `pyproject.toml`, updates all 4 version files. Supports `--version X.Y.Z` and `--bump {major,minor,patch}`.
- `scripts/changelog-gen.py` — Parses conventional commits since last tag, groups by type, generates markdown changelog with breaking change detection.
- `scripts/verify-versions.py` — Pre-release sanity check that all 4 version files are in sync.
- `scripts/tests/test_version_sync.py` — 5 tests for version-sync.py
- `scripts/tests/test_changelog_gen.py` — 23 tests for changelog-gen.py

**Workflows:**
- `.github/workflows/release.yml` — Unified release pipeline: detect → build → publish-pypi → publish-npm → publish-github-packages → create-release
- `.commitlintrc.json` — Conventional commit enforcement via `@commitlint/config-conventional`

**Local Testing (act):**
- `.actrc` — Default `act` flags (Ubuntu runner, reuse, quiet)
- `.github/act/dry-run.json` — `act` event file for dry-run testing
- `.github/act/push-feat.json` — `act` event file for simulating a feat commit
- `.actrc.local.example` — Local override template for `act`
- `.env.act.example` — Secrets documentation template for `act` local testing

**Documentation:**
- `docs/content/docs/releases.mdx` — Full documentation for the release pipeline, testing guide, and configuration reference

### Modified Files

- `.github/workflows/ci.yml` — Added `commitlint` job to enforce conventional commits
- `.github/workflows/publish.yml` — Changed from `release` trigger to `workflow_dispatch` only (superseded by `release.yml`)
- `.github/workflows/release.yml` — **Rewritten** with canonical+commit-height algorithm (no more commit loop)
- `.gitignore` — Added `!scripts/version-sync.py`, `!scripts/changelog-gen.py`, `!scripts/verify-versions.py`, `!scripts/tests/`, `.env.act`, `.actrc.local`

## Testing

- [x] Unit tests pass (`pytest`)
  - `scripts/tests/test_version_sync.py` — 5/5 passing
  - `scripts/tests/test_changelog_gen.py` — 23/23 passing
- [x] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy headroom`) — pre-existing issue in `headroom/cli/wrap.py:487` (unrelated)
- [x] New tests added for new functionality
- [x] Workflow tested with `act` (dry-run passes all jobs through build step — no infinite loop)

## Algorithm Validation

The canonical+commit-height algorithm was validated with test cases:
- Canonical `0.5.25`, no prior tag, `feat:` commit → git tag `v0.6.0.0`, npm `0.6.0` ✅
- Canonical `0.5.25`, tag `v0.5.25.2`, `fix:` commit → git tag `v0.5.25.3`, npm `0.5.26` ✅
- Canonical `0.5.25`, no prior tag, `fix:` commit → git tag `v0.5.25.0`, npm `0.5.25` ✅
- Manual override `1.2.3` → git tag `v1.2.3`, npm `1.2.3` ✅

## Test Output

```
scripts/tests/test_version_sync.py .....
scripts/tests/test_changelog_gen.py .......................
```

## Checklist

- [x] My code follows the project's style guidelines
- [x] I have performed a self-review of my code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
- [x] I have made corresponding changes to the documentation
- [ ] I have updated the CHANGELOG.md if applicable

## Additional Notes

### Version Bump Logic

**Canonical + Commit Height Algorithm** — The workflow NEVER commits back to the repo. `pyproject.toml` is the canonical source of truth, updated manually before merging.

| Commit | Bump | Git Tag | npm Version |
|--------|------|---------|-------------|
| `fix:`, `ci:`, `chore:`, `perf:`, `refactor:` | patch | `v0.5.25.3` | `0.5.26` |
| `feat:` | minor | `v0.6.0.0` | `0.6.0` |
| `feat!:` or `feat:` + `BREAKING CHANGE` body | major | `v1.0.0.0` | `1.0.0` |

The git tag uses `v{canonical}.{height}` (e.g., `v0.5.25.3` = 3 commits since canonical `0.5.25`). npm versions use 3-part semver, bumped from canonical.

### Package Publishing Targets

| Package | Target | Status |
|---------|--------|--------|
| `headroom-ai` (Python) | PyPI | ✅ via `pypa/gh-action-pypi-publish` |
| `headroom-ai` (TypeScript SDK) | npmjs.org | ✅ via `npm publish` |
| `headroom-openclaw` | npmjs.org | ✅ via `npm publish` |
| `headroom-openclaw` | GitHub Package Registry | ✅ via `npm publish --registry npm.pkg.github.com` |

### Safety Gates

Each publish job requires both `dry_run != 'true'` **and** the corresponding skip variable not set:

| Variable | Effect |
|----------|--------|
| `PYPI_SKIP=true` | Skip PyPI publish |
| `NPM_SKIP=true` | Skip both npm publishes |
| `GH_PACKAGES_SKIP=true` | Skip GitHub Package Registry publish |

Set in: **GitHub repo → Settings → Variables → Actions Variables**.

### Workflow Triggers

- **Auto:** On push to `main` — analyzes latest commit, bumps version, builds, publishes, creates GitHub Release
- **Manual:** `workflow_dispatch` with optional `version` override and `dry_run` flag
- **Paths ignore:** Skips runs when only `docs/`, `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `scripts/`, `.commitlintrc.json`, `.actrc`, `.github/act/`, or `.env.act.example` change

### Local Testing

```bash
# Install act
winget install act

# Dry-run (no publishes)
act -W .github/workflows/release.yml -e .github/act/dry-run.json

# Test feat: commit (minor bump)
act -W .github/workflows/release.yml -e .github/act/push-feat.json
```

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `NPM_TOKEN` | Publishing to npmjs.org |
| `GITHUB_TOKEN` | GitHub Package Registry (auto-provided by GitHub Actions) |

PyPI uses trusted publisher OIDC — no secret required, only the `pypi` GitHub Environment must be configured.

### First Release Note

The TypeScript packages are currently at `0.1.0` while Python is at `0.5.25`. The first release will align all three to the same version. Update `pyproject.toml` to the desired canonical version before merging, then use `workflow_dispatch` with a manual `version` input to set the target explicitly.

After each release, update `pyproject.toml` to match the published version to keep the canonical current and ensure unique git tags.

### Parameterized Configuration

All package names and registries are top-level `env` constants in `release.yml`:

```yaml
env:
  PYPI_PACKAGE: headroom-ai
  PYPI_ENVIRONMENT: pypi
  NPM_REGISTRY_URL: https://registry.npmjs.org
  NPM_SDK_PACKAGE: headroom-ai
  NPM_OPENCLAW_PACKAGE: headroom-openclaw
  GITHUB_PACKAGES_REGISTRY_URL: https://npm.pkg.github.com
```
