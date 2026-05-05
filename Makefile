# Headroom Rust build targets. `just` is not installed on dev boxes; this
# Makefile is the source of truth and is mirrored by .github/workflows/rust.yml.

SHELL := /bin/bash
CARGO ?= cargo
MATURIN ?= maturin
PYTHON ?= python3
FIXTURES ?= tests/parity/fixtures

.PHONY: help test test-parity bench build-proxy build-wheel fmt fmt-check lint clippy clean ci-precheck ci-precheck-rust ci-precheck-python ci-precheck-commitlint install-git-hooks verify-rust-core

help:
	@echo "Headroom Rust targets:"
	@echo "  make test               - cargo test --workspace"
	@echo "  make test-parity        - maturin develop + parity-run against fixtures"
	@echo "  make bench              - cargo bench --workspace"
	@echo "  make build-proxy        - release build + strip headroom-proxy, print size"
	@echo "  make build-wheel        - release wheel for headroom-py"
	@echo "  make verify-rust-core   - build + install + import-verify headroom._core"
	@echo "  make fmt                - cargo fmt --all"
	@echo "  make fmt-check          - cargo fmt --all -- --check"
	@echo "  make lint               - cargo clippy --workspace -- -D warnings"
	@echo "  make clean              - cargo clean"
	@echo ""
	@echo "Pre-push verification (run BEFORE git push to catch CI failures locally):"
	@echo "  make ci-precheck        - run all CI gates (rust + python + commitlint)"
	@echo "  make ci-precheck-rust   - cargo fmt --check + clippy + test"
	@echo "  make ci-precheck-python - smart_crusher-affected python tests"
	@echo "  make ci-precheck-commitlint - lint commits since origin/main"
	@echo "  make install-git-hooks  - install a pre-push hook that runs ci-precheck"

test:
	$(CARGO) test --workspace

test-parity:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "error: activate a venv first (e.g. source .venv/bin/activate)"; \
		exit 1; \
	fi
	$(MATURIN) develop -m crates/headroom-py/Cargo.toml
	$(CARGO) run -p headroom-parity -- run --fixtures $(FIXTURES)

bench:
	$(CARGO) bench --workspace

build-proxy:
	$(CARGO) build --release -p headroom-proxy
	@BIN=target/release/headroom-proxy; \
	if command -v strip >/dev/null 2>&1; then strip "$$BIN" || true; fi; \
	SIZE=$$(wc -c < "$$BIN"); \
	printf 'headroom-proxy: %s bytes (%.1f MiB)\n' "$$SIZE" "$$(echo "$$SIZE / 1048576" | bc -l)"

build-wheel:
	$(MATURIN) build --release -m crates/headroom-py/Cargo.toml

# Hotfix-A0: maturin-develop + symlink + import-verify in one shot. Run this
# any time you suspect the proxy is silently falling back to Python-only
# mode (Finding #2 in HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md). The
# proxy itself runs the same check at lifespan startup; this target
# exposes it as a developer-facing one-liner.
verify-rust-core:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "error: activate a venv first (e.g. source .venv/bin/activate)"; \
		exit 1; \
	fi
	bash scripts/build_rust_extension.sh

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy lint:
	$(CARGO) clippy --workspace -- -D warnings

clean:
	$(CARGO) clean

# ─── Pre-push CI gate ──────────────────────────────────────────────────────
#
# These targets run the same checks GitHub Actions runs, locally. The intent
# is: if `make ci-precheck` is green, `git push` will not turn red. The
# 2026-04-27 push surfaced five CI breaks (cargo fmt drift, x86_64-apple-
# darwin wheel, headroom._core not built in test-extras + smoke-test,
# commitlint footer-leading-blank). The first three are caught by the gates
# below; the last two are caught by the workflow fixes themselves.
#
# Run before EVERY `git push`. Install the git hook (one-time) with:
#   make install-git-hooks

ci-precheck: ci-precheck-rust ci-precheck-python ci-precheck-commitlint
	@echo ""
	@echo "✅ ci-precheck PASSED — safe to push."

ci-precheck-rust:
	@echo "── ci-precheck-rust ────────────────────────────────────────────"
	$(CARGO) fmt --all -- --check
	$(CARGO) clippy --workspace -- -D warnings
	$(CARGO) test --workspace

# Mirrors the smart_crusher-affected test files we expect green on every
# push. Builds the Rust extension first because most of these tests
# instantiate `SmartCrusher`, which hard-imports `headroom._core`.
ci-precheck-python:
	@echo "── ci-precheck-python ─────────────────────────────────────────"
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "error: activate a venv first (e.g. source .venv/bin/activate)"; \
		exit 1; \
	fi
	bash scripts/build_rust_extension.sh
	$(PYTHON) -m pytest -q \
		tests/test_transforms/test_smart_crusher_bugs.py \
		tests/test_transforms/test_smart_crusher_rust_parity.py \
		tests/test_transforms/test_diff_compressor.py \
		tests/test_transforms/test_diff_compressor_rust_parity.py \
		tests/test_relevance.py \
		tests/test_relevance_extra.py \
		tests/test_ccr.py \
		tests/test_acceptance.py \
		tests/test_critical_fixes.py \
		tests/test_quality_retention.py \
		tests/test_toin_integration.py

# Lint commits since `origin/main`. Requires npx (Node 18+) on PATH.
# Skips silently if npx is unavailable; install nodejs to enable.
ci-precheck-commitlint:
	@echo "── ci-precheck-commitlint ─────────────────────────────────────"
	@if ! command -v npx >/dev/null 2>&1; then \
		echo "skip: npx not on PATH (install node 18+ to enable commitlint pre-check)"; \
		exit 0; \
	fi
	@if ! git rev-parse --verify origin/main >/dev/null 2>&1; then \
		echo "skip: origin/main not fetched (run 'git fetch origin main')"; \
		exit 0; \
	fi
	npx --yes --package=@commitlint/cli --package=@commitlint/config-conventional -- \
		commitlint --from origin/main --to HEAD --config .commitlintrc.json

install-git-hooks:
	@scripts/install-git-hooks.sh
