# Headroom (Claude-only, local-only fork) — Rust core build targets.
#
# The Python package hard-requires the compiled `headroom._core` extension
# (built from crates/headroom-py via maturin). Two ways to get it:
#   1. `make build-ext`   — needs a Rust toolchain (rustup) + maturin in the venv
#   2. copy `headroom/_core.abi3.so` out of the matching upstream PyPI wheel
#      (no Rust toolchain needed; see README "Install").

SHELL := /bin/bash
CARGO ?= cargo
MATURIN ?= maturin
PYTHON ?= python3

.PHONY: help test build-ext build-wheel fmt fmt-check lint clean pytest

help:
	@echo "Headroom targets:"
	@echo "  make build-ext    - maturin develop: build + install headroom._core into the active venv"
	@echo "  make build-wheel  - release wheel for headroom-py"
	@echo "  make test         - cargo test (headroom-core)"
	@echo "  make pytest       - python test suite (needs .venv with deps + headroom._core)"
	@echo "  make fmt / fmt-check / lint / clean"

test:
	$(CARGO) test -p headroom-core

build-ext:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "error: activate a venv first (e.g. source .venv/bin/activate)"; \
		exit 1; \
	fi
	bash scripts/build_rust_extension.sh

build-wheel:
	$(MATURIN) build --release -m crates/headroom-py/Cargo.toml

pytest:
	$(PYTHON) -m pytest -q -m "not slow and not real_llm and not live"

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy -p headroom-core -- -D warnings

clean:
	$(CARGO) clean
