.PHONY: help install lint fmt typecheck test test-live gate

help:
	@echo "make install   - pip install -e .[dev,stats]"
	@echo "make lint      - ruff check src tests"
	@echo "make fmt       - ruff format src tests"
	@echo "make typecheck - mypy src"
	@echo "make test      - pytest (unit only; live/real_llm skipped without keys)"
	@echo "make test-live - pytest incl. live tests (requires provider keys)"
	@echo "make gate      - lint + typecheck + test  (the agent-evals push gate)"

install:
	pip install -e ".[dev,stats]"

lint:
	ruff check src tests

fmt:
	ruff format src tests

typecheck:
	mypy src

# Unit-only by default: live tests self-skip when keys are absent, but we also exclude
# them here so a no-key machine never even collects them.
test:
	pytest -m "not live and not real_llm"

test-live:
	pytest

gate: lint typecheck test
	@echo "✅ agent-evals gate PASSED"
