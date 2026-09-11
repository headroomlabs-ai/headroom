# Anthropic CCR usage reproduction

Synthetic loopback HTTP only; no paid model, credentials or private fixtures.
Tested on Ubuntu, Python 3.13.12, base main
`04cdf79ab0a8423d88148ba63e960ac6b4007b9c` (declares 0.37.0).

## Run

Use separate environments for base and patched source. From the relevant checkout:

```bash
python3.13 -m venv /tmp/ccr-check-venv
/tmp/ccr-check-venv/bin/pip install -c experiments/ccr_usage/dependencies.lock '.[proxy]'
export TIKTOKEN_CACHE_DIR="$PWD/tokenizer-cache"
/tmp/ccr-check-venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); tiktoken.get_encoding("o200k_base")'
/tmp/ccr-check-venv/bin/python experiments/ccr_usage/test_decoder.py
/tmp/ccr-check-venv/bin/python experiments/ccr_usage/reproduce_usage.py --output actual.json
```

The base lacks these files: copy this directory to it unchanged before running.
Build/install and tokenizer preparation need network and a Rust toolchain.
The reproduction blocks non-loopback socket connections. It starts a real Uvicorn
proxy and HTTP provider; handlers and transport are not mocked. Optimization is
disabled to isolate CCR. A seeded in-memory entry is successfully retrieved;
every continuation checks its exact synthetic content. Main buffers these seeded
SSE requests into upstream JSON; `upstream[].stream` records the route.

Each upstream call n returns input/output/cache-read/cache-create = 11n/7n/13n/17n.
Both client JSON and SSE are tested with 1, 2 and 3 calls. Expected totals are
11/7/13/17, 33/21/39/51 and 66/42/78/102. Base returns only the final call's usage
for multi-call cases: two controls pass, four fail (exit 1). Patched source passes
all six (exit 0). Exit 2 means invalid setup/transport/parsing, not a scored bug.
`usage-main.json` and `usage-main-fixed.json` preserve those observed results.
The two decoder tests validate cumulative replacement and stream termination.

## Fix and limits

The shared Anthropic response handler sums usage after each completed continuation.
Missing required or malformed counters stay unknown; optional absent cache counters
use zero. Nested 5-minute and 1-hour cache creation counters are summed separately.
Content and other final-response metadata remain unchanged. No extra calls, retries,
persistent writes or process-death recovery behavior are introduced.

Affected tests: 106 pass, two upstream deprecation warnings. New tests cover 1–3
calls, missing/malformed usage, non-mutation, nested cache duration, failed
continuation and round limits. Run:

```bash
python -m pytest tests/test_ccr_anthropic_usage_accounting.py \
 tests/test_ccr_response_handler.py tests/test_ccr_response_handler_extra.py \
 tests/test_ccr_response_handler_openai_responses.py \
 tests/test_ccr_buffered_stream_signed_thinking.py \
 tests/test_no_ccr_disables_response_handling.py \
 tests/test_proxy/test_anthropic_streaming_ccr_retrieve.py -q
```

Not validated: full upstream CI, live provider invoices, live agent integration,
concurrency, retry accounting or all providers. This is a token-accounting fix,
not a measured compression or cost improvement. Self-reviewed only.

Related open PR #3013 changes event-level SSE handling. Its pinned revision
`b59ded54e51f81593a17f9433fdac808c9db0e4e` also failed four of these cases.
It needs coordination: once the shared handler sums all calls, its own usage
combination must not count the initial response twice, and terminal SSE usage
must carry cumulative input/cache counters. This main-based PR does not modify
that unmerged implementation.

## Broader validation follow-up

Whole-repository Ruff check and format-check pass after formatting the reproduction.
The repository's Rust checks pass: 1,493 tests passed, three ignored, including
doctests; formatting and Clippy also pass. `make ci-precheck-python` rebuilt and
verified the native extension and passed 175 tests, with four skipped. Commitlint
and the installed commit hooks (including whole-project mypy) pass.

An initial host-sandbox `python -m pytest --maxfail=5 -q` run reached 1,501 passing tests and 103
skips before stopping at five failures in bundled-tool and CLI-install tests.
Those five failures also occur on unmodified base `04cdf79`: tests try to write
binary caches or deployment state outside this environment's writable sandbox.
This is not a green full-suite result; remaining tests were not executed. No
sandbox restriction was relaxed to permit changes to the developer's environment.

The development environment required pip plus lightweight test extras after initial
collection errors. The CUDA-heavy full dev install remains incomplete. Provider
API keys were removed for the full-suite attempt; model downloads were disabled.
The formatted standalone reproduction was rechecked against both baseline and
patched source: two passes/four failures before, six passes after; decoder tests pass.

## Complete isolated-suite attempt

The later Docker run removes the host filesystem restriction. It used Ubuntu
26.04 (base digest `sha256:513c074113a871b51a8d16ab445c88779d6452d937a164fb5cc479f32668a41d`),
Python 3.13.12, four CPUs, 8 GiB memory, a copied checkout/environment and its own
writable home. No provider credentials or host Docker socket were supplied.
CPU-only Torch plus dev dependencies were installed; the public all-MiniLM-L6-v2
cache passed the repository's offline verification. This is not an exact copy of
CI's Python 3.12 runner. Final test dependencies are in
`isolated-test-dependencies.lock`; install its CPU Torch from the PyTorch CPU index
and the matching Headroom source separately.

With `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and Hugging Face telemetry disabled:

```bash
python -m pytest tests scripts/tests --timeout=120 --timeout-method=signal \
  --tb=short -q --junitxml=full.xml
```

Result: **12,091 passed, 594 skipped, seven failed** in 842.80 seconds.
The timeout plugin was added after an unbounded exploratory run hung in the CLI
dependency-check test. The reported bounded run reached the end of the suite;
this is not a clean full-suite pass. Counts in `isolated-results.json` include
collection skips represented by pytest's JUnit output; do not add rerun counts
to the original run.

Five failures were container setup effects. Three wrapper tests require a normal
parent process (direct `docker exec python` reports parent PID zero); run pytest
through a shell that waits for it. The Windows-supervisor fixture mocks the UID
and needs a username environment variable; `LOGNAME=root` matches this container's
user. The image test requires Pillow, which dev dependencies did not install.
After these corrections, rerunning all three affected files produced **111 passed,
seven skipped**. Four originally failing cases passed; the image case was skipped
because its optional image model is unavailable offline, so it remains unvalidated.

Two failures remain and reproduce in a separate unmodified-base container:

- `TestMissingProxyDepsError.test_proxy_command_exits_when_mcp_missing` times out.
- `TestMissingProxyDepsError.test_ensure_proxy_dependencies_exits_when_fastapi_missing`
  does not raise the expected SystemExit.

Both are in `tests/test_cli_proxy_improvements.py`: the tests mock
`builtins.__import__`, whereas the CLI calls `importlib.import_module`. Already
imported modules bypass that mock; the first test can start a real proxy instead
of exiting. The baseline file run yielded two failures and 57 passes with a
15-second timeout. These tests were not changed or excluded to manufacture a
passing result. The accounting fix and its tests passed in the full run.
