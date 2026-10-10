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

### Follow-up: unknown cache-duration propagation

Self-review found a hole in the proposed fix: if the first or second of three
completed calls omitted its entire usage object, a later call could publish its
own nested cache-duration counts as though they were complete. Both cases failed
before the follow-up. The missing-usage result now explicitly marks
`cache_creation` unknown, preserving that state through subsequent calls. Three
regression cases cover each missing-call position; optional cache fields remain
absent when neither call supplied them.

The offline SDK check uses the pinned Anthropic 1.5.0 / httpx2 2.12.0 environment
in `isolated-test-dependencies.lock`:

```sh
PYTHONPATH="$PWD" python -m pytest experiments/ccr_usage/test_sdk_usage.py -q
```

It verifies final content and unknown usage fields through the SDK's JSON and
SSE decoders using an in-process synthetic transport. This proves those decoder
paths tolerate nulls; it does not prove every client, strict schema validation,
live provider billing, or proxy metrics preserve unknown values.

The six real loopback HTTP accounting cases still pass after this follow-up.
Whole-project CI-version mypy 1.20.2 passes (532 files); Ruff check/format pass.
The earlier isolated full-suite results remain tied to their recorded revision,
not this follow-up. Upstream workflows require approval and have not executed;
no clean full-suite or upstream CI pass is claimed.

Combined follow-up check: **114 passed, one upstream deprecation warning**.
To repeat the affected tests and standalone decoder checks:

```sh
HF_HUB_OFFLINE=1 python -m pytest \
  tests/test_ccr_anthropic_usage_accounting.py \
  tests/test_ccr_response_handler.py tests/test_ccr_response_handler_extra.py \
  tests/test_ccr_response_handler_openai_responses.py \
  tests/test_ccr_buffered_stream_signed_thinking.py \
  tests/test_no_ccr_disables_response_handling.py \
  tests/test_proxy/test_anthropic_streaming_ccr_retrieve.py \
  experiments/ccr_usage/test_decoder.py experiments/ccr_usage/test_sdk_usage.py -q
```

Coverage was also measured with pytest-cov after pre-importing
`pydantic.root_model` (avoids an installed Pydantic/coverage collection error).
`followup-results.json` records helper coverage separately from whole-module
coverage. The earlier full-suite result has not been rerun or relabeled green.

### Real proxy check with missing provider usage

The reproduction also supports omitting the entire first or second upstream
usage object. It runs two- and three-call requests through the real loopback
proxy, with both JSON and SSE clients:

```sh
PYTHONPATH="$PWD" python experiments/ccr_usage/reproduce_usage.py \
  --missing-usage-call 1 --output missing-first.json
PYTHONPATH="$PWD" python experiments/ccr_usage/reproduce_usage.py \
  --missing-usage-call 2 --output missing-second.json
```

Both runs pass all four cases: eight additional HTTP cases preserve unknown
principal counters. Versioned evidence: `usage-missing-first.json` and
`usage-missing-second.json`. The default six complete-usage cases also pass.
The score compares only the four declared principal counters, retaining extra
JSON usage metadata in the evidence. Comparing the entire JSON usage object
against only four expected fields previously produced a false failure when
`cache_creation` was also present.

This still uses the seeded CCR path that buffers upstream JSON, including for
SSE clients. Missing-usage upstream SSE events are not covered. Proxy metrics
are also outside this contract: existing metrics code converts null counters
to zero with `int(value or 0)`. Do not use these results to claim those internal
metrics distinguish unknown usage from zero, or that all billing is correct.

### Observed metrics fallback (not a new defect claim)

`--capture-metrics` wraps the live proxy's `metrics.record_request`, calls the
original unchanged, then retains four numeric arguments after it completes.
It requires exactly one emission per request. Response-usage scoring remains
separate; capturing metrics does not claim they equal provider billing.

```sh
PYTHONPATH="$PWD" python experiments/ccr_usage/reproduce_usage.py \
  --capture-metrics --output complete-metrics.json
PYTHONPATH="$PWD" python experiments/ccr_usage/reproduce_usage.py \
  --capture-metrics --missing-usage-call 1 --output missing-metrics.json
```

The six complete-usage cases emit the expected cumulative metrics. For three
calls, input volume is 246 (66 uncached + 78 cache reads + 102 cache writes),
output 42. The four missing-usage cases preserve null response counters, but
emit 33 input tokens (the local estimate), zero output and zero cache counts.
`metrics-observations.json` preserves those ten observations. The local estimate
is fixture/environment-specific, not a provider usage measurement.

This is the documented `RequestOutcome` contract in `headroom/proxy/outcome.py`:
unavailable provider input is zero, and the emitter uses
`provider_input_tokens or optimized_tokens`. The earlier statement that metrics
turn unknown into zero needs this qualification: **input volume falls back to an
estimate**. Output/cache counts remain zero. Cost tracking is disabled in this
reproduction; these observations are not invoice or cost-tracker validation.

Replacing these defaults would require a shared missing-data contract for
metrics, cost tracking, logs and their consumers across providers. It is not a
one-line extension of the CCR response fix. Per CONTRIBUTING.md, architectural
changes require core-maintainer agreement before implementation; none is sought
or included by this reproduction.
