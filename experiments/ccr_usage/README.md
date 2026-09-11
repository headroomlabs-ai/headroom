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
