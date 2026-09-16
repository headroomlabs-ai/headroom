"""Label-value escaping in the Prometheus text exposition output.

``PrometheusMetrics.export()`` builds the exposition text by hand, so every label
value has to pass through ``_escape_label_value`` before it is interpolated. The
format reserves ``"``, ``\\`` and the line feed, and a standard scraper does not
degrade gracefully on a malformed line — it aborts the parse, losing every
family emitted at or after the bad sample.

``model`` reaches ``requests_by_model`` straight from the parsed client request
body (``handlers/openai.py`` reads ``body.get("model", "unknown")`` with no
sanitisation, and the Anthropic path's ``sanitize_anthropic_model_id`` only
strips ANSI sequences and whitespace), so an unescaped value is remotely
reachable.

Imports only the metrics module so the test stays free of heavy ML deps.
"""

from __future__ import annotations

import re

import pytest

from headroom.proxy.prometheus_metrics import PrometheusMetrics

# A label whose value contains only unreserved characters or well-formed escape
# pairs. An unescaped quote inside a value stops this matching, which is exactly
# the failure a scraper hits.
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?P<labels>.*)\} \S+$")
_ESCAPE_RE = re.compile(r"\\(.)")
_UNESCAPE = {"n": "\n", '"': '"', "\\": "\\"}


def _unescape(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        char = match.group(1)
        if char not in _UNESCAPE:
            raise ValueError(f"undefined escape sequence '\\{char}' in {value!r}")
        return _UNESCAPE[char]

    return _ESCAPE_RE.sub(replace, value)


def _parse_label_block(block: str) -> dict[str, str]:
    """Parse ``key="value",key="value"`` the way a scraper would.

    Raises ``ValueError`` on anything the exposition grammar rejects, so a line
    carrying an unescaped quote fails loudly instead of yielding a
    plausible-looking dict.
    """
    labels: dict[str, str] = {}
    pos = 0
    while pos < len(block):
        match = _LABEL_RE.match(block, pos)
        if match is None:
            raise ValueError(f"malformed label block at offset {pos}: {block!r}")
        labels[match.group(1)] = _unescape(match.group(2))
        pos = match.end()
        if pos < len(block):
            if block[pos] != ",":
                raise ValueError(f"expected ',' at offset {pos}: {block!r}")
            pos += 1
    return labels


def _labelled_samples(text: str) -> list[tuple[str, dict[str, str]]]:
    """Every labelled sample in a scrape, as (metric name, decoded labels).

    Raises on any line a scraper would reject — including the fragments an
    unescaped line feed splits a sample into.
    """
    samples: list[tuple[str, dict[str, str]]] = []
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" not in line:
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            raise ValueError(f"malformed sample line: {line!r}")
        samples.append((match.group("name"), _parse_label_block(match.group("labels"))))
    return samples


async def _record(metrics: PrometheusMetrics, **overrides: object) -> None:
    kwargs: dict[str, object] = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "input_tokens": 100,
        "output_tokens": 20,
        # tokens_saved=0 keeps the durable savings-ledger write out of the test.
        "tokens_saved": 0,
        "latency_ms": 10.0,
    }
    kwargs.update(overrides)
    await metrics.record_request(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_quote_in_model_is_escaped() -> None:
    metrics = PrometheusMetrics()

    await _record(metrics, model='claude-sonnet-4-5"evil')

    text = await metrics.export()

    assert 'headroom_requests_by_model{model="claude-sonnet-4-5\\"evil"} 1' in text
    assert 'headroom_requests_by_model{model="claude-sonnet-4-5"evil"}' not in text


@pytest.mark.asyncio
async def test_quote_in_provider_is_escaped() -> None:
    metrics = PrometheusMetrics()

    await _record(metrics, provider='anth"ropic')

    text = await metrics.export()

    assert 'headroom_requests_by_provider{provider="anth\\"ropic"} 1' in text
    assert 'headroom_requests_by_provider{provider="anth"ropic"}' not in text


@pytest.mark.asyncio
async def test_backslash_and_newline_in_model_are_escaped() -> None:
    metrics = PrometheusMetrics()

    await _record(metrics, model="back\\slash")
    await _record(metrics, model="line\nfeed")

    text = await metrics.export()

    # Backslash first, so the escapes this inserts are not re-escaped.
    assert 'headroom_requests_by_model{model="back\\\\slash"} 1' in text
    assert 'headroom_requests_by_model{model="line\\nfeed"} 1' in text
    # The line feed must not survive as a real newline splitting the sample.
    assert "line\nfeed" not in text


@pytest.mark.asyncio
async def test_provider_cache_families_escape_provider() -> None:
    # The families PR #2450 added inherit `provider` from the same parameter,
    # so they need naming explicitly rather than assuming coverage.
    metrics = PrometheusMetrics()

    await _record(
        metrics,
        provider='anth"ropic',
        cache_read_tokens=40,
        cache_write_tokens=60,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=50,
        uncached_input_tokens=20,
    )

    text = await metrics.export()

    families = [
        "headroom_cache_read_tokens_total",
        "headroom_cache_write_tokens_total",
        "headroom_cache_write_ttl_tokens_total",
        "headroom_cache_write_ttl_requests_total",
        "headroom_uncached_input_tokens_total",
        "headroom_provider_cache_requests_total",
        "headroom_provider_cache_hit_requests_total",
        "headroom_provider_cache_bust_total",
        "headroom_provider_cache_bust_write_tokens_total",
    ]
    for family in families:
        assert f'{family}{{provider="anth\\"ropic"' in text, f"{family} left provider raw"


@pytest.mark.asyncio
async def test_cache_miss_attribution_escapes_both_labels() -> None:
    metrics = PrometheusMetrics()

    await metrics.record_cache_miss_attribution('anth"ropic', 'ttl"expiry')

    text = await metrics.export()

    assert (
        'headroom_cache_miss_attribution_total{provider="anth\\"ropic",reason="ttl\\"expiry"} 1'
        in text
    )


@pytest.mark.asyncio
async def test_no_emitted_label_value_is_malformed() -> None:
    # The regression guard: poison every reachable label input, then read the
    # whole scrape the way a scraper does. A future emission that forgets to
    # escape fails here even when no assertion above names it.
    metrics = PrometheusMetrics()

    # The model poison carries a comma and an inner quote. The parse alone
    # can't catch comma-injection (this value raises on the quote first), so the
    # round-trip assertion below is the real guard: after escaping, the value
    # must decode back to the exact raw string, comma and all, rather than
    # splitting into extra labels.
    await _record(
        metrics,
        provider='pro"vider\\one',
        model='mo"del,evil="1',
        cache_read_tokens=40,
        cache_write_tokens=60,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=50,
        uncached_input_tokens=20,
    )
    await metrics.record_cache_miss_attribution('pro"vider\\one', 'rea"son')

    samples = _labelled_samples(await metrics.export())

    values = {value for _, labels in samples for value in labels.values()}
    assert 'pro"vider\\one' in values, "provider did not round-trip through the escape"
    assert 'mo"del,evil="1' in values, "model did not round-trip through the escape"


@pytest.mark.asyncio
async def test_non_string_label_values_are_coerced() -> None:
    # A JSON body can carry `"model": 123`, and the handlers pass the decoded
    # value through untouched (handlers/openai.py reads body.get("model")). The
    # hand-rolled f-strings used to call str() implicitly, so escaping has to
    # keep tolerating a non-str. /metrics has no error handling around export(),
    # and the key survives in the dict, so a raise here would take out every
    # later scrape too.
    metrics = PrometheusMetrics()

    await _record(metrics, provider=456, model=123, cache_read_tokens=5, cache_write_tokens=5)
    await metrics.record_cache_miss_attribution(456, 789)

    text = await metrics.export()

    assert 'headroom_requests_by_model{model="123"} 1' in text
    assert 'headroom_requests_by_provider{provider="456"} 1' in text
    assert 'headroom_cache_read_tokens_total{provider="456"}' in text
    assert 'headroom_cache_miss_attribution_total{provider="456",reason="789"} 1' in text


@pytest.mark.asyncio
async def test_well_formed_values_are_emitted_unchanged() -> None:
    metrics = PrometheusMetrics()

    await _record(metrics)

    text = await metrics.export()

    assert 'headroom_requests_by_provider{provider="anthropic"} 1' in text
    assert 'headroom_requests_by_model{model="claude-sonnet-4-5"} 1' in text


@pytest.mark.asyncio
async def test_export_is_utf8_encodable_with_surrogate_model() -> None:
    # `/metrics` renders the whole body with `.encode("utf-8")` (server.py). A
    # client can decode a lone surrogate from JSON (`{"model": "x-\ud83d-y"}`) —
    # a valid str that is NOT UTF-8-encodable and passes escaping untouched. It
    # would raise in the response encoder and, because the poisoned key persists
    # in requests_by_model, 500 every later scrape until restart. Escaping must
    # leave the whole export encodable.
    metrics = PrometheusMetrics()

    await _record(metrics, model="x-\ud83d-y")
    await _record(metrics, model="clean-model")  # a healthy series alongside

    text = await metrics.export()

    # The load-bearing assertion: the body a scraper receives must encode.
    text.encode("utf-8")
    # And the healthy series is still readable, i.e. the poison did not corrupt
    # the surrounding output.
    assert 'headroom_requests_by_model{model="clean-model"} 1' in text
    # Legitimate astral characters (a real emoji is one code point, encodable)
    # are preserved, not scrubbed — only un-encodable lone surrogates change.
    metrics2 = PrometheusMetrics()
    await _record(metrics2, model="gpt-\U0001f600")
    assert 'headroom_requests_by_model{model="gpt-\U0001f600"} 1' in await metrics2.export()
