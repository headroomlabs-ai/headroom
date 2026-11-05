"""Integration metric timestamps must be recorded in UTC, not local time.

The SDK's canonical recorder (`headroom/client.py`) and the agno/strands
adapters stamp metric timestamps in UTC, and `langchain/langsmith.py` exports a
metric timestamp via `.isoformat()` as the absolute `headroom.optimization_timestamp`
telemetry field. The autogen/crewai/langchain tool wrappers, and the langsmith
`PendingMetrics` default, used a naive local `datetime.now()`, which records the
local wall-clock time with no offset — read downstream as UTC, it is wrong by the
local UTC offset for every non-UTC user.

These adapters import their frameworks lazily, so the recording and serialization
seams are exercised directly here without installing autogen/crewai/langchain.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from headroom.integrations.autogen.agents import ToolMetricsCollector as AutogenCollector
from headroom.integrations.autogen.agents import _record_metrics as autogen_record
from headroom.integrations.crewai.agents import HeadroomToolWrapper as CrewaiWrapper
from headroom.integrations.crewai.agents import ToolMetricsCollector as CrewaiCollector
from headroom.integrations.langchain.agents import HeadroomToolWrapper as LangchainWrapper
from headroom.integrations.langchain.agents import ToolMetricsCollector as LangchainCollector


def _assert_utc(ts: object) -> None:
    assert isinstance(ts, datetime)
    # Aware and anchored to UTC (naive local time would have tzinfo is None).
    assert ts.tzinfo is not None
    assert ts.utcoffset() == timedelta(0)
    # Serialization contract: langchain/langsmith exports this value verbatim via
    # ``metrics.timestamp.isoformat()``, so the persisted string must carry an
    # explicit UTC offset — otherwise a downstream reader silently treats it as
    # local time. A naive datetime serializes with no offset at all, and an aware
    # ``isoformat()`` yields a valid ``+00:00`` string (never the malformed
    # ``+00:00Z`` that appending "Z" to an aware datetime would produce). This
    # path does not go through ``headroom.utils.format_timestamp``, so it is
    # correct independently of that helper.
    serialized = ts.isoformat()
    assert serialized.endswith("+00:00"), serialized
    # And it round-trips through a strict parser unchanged.
    assert datetime.fromisoformat(serialized) == ts


def test_autogen_record_metrics_timestamp_is_utc() -> None:
    collector = AutogenCollector()
    autogen_record(collector, "search", "the original tool output", "short", True)
    _assert_utc(collector.metrics[0].timestamp)


def test_crewai_record_metrics_timestamp_is_utc() -> None:
    collector = CrewaiCollector()
    wrapper = SimpleNamespace(name="search", _metrics=collector)
    # Unbound call with a duck-typed self avoids constructing the CrewAI BaseTool.
    CrewaiWrapper._record_metrics(wrapper, "the original tool output", "short", True)
    _assert_utc(collector.metrics[0].timestamp)


def test_langchain_record_metrics_timestamp_is_utc() -> None:
    collector = LangchainCollector()
    wrapper = SimpleNamespace(name="search", _metrics=collector)
    LangchainWrapper._record_metrics(wrapper, "the original tool output", "short", True)
    _assert_utc(collector.metrics[0].timestamp)


def test_langsmith_exported_optimization_timestamp_is_utc() -> None:
    """The actual telemetry serialization seam must emit an aware UTC offset.

    ``HeadroomLangSmithCallbackHandler._attach_metrics_to_run`` is the real
    persistence/format path: it serializes ``PendingMetrics.timestamp`` via
    ``.isoformat()`` into the exported ``headroom.optimization_timestamp``
    LangSmith metadata field. Regression: ``PendingMetrics`` defaulted its
    timestamp to a naive local ``datetime.now()``, so the exported string
    carried no offset and a downstream reader silently mis-read it as UTC.

    The handler imports LangChain lazily and the export seam does not touch
    the framework, so it is exercised unbound with a duck-typed ``self`` —
    no langchain/langsmith install required.
    """
    from headroom.integrations.langchain.langsmith import (
        HeadroomLangSmithCallbackHandler,
        PendingMetrics,
    )

    # The default-constructed timestamp (the recording seam) is UTC-aware.
    metrics = PendingMetrics(
        tokens_before=100,
        tokens_after=40,
        tokens_saved=60,
        savings_percent=60.0,
        transforms_applied=["code"],
    )
    _assert_utc(metrics.timestamp)

    # Drive the real export code path (metrics.timestamp.isoformat()).
    handler = SimpleNamespace(_run_metrics={}, _client=None, _auto_update=False)
    HeadroomLangSmithCallbackHandler._attach_metrics_to_run(handler, "run-1", metrics)

    serialized = handler._run_metrics["run-1"]["headroom.optimization_timestamp"]
    assert serialized.endswith("+00:00"), serialized
    assert datetime.fromisoformat(serialized).utcoffset() == timedelta(0)
