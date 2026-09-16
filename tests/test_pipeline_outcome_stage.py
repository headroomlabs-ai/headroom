"""The pipeline seam that lets an extension learn from what a model wrote.

``OUTCOME_OBSERVED`` carries an :class:`OutcomeSnapshot` — what actually came
back — so an extension can measure its own effect.

This file used to also cover ``PRE_SEND_PARAMS`` and ``PipelineEvent.body``,
which let an extension rewrite outbound request parameters. Both were removed
along with the effort router: measurement showed per-turn parameter shaping
does not pay (on mechanical turns it saved ~$0.0007 while one effort switch
costs ~$0.011 in cache re-writes), so nothing wrote to that seam any more and
an unused writable-body hook is a liability rather than an extension point.

The discovery gate is still tested here. It matters even with the body gone:
an unaudited package installed as a transitive dependency must not be able to
switch itself on and start rewriting the messages of live traffic.
"""

from __future__ import annotations

import dataclasses

import pytest

from headroom.pipeline import (
    CANONICAL_PIPELINE_STAGES,
    OutcomeSnapshot,
    PipelineEvent,
    PipelineExtensionManager,
    PipelineStage,
    discover_pipeline_extensions,
)


class _Recorder:
    """Minimal extension: records every event it is handed."""

    def __init__(self) -> None:
        self.seen: list[PipelineEvent] = []

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        self.seen.append(event)

    def stages(self) -> list[PipelineStage]:
        return [e.stage for e in self.seen]


class TestStages:
    def test_outcome_stage_is_canonical(self):
        assert PipelineStage.OUTCOME_OBSERVED in CANONICAL_PIPELINE_STAGES

    def test_outcome_stage_runs_last(self):
        order = list(CANONICAL_PIPELINE_STAGES)
        assert order.index(PipelineStage.OUTCOME_OBSERVED) == len(order) - 1

    def test_params_stage_is_gone(self):
        """The writable-body seam was removed, not merely left unused."""
        assert not hasattr(PipelineStage, "PRE_SEND_PARAMS")

    def test_event_has_no_writable_body(self):
        assert not hasattr(PipelineEvent(stage=PipelineStage.PRE_SEND, operation="x"), "body")


class TestOutcomeSnapshot:
    def test_truncated_reads_both_provider_spellings(self):
        assert OutcomeSnapshot(stop_reason="max_tokens").truncated
        assert OutcomeSnapshot(stop_reason="length").truncated
        assert not OutcomeSnapshot(stop_reason="end_turn").truncated
        assert not OutcomeSnapshot().truncated

    def test_visible_split(self):
        assert OutcomeSnapshot(output_tokens=900, thinking_tokens=700).visible_output_tokens == 200

    def test_unknown_thinking_yields_unknown_visible_not_the_total(self):
        """Defaulting to output_tokens would credit steering with reductions it
        did not produce."""
        assert OutcomeSnapshot(output_tokens=900).visible_output_tokens is None

    def test_snapshot_is_frozen(self):
        """Extensions learn from the measurement; they must not rewrite it."""
        snap = OutcomeSnapshot(output_tokens=100)
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.output_tokens = 5  # type: ignore[misc]

    def test_snapshot_reaches_the_extension(self):
        rec = _Recorder()
        mgr = PipelineExtensionManager(extensions=[rec], discover=False)
        snap = OutcomeSnapshot(output_tokens=250, stop_reason="max_tokens")
        mgr.emit(PipelineStage.OUTCOME_OBSERVED, operation="proxy.outcome", outcome=snap)
        assert rec.seen[0].outcome is snap
        assert rec.seen[0].outcome.truncated

    def test_a_raising_extension_does_not_break_the_emit(self):
        class _Boom:
            def on_pipeline_event(self, event):
                raise RuntimeError("bad extension")

        mgr = PipelineExtensionManager(extensions=[_Boom(), _Recorder()], discover=False)
        snap = OutcomeSnapshot(output_tokens=1)
        event = mgr.emit(PipelineStage.OUTCOME_OBSERVED, operation="proxy.outcome", outcome=snap)
        assert event.outcome is snap


class TestDiscoveryGate:
    """Installing a package must not silently start rewriting live requests."""

    def test_discovery_is_off_without_an_explicit_enable(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_PIPELINE_EXTENSIONS", raising=False)
        assert discover_pipeline_extensions() == []

    def test_empty_env_does_not_enable(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_PIPELINE_EXTENSIONS", "   ,  ,")
        assert discover_pipeline_extensions() == []

    def test_explicit_argument_beats_the_env(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_PIPELINE_EXTENSIONS", "something")
        assert discover_pipeline_extensions(["definitely-not-installed"]) == []

    def test_directly_passed_extensions_are_unaffected_by_the_gate(self, monkeypatch):
        """Constructing an extension and handing it over is already explicit
        consent — the gate covers entry-point discovery only."""
        monkeypatch.delenv("HEADROOM_PIPELINE_EXTENSIONS", raising=False)
        rec = _Recorder()
        mgr = PipelineExtensionManager(extensions=[rec], discover=True)
        assert mgr.enabled
        mgr.emit(PipelineStage.OUTCOME_OBSERVED, operation="proxy.outcome")
        assert rec.stages() == [PipelineStage.OUTCOME_OBSERVED]
