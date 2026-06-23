"""Phase-0 cross-module integration: savings capture -> orchestrator -> scorecard.

Proves the leaf modules compose with NO real I/O. Everything is real except the proxy spawn,
the harness rollout, and the grader: real ``parse_savings_headers`` + ``SavingsStore`` +
``make_response_hook`` (exercised through a synthetic httpx.Response carrying real
``x-headroom-*`` headers), the real ``Orchestrator``/``Journal``, the real ``ArmHandle``
savings delegation, and the real ``build_scorecard``/``render_scorecard``.

Each arm owns its own SavingsStore (mirroring reality: one proxy + one client shim + one store
per arm), so the same task id running under multiple arms never collides.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import httpx
import pytest

from agent_evals.arms import ArmHandle
from agent_evals.config import Settings
from agent_evals.metrics.savings import (
    HEADER_TOKENS_AFTER,
    HEADER_TOKENS_BEFORE,
    HEADER_TRANSFORMS,
    SavingsStore,
    make_response_hook,
)
from agent_evals.models import (
    ArmName,
    ArmSpec,
    BenchTask,
    GradeResult,
    Pricing,
    Provider,
    ProxyMode,
    RolloutResult,
)
from agent_evals.orchestrator import Journal, Orchestrator
from agent_evals.report.scorecard import build_scorecard, render_scorecard


class _Active:
    """Shared holder for the arm currently inside ``async with`` (arms run sequentially)."""

    store: SavingsStore | None = None
    emits_headers: bool = False


class FakeArm:
    """Implements the Arm protocol without spawning a proxy; uses the real ArmHandle."""

    def __init__(self, spec: ArmSpec, pricing: Pricing, active: _Active) -> None:
        self.spec = spec
        self.pricing = pricing
        self.store = SavingsStore()
        self._active = active
        self.handle = ArmHandle(
            base_url="http://127.0.0.1:0",
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:0"},
            savings_provider=lambda tid: self.store.aggregate(tid, pricing),
        )

    async def __aenter__(self) -> ArmHandle:
        self._active.store = self.store
        self._active.emits_headers = self.spec.name == ArmName.B_HEADROOM
        return self.handle

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._active.store = None


class FakeHarness:
    """Rollout fake. For the compression arm it simulates one optimized response by driving the
    REAL response hook with a synthetic httpx.Response carrying real x-headroom headers."""

    name = "fake"
    version = "0.0.0"
    supported_providers = {Provider.ANTHROPIC, Provider.OPENAI}

    def __init__(self, pricing: Pricing, active: _Active) -> None:
        self.pricing = pricing
        self._active = active
        self.calls = 0

    async def run_task(
        self, task: BenchTask, env: dict[str, str], workdir: Path, task_tag: str
    ) -> RolloutResult:
        self.calls += 1
        if self._active.emits_headers and self._active.store is not None:
            hook = make_response_hook(self._active.store, lambda: task.task_id, self.pricing)
            resp = httpx.Response(
                200,
                headers={
                    HEADER_TOKENS_BEFORE: "1000",
                    HEADER_TOKENS_AFTER: "400",
                    HEADER_TRANSFORMS: "smart_crusher,read_lifecycle",
                },
                request=httpx.Request("POST", "http://127.0.0.1:0/v1/messages"),
            )
            hook(resp)
        # arm/run_index are advisory — the orchestrator stamps the authoritative values.
        return RolloutResult(
            task_id=task.task_id,
            arm=ArmName.A0_DIRECT,
            run_index=0,
            prediction=f"patch-{task.task_id}-{task_tag}",
            trajectory_path=workdir,
            savings=None,
            wall_ms=1.0,
        )


class FakeGrader:
    """Grades by a per-arm-unaware resolved map keyed on task_id."""

    name = "fake"
    benchmark_ref = "fake@v0"

    def __init__(self, resolved_by_task: dict[str, bool]) -> None:
        self._resolved = resolved_by_task

    def grade(self, predictions: dict[str, str], tasks: list[BenchTask]) -> dict[str, GradeResult]:
        return {
            t.task_id: GradeResult(task_id=t.task_id, resolved=self._resolved.get(t.task_id, False))
            for t in tasks
        }


def _arm(name: ArmName, mode: ProxyMode | None, pricing: Pricing, active: _Active) -> FakeArm:
    return FakeArm(
        ArmSpec(name=name, provider=Provider.ANTHROPIC, proxy_mode=mode, label=name.value),
        pricing,
        active,
    )


async def test_phase0_pipeline_composes(tmp_path: Path) -> None:
    pricing = Pricing(input_usd_per_1m=2.0)
    active = _Active()
    arms = [
        _arm(ArmName.A0_DIRECT, None, pricing, active),
        _arm(ArmName.A1_PASSTHROUGH, ProxyMode.OFF, pricing, active),
        _arm(ArmName.B_HEADROOM, ProxyMode.TOKEN, pricing, active),
    ]
    # B resolves both tasks; A1 resolves only t1 -> accuracy delta = 1.0 - 0.5 = 0.5.
    grader = FakeGrader({"t1": True, "t2": True})
    grader_a1 = {"t1": True, "t2": False}

    settings = Settings(stats={"k_runs": 2})  # type: ignore[arg-type]
    journal = Journal(tmp_path)
    harness = FakeHarness(pricing, active)

    # Run B + A0 with the all-resolve grader, A1 with its own grader, sharing the journal so the
    # final scorecard sees all three arms. (Three orchestrators, one journal — like resuming.)
    tasks = [BenchTask(task_id="t1"), BenchTask(task_id="t2")]
    await Orchestrator(settings, [arms[0]], harness, grader, journal).run(tasks)
    await Orchestrator(settings, [arms[1]], harness, FakeGrader(grader_a1), journal).run(tasks)
    await Orchestrator(settings, [arms[2]], harness, grader, journal).run(tasks)

    results = journal.all_results()
    # 3 arms x 2 tasks x 2 runs = 12 cells.
    assert len(results) == 12

    scorecard = build_scorecard(results, experiment_id="phase0-integration")
    by_arm = {s.arm: s for s in scorecard.arms}

    # All three arms present.
    assert set(by_arm) == {ArmName.A0_DIRECT, ArmName.A1_PASSTHROUGH, ArmName.B_HEADROOM}
    # Resolved rates reflect the graders.
    assert by_arm[ArmName.B_HEADROOM].resolved_rate == pytest.approx(1.0)
    assert by_arm[ArmName.A1_PASSTHROUGH].resolved_rate == pytest.approx(0.5)
    # Only the compression arm captured savings (the others produced no x-headroom headers).
    assert by_arm[ArmName.B_HEADROOM].median_savings_percent == pytest.approx(60.0)
    assert by_arm[ArmName.A0_DIRECT].median_savings_percent == pytest.approx(0.0)
    assert by_arm[ArmName.A1_PASSTHROUGH].median_savings_percent == pytest.approx(0.0)
    # Naive accuracy delta = B - A1.
    assert scorecard.accuracy_delta_b_vs_a1 == pytest.approx(0.5)

    rendered = render_scorecard(scorecard)
    assert "b_headroom" in rendered
    assert "Phase 1" in rendered  # honest: no CI/verdict yet


async def test_savings_attributed_per_task_via_real_handle(tmp_path: Path) -> None:
    """The ArmHandle's capture_savings (injected provider -> SavingsStore.aggregate) is what the
    orchestrator stores per cell — verify it carries the summed token counts end-to-end."""

    pricing = Pricing(input_usd_per_1m=2.0)
    active = _Active()
    b = _arm(ArmName.B_HEADROOM, ProxyMode.TOKEN, pricing, active)
    settings = Settings(stats={"k_runs": 1})  # type: ignore[arg-type]
    journal = Journal(tmp_path)
    harness = FakeHarness(pricing, active)

    await Orchestrator(settings, [b], harness, FakeGrader({"t1": True}), journal).run(
        [BenchTask(task_id="t1")]
    )
    [cell] = journal.all_results()
    assert cell.savings is not None
    assert cell.savings.tokens_before == 1000
    assert cell.savings.tokens_after == 400
    assert cell.savings.tokens_saved == 600
    assert cell.savings.source == "headers"
