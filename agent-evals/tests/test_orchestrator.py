"""Unit tests for the resumable orchestrator + journal.

No real I/O: the harness/grader/arm are fakes implementing the protocols. The only filesystem
touched is a tmp_path journal. No keys, no network, no subprocess.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_evals.config import Settings
from agent_evals.models import (
    ArmName,
    ArmSpec,
    BenchTask,
    GradeResult,
    Pricing,
    Provider,
    ProxyMode,
    RolloutResult,
    TaskResult,
    TaskSavings,
)
from agent_evals.orchestrator import Journal, Orchestrator

# --------------------------------------------------------------------------------------------- #
# Fakes implementing the protocols.
# --------------------------------------------------------------------------------------------- #


class FakeHandle:
    """An ArmHandle: a base_url + env and a canned per-task savings."""

    def __init__(self, base_url: str, env: dict[str, str], pricing: Pricing) -> None:
        self.base_url = base_url
        self.env = env
        self._pricing = pricing
        self.savings_calls: list[str] = []

    def capture_savings(self, task_id: str) -> TaskSavings | None:
        self.savings_calls.append(task_id)
        return TaskSavings.from_token_counts(
            tokens_before=1000,
            tokens_after=600,
            pricing=self._pricing,
            transforms=["fake"],
        )


class FakeArm:
    """An Arm: async context manager that yields a FakeHandle."""

    def __init__(self, spec: ArmSpec, pricing: Pricing) -> None:
        self.spec = spec
        self._pricing = pricing
        self.handle: FakeHandle | None = None
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self) -> FakeHandle:
        self.enter_count += 1
        self.handle = FakeHandle(
            base_url="http://127.0.0.1:18800",
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:18800"},
            pricing=self._pricing,
        )
        return self.handle

    async def __aexit__(self, *exc: object) -> None:
        self.exit_count += 1


class FakeHarness:
    """A Harness: returns a canned RolloutResult; can raise for specific tasks; tracks concurrency."""

    name = "fake-harness"
    version = "0.0.0"
    supported_providers = {Provider.ANTHROPIC, Provider.OPENAI}

    def __init__(
        self,
        *,
        raise_on: set[str] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.raise_on = raise_on or set()
        self.delay_s = delay_s
        self.call_count = 0
        self.calls: list[tuple[str, int]] = []
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def run_task(
        self, task: BenchTask, env: dict[str, str], workdir: Path, task_tag: str
    ) -> RolloutResult:
        async with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            self.call_count += 1
            # run_index is encoded into the tag by the orchestrator: "{arm}-r{run_index}-{task_id}".
            run_index = int(task_tag.split("-r", 1)[1].split("-", 1)[0])
            self.calls.append((task.task_id, run_index))
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if task.task_id in self.raise_on:
                raise RuntimeError(f"boom: {task.task_id}")
            return RolloutResult(
                task_id=task.task_id,
                arm=ArmName.B_HEADROOM,
                run_index=run_index,
                prediction=f"patch-{task.task_id}-{run_index}",
                trajectory_path=workdir / "trajectory.json",
                wall_ms=12.5,
            )
        finally:
            async with self._lock:
                self._in_flight -= 1


class FakeGrader:
    """A Grader: returns resolved verdicts per a configurable map; counts invocations."""

    name = "fake-grader"
    benchmark_ref = "fake@v0"

    def __init__(self, resolved_map: dict[str, bool]) -> None:
        self.resolved_map = resolved_map
        self.grade_calls = 0
        self.last_predictions: dict[str, str] = {}

    def grade(self, predictions: dict[str, str], tasks: list[BenchTask]) -> dict[str, GradeResult]:
        self.grade_calls += 1
        self.last_predictions = dict(predictions)
        return {
            t.task_id: GradeResult(
                task_id=t.task_id,
                resolved=self.resolved_map.get(t.task_id, False),
            )
            for t in tasks
        }


# --------------------------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------------------------- #


@pytest.fixture
def headroom_spec() -> ArmSpec:
    return ArmSpec(
        name=ArmName.B_HEADROOM,
        provider=Provider.ANTHROPIC,
        proxy_mode=ProxyMode.TOKEN,
        label="headroom",
    )


def _settings(run_dir: Path, *, k_runs: int = 2, concurrency: int = 4) -> Settings:
    return Settings(
        run_dir=run_dir,
        concurrency=concurrency,
        stats={"k_runs": k_runs},  # type: ignore[arg-type]
    )


def _tasks(*ids: str) -> list[BenchTask]:
    return [BenchTask(task_id=i) for i in ids]


# --------------------------------------------------------------------------------------------- #
# Journal tests.
# --------------------------------------------------------------------------------------------- #


def test_journal_missing_file_is_empty(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    assert journal.all_results() == []
    assert journal.load_completed() == set()


def test_journal_append_and_load(tmp_path: Path, pricing: Pricing) -> None:
    journal = Journal(tmp_path)
    r = TaskResult(task_id="t1", arm=ArmName.B_HEADROOM, run_index=0, resolved=True)
    journal.append(r)
    journal.append(TaskResult(task_id="t2", arm=ArmName.B_HEADROOM, run_index=1, resolved=False))

    results = journal.all_results()
    assert [x.cell_key for x in results] == [
        ("t1", "b_headroom", 0),
        ("t2", "b_headroom", 1),
    ]
    assert journal.load_completed() == {
        ("t1", "b_headroom", 0),
        ("t2", "b_headroom", 1),
    }


def test_journal_tolerates_blank_lines(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append(TaskResult(task_id="t1", arm=ArmName.A0_DIRECT, run_index=0, resolved=True))
    # Inject a stray blank line — load must skip it, not raise.
    with journal.path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert len(journal.all_results()) == 1


# --------------------------------------------------------------------------------------------- #
# Orchestrator tests.
# --------------------------------------------------------------------------------------------- #


async def test_full_run(tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec) -> None:
    tasks = _tasks("t1", "t2")
    arm = FakeArm(headroom_spec, pricing)
    harness = FakeHarness()
    grader = FakeGrader({"t1": True, "t2": False})
    journal = Journal(tmp_path)
    orch = Orchestrator(_settings(tmp_path, k_runs=2), [arm], harness, grader, journal)

    results = await orch.run(tasks)

    # 2 tasks x 1 arm x k_runs=2 = 4 cells.
    assert len(results) == 4
    keys = {r.cell_key for r in results}
    assert keys == {
        ("t1", "b_headroom", 0),
        ("t2", "b_headroom", 0),
        ("t1", "b_headroom", 1),
        ("t2", "b_headroom", 1),
    }
    # resolved flags match the grader map.
    for r in results:
        assert r.resolved == (r.task_id == "t1")
        assert r.error is None
        # capture_savings flowed into the cell.
        assert r.savings is not None
        assert r.savings.tokens_saved == 400

    assert arm.enter_count == 1
    assert arm.exit_count == 1
    # All 4 cells journaled and persisted.
    assert len(journal.all_results()) == 4


async def test_resume_skips_completed_cells(
    tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec
) -> None:
    tasks = _tasks("t1", "t2")
    journal = Journal(tmp_path)
    # Pre-write 2 of the 4 cells (run_index 0 for both tasks).
    journal.append(TaskResult(task_id="t1", arm=ArmName.B_HEADROOM, run_index=0, resolved=True))
    journal.append(TaskResult(task_id="t2", arm=ArmName.B_HEADROOM, run_index=0, resolved=False))

    arm = FakeArm(headroom_spec, pricing)
    harness = FakeHarness()
    grader = FakeGrader({"t1": True, "t2": True})
    orch = Orchestrator(_settings(tmp_path, k_runs=2), [arm], harness, grader, journal)

    results = await orch.run(tasks)

    # Harness only ran the 2 MISSING cells (run_index 1 for both tasks).
    assert harness.call_count == 2
    assert sorted(harness.calls) == [("t1", 1), ("t2", 1)]

    # Final results cover all 4 cells.
    assert len({r.cell_key for r in results}) == 4
    assert len(results) == 4
    # Grader was only invoked for the missing run_index (1), once.
    assert grader.grade_calls == 1


async def test_error_capture(tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec) -> None:
    tasks = _tasks("t1", "t2")
    arm = FakeArm(headroom_spec, pricing)
    harness = FakeHarness(raise_on={"t2"})
    grader = FakeGrader({"t1": True, "t2": True})
    journal = Journal(tmp_path)
    orch = Orchestrator(_settings(tmp_path, k_runs=1), [arm], harness, grader, journal)

    results = await orch.run(tasks)

    assert len(results) == 2
    by_id = {r.task_id: r for r in results}
    # t2 errored -> recorded as TaskResult(error=..., resolved=False).
    assert by_id["t2"].error is not None
    assert "boom: t2" in by_id["t2"].error
    assert by_id["t2"].resolved is False
    # t1 still completes successfully (the run did not crash).
    assert by_id["t1"].error is None
    assert by_id["t1"].resolved is True
    # The errored task was never sent to the grader.
    assert "t2" not in grader.last_predictions
    assert "t1" in grader.last_predictions


async def test_timeout_capture(tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec) -> None:
    tasks = _tasks("t1")
    arm = FakeArm(headroom_spec, pricing)
    harness = FakeHarness(delay_s=0.2)
    grader = FakeGrader({"t1": True})
    journal = Journal(tmp_path)
    orch = Orchestrator(
        _settings(tmp_path, k_runs=1),
        [arm],
        harness,
        grader,
        journal,
        cell_timeout_s=0.01,
    )

    results = await orch.run(tasks)

    assert len(results) == 1
    assert results[0].error is not None
    assert "timeout" in results[0].error
    assert results[0].resolved is False
    # Nothing to grade since the only cell timed out.
    assert grader.grade_calls == 0


async def test_grader_called_once_per_run_index(
    tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec
) -> None:
    tasks = _tasks("t1", "t2", "t3")
    arm = FakeArm(headroom_spec, pricing)
    harness = FakeHarness()
    grader = FakeGrader({"t1": True, "t2": True, "t3": False})
    journal = Journal(tmp_path)
    orch = Orchestrator(_settings(tmp_path, k_runs=3), [arm], harness, grader, journal)

    await orch.run(tasks)

    # Exactly one grade() per (arm, run_index): 1 arm x 3 run_indices.
    assert grader.grade_calls == 3


async def test_concurrency_cap_respected(
    tmp_path: Path, pricing: Pricing, headroom_spec: ArmSpec
) -> None:
    tasks = _tasks("t1", "t2", "t3", "t4", "t5", "t6")
    arm = FakeArm(headroom_spec, pricing)
    # Each rollout holds for a beat so several overlap, exposing the semaphore cap.
    harness = FakeHarness(delay_s=0.05)
    grader = FakeGrader(dict.fromkeys(["t1", "t2", "t3", "t4", "t5", "t6"], True))
    journal = Journal(tmp_path)
    orch = Orchestrator(
        _settings(tmp_path, k_runs=1, concurrency=2), [arm], harness, grader, journal
    )

    await orch.run(tasks)

    # Never more than the configured concurrency in flight at once.
    assert harness.max_in_flight <= 2
    # And we actually exercised concurrency (more than one ran together).
    assert harness.max_in_flight == 2


async def test_multiple_arms_each_provisioned_once(tmp_path: Path, pricing: Pricing) -> None:
    tasks = _tasks("t1")
    specs = [
        ArmSpec(
            name=ArmName.A0_DIRECT, provider=Provider.ANTHROPIC, proxy_mode=None, label="direct"
        ),
        ArmSpec(
            name=ArmName.A1_PASSTHROUGH,
            provider=Provider.ANTHROPIC,
            proxy_mode=ProxyMode.OFF,
            label="passthrough",
        ),
    ]
    arms = [FakeArm(s, pricing) for s in specs]
    harness = FakeHarness()
    grader = FakeGrader({"t1": True})
    journal = Journal(tmp_path)
    orch = Orchestrator(_settings(tmp_path, k_runs=1), list(arms), harness, grader, journal)

    results = await orch.run(tasks)

    # One cell per arm.
    assert len(results) == 2
    assert {r.arm for r in results} == {ArmName.A0_DIRECT, ArmName.A1_PASSTHROUGH}
    for arm in arms:
        assert arm.enter_count == 1
        assert arm.exit_count == 1
