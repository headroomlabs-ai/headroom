"""Resumable experiment loop.

The orchestrator drives the cross-product of (arm x run_index x task), rolling out tasks
through a :class:`~agent_evals.protocols.Harness`, grading them with a
:class:`~agent_evals.protocols.Grader`, and joining the two into :class:`TaskResult` cells that
are appended to an append-only :class:`Journal`. It types exclusively against the PROTOCOLS
(``Arm``/``ArmHandle``/``Harness``/``Grader``) so real adapters and test fakes are interchangeable.

Resume guarantee: every completed cell is identified by ``TaskResult.cell_key`` and persisted to
the journal as it finishes. On a re-run, already-completed cells are skipped before any harness or
grader work happens, so an interrupted experiment resumes exactly where it left off.

No global time/RNG and no hardcoded I/O: concurrency, run count and the per-cell timeout are all
injected (Settings + an explicit ``cell_timeout_s`` parameter).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import Settings
from .logging import get_logger
from .models import (
    ArmName,
    BenchTask,
    GradeResult,
    RolloutResult,
    TaskResult,
)
from .protocols import Arm, ArmHandle, Grader, Harness

logger = get_logger("orchestrator")


class Journal:
    """Append-only JSONL of :class:`TaskResult` rows at ``run_dir/journal.jsonl``.

    Each line is one ``TaskResult`` serialized via pydantic. The journal is the single source of
    truth for resume: ``load_completed`` returns the set of cell keys already persisted so the
    orchestrator can skip them. A missing journal file is treated as an empty journal.
    """

    def __init__(self, run_dir: Path, *, filename: str = "journal.jsonl") -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / filename

    def append(self, result: TaskResult) -> None:
        """Write one ``TaskResult`` as a JSON line and flush to disk immediately."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = result.model_dump_json()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
            fh.flush()

    def all_results(self) -> list[TaskResult]:
        """Return every persisted ``TaskResult`` in journal order. Empty if the file is missing."""

        if not self.path.exists():
            return []
        results: list[TaskResult] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                results.append(TaskResult.model_validate_json(line))
        return results

    def load_completed(self) -> set[tuple[str, str, int]]:
        """Return the set of ``cell_key`` tuples already persisted (the resume frontier)."""

        return {r.cell_key for r in self.all_results()}


class Orchestrator:
    """Drives the resumable (arm x run_index x task) experiment loop.

    Typed against the protocols only. The harness produces predictions + trajectories; the grader
    turns predictions into resolved/unresolved verdicts; the arm handle attributes Layer-1 savings
    to a task. The orchestrator joins all three into journal cells.
    """

    def __init__(
        self,
        settings: Settings,
        arms: list[Arm],
        harness: Harness,
        grader: Grader,
        journal: Journal,
        *,
        cell_timeout_s: float | None = None,
    ) -> None:
        self.settings = settings
        self.arms = arms
        self.harness = harness
        self.grader = grader
        self.journal = journal
        # Default to the frozen Settings value; an explicit override (e.g. tests) wins.
        self.cell_timeout_s = (
            cell_timeout_s if cell_timeout_s is not None else settings.cell_timeout_s
        )

    async def run(self, tasks: list[BenchTask]) -> list[TaskResult]:
        """Execute the full experiment, resuming any already-completed cells from the journal.

        For each arm, the proxy is provisioned once via ``async with arm as handle``. For each
        ``run_index`` in ``range(settings.stats.k_runs)`` the orchestrator rolls out only the
        tasks whose cell is not yet in the journal, grades them once, and appends a ``TaskResult``
        per task. Returns every persisted result (including those resumed from prior runs).
        """

        k_runs = self.settings.stats.k_runs
        concurrency = self.settings.concurrency

        for arm in self.arms:
            arm_name = arm.spec.name.value
            logger.info(
                "arm_enter",
                extra={"fields": {"arm": arm_name, "k_runs": k_runs, "n_tasks": len(tasks)}},
            )
            try:
                async with arm as handle:
                    for run_index in range(k_runs):
                        await self._run_cell_group(
                            arm=arm,
                            handle=handle,
                            run_index=run_index,
                            tasks=tasks,
                            concurrency=concurrency,
                        )
            finally:
                logger.info("arm_exit", extra={"fields": {"arm": arm_name}})

        return self.journal.all_results()

    async def _run_cell_group(
        self,
        *,
        arm: Arm,
        handle: ArmHandle,
        run_index: int,
        tasks: list[BenchTask],
        concurrency: int,
    ) -> None:
        """Roll out + grade the missing tasks for one (arm, run_index), appending each cell."""

        arm_name = arm.spec.name.value
        completed = self.journal.load_completed()
        missing = [t for t in tasks if (t.task_id, arm_name, run_index) not in completed]

        if not missing:
            logger.info(
                "cell_group_skip",
                extra={
                    "fields": {
                        "arm": arm_name,
                        "run_index": run_index,
                        "reason": "all_cells_completed",
                    }
                },
            )
            return

        logger.info(
            "cell_group_start",
            extra={
                "fields": {
                    "arm": arm_name,
                    "run_index": run_index,
                    "n_missing": len(missing),
                    "concurrency": concurrency,
                }
            },
        )

        semaphore = asyncio.Semaphore(concurrency)
        rollouts: list[RolloutResult] = await asyncio.gather(
            *(
                self._rollout_one(
                    arm=arm,
                    handle=handle,
                    run_index=run_index,
                    task=task,
                    semaphore=semaphore,
                )
                for task in missing
            )
        )

        # Grade once per (arm, run_index). Only tasks that produced a prediction without a rollout
        # error are sent to the grader; errored cells are recorded as unresolved without grading.
        predictions: dict[str, str] = {}
        for rollout in rollouts:
            if rollout.error is None:
                predictions[rollout.task_id] = rollout.prediction

        grade_tasks = [t for t in missing if t.task_id in predictions]
        grades: dict[str, GradeResult] = {}
        if grade_tasks:
            logger.info(
                "grade_start",
                extra={
                    "fields": {
                        "arm": arm_name,
                        "run_index": run_index,
                        "n_graded": len(grade_tasks),
                    }
                },
            )
            grades = await asyncio.to_thread(self.grader.grade, predictions, grade_tasks)

        for rollout in rollouts:
            result = self._join_cell(
                arm_name=arm.spec.name,
                handle=handle,
                rollout=rollout,
                grade=grades.get(rollout.task_id),
            )
            self.journal.append(result)
            logger.info(
                "cell_done",
                extra={
                    "fields": {
                        "arm": arm_name,
                        "run_index": run_index,
                        "task_id": result.task_id,
                        "resolved": result.resolved,
                        "error": result.error,
                    }
                },
            )

    async def _rollout_one(
        self,
        *,
        arm: Arm,
        handle: ArmHandle,
        run_index: int,
        task: BenchTask,
        semaphore: asyncio.Semaphore,
    ) -> RolloutResult:
        """Roll out a single task under the concurrency cap with a per-cell timeout.

        Any harness exception or timeout is caught and surfaced as a ``RolloutResult`` with
        ``error`` set so a single bad cell never crashes the whole run.
        """

        arm_name = arm.spec.name.value
        task_tag = f"{arm_name}-r{run_index}-{task.task_id}"
        workdir = self.journal.run_dir / arm_name / f"run-{run_index}" / task.task_id

        async with semaphore:
            logger.info(
                "rollout_start",
                extra={
                    "fields": {"arm": arm_name, "run_index": run_index, "task_id": task.task_id}
                },
            )
            try:
                rollout = await asyncio.wait_for(
                    self.harness.run_task(task, handle.env, workdir, task_tag),
                    timeout=self.cell_timeout_s,
                )
                # The orchestrator owns cell identity: a harness only sees task_tag, so its
                # rollout.arm/run_index are advisory. Stamp the authoritative values here so a
                # cell can never be misattributed regardless of what the harness returned.
                rollout.arm = arm.spec.name
                rollout.run_index = run_index
                return rollout
            except asyncio.TimeoutError:
                logger.error(
                    "rollout_timeout",
                    extra={
                        "fields": {
                            "arm": arm_name,
                            "run_index": run_index,
                            "task_id": task.task_id,
                            "timeout_s": self.cell_timeout_s,
                        }
                    },
                )
                return RolloutResult(
                    task_id=task.task_id,
                    arm=arm.spec.name,
                    run_index=run_index,
                    prediction="",
                    trajectory_path=workdir,
                    error=f"timeout after {self.cell_timeout_s}s",
                )
            except Exception as exc:  # noqa: BLE001 - record-and-continue is the contract here.
                logger.error(
                    "rollout_error",
                    extra={
                        "fields": {
                            "arm": arm_name,
                            "run_index": run_index,
                            "task_id": task.task_id,
                            "error": repr(exc),
                        }
                    },
                    exc_info=True,
                )
                return RolloutResult(
                    task_id=task.task_id,
                    arm=arm.spec.name,
                    run_index=run_index,
                    prediction="",
                    trajectory_path=workdir,
                    error=repr(exc),
                )

    def _join_cell(
        self,
        *,
        arm_name: ArmName,
        handle: ArmHandle,
        rollout: RolloutResult,
        grade: GradeResult | None,
    ) -> TaskResult:
        """Join a rollout, its grade, and captured savings into one ``TaskResult`` cell.

        The cell's ``arm`` identity is owned by the orchestrator loop (``arm_name``), not taken
        from the rollout, so a (arm, run_index) cell is always attributed to the arm actually
        being run. An errored rollout is always recorded ``resolved=False`` and is never graded. A
        non-errored rollout that was somehow not graded is also ``resolved=False`` (loud, not
        silently dropped) — every missing cell yields exactly one journal row.
        """

        resolved = grade.resolved if (grade is not None and rollout.error is None) else False

        # Prefer the savings the harness attached to the rollout; otherwise ask the live handle.
        savings = rollout.savings
        if savings is None:
            savings = handle.capture_savings(rollout.task_id)

        return TaskResult(
            task_id=rollout.task_id,
            arm=arm_name,
            run_index=rollout.run_index,
            resolved=resolved,
            savings=savings,
            wall_ms=rollout.wall_ms,
            error=rollout.error,
        )
