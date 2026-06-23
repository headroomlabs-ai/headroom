"""Structural typing contracts (Protocols).

One real implementation per concrete type; no stub/fallback implementations are shipped
(house rule). These are interfaces only — the orchestrator types against ``Harness``/``Grader``
and ``Arm``/``ArmHandle`` so real adapters (Phase 1/2) and test fakes are interchangeable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import ArmSpec, BenchTask, GradeResult, Provider, RolloutResult, TaskSavings


@runtime_checkable
class ArmHandle(Protocol):
    """A live arm: the ``base_url`` + ``env`` a harness uses, plus per-task savings capture."""

    base_url: str
    env: dict[str, str]

    def capture_savings(self, task_id: str) -> TaskSavings | None:
        """Return the Layer-1 savings attributed to ``task_id``, or None if unavailable."""
        ...


@runtime_checkable
class Arm(Protocol):
    """Async context manager that provisions an :class:`ArmHandle` (spawns/tears down a proxy)."""

    spec: ArmSpec

    async def __aenter__(self) -> ArmHandle: ...

    async def __aexit__(self, *exc: object) -> None: ...


@runtime_checkable
class Harness(Protocol):
    """Rollout only — produces a prediction + trajectory for a task. Never grades."""

    name: str
    version: str
    supported_providers: set[Provider]

    async def run_task(
        self, task: BenchTask, env: dict[str, str], workdir: Path, task_tag: str
    ) -> RolloutResult: ...


@runtime_checkable
class Grader(Protocol):
    """Wraps the official, execution-based grader for a benchmark (run in a thread executor)."""

    name: str
    benchmark_ref: str

    def grade(
        self, predictions: dict[str, str], tasks: list[BenchTask]
    ) -> dict[str, GradeResult]: ...
