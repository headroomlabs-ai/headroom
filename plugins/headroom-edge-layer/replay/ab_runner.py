#!/usr/bin/env python3
"""Phase 3 — online A/B runner. UNEXECUTED SCAFFOLD.

Nothing in this file has been run against a live Anthropic API key, and it
refuses to run without an explicit, human-set spending cap (see `main()`).
This exists so the runner is ready to point at real credentials once you
decide to spend real API money on Phase 3 — that decision is yours, not
something this session makes for you.

Real agent runs against the fixed task set (20 SWE-bench Verified + 10 of
your own repo's tasks, each with a written pass check — see
`scripts/census.py`'s docstring and the plan doc's Phase 0 for why the task
set has to be fixed before any of this runs), through 4 arms:

  A — direct to the provider, no proxy.
  B — stock Headroom (`--mode cache`).
  C — Headroom + the edges that passed Phase 1's replay-harness gate.
  D — direct, with Anthropic's native context editing switched on.

3 repeats per (task, arm) pair, interleaved in random order, with repeats of
the same task spaced beyond the provider's cache TTL and the CCR store
cleared between runs — per the protocol's own "avoid unfair cache effects."
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _ROOT.parents[1]
for path in (_REPO_ROOT, _ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class Arm(str, Enum):
    A_DIRECT = "a_direct"
    B_STOCK = "b_stock"
    C_EDGE_LAYER = "c_edge_layer"
    D_CONTEXT_EDITING = "d_context_editing"


@dataclass
class Task:
    task_id: str
    source: str  # "swebench_verified" | "own_repo"
    repo_url: str | None
    pass_check_command: str  # e.g. a pytest invocation; must exit 0 on pass


@dataclass
class RunRecord:
    task_id: str
    arm: Arm
    repeat_index: int
    passed: bool | None
    requests: int
    headroom_retrieve_calls: int
    failed_edits: int
    wall_time_s: float
    time_to_first_token_s: float | None
    usage: dict[str, Any]
    error: str | None = None


def load_task_manifest(path: Path) -> list[Task]:
    """Loads the fixed task set from a JSON manifest.

    This repo ships no manifest — the protocol is explicit that the task set
    "must be fixed now, before you build anything, so you can't pick tasks
    that flatter your approach later," and the 10 own-repo tasks can only
    come from you. See `README.md` for what the manifest schema expects.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No task manifest at {path}. This script will not invent a task "
            "set — the 20 SWE-bench Verified + 10 own-repo tasks (each with a "
            "written pass check) have to be fixed and committed before any "
            "run, per the protocol's own anti-cherry-picking rule."
        )
    raw = json.loads(path.read_text())
    return [Task(**item) for item in raw]


def _env_for_arm(arm: Arm, *, proxy_base_url: str | None, direct_base_url: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if arm is Arm.A_DIRECT:
        if direct_base_url:
            env["ANTHROPIC_BASE_URL"] = direct_base_url
        env.pop("HEADROOM_PIPELINE_EXTENSIONS", None)
    elif arm is Arm.B_STOCK:
        assert proxy_base_url, "arm B needs a running headroom proxy"
        env["ANTHROPIC_BASE_URL"] = proxy_base_url
        env.pop("HEADROOM_PIPELINE_EXTENSIONS", None)
    elif arm is Arm.C_EDGE_LAYER:
        assert proxy_base_url, "arm C needs a running headroom proxy"
        env["ANTHROPIC_BASE_URL"] = proxy_base_url
        env["HEADROOM_PIPELINE_EXTENSIONS"] = "edge-layer"
    elif arm is Arm.D_CONTEXT_EDITING:
        if direct_base_url:
            env["ANTHROPIC_BASE_URL"] = direct_base_url
        env.pop("HEADROOM_PIPELINE_EXTENSIONS", None)
        # Anthropic's native context-editing beta — see
        # https://docs.claude.com/en/docs/build-with-claude/context-editing.
        # Actually enabling it is a request-body/header concern for whatever
        # agent runner is used (claude -p, mini-swe-agent); this only marks
        # intent for that runner to pick up.
        env["HEADROOM_EDGE_LAYER_AB_ARM"] = "context_editing_native"
    return env


def run_single(
    task: Task,
    arm: Arm,
    repeat_index: int,
    *,
    proxy_base_url: str | None,
    direct_base_url: str | None,
    workdir: Path,
) -> RunRecord:
    """Runs one (task, arm, repeat) triple via `claude -p` and scores it.

    NEVER CALLED by anything in this session — see `main()`'s spending-cap
    guard. Left fully written so pointing it at real credentials later is a
    config change, not a rewrite.
    """
    env = _env_for_arm(arm, proxy_base_url=proxy_base_url, direct_base_url=direct_base_url)
    session_marker = uuid.uuid4().hex
    env["HEADROOM_EDGE_LAYER_RUN_ID"] = session_marker

    start = time.monotonic()
    try:
        result = subprocess.run(
            ["claude", "-p", f"Solve this task: {task.task_id}", "--output-format", "json"],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        wall_time = time.monotonic() - start
        stdout_json = json.loads(result.stdout) if result.stdout.strip() else {}
        error = None if result.returncode == 0 else f"claude -p exited {result.returncode}: {result.stderr[-2000:]}"
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        wall_time = time.monotonic() - start
        stdout_json = {}
        error = str(exc)

    passed = _run_pass_check(task, workdir) if error is None else False

    return RunRecord(
        task_id=task.task_id,
        arm=arm,
        repeat_index=repeat_index,
        passed=passed,
        requests=stdout_json.get("num_turns", 0),
        headroom_retrieve_calls=_count_retrieve_calls(stdout_json),
        failed_edits=_count_failed_edits(stdout_json),
        wall_time_s=wall_time,
        time_to_first_token_s=stdout_json.get("time_to_first_token_s"),
        usage=stdout_json.get("usage", {}),
        error=error,
    )


def _run_pass_check(task: Task, workdir: Path) -> bool:
    result = subprocess.run(task.pass_check_command, shell=True, cwd=workdir, capture_output=True, timeout=600)
    return result.returncode == 0


def _count_retrieve_calls(stdout_json: dict[str, Any]) -> int:
    # Placeholder: real accounting needs claude -p's per-turn tool-call log,
    # which is where `headroom_retrieve` calls would show up by name.
    return int(stdout_json.get("headroom_retrieve_calls", 0) or 0)


def _count_failed_edits(stdout_json: dict[str, Any]) -> int:
    return int(stdout_json.get("failed_edits", 0) or 0)


def clear_ccr_store() -> None:
    """Clears the compression store between runs — required by the
    protocol's "avoid unfair cache effects" rule, since the CCR store and
    the provider's own prompt cache are shared across runs on the same
    account/process otherwise.
    """
    from headroom.cache.compression_store import get_compression_store

    store = get_compression_store()
    clear = getattr(store, "clear", None)
    if callable(clear):
        clear()


def build_run_plan(tasks: list[Task], arms: list[Arm], *, repeats: int, seed: int) -> list[tuple[Task, Arm, int]]:
    """Interleaves (task, arm, repeat) triples in random order and returns
    them as a flat plan — arms must not run in fixed order back-to-back, per
    the protocol's cache-fairness rule.
    """
    plan = [(task, arm, repeat) for task in tasks for arm in arms for repeat in range(repeats)]
    random.Random(seed).shuffle(plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the fixed task-set JSON manifest.")
    parser.add_argument("--budget-cap-usd", type=float, required=True)
    parser.add_argument(
        "--confirm-live-spend",
        action="store_true",
        help="Required. Without this flag the script refuses to run — see the module docstring.",
    )
    parser.add_argument("--proxy-base-url", type=str, default=None)
    parser.add_argument("--direct-base-url", type=str, default=None)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.confirm_live_spend:
        print(
            "Refusing to run: pass --confirm-live-spend to acknowledge this will spend "
            f"real API money up to your stated cap (${args.budget_cap_usd:.2f}). "
            "Nothing has been called.",
            file=sys.stderr,
        )
        return 1

    tasks = load_task_manifest(args.manifest)
    arms = [Arm.A_DIRECT, Arm.B_STOCK, Arm.C_EDGE_LAYER, Arm.D_CONTEXT_EDITING]
    plan = build_run_plan(tasks, arms, repeats=args.repeats, seed=args.seed)

    print(f"Plan: {len(plan)} runs across {len(tasks)} tasks x {len(arms)} arms x {args.repeats} repeats.")
    print(f"Budget cap: ${args.budget_cap_usd:.2f}. Estimate this against your Phase 0 cost-per-session BEFORE running.")
    print("This scaffold stops here — wire in a real cost meter that aborts at the cap before removing this line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
