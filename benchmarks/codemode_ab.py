#!/usr/bin/env python3
"""Live A/B for the headroom-codemode probe-batching directive.

Runs a fixed exploration-task suite through `claude -p` twice per task -- once
with the steering plugin active (HEADROOM_CODEMODE=1) and once without -- in a
clean detached worktree, then scores cost, turns, burst structure, bytes
fetched and answer correctness from the resulting session transcripts.

Replay benchmarks cannot measure this: steering changes which calls the model
emits, so the counterfactual is not present in recorded sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"

# (id, prompt, ground truth, kind)
TASKS = [
    (
        "ccr_py_count",
        "How many .py files are directly in headroom/ccr/? Answer with just the number.",
        "10",
        "num",
    ),
    (
        "max_rounds",
        "What is the default value of max_retrieval_rounds in ResponseHandlerConfig? Answer with just the number.",
        "3",
        "num",
    ),
    (
        "plugin_dirs",
        "How many directories are directly under plugins/? Answer with just the number.",
        "5",
        "num",
    ),
    (
        "logcompressor_lines",
        "How many lines are in crates/headroom-core/src/transforms/log_compressor.rs? Answer with just the number.",
        "1795",
        "num",
    ),
    (
        "openclaw_version",
        "What version is declared in plugins/openclaw/package.json? Answer with just the version string.",
        "0.37.0",
        "exact",
    ),
    (
        "test_ccr_files",
        "How many files directly in tests/ (not in subdirectories) have names starting with test_ccr? Answer with just the number.",
        "24",
        "num",
    ),
    (
        "crate_names",
        "List the directory names under crates/ as a comma-separated sorted list, nothing else.",
        "headroom-core,headroom-parity,headroom-proxy,headroom-py,headroom-simulators",
        "set",
    ),
    (
        "agenthooks_events",
        "How many distinct hook event names are defined in plugins/headroom-agent-hooks/hooks/hooks.json? Answer with just the number.",
        "2",
        "num",
    ),
]

# Multi-probe suite: each task needs several small independent facts, i.e. the
# exact shape the directive targets. This measures the CEILING for steering,
# not a field average -- real observed bursts average 2.6 calls.
TASKS_MULTI = [
    (
        "crate_rs_counts",
        "For each directory under crates/, report how many .rs files it contains recursively. "
        "Answer as name=count pairs, comma-separated, sorted by name, nothing else.",
        "headroom-core=99,headroom-parity=3,headroom-proxy=86,headroom-py=2,headroom-simulators=7",
        "seq",
    ),
    (
        "file_existence",
        "Do these paths exist? pyproject.toml, uv.lock, Cargo.toml, mkdocs.yml, tsconfig.json. "
        "Answer as five yes/no values, comma-separated, in that order, nothing else.",
        "yes,yes,yes,no,no",
        "seq",
    ),
    (
        "dir_counts",
        "Report three numbers: how many .py files are directly in benchmarks/, how many .md files "
        "are directly in wiki/, and how many directories are directly under plugins/. "
        "Answer as three comma-separated numbers in that order, nothing else.",
        "25,35,5",
        "seq",
    ),
    (
        "ccr_symbols",
        "In headroom/ccr/, report how many .py files contain the string 'CCR_TOOL_NAME', how many "
        "contain 'async def', and how many contain 'class '. Answer as three comma-separated numbers "
        "in that order, nothing else.",
        "6,5,8",
        "seq",
    ),
    (
        "plugin_versions",
        "Report the version field from each of: plugins/openclaw/package.json, "
        "plugins/headroom-agent-hooks/.claude-plugin/plugin.json. "
        "Answer as two comma-separated version strings in that order, nothing else.",
        "0.37.0,0.37.0",
        "seq",
    ),
    (
        "line_counts",
        "Report the line count of each of these files: headroom/ccr/response_handler.py, "
        "headroom/ccr/tool_injection.py, headroom/ccr/mcp_server.py. "
        "Answer as three comma-separated numbers in that order, nothing else.",
        "1095,615,1199",
        "seq",
    ),
]


def score(kind: str, truth: str, answer: str) -> bool:
    """Grade an answer.

    Answers often carry a trailing citation ("3\n\n(file.py:72)"), so numeric
    grading reads the FIRST number, not the last -- otherwise a line number in
    the citation is graded as the answer.
    """
    a = (answer or "").strip().replace("`", "")
    if kind == "num":
        nums = re.findall(r"\d[\d,]*", a)
        return bool(nums) and nums[0].replace(",", "") == truth
    if kind == "exact":
        return truth in a
    if kind == "seq":
        want = [t.strip().lower() for t in truth.split(",")]
        # take the first len(want) comma/whitespace-separated fields of the
        # first line that has at least that many -- ignores trailing prose
        for line in [a] + a.splitlines():
            got = [t.strip().lower() for t in re.split(r"[,\s]+", line) if t.strip()]
            if len(got) >= len(want) and got[: len(want)] == want:
                return True
        return False
    if kind == "set":
        got = {t.strip().lower() for t in re.split(r"[,\s]+", a) if t.strip()}
        want = {t.strip().lower() for t in truth.split(",")}
        return want.issubset(got)
    return False


def trace(session_id: str) -> dict:
    """Pull burst structure and bytes fetched out of the session transcript."""
    hits = list(PROJECTS.glob(f"**/{session_id}.jsonl"))
    if not hits:
        return {"trace_found": False}
    rows = []
    for line in hits[0].open(errors="replace"):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    calls = bash_calls = 0
    bursts: list[int] = []
    run = 0
    fetched = 0
    pend: set[str] = set()
    batched = 0  # compound shell commands (a proxy for directive compliance)

    def text_of(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return "\n".join(
                b.get("text", "")
                if isinstance(b, dict) and b.get("type") == "text"
                else json.dumps(b)
                for b in x
            )
        return json.dumps(x) if x is not None else ""

    for d in rows:
        t = d.get("type")
        if t == "assistant":
            tus = [
                b
                for b in (d.get("message") or {}).get("content") or []
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if tus:
                run += 1
                for b in tus:
                    calls += 1
                    pend.add(b.get("id"))
                    if b.get("name") == "Bash":
                        bash_calls += 1
                        cmd = (b.get("input") or {}).get("command") or ""
                        if re.search(r";|&&", cmd):
                            batched += 1
            else:
                if run:
                    bursts.append(run)
                run = 0
        elif t in ("user", "attachment"):
            for b in (d.get("message") or {}).get("content") or []:
                if (
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("tool_use_id") in pend
                ):
                    pend.discard(b.get("tool_use_id"))
                    fetched += len(text_of(b.get("content")))
    if run:
        bursts.append(run)
    return {
        "trace_found": True,
        "tool_calls": calls,
        "bash_calls": bash_calls,
        "compound_bash": batched,
        "bursts": len(bursts),
        "max_burst": max(bursts) if bursts else 0,
        "mean_burst": round(sum(bursts) / len(bursts), 2) if bursts else 0,
        "fetched_chars": fetched,
    }


def run_one(task, arm: str, wt: Path, plugin_dir: Path, model: str | None) -> dict:
    tid, prompt, truth, kind = task
    env = dict(os.environ)
    env["HEADROOM_CODEMODE"] = "1" if arm == "steered" else "0"
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        "auto",
        "--plugin-dir",
        str(plugin_dir),
    ]
    if model:
        cmd += ["--model", model]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=wt, env=env, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    try:
        d = json.loads(proc.stdout)
    except Exception:
        return {
            "task": tid,
            "arm": arm,
            "error": (proc.stdout or proc.stderr)[:300],
            "wall_s": wall,
        }
    u = d.get("usage") or {}
    rec = {
        "task": tid,
        "arm": arm,
        "session_id": d.get("session_id"),
        "cost_usd": d.get("total_cost_usd"),
        "num_turns": d.get("num_turns"),
        "wall_s": round(wall, 1),
        "input_tokens": u.get("input_tokens", 0),
        "cache_creation": u.get("cache_creation_input_tokens", 0),
        "cache_read": u.get("cache_read_input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "answer": str(d.get("result", ""))[:200],
    }
    rec["correct"] = score(kind, truth, rec["answer"])
    rec.update(trace(rec["session_id"] or ""))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--plugin-dir", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="benchmark_results/codemode_ab.json")
    ap.add_argument("--suite", choices=["simple", "multi"], default="simple")
    args = ap.parse_args()

    wt = Path(args.worktree).resolve()
    pd = Path(args.plugin_dir).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    done = {(r["task"], r["arm"], r.get("rep")) for r in results}

    suite = TASKS_MULTI if args.suite == "multi" else TASKS
    for rep in range(args.reps):
        for task in suite:
            # interleave arms so prompt-cache warmth cannot favour one side
            for arm in ("steered", "control") if rep % 2 == 0 else ("control", "steered"):
                if (task[0], arm, rep) in done:
                    continue
                rec = run_one(task, arm, wt, pd, args.model)
                rec["rep"] = rep
                results.append(rec)
                out.write_text(json.dumps(results, indent=2))
                print(
                    f"  {task[0]:22s} {arm:8s} rep{rep}  "
                    f"${rec.get('cost_usd', 0) or 0:.4f}  turns={rec.get('num_turns')}  "
                    f"calls={rec.get('tool_calls')}  ok={rec.get('correct')}",
                    flush=True,
                )
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"], cwd=wt, capture_output=True, text=True
                ).stdout.strip()
                if dirty:
                    print(f"    !! worktree dirtied, resetting: {dirty[:120]}", flush=True)
                    subprocess.run(["git", "reset", "--hard", "-q"], cwd=wt)
                    subprocess.run(["git", "clean", "-qfd"], cwd=wt)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
