#!/usr/bin/env python3
"""Claude Code native tool search vs Headroom server-side tool search.

    python3 run.py --arms native,hr_server --reps 3 --model claude-sonnet-5-5

Every run is a fresh, isolated Claude Code (throwaway HOME, fresh copy of the fixture
repo, six fake enterprise MCP servers) doing the same task: read a Jira ticket, fix the
bug, run the tests, comment on the ticket, post to Slack. Usage is read from Claude
Code's own transcript (what the API billed), so the numbers include subagents.

Arms:
  native     Claude Code straight to api.anthropic.com, its own ToolSearch (default on)
  eager      Claude Code straight to Anthropic with ENABLE_TOOL_SEARCH=false (no deferral)
  hr_client  Claude Code -> Headroom, ENABLE_TOOL_SEARCH=true (today's `headroom wrap`)
  hr_server  Claude Code -> Headroom, ENABLE_TOOL_SEARCH=false; Headroom defers server-side
  hr_hot     as hr_server, but the MCP tools this developer used before stay loaded
             (HEADROOM_TOOL_SEARCH_CORE_TOOLS): no lookup at all for routine tools
  hr_ext     as hr_server, plus the proxy env given with --proxy-env KEY=VALUE
             (repeatable), e.g. to measure a proxy extension

Needs ANTHROPIC_API_KEY (or ~/env.txt). Results go to results/<timestamp>/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SERVERS = ["github", "jira", "slack", "datadog", "pagerduty", "confluence"]

TASK = (
    "Jira ticket BENCH-42 reports a pricing bug in this repository. Look the ticket up in Jira, "
    "fix the bug, and run the tests with `python3 -m unittest`. Then search Datadog logs for "
    "errors from service billing in the last hour, add a comment on BENCH-42 with a one-line "
    "summary of the fix and the Datadog result, and post a short note about the fix to the "
    "#eng-alerts Slack channel. Do not ask questions; just do it."
)

# Headroom's default resident set (headroom/proxy/helpers.py _TOOL_SEARCH_CORE_TOOLS) plus the
# MCP tools a developer doing this kind of work has used before: a learned working set.
CORE = [
    "bash",
    "bash_background",
    "bash_background_output",
    "bash_background_wait",
    "bash_background_kill",
    "read",
    "write",
    "edit",
    "multiedit",
    "apply_patch",
    "glob",
    "grep",
    "task",
    "todowrite",
    "todoread",
    "webfetch",
    "question",
    "skill",
    "toolsearch",
]
HOT = [
    "mcp__jira__get_issue",
    "mcp__jira__add_comment",
    "mcp__slack__post_message",
    "mcp__datadog__search_logs",
]
PROXY_ENV = {
    "hr_client": {},
    "hr_server": {},
    "hr_hot": {"HEADROOM_TOOL_SEARCH_CORE_TOOLS": ",".join(CORE + HOT)},
    # As hr_server plus whatever --proxy-env adds, e.g. a proxy extension under test.
    "hr_ext": {},
}

ALLOWED = ["Bash", "Read", "Edit", "Write", "Glob", "Grep", "ToolSearch"] + [
    f"mcp__{s}" for s in SERVERS
]

# Billed input-equivalent weights (Anthropic): cache write 5m 1.25x, 1h 2x, read 0.1x.
W_5M, W_1H, W_READ = 1.25, 2.0, 0.1


def api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for line in Path.home().joinpath("env.txt").read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("ANTHROPIC_API_KEY not found")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_proxy(out: Path, key: str, name: str, extra: dict) -> tuple[subprocess.Popen, int]:
    port = free_port()
    ws = out / f"headroom-workspace-{name}"
    ws.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HEADROOM_WORKSPACE_DIR": str(ws),
        "HEADROOM_TOOL_SEARCH": "1",
        "HEADROOM_TELEMETRY": "off",
        "ANTHROPIC_API_KEY": key,
        **extra,
    }
    env.pop("ANTHROPIC_BASE_URL", None)
    log = open(out / f"proxy-{name}.log", "w")
    proc = subprocess.Popen(
        [str(REPO / ".venv/bin/headroom"), "proxy", "--port", str(port)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=ws,
    )
    for _ in range(120):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc, port
        except OSError:
            time.sleep(0.5)
    proc.kill()
    sys.exit(f"proxy did not start; see {out / f'proxy-{name}.log'}")


def usage_from_transcripts(home: Path) -> dict:
    seen, calls = set(), []
    for f in glob.glob(str(home / ".claude/projects/*/**/*.jsonl"), recursive=True):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message") or {}
            if m.get("role") != "assistant" or not m.get("usage") or m.get("id") in seen:
                continue
            seen.add(m.get("id"))
            u = m["usage"]
            cc = u.get("cache_creation") or {}
            w1h = cc.get("ephemeral_1h_input_tokens", 0)
            w5m = cc.get("ephemeral_5m_input_tokens", u.get("cache_creation_input_tokens", 0) - w1h)
            tool_names = [
                b.get("name")
                for b in m.get("content") or []
                if isinstance(b, dict) and b.get("type") in ("tool_use", "server_tool_use")
            ]
            calls.append(
                {
                    "input": u.get("input_tokens", 0),
                    "w5m": w5m,
                    "w1h": w1h,
                    "read": u.get("cache_read_input_tokens", 0),
                    "output": u.get("output_tokens", 0),
                    "sidechain": bool(r.get("isSidechain")),
                    "tools": tool_names,
                }
            )
    t = {k: sum(c[k] for c in calls) for k in ("input", "w5m", "w1h", "read", "output")}
    t["api_calls"] = len(calls)
    t["toolsearch_calls"] = sum(c["tools"].count("ToolSearch") for c in calls)
    t["toolsearch_only_turns"] = sum(
        1 for c in calls if c["tools"] and set(c["tools"]) == {"ToolSearch"}
    )
    t["server_searches"] = sum(
        1 for c in calls for n in c["tools"] if n and n.startswith("tool_search_tool")
    )
    t["billed_input_equiv"] = round(
        t["input"] + W_5M * t["w5m"] + W_1H * t["w1h"] + W_READ * t["read"]
    )
    return t


def check_task(work: Path, calls_log: Path) -> dict:
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "-q"], cwd=work, capture_output=True, text=True
    )
    calls = (
        [json.loads(line) for line in calls_log.read_text().splitlines()]
        if calls_log.exists()
        else []
    )
    did = lambda svc, tool: any(c["service"] == svc and c["tool"] == tool for c in calls)  # noqa: E731
    return {
        "tests_pass": tests.returncode == 0,
        "read_ticket": did("jira", "get_issue"),  # a search result has no description
        "searched_logs": did("datadog", "search_logs"),
        "commented": did("jira", "add_comment"),
        "posted_slack": did("slack", "post_message"),
        "mcp_calls": len(calls),
    }


def run_one(
    arm: str, rep: int, out: Path, key: str, model: str, port: int | None, context_kb: int
) -> dict:
    d = out / f"{arm}-{rep}"
    home, work = d / "home", d / "work"
    home.mkdir(parents=True)
    shutil.copytree(HERE / "fixture", work)
    task = TASK
    if context_kb:  # put the MCP lookups deep in the session, where real ones happen
        (work / "docs").mkdir()
        (work / "docs/ARCHITECTURE.md").write_text(architecture_doc(context_kb))
        task = "First read docs/ARCHITECTURE.md in full (all of it, in chunks if needed). " + TASK
    subprocess.run(["git", "init", "-q"], cwd=work)
    subprocess.run(["git", "add", "-A"], cwd=work)
    subprocess.run(
        ["git", "-c", "user.email=b@b", "-c", "user.name=b", "commit", "-qm", "init"], cwd=work
    )
    calls_log = d / "mcp_calls.jsonl"
    cfg = {
        "mcpServers": {
            s: {"command": sys.executable, "args": [str(HERE / "mcp_server.py"), s, str(calls_log)]}
            for s in SERVERS
        }
    }
    (d / "mcp.json").write_text(json.dumps(cfg))

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "CLAUDE_", "ENABLE_TOOL_SEARCH", "HEADROOM_"))
    }
    env.update(
        {
            "HOME": str(home),
            "ANTHROPIC_API_KEY": key,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
        }
    )
    if arm == "eager":
        env["ENABLE_TOOL_SEARCH"] = "false"
    elif arm in PROXY_ENV:
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
        env["ENABLE_TOOL_SEARCH"] = "true" if arm == "hr_client" else "false"

    cmd = [
        "claude",
        "-p",
        task,
        "--model",
        model,
        "--mcp-config",
        str(d / "mcp.json"),
        "--strict-mcp-config",
        "--output-format",
        "json",
        "--allowedTools",
        *ALLOWED,
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    (d / "stdout.json").write_text(p.stdout)
    (d / "stderr.txt").write_text(p.stderr)
    try:
        cc = json.loads(p.stdout)
    except ValueError:
        cc = {}
    row = {
        "arm": arm,
        "rep": rep,
        "wall_s": round(wall, 1),
        "exit": p.returncode,
        "cc_cost_usd": cc.get("total_cost_usd"),
        "cc_turns": cc.get("num_turns"),
        **usage_from_transcripts(home),
        **check_task(work, calls_log),
    }
    # A session that errored out does not pass, even if it got the actions done.
    row["task_ok"] = row["exit"] == 0 and all(
        row[k] for k in ("tests_pass", "read_ticket", "searched_logs", "commented", "posted_slack")
    )
    return row


def architecture_doc(kb: int) -> str:
    """Deterministic, non-repetitive prose so compression cannot trivially erase it."""
    import random

    rng = random.Random(42)
    words = (
        "ledger invoice coupon tenant region shard replica queue retry idempotency webhook settlement "
        "refund currency rounding audit partition latency throughput cache eviction migration rollout "
        "flag canary checkout cart session token gateway schema index backfill reconciliation"
    ).split()
    out, size, n = ["# Billing platform architecture\n"], 0, 0
    while size < kb * 1024:
        n += 1
        para = f"\n## Section {n}\n" + " ".join(rng.choice(words) for _ in range(120)) + ".\n"
        out.append(para)
        size += len(para)
    return "".join(out)


def summarize(rows: list[dict]) -> str:
    keys = [
        "task_ok",
        "cc_cost_usd",
        "billed_input_equiv",
        "output",
        "api_calls",
        "toolsearch_only_turns",
        "server_searches",
        "read",
        "w5m",
        "w1h",
        "input",
        "wall_s",
    ]
    lines = ["| arm | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    for arm in dict.fromkeys(r["arm"] for r in rows):
        rs = [r for r in rows if r["arm"] == arm]
        cells = []
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if k == "task_ok":
                cells.append(f"{sum(vals)}/{len(rs)}")
            elif not vals:
                cells.append("-")
            else:
                med = statistics.median(vals)
                cells.append(
                    f"{med:.3f}"
                    if k == "cc_cost_usd"
                    else f"{med:,.0f}"
                    if k != "wall_s"
                    else f"{med:.0f}"
                )
        lines.append(f"| {arm} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n(median per run)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="native,hr_server,hr_client,eager")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument(
        "--context-kb", type=int, default=0, help="add a doc of this size to read before the task"
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="run each arm once first and discard it (warms caches)",
    )
    ap.add_argument(
        "--proxy-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra proxy environment for the hr_ext arm (repeatable)",
    )
    ap.add_argument("--out", default=str(HERE / "results" / time.strftime("%Y%m%d-%H%M%S")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = api_key()
    arms = a.arms.split(",")
    PROXY_ENV["hr_ext"] = dict(kv.split("=", 1) for kv in a.proxy_env)
    proxies = {arm: start_proxy(out, key, arm, PROXY_ENV[arm]) for arm in arms if arm in PROXY_ENV}
    rows = []
    try:
        if a.warmup:
            for arm in arms:
                run_one(arm, 0, out, key, a.model, proxies.get(arm, (None, None))[1], a.context_kb)
        for rep in range(1, a.reps + 1):  # interleave arms so drift hits every arm alike
            for arm in arms:
                row = run_one(
                    arm, rep, out, key, a.model, proxies.get(arm, (None, None))[1], a.context_kb
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
                (out / "rows.jsonl").open("a").write(json.dumps(row) + "\n")
    finally:
        for proc, _ in proxies.values():
            proc.terminate()
    s = summarize(rows)
    (out / "summary.md").write_text(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
