"""Live before/after eval for the output shaper.

Sends the SAME request to the Anthropic API twice — once as a client would
send it (baseline) and once after `shape_request` rewrites it (exactly what
the proxy forwards upstream) — and compares `usage.output_tokens`, which
includes thinking tokens.

Scenario A (verbosity steering): a complex code-review ask. Baseline vs
verbosity levels 2 and 3.

Scenario B (effort routing): an agentic transcript whose last message is a
clean tool_result (mechanical continuation) with `output_config.effort` set
to "xhigh" the way Claude Code pins it. The shaper lowers effort to "low"
for this turn only.

Usage:
    source .venv/bin/activate && python scripts/eval_output_shaper.py
Requires ANTHROPIC_API_KEY in the environment or in ./.env.
"""

from __future__ import annotations

import copy
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from headroom.proxy.output_shaper import OutputShaperSettings, shape_request  # noqa: E402

MODEL = "claude-opus-4-8"
TRIALS = 2

BUGGY_CODE = '''\
import threading
from collections import OrderedDict

class TTLCache:
    """LRU cache with per-entry TTL."""

    def __init__(self, max_size=128, ttl=300):
        self.max_size = max_size
        self.ttl = ttl
        self._store = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, now):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if now > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key, value, now):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, now + self.ttl)
            if len(self._store) > self.max_size:
                self._store.popitem(last=True)

    def cleanup(self, now):
        for key, (_, expires_at) in self._store.items():
            if now > expires_at:
                del self._store[key]
'''


def load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists() or os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            os.environ.setdefault(key.strip(), value)


def scenario_a_body() -> dict[str, Any]:
    """Complex single-turn ask — exercises verbosity steering."""
    return {
        "model": MODEL,
        "max_tokens": 8000,
        "system": "You are a senior Python engineer doing code review.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Review this cache implementation. Identify every bug and "
                    "thread-safety issue, then show how to fix each one:\n\n"
                    f"```python\n{BUGGY_CODE}```"
                ),
            }
        ],
    }


def scenario_b_body() -> dict[str, Any]:
    """Agentic mechanical continuation — exercises effort routing."""
    return {
        "model": MODEL,
        "max_tokens": 8000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "system": (
            "You are a coding agent. Use the Read tool to inspect files, then "
            "report findings concisely."
        ),
        "tools": [
            {
                "name": "Read",
                "description": "Read a file from the repository.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Check whether cache.py has thread-safety issues.",
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading cache.py first."},
                    {
                        "type": "tool_use",
                        "id": "toolu_eval_01",
                        "name": "Read",
                        "input": {"path": "cache.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_eval_01",
                        "content": BUGGY_CODE,
                    }
                ],
            },
        ],
    }


def run(client: anthropic.Anthropic, body: dict[str, Any]) -> dict[str, int]:
    # The installed SDK may predate output_config as a typed kwarg; the API
    # accepts it either way, so pass it through extra_body.
    body = dict(body)
    extra_body = None
    if "output_config" in body:
        extra_body = {"output_config": body.pop("output_config")}
    response = client.messages.create(**body, extra_body=extra_body)
    if response.stop_reason == "refusal":
        raise RuntimeError("request was refused by safety classifiers")
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main() -> int:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found (env or .env)", file=sys.stderr)
        return 1
    client = anthropic.Anthropic()
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"

    conditions: list[tuple[str, str, dict[str, Any]]] = []

    if which in ("A", "ALL"):
        # Scenario A: baseline vs steered.
        conditions.append(("A:verbosity", "baseline", scenario_a_body()))
        for level in (2, 3):
            body = scenario_a_body()
            shape_request(body, OutputShaperSettings(enabled=True, verbosity_level=level))
            conditions.append(("A:verbosity", f"shaped L{level}", body))

    if which in ("B", "ALL"):
        # Scenario B: baseline (effort=xhigh) vs shaped (effort routed to low).
        conditions.append(("B:effort-routing", "baseline xhigh", scenario_b_body()))
        body = scenario_b_body()
        result = shape_request(body, OutputShaperSettings(enabled=True, verbosity_level=0))
        assert body["output_config"]["effort"] == "low", result.labels
        conditions.append(("B:effort-routing", "shaped low", body))

    print(f"model={MODEL}  trials={TRIALS}\n")
    print(f"{'scenario':<18} {'condition':<16} {'trial':<6} {'in_tok':>7} {'out_tok':>8}")
    print("-" * 60)

    results: dict[tuple[str, str], list[int]] = {}
    for scenario, condition, body in conditions:
        for trial in range(1, TRIALS + 1):
            usage = run(client, copy.deepcopy(body))
            results.setdefault((scenario, condition), []).append(usage["output_tokens"])
            print(
                f"{scenario:<18} {condition:<16} {trial:<6} "
                f"{usage['input_tokens']:>7} {usage['output_tokens']:>8}"
            )

    print("\n=== Summary (mean output tokens, reduction vs baseline) ===")
    baselines: dict[str, float] = {}
    for (scenario, condition), outs in results.items():
        if condition.startswith("baseline"):
            baselines[scenario] = statistics.mean(outs)
    for (scenario, condition), outs in results.items():
        mean = statistics.mean(outs)
        base = baselines.get(scenario, 0)
        if condition.startswith("baseline") or not base:
            print(f"{scenario:<18} {condition:<16} {mean:>8.0f}  (baseline)")
        else:
            pct = (base - mean) / base * 100
            print(f"{scenario:<18} {condition:<16} {mean:>8.0f}  ({pct:+.1f}% vs baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
