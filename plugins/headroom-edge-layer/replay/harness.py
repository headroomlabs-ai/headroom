"""Phase 1 — offline replay harness.

Replays a recorded session through a real local `headroom proxy` (via
`headroom/testing/harness.py`'s `deploy_local()`) pointed at a fake upstream
(`fake_upstream.py`) that plays back the session's own logged assistant
replies and records every request body it actually receives. That's the one
piece the existing benchmark/harness infra doesn't already do — everything
else here (arm definitions, cache simulation, recall scoring) is new but
built to slot into what's already there.

Four arms:
  A — raw, `--no-optimize` (passthrough).
  B — stock Headroom, `--mode cache`.
  C — Headroom + the edge layer (`HEADROOM_PIPELINE_EXTENSIONS=edge-layer`).
  D — Anthropic context editing, *simulated* in pure Python per its
      documented rules (no live proxy, no real beta feature) — see
      `simulate_context_editing_arm`.

Limitation carried over from the protocol itself: offline replay can't show
behavior changes (re-reads, retrievals, extra turns, failed edits) — a good
replay score earns a place in Phase 3, it doesn't prove the idea.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _ROOT.parents[1]
for path in (_REPO_ROOT, _ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.claude_session_mode_benchmark import SessionReplay  # noqa: E402
from headroom.testing.harness import Headroom  # noqa: E402
from headroom_edge_layer.message_utils import set_block_text  # noqa: E402

from cache_simulator import CacheSimResult, simulate_session_cache  # noqa: E402
from fake_upstream import FakeUpstream, anthropic_response_from_replay_turn  # noqa: E402
from recall import next_step_recall  # noqa: E402


class Arm(str, Enum):
    A_RAW = "a_raw"
    B_STOCK = "b_stock"
    C_EDGE_LAYER = "c_edge_layer"
    D_CONTEXT_EDITING = "d_context_editing"


@dataclass
class TurnResult:
    request_id: str
    sent_messages: list[dict[str, Any]]
    received_by_upstream: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    cache: CacheSimResult | None = None
    recall: float | None = None


@dataclass
class ArmRunResult:
    arm: Arm
    turns: list[TurnResult] = field(default_factory=list)
    received_bodies: list[bytes] = field(default_factory=list)

    @property
    def median_cache_read_share(self) -> float:
        shares = sorted(t.cache.cache_read_share for t in self.turns if t.cache is not None)
        if not shares:
            return 0.0
        mid = len(shares) // 2
        return shares[mid] if len(shares) % 2 else (shares[mid - 1] + shares[mid]) / 2

    @property
    def recall_scores(self) -> list[float]:
        return [t.recall for t in self.turns if t.recall is not None]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post_messages(base_url: str, body: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=data,
        headers={
            "content-type": "application/json",
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _compute_recall(turn_results: list[TurnResult], session: SessionReplay) -> None:
    for i, turn_result in enumerate(turn_results):
        if turn_result.received_by_upstream is None:
            continue
        next_original_messages = [
            m for t in session.turns[i + 1 : i + 4] for m in t.input_messages
        ]
        if not next_original_messages:
            continue
        compressed_text = json.dumps(turn_result.received_by_upstream.get("messages", []))
        turn_result.recall = next_step_recall(compressed_text, next_original_messages)


def run_live_arm(
    session: SessionReplay,
    arm: Arm,
    *,
    port: int | None = None,
    deploy_timeout_s: float = 30.0,
) -> ArmRunResult:
    """Runs arm A, B, or C against a real local proxy + fake upstream.

    Arm D never touches a live proxy — see `simulate_context_editing_arm`.
    """
    if arm is Arm.D_CONTEXT_EDITING:
        raise ValueError("arm D is simulated offline; call simulate_context_editing_arm() instead")

    responses = [anthropic_response_from_replay_turn(t) for t in session.turns]
    port = port or _free_port()
    env_overrides = {"HEADROOM_PIPELINE_EXTENSIONS": "edge-layer"} if arm is Arm.C_EDGE_LAYER else {}

    with FakeUpstream(responses) as upstream:
        builder = Headroom.with_anthropic(api_url=upstream.base_url)
        if arm is Arm.A_RAW:
            builder._proxy_config.optimize = False
        else:
            builder._proxy_config.optimize = True
            builder._proxy_config.mode = "cache"
        scenario = builder.build()

        previous_env = {key: os.environ.get(key) for key in env_overrides}
        os.environ.update(env_overrides)
        turn_results: list[TurnResult] = []
        try:
            with scenario.deploy_local(
                port=port,
                headroom_cmd=("headroom", "proxy", "--anthropic-api-url", upstream.base_url),
                timeout_s=deploy_timeout_s,
            ) as handle:
                # A real client resends the full accumulated history on every
                # call — `turn.input_messages` is only the tail new since the
                # previous turn (see benchmarks/claude_session_mode_benchmark.py's
                # `_finalize_group`), so it has to be appended onto a running
                # transcript, with each turn's own assistant reply folded back
                # in afterward for the *next* turn's history. Sending only the
                # tail would give the proxy nothing to find a cache prefix in.
                accumulated_messages: list[dict[str, Any]] = []
                for turn in session.turns:
                    accumulated_messages.extend(copy.deepcopy(turn.input_messages))
                    body = {
                        "model": turn.model,
                        "max_tokens": 1024,
                        "messages": copy.deepcopy(accumulated_messages),
                    }
                    try:
                        response = _post_messages(handle.base_url, body)
                    except (urllib.error.URLError, OSError, TimeoutError) as exc:
                        response = {"error": str(exc)}
                    turn_results.append(
                        TurnResult(
                            request_id=turn.request_id,
                            sent_messages=copy.deepcopy(accumulated_messages),
                            response=response,
                        )
                    )
                    accumulated_messages.append(copy.deepcopy(turn.assistant_message))
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        received = upstream.received_requests()
        for turn_result, recorded in zip(turn_results, received):
            try:
                turn_result.received_by_upstream = recorded.body_json
            except (json.JSONDecodeError, ValueError):
                turn_result.received_by_upstream = None

        timestamps = [float(i) for i in range(len(received))]
        cache_results = simulate_session_cache(
            [r.received_by_upstream or {} for r in turn_results], timestamps
        )
        for turn_result, cache_result in zip(turn_results, cache_results):
            turn_result.cache = cache_result

        _compute_recall(turn_results, session)
        received_bodies = [r.body for r in received]

    return ArmRunResult(arm=arm, turns=turn_results, received_bodies=received_bodies)


def simulate_context_editing_arm(
    session: SessionReplay,
    *,
    trigger_tokens: int = 100_000,
    keep_last_n: int = 3,
) -> ArmRunResult:
    """Arm D: Anthropic's documented context-editing rules, simulated.

    "Clear old tool results and thinking blocks server-side" once input
    passes the trigger — approximated here as: once cumulative observed
    input tokens exceed `trigger_tokens`, replace every `tool_result` block's
    text except the last `keep_last_n` with a cleared-content marker. No
    live proxy involved — this arm exists to compare against the built
    edges, not to be built itself.
    """
    turn_results: list[TurnResult] = []
    accumulated_messages: list[dict[str, Any]] = []
    cumulative_input_tokens = 0

    for turn in session.turns:
        # Same accumulation as run_live_arm: `turn.input_messages` is only the
        # tail new since the previous turn, so it has to be appended onto a
        # running transcript, not sent alone.
        accumulated_messages.extend(copy.deepcopy(turn.input_messages))
        cumulative_input_tokens += turn.observed_input_tokens

        if cumulative_input_tokens > trigger_tokens:
            positions = [
                (mi, bi)
                for mi, message in enumerate(accumulated_messages)
                if message.get("role") == "user" and isinstance(message.get("content"), list)
                for bi, block in enumerate(message["content"])
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            to_clear = positions[:-keep_last_n] if keep_last_n else positions
            for mi, bi in to_clear:
                block = accumulated_messages[mi]["content"][bi]
                set_block_text(block, "[cleared by simulated context editing]")

        body = {"model": turn.model, "max_tokens": 1024, "messages": copy.deepcopy(accumulated_messages)}
        turn_results.append(
            TurnResult(request_id=turn.request_id, sent_messages=accumulated_messages, received_by_upstream=body)
        )
        accumulated_messages.append(copy.deepcopy(turn.assistant_message))

    timestamps = [float(i) for i in range(len(turn_results))]
    cache_results = simulate_session_cache([t.received_by_upstream or {} for t in turn_results], timestamps)
    for turn_result, cache_result in zip(turn_results, cache_results):
        turn_result.cache = cache_result
    _compute_recall(turn_results, session)

    return ArmRunResult(
        arm=Arm.D_CONTEXT_EDITING,
        turns=turn_results,
        received_bodies=[json.dumps(t.received_by_upstream, sort_keys=True).encode() for t in turn_results],
    )


def check_determinism(session: SessionReplay, arm: Arm) -> bool:
    """Replays the same arm twice and diffs the bodies the upstream actually
    received. Any difference is a bug — it would break the provider's cache
    in production (cache-safety rule #2).
    """
    if arm is Arm.D_CONTEXT_EDITING:
        first = simulate_context_editing_arm(session)
        second = simulate_context_editing_arm(session)
    else:
        first = run_live_arm(session, arm)
        second = run_live_arm(session, arm)
    return first.received_bodies == second.received_bodies
