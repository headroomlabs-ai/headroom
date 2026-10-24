"""One-command verification of catalog-driven Copilot model routing.

Run:
    python scripts/verify_copilot_model_routing.py

Fully isolated: every file it writes goes to a throwaway temp directory, it picks
its own free port, and it removes both when done. It never touches your
repository, your `~/.headroom` state, or your Copilot instruction files.

Two tiers of check:

  PART A - hermetic. Driven entirely off the committed 40-model catalog capture
  (`tests/fixtures/copilot_models/models_list.json`). No network, no token, no
  subscription. **Anyone running this on any machine gets byte-identical
  results**, which is what makes it a shareable proof rather than an anecdote.

  PART B - live. Requires a real GitHub Copilot subscription. Numbers here
  legitimately differ between accounts (entitlements differ), so each check
  asserts a *property* -- "the wire API for a /responses-only model is
  responses" -- rather than a fixed count. Skipped cleanly with no token.

Exit code 0 = everything that ran passed. Non-zero = at least one real failure.
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "copilot_models" / "models_list.json"

_results: list[tuple[str, bool, str]] = []
_skipped: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}")
    if detail:
        print(f"         {detail}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ===========================================================================
# PART A — hermetic. Identical on every machine.
# ===========================================================================
def part_a() -> None:
    print("=" * 78)
    print("PART A - hermetic checks (no network, no token; identical everywhere)")
    print("=" * 78)

    from headroom.models.copilot_catalog import parse_models_payload, resolve_model_id
    from headroom.proxy.handlers.openai import (
        _KNOWN_COPILOT_MODEL_IDS,
        _responses_body_to_chat_completion_body,
        resolve_copilot_model_id,
    )
    from headroom.proxy.transport_planner import (
        CHAT_COMPLETIONS_PATH,
        RESPONSES_PATH,
        plan_transport,
    )

    cards = parse_models_payload(json.loads(FIXTURE.read_text(encoding="utf-8")))
    check(
        "catalog parses the committed capture", len(cards) == 40, f"{len(cards)} models (expect 40)"
    )

    # The single most common live shape: no published endpoint list. Reading it
    # as an empty allow-list would break the whole gpt-4*/gpt-3.5* family.
    unconstrained = [c for c in cards.values() if not c.constrains_endpoints()]
    check(
        "absent supported_endpoints means 'no constraint'",
        len(unconstrained) == 17 and cards["gpt-4o"].supports_endpoint(CHAT_COMPLETIONS_PATH),
        f"{len(unconstrained)}/40 publish none; gpt-4o still routable",
    )

    # gpt-5.3-codex has model_picker_enabled but NO policy object.
    check(
        "absent policy means 'no gate', not 'denied'",
        cards["gpt-5.3-codex"].selectable is True,
        "gpt-5.3-codex is selectable",
    )

    # THE headline regression. mai-code is /responses-only but does not match
    # gpt-5*/o1*/o3*, so the old name heuristic downgraded it -> upstream 400.
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["mai-code-1-flash-picker"],
        heuristic_prefers_responses=False,  # what the old heuristic said
    )
    check(
        "/responses-only model is NOT downgraded to /chat/completions",
        plan.upstream_path == RESPONSES_PATH and plan.request_bridge is None,
        f"mai-code-1-flash-picker -> {plan.upstream_path} (old heuristic said chat/completions)",
    )

    # Published endpoint ORDER is not semantic; selection must not follow it.
    ok_order = True
    for model in ("claude-opus-4.6", "claude-sonnet-4.6"):
        p = plan_transport(
            inbound_path=RESPONSES_PATH, card=cards[model], heuristic_prefers_responses=True
        )
        ok_order &= p.upstream_path == CHAT_COMPLETIONS_PATH
    check(
        "endpoint choice ignores published order",
        ok_order,
        "claude-opus-4.6 lists /v1/messages first, claude-sonnet-4.6 lists /chat/completions first; both bridge to chat",
    )

    # Every emitted plan must name a bridge that actually exists.
    unexecutable = [
        c.id
        for c in cards.values()
        if c.is_chat_model
        for inbound in (RESPONSES_PATH, CHAT_COMPLETIONS_PATH)
        for pref in (True, False)
        if not plan_transport(
            inbound_path=inbound, card=c, heuristic_prefers_responses=pref
        ).executable
    ]
    check(
        "no plan requires an unimplemented bridge",
        not unexecutable,
        f"checked all chat models x 2 wires x 2 heuristics; offenders: {unexecutable or 'none'}",
    )

    # reasoning_effort is a per-model VALUE SET. Each row below was confirmed
    # against the live API (the 'reject' rows return 400).
    clamp_rows = [
        ("claude-opus-4.6", "xhigh", "high"),
        ("claude-opus-4.6", "max", "max"),
        ("gpt-5.4", "max", "xhigh"),
        ("gpt-5.4", "xhigh", "xhigh"),
        ("gemini-3.5-flash", "minimal", "minimal"),
        ("kimi-k2.7-code", "medium", None),
    ]
    bad = [
        f"{m}+{eff}->{cards[m].clamp_reasoning_effort(eff)!r} (want {want!r})"
        for m, eff, want in clamp_rows
        if cards[m].clamp_reasoning_effort(eff) != want
    ]
    check(
        "reasoning_effort clamps per-model instead of being stripped",
        not bad,
        f"{len(clamp_rows)} live-confirmed rows; mismatches: {bad or 'none'}",
    )

    # Retired / foreign names must never be "corrected" into something plausible.
    retired = ["claude-3.5-sonnet", "gemini-2.5-pro", "glm-5.2", "gpt-5.2", "o1-experimental"]
    invented = [r for r in retired if resolve_model_id(r, cards) is not None]
    check(
        "retired/foreign model names are not invented",
        not invented,
        f"all reached for by real agents; invented: {invented or 'none'}",
    )
    stale = [
        r for r in ("gemini-2.5-pro", "gpt-5.2", "o1-experimental") if r in _KNOWN_COPILOT_MODEL_IDS
    ]
    check("static fallback table lists no retired models", not stale, f"stale: {stale or 'none'}")

    # Separator confusion, observed live (`claude-sonnet-4-6`).
    sep = {"claude-sonnet-4-6": "claude-sonnet-4.6", "gpt-5-5": "gpt-5.5"}
    wrong = {
        k: resolve_model_id(k, cards) for k, v in sep.items() if resolve_model_id(k, cards) != v
    }
    check("version-separator confusion resolves", not wrong, f"mismatches: {wrong or 'none'}")

    # Copilot-only normalization must not leak onto other upstreams.
    leaked = [
        u
        for u in ("https://api.openai.com", "https://my-gateway.internal/v1")
        if resolve_copilot_model_id("Sol", upstream_base_url=u) != "Sol"
    ]
    check(
        "non-Copilot upstreams are never rewritten",
        not leaked
        and resolve_copilot_model_id(
            "Claude Opus 4.8", upstream_base_url="https://api.githubcopilot.com"
        )
        == "claude-opus-4.8",
        f"leaked on: {leaked or 'none'}; Copilot upstream still corrects labels",
    )

    # gpt-5.4 rejects max_tokens; max_completion_tokens is accepted by all.
    body = _responses_body_to_chat_completion_body(
        "gpt-5.4", {"input": "hi", "max_output_tokens": 4096}
    )
    check(
        "bridge emits max_completion_tokens, not max_tokens",
        "max_tokens" not in body and body.get("max_completion_tokens") == 4096,
        f"keys: max_tokens={'max_tokens' in body}, max_completion_tokens={body.get('max_completion_tokens')}",
    )

    # reasoning_effort must never go out as a dict.
    body2 = _responses_body_to_chat_completion_body(
        "claude-opus-4.8", {"input": "hi", "reasoning": {"effort": "high", "summary": "detailed"}}
    )
    check(
        "reasoning_effort is never a dict on the wire",
        not isinstance(body2.get("reasoning_effort"), dict),
        f"type: {type(body2.get('reasoning_effort')).__name__}",
    )

    # Instruction injection must never damage the user's own content.
    from headroom.cli.wrap import (
        _inject_copilot_models_instructions,
        _remove_copilot_models_instructions,
    )

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / ".github" / "copilot-instructions.md"
        f.parent.mkdir(parents=True)
        mine = "# My rules\n\nAlways use tabs.\n"
        f.write_text(mine, encoding="utf-8")
        _inject_copilot_models_instructions(f, ["gpt-5.4", "claude-opus-5"])
        _inject_copilot_models_instructions(f, ["gpt-5.5"])  # relaunch: must refresh, not duplicate
        after = f.read_text(encoding="utf-8")
        _remove_copilot_models_instructions(f)
        restored = f.read_text(encoding="utf-8")
        check(
            "instruction injection is non-destructive and reversible",
            mine.strip() in after
            and after.count("<!-- headroom:available-models -->") == 1
            and "gpt-5.4" not in after
            and restored.strip() == mine.strip(),
            "user content preserved, refreshed in place (no duplicate), fully removed on unwrap",
        )


# ===========================================================================
# PART B — live. Needs a real Copilot subscription.
# ===========================================================================
def part_b() -> None:
    print()
    print("=" * 78)
    print("PART B - live checks (needs a GitHub Copilot subscription)")
    print("=" * 78)

    try:
        from headroom.copilot_auth import resolve_subscription_bearer_token_details

        res = resolve_subscription_bearer_token_details()
    except Exception as exc:  # noqa: BLE001
        _skipped.append(f"live checks skipped: token resolution raised {exc}")
        print(f"  [SKIP] {_skipped[-1]}")
        return
    if res is None:
        _skipped.append(
            "live checks skipped: no Copilot token resolved (run `headroom copilot-auth login`)"
        )
        print(f"  [SKIP] {_skipped[-1]}")
        return

    import httpx

    from headroom.copilot_auth import _copilot_chat_header_defaults
    from headroom.models.copilot_catalog import parse_models_payload
    from headroom.providers.copilot.wrap import (
        default_wire_api_for_model,
        resolve_wire_api_for_model,
    )

    try:
        r = httpx.get(
            f"{res.api_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {res.token}", **_copilot_chat_header_defaults()},
            timeout=20,
        )
        live = parse_models_payload(r.json()) if r.status_code == 200 else {}
    except Exception as exc:  # noqa: BLE001
        check("live /models is reachable", False, f"{type(exc).__name__}: {exc}")
        return

    selectable = [c for c in live.values() if c.is_chat_model and c.selectable]
    check(
        "live /models is reachable and parses",
        bool(selectable),
        f"{len(live)} entries, {len(selectable)} selectable chat models on THIS account "
        "(count varies by entitlement — that is expected)",
    )

    # Property, not a fixed value: any /responses-only model must resolve to
    # 'responses' even though its NAME says otherwise.
    only_responses = [
        c
        for c in live.values()
        if c.is_chat_model
        and c.constrains_endpoints()
        and "/responses" in c.endpoints
        and "/chat/completions" not in c.endpoints
        and default_wire_api_for_model(c.id) == "completions"
    ]
    if not only_responses:
        _skipped.append(
            "no /responses-only model on this account that the name heuristic gets wrong"
        )
        print(f"  [SKIP] launcher wire-API check: {_skipped[-1]}")
    else:
        target = only_responses[0]
        chosen = resolve_wire_api_for_model(target.id, api_url=res.api_url, token=res.token)
        check(
            "launcher picks the wire API from published endpoints, not the name",
            chosen == "responses",
            f"{target.id}: name heuristic says 'completions', catalog says {chosen!r}",
        )

    # A real bridged completion: chat-only model on the /responses wire.
    chat_only = [
        c
        for c in live.values()
        if c.is_chat_model
        and c.selectable
        and c.constrains_endpoints()
        and "/chat/completions" in c.endpoints
        and "/responses" not in c.endpoints
    ]
    if not chat_only:
        _skipped.append("no chat-only model available to exercise the bridge")
        print(f"  [SKIP] bridged completion: {_skipped[-1]}")
        return

    model = chat_only[0].id
    port = free_port()
    import os
    import subprocess
    import time

    env = dict(os.environ)
    env["GITHUB_COPILOT_API_TOKEN"] = res.token
    env["HEADROOM_SKIP_UPSTREAM_CHECK"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "headroom.cli",
            "proxy",
            "--port",
            str(port),
            "--openai-api-url",
            res.api_url,
        ],
        env=env,
        cwd=str(REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ready = False
        for _ in range(60):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        if not ready:
            check(
                "bridged /responses request returns a real completion",
                False,
                "proxy never became healthy",
            )
            return

        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/responses",
            json={
                "model": model,
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "Reply with: ok"}]}
                ],
                "stream": False,
            },
            timeout=180,
        )
        text = ""
        if resp.status_code == 200:
            for item in resp.json().get("output") or []:
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("text"):
                        text += c["text"]
        check(
            "bridged /responses request returns a real completion",
            resp.status_code == 200 and bool(text.strip()),
            f"{model} on the /responses wire -> HTTP {resp.status_code}, text={text.strip()[:40]!r}",
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    # Some checks call into code that prints via click.echo, which flushes on a
    # different schedule to print(). Piping the output (`| tail`, `> log.txt`)
    # then interleaves the two and scrambles the report. Line-buffer stdout so
    # the transcript reads in order wherever it is sent.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001 — cosmetic only
        pass
    if not FIXTURE.exists():
        print(f"ERROR: fixture missing at {FIXTURE}")
        print("Run this from a checkout of the branch that adds it.")
        return 2
    part_a()
    part_b()

    print()
    print("=" * 78)
    failed = [n for n, ok, _ in _results if not ok]
    print(
        f"RESULT: {len(_results) - len(failed)}/{len(_results)} checks passed"
        + (f", {len(_skipped)} skipped" if _skipped else "")
    )
    for name in failed:
        print(f"   FAILED: {name}")
    for note in _skipped:
        print(f"   SKIPPED: {note}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
