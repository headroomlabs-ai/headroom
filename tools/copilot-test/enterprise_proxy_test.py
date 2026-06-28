"""Headroom × Copilot Enterprise — Phase B live-proxy harness.

Starts a real Headroom proxy pointed at the Copilot host you give it, then runs
premium models THROUGH the proxy (chat + /responses fallback), reads the
secret-free outbound capture, and prints a PASS/FAIL matrix. Host-agnostic:
point it at business / enterprise / data-residency hosts.

Usage (on the entitled machine, from a checkout of this branch, in your venv):
  GITHUB_COPILOT_API_URL=https://api.business.githubcopilot.com \
    .venv/bin/python tools/copilot-test/enterprise_proxy_test.py

It relies on Headroom's own token discovery (keychain/env/gh) — you do NOT pass a
token. Output contains no secrets (token shown as kind only). Override the CLI
path with HEADROOM_BIN if `headroom` is not on PATH or in ./.venv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

HOST = os.environ.get("GITHUB_COPILOT_API_URL", "https://api.githubcopilot.com").rstrip("/")
PORT = int(os.environ.get("HR_TEST_PORT", "8911"))
CAP = Path("/tmp/hr_enterprise_capture.jsonl")
PROBES = ["gpt-4o", "gpt-5.5", "claude-sonnet-4.6"]
REASONING = ("gpt-5", "o1", "o3")
# Resolve the headroom CLI: explicit override → source-checkout venv → PATH.
HEADROOM_BIN = (
    os.environ.get("HEADROOM_BIN")
    or (".venv/bin/headroom" if Path(".venv/bin/headroom").exists() else None)
    or shutil.which("headroom")
    or "headroom"
)

CAP.unlink(missing_ok=True)
env = {
    **os.environ,
    "OPENAI_TARGET_API_URL": HOST,
    "GITHUB_COPILOT_API_URL": HOST,
    "HEADROOM_COPILOT_DEBUG_OUTBOUND": "1",
    "HEADROOM_COPILOT_DEBUG_OUTBOUND_FILE": str(CAP),
}

print("=" * 64)
print(" HEADROOM × COPILOT — PHASE B (live proxy through-path)")
print(f" host = {HOST}   port = {PORT}")
print("=" * 64)

proc = subprocess.Popen(
    [HEADROOM_BIN, "proxy", "--port", str(PORT), "--no-rate-limit"],
    env=env,
    stdout=open("/tmp/hr_enterprise_proxy.log", "w"),
    stderr=subprocess.STDOUT,
)
base = f"http://127.0.0.1:{PORT}"
rows: list[tuple[str, int, str]] = []
try:
    # wait for readiness
    ready = False
    with httpx.Client(timeout=2) as c:
        for _ in range(60):
            try:
                if c.get(f"{base}/health").status_code == 200:
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
    if not ready:
        raise SystemExit("proxy did not become ready — see /tmp/hr_enterprise_proxy.log")

    def call(path: str, payload: dict) -> int:
        try:
            with httpx.Client(timeout=45) as c:
                # no Authorization header → proxy attaches the discovered token (like `wrap`)
                return c.post(f"{base}/v1/{path}", json=payload).status_code
        except Exception:  # noqa: BLE001
            return -1

    print("\nThrough-proxy inference:")
    for m in PROBES:
        sc = call(
            "chat/completions",
            {"model": m, "messages": [{"role": "user", "content": "reply OK"}], "max_tokens": 8},
        )
        via = "chat"
        if sc == 400 and m.startswith(REASONING):
            sc2 = call("responses", {"model": m, "input": "reply OK"})
            if sc2 != 400:
                sc, via = sc2, "responses"
            else:
                via = "chat+responses (both 400)"
        rows.append((m, sc, via))
        mark = "✅" if sc == 200 else ("🔒" if sc == 403 else "❌")
        print(f"  {mark} {m:20s} {sc} via={via}")

    print("\nOutbound capture (what Headroom actually sent — secret-free):")
    if CAP.exists():
        seen = set()
        for line in CAP.read_text().splitlines():
            import json

            r = json.loads(line)
            key = (r["host"], r["url"].rsplit("/", 1)[-1], r["token_kind"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  → {r['url']}  token={r['auth_scheme']}/{r['token_kind']}")
    else:
        print("  (no capture written — requests may not have reached the Copilot auth path)")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()

# Report --------------------------------------------------------------------
print("\n" + "=" * 64)
print(" PHASE B VERDICT")
ok = {m: sc for m, sc, _ in rows}
print(f"  host reached        : {HOST}")
print(f"  gpt-4o through proxy: {'✅ PASS' if ok.get('gpt-4o') == 200 else '❌ FAIL'}")
prem_ok = any(sc == 200 for m, sc, _ in rows if m != "gpt-4o")
print(
    f"  premium through proxy: {'✅ PASS' if prem_ok else '❌ FAIL (entitlement/policy OR Headroom routing)'}"
)
need_resp = [m for m, sc, via in rows if via == "responses" and sc == 200]
if need_resp:
    print(
        f"  wire-API split      : {', '.join(need_resp)} worked ONLY via /responses — confirm wrap auto-selects it"
    )
if not prem_ok and ok.get("gpt-4o") == 200:
    print("  → gpt-4o works but premium doesn't: if premium ALSO fails natively (run the doctor),")
    print(
        "    it's org policy/entitlement; if premium works natively but not here, it's a Headroom bug."
    )
print("=" * 64)
