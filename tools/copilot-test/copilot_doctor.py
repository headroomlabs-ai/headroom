"""headroom copilot doctor.

Run on the machine where Copilot is logged in. Read-only. Prints token
PREFIXES only, never secrets. Safe to hand to an Enterprise contact and have
them paste the output back.

Answers:
  - WHERE does the Copilot credential live (env / files / keychain)?
  - WHICH host will Headroom route to (standard vs enterprise / data-residency)?
  - WHAT does Headroom actually FORWARD (token kind + identity headers)?  ← outbound capture
  - WHAT can this seat RUN — on /chat/completions AND /responses (reasoning models)?
  - Does Headroom's PASS-THROUGH path forward a client token untouched (no keystore)?

Enterprise host:
  GITHUB_COPILOT_API_URL=https://api.enterprise.githubcopilot.com \
    .venv/bin/python tools/copilot-test/copilot_doctor.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

try:
    from headroom import copilot_auth
except Exception as e:  # noqa: BLE001
    raise SystemExit(
        f"Run me from the headroom repo via .venv/bin/python — import failed: {e}"
    ) from e

PREMIUM = ("gpt-5", "claude", "gemini", "o1", "o3")
REASONING = ("gpt-5", "o1", "o3")  # these need /responses, not /chat/completions (#644/#647)
PROBE = ["gpt-4o", "gpt-5.5", "claude-sonnet-4.6"]
# GitHub token TYPE prefixes — these are not secret (the random bytes after are).
TOKEN_PREFIXES = ("github_pat_", "gho_", "ghu_", "ghs_", "ghp_", "tid_")


def redact(t: str | None) -> str:
    """Show only the non-secret type prefix + length — never any token bytes."""
    if not t:
        return "—"
    for p in TOKEN_PREFIXES:
        if t.startswith(p):
            return f"{p}…(len {len(t)})"
    return f"…(len {len(t)})"


def head(n: str) -> None:
    print(f"\n{'─' * 64}\n{n}")


def status_msg(r: httpx.Response) -> tuple[int, str]:
    b = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    er = b.get("error")
    msg = (er.get("message") if isinstance(er, dict) else er) or (
        "OK" if r.status_code == 200 else r.text[:40]
    )
    return r.status_code, str(msg)[:46]


print("=" * 64)
print(" HEADROOM · COPILOT DOCTOR")
print("=" * 64)

# [1] Env -------------------------------------------------------------------
head("[1] Environment variables")
for v in copilot_auth._API_TOKEN_ENV_VARS + copilot_auth._COPILOT_OAUTH_TOKEN_ENV_VARS:
    # presence by KEY only — never read the value of a token env var
    print(f"    {v:38s} {'SET' if v in os.environ else 'unset'}")
for v in (
    "GITHUB_COPILOT_API_URL",
    "GITHUB_COPILOT_ENTERPRISE_URL",
    "GITHUB_COPILOT_ENTERPRISE_DOMAIN",
):
    print(f"    {v:38s} {os.environ.get(v) or 'unset'}")

# [2] Credential files ------------------------------------------------------
head("[2] Credential files on this machine")
home = Path.home()
candidates_files = {
    home
    / ".copilot/config.json": "new Copilot CLI (plaintext fallback)  ← Headroom does NOT read token here",
    home / ".config/github-copilot/apps.json": "older layout  ← Headroom DOES read this",
    home / ".config/github-copilot/hosts.json": "older layout  ← Headroom DOES read this",
    home / ".config/gh/hosts.yml": "gh CLI creds (keychain-backed on mac)",
    Path(
        "/workspaces/.codespaces/shared/user-secrets-envs.json"
    ): "Codespaces secrets (REMOTE host)",
}
for f, note in candidates_files.items():
    print(f"    {'✓ EXISTS' if f.exists() else '·       '}  {str(f):46s} {note}")

# [3] What Headroom discovers ----------------------------------------------
head("[3] OAuth token candidates Headroom discovers (priority order)")
cands = copilot_auth.iter_oauth_token_candidates()
for c in cands or []:
    print(f"    {c.source:42s} conf={getattr(c, 'confidence', '?'):14s} {redact(c.token)}")
if not cands:
    print("    (NONE — Headroom cannot find a credential on this machine)")
cached = copilot_auth.read_cached_oauth_token()
print(
    f"\n    → resolved = {redact(cached)}  kind={'tid_(session)' if cached and cached.startswith('tid_') else ('gho_(OAuth)' if cached else 'none')}"
)

# [4] Resolved token + host -------------------------------------------------
head("[4] Resolved API token + host (Headroom's live path)")
api_token, api_url = None, "https://api.githubcopilot.com"
try:
    tok = asyncio.run(copilot_auth.get_copilot_token_provider().get_api_token())
    api_token, api_url = tok.token, (tok.api_url or api_url).rstrip("/")
    print(
        f"    api token : {redact(api_token)}  kind={'tid_(exchanged)' if api_token.startswith('tid_') else 'gho_(used DIRECTLY — no exchange)'}"
    )
    print(f"    api host  : {api_url}")
    _host = (urlparse(api_url).hostname or "").lower()
    if _host == "ghe.com" or _host.endswith(".ghe.com") or "enterprise" in _host:
        print("    host type : ENTERPRISE / data-residency")
    elif "individual" in _host:
        print("    host type : individual segmented host")
    else:
        print(
            "    host type : generic public host  (set GITHUB_COPILOT_API_URL to test an enterprise host)"
        )
except Exception as e:  # noqa: BLE001
    print(f"    ERROR: {e}")

# [5] Exact OUTBOUND request Headroom would forward (capture) ---------------
head("[5] OUTBOUND capture — exactly what Headroom forwards to GitHub")
try:
    sample = asyncio.run(copilot_auth.apply_copilot_api_auth({}, url=f"{api_url}/chat/completions"))
    auth = next((v for k, v in sample.items() if k.lower() == "authorization"), "")
    sch, _, raw = auth.partition(" ")
    print(f"    POST {api_url}/chat/completions")
    print(
        f"    Authorization        : {sch} {redact(raw)}  ({copilot_auth._token_kind(raw) if raw else 'none'})"
    )
    for k in ("Copilot-Integration-Id", "Editor-Version", "Editor-Plugin-Version", "User-Agent"):
        v = next((vv for kk, vv in sample.items() if kk.lower() == k.lower()), None)
        print(f"    {k:21s}: {v or '(absent)'}")
except Exception as e:  # noqa: BLE001
    print(f"    capture error: {e}")

# [6] Catalog ---------------------------------------------------------------
hdr = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json",
    "Copilot-Integration-Id": "vscode-chat",
    "Editor-Version": "vscode/1.107.0",
}
head("[6] /models catalog (what the picker SHOWS)")
if api_token:
    try:
        with httpx.Client(timeout=20) as c:
            r = c.get(f"{api_url}/models", headers=hdr)
        if r.status_code == 403:
            print(
                "    http=403 → SSO authorization likely required (authorize the org in your IdP / re-login)"
            )
        ids = [m.get("id") for m in (r.json().get("data") or [])] if r.status_code == 200 else []
        prem = [i for i in ids if any(k in i for k in PREMIUM)]
        print(
            f"    http={r.status_code}  total={len(ids)}  premium_listed={', '.join(prem[:10]) or '(none)'}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"    error: {e}")

# [7] Inference entitlement — chat + /responses fallback --------------------
head("[7] Inference entitlement (RUN: /chat/completions, then /responses for reasoning models)")
results: dict[str, int] = {}
endpoint_used: dict[str, str] = {}
sso = False
if api_token:

    def post(path: str, payload: dict) -> httpx.Response:
        with httpx.Client(timeout=30) as c:
            return c.post(f"{api_url}/{path}", headers=hdr, json=payload)

    for m in PROBE:
        sc, msg = status_msg(
            post(
                "chat/completions",
                {
                    "model": m,
                    "messages": [{"role": "user", "content": "reply OK"}],
                    "max_tokens": 8,
                },
            )
        )
        used = "chat"
        # reasoning models often 400 on chat but work on /responses (the #644/#647 wire-API split)
        if sc == 400 and any(m.startswith(k) for k in REASONING):
            sc2, msg2 = status_msg(post("responses", {"model": m, "input": "reply OK"}))
            if sc2 != 400:
                sc, msg, used = sc2, msg2, "responses"
            else:
                used = "chat+responses (both 400)"
        results[m] = sc
        endpoint_used[m] = used
        sso = sso or sc == 403
        mark = "✅" if sc == 200 else ("🔒" if sc == 403 else "❌")
        print(f"    {mark} {m:20s} {sc} via={used:24s} {msg}")

# [8] Pass-through contract (platform-agnostic path) ------------------------
head("[8] Pass-through contract — does Headroom forward a CLIENT token untouched?")
sim = "tid_SIMULATED_client_session"
try:
    resolved = asyncio.run(
        copilot_auth.apply_copilot_api_auth(
            {"authorization": f"Bearer {sim}", "x-api-key": "sk-should-drop"},
            url=f"{api_url}/chat/completions",
        )
    )
    fwd = next((v for k, v in resolved.items() if k.lower() == "authorization"), "")
    passthru = fwd == f"Bearer {sim}"
    xapi_dropped = not any(k.lower() == "x-api-key" for k in resolved)
    print(f"    client presents : Bearer {sim[:14]}…  (a tid_ session token)")
    print(
        f"    Headroom sends  : {'UNCHANGED ✅' if passthru else 'REPLACED ❌ → ' + fwd[:24]}   (x-api-key dropped: {xapi_dropped})"
    )
    print(
        "    → "
        + (
            "pass-through works with NO keystore read — identical on mac/win/linux/containers/Codespaces."
            if passthru
            else "client token NOT preserved — investigate before relying on pass-through."
        )
    )
except Exception as e:  # noqa: BLE001
    print(f"    error: {e}")

# [9] Verdict ---------------------------------------------------------------
head("VERDICT")
print(
    f"    Credential found    : {cands[0].source if cands else 'NONE — discovery gap (run where the token lives, or use env/pass-through)'}"
)
print(
    f"    Token forwarded     : {'tid_ session' if (api_token or '').startswith('tid_') else 'gho_ OAuth (direct, no exchange)'}"
)
print(f"    API host            : {api_url}")
print(
    f"    USABLE now          : {', '.join(m for m, sc in results.items() if sc == 200) or '(none)'}"
)
print(
    f"    BLOCKED             : {', '.join(m for m, sc in results.items() if sc == 400) or '(none)'}"
)
needed_responses = [
    m for m, ep in endpoint_used.items() if ep == "responses" and results.get(m) == 200
]
if needed_responses:
    print(
        f"    ⚠ wire-API split    : {', '.join(needed_responses)} only work via /responses — confirm `headroom wrap`"
    )
    print("                          auto-selects --wire-api responses for these (#644/#647).")
if sso:
    print("    → 403 seen: SSO authorization needed — authorize the org via your IdP, then re-run.")
if any(sc == 400 for sc in results.values()):
    print(
        "    → remaining 400s are an upstream GitHub ENTITLEMENT/policy limit, NOT a Headroom bug."
    )
    print(
        "      Unlock = Copilot Pro/Business/Enterprise + admin enables the models in Copilot policy."
    )
if any(sc == 200 for m, sc in results.items() if m != "gpt-4o"):
    print(
        "    → a premium model RUNS on this seat — now verify it also flows THROUGH `headroom wrap copilot`."
    )
print("\n" + "=" * 64)
