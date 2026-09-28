#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run-all-plugins.sh — install + configure + run the Headroom proxy with ALL 5
# enterprise plugins, the coding savings-profile, and ML compression offloaded
# to the Kompress-v2 Modal endpoint. Then confirm everything loaded.
#
#   Plugins : lossless_guard, skill_search, observability, tier_router, tool_search
#   Extra   : headroom-ai[sandbox]  (torch-free proxy; ML offloaded to Modal)
#   Profile : coding  (HEADROOM_SAVINGS_PROFILE) + cache mode (prefix-cache safe)
#
# Secrets are SOURCED from ~/env.txt and ~/.headroom/plugins.env — never inlined.
# Re-runnable: install is skipped when already satisfied (FORCE_INSTALL=1 forces).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HR=/Users/tcms/demo/headroom
VENV="$HR/.venv"
PORT="${HEADROOM_PORT:-8787}"
ENV_TXT="${ENV_TXT:-$HOME/env.txt}"
PLUGINS_ENV="$HOME/.headroom/plugins.env"
LOG="${HEADROOM_LOG:-$HOME/.headroom/logs/proxy-all-plugins.log}"
mkdir -p "$(dirname "$LOG")"

# ── 1. venv ──────────────────────────────────────────────────────────────────
# `python`/`pip`/`uv` are broken system-wide on this box — always use the venv,
# and `python -m pip` (the .venv/bin/pip shim is broken too).
# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"

# ── 2. install (guarded) ──────────────────────────────────────────────────────
# headroom-ai[sandbox] pulls proxy,code,relevance,reports,otel,html,mcp,spreadsheet
# (all torch-free — heavy ML is offloaded to the Modal Kompress endpoint below).
# The 5 plugins install --no-deps so pip won't drag PyPI's headroom-ai over the
# local editable one; headroom-license is their shared Ed25519 verifier.
need_install=1
if [ "${FORCE_INSTALL:-0}" != "1" ]; then
  n=$("$PY" -c 'import opentelemetry; from headroom.proxy.extensions import discover; print(len(list(discover())))' 2>/dev/null || echo 0)
  [ "$n" = "5" ] && need_install=0
fi
if [ "$need_install" = "1" ]; then
  avail=$(df -g "$HR" 2>/dev/null | awk 'NR==2{print $4}')
  echo "▶ disk: ${avail:-?}Gi free before install"
  if [ -n "$avail" ] && [ "$avail" -lt 2 ]; then
    echo "!! <2Gi free — aborting before heavy install (free space, then re-run)"; exit 1
  fi
  echo "▶ installing pip+maturin, then headroom-ai[sandbox] + license + 5 plugins (editable)…"
  "$PY" -m pip install -U pip maturin
  # litellm >=1.92 ships an sdist-only Rust bridge whose AWS-SDK crates need rustc>=1.94.1;
  # the default rustup toolchain here is older (pip builds litellm in a temp dir that misses
  # the repo's 1.95 pin), so pin to the last pure-Python wheel line (1.91.4). Satisfies
  # headroom's litellm>=1.86.2,<2.0 and skips the Rust build entirely.
  "$PY" -m pip install "litellm<1.92"
  "$PY" -m pip install -e "${HR}[sandbox]" "litellm<1.92"
  "$PY" -m pip install -e /Users/tcms/demo/headroom-license
  for p in lossless-guard skill-search observability tier-router tool-search; do
    "$PY" -m pip install -e "/Users/tcms/demo/headroom-${p}" --no-deps
  done
else
  echo "▶ install satisfied (5 extensions discovered) — skipping (FORCE_INSTALL=1 to force)"
fi

# ── 3. secrets from ~/env.txt ─────────────────────────────────────────────────
# Provides: OPENAI_API_KEY, ANTHROPIC_API_KEY, FIREWORKS_API_KEY (upstream creds);
#           LANGFUSE_{PUBLIC,SECRET}_KEY + LANGFUSE_BASE_URL (observability sink);
#           HEADROOM_KOMPRESS_ENDPOINT + _TOKEN (Modal ML offload).
[ -f "$ENV_TXT" ] || { echo "!! $ENV_TXT not found"; exit 1; }
set -a; # shellcheck disable=SC1090
source "$ENV_TXT"; set +a

# ── 4. plugin license (Ed25519, offline, wildcard) ────────────────────────────
# HEADROOM_LICENSE + HEADROOM_LICENSE_PUBKEY. This is SEPARATE from the OSS cloud
# key (HEADROOM_LICENSE_KEY) — the banner will still say "OSS (no license key)",
# but each plugin prints "license accepted". Fallback: skip verification entirely.
if [ -f "$PLUGINS_ENV" ]; then
  set -a; # shellcheck disable=SC1090
  source "$PLUGINS_ENV"; set +a
else
  echo "▶ $PLUGINS_ENV missing — using dev license bypass"
  export HEADROOM_LICENSE_DEV=1
fi

# ── 5. Kompress ML offload → Modal ────────────────────────────────────────────
# Setting HEADROOM_KOMPRESS_ENDPOINT (+_TOKEN) alone routes Kompress inference to
# the Modal endpoint (content_router._get_kompress_remote). No other flag needed;
# HEADROOM_COMPRESS_ALLOW_REMOTE is a different thing (remote upstreams, not this).
: "${HEADROOM_KOMPRESS_ENDPOINT:?must be set in $ENV_TXT}"
export HEADROOM_KOMPRESS_ENDPOINT_TOKEN="${HEADROOM_KOMPRESS_ENDPOINT_TOKEN:-}"

# ── 6. observability sink → Langfuse + spend attribution ──────────────────────
# HEADROOM_LANGFUSE_ENABLED must be explicitly truthy (LANGFUSE_* creds come from
# env.txt). Traces (agent.turn / llm.turn spans with gen_ai.usage.cost) land in
# Langfuse. Spend is opt-in: HEADROOM_MODEL_PRICES is {model-substr:{in,out}} in
# USD per 1K tokens. (Per-request identity — org/team/user/session — is supplied
# by the CLIENT via x-headroom-* headers, not settable here.)
export HEADROOM_LANGFUSE_ENABLED=1
export HEADROOM_LANGFUSE_SERVICE_NAME=headroom-proxy
export HEADROOM_MODEL_PRICES='{"claude-opus":{"in":0.015,"out":0.075},"claude-sonnet":{"in":0.003,"out":0.015},"gpt-5":{"in":0.00125,"out":0.01},"gpt-4":{"in":0.003,"out":0.012}}'
# Metrics (counters) need a separate OTLP endpoint — none in env.txt, so left off:
# export HEADROOM_OTEL_METRICS_ENABLED=1 HEADROOM_OTEL_METRICS_ENDPOINT=http://localhost:4318

# ── 7. tier_router ─────────────────────────────────────────────────────────────
# Only stamps service_tier on the wire (no token delta). OpenAI 'flex' is only
# auto-selected for models declared eligible here. Anthropic tiers are a no-op by
# default. Clients force a tier with x-headroom-tier / x-headroom-background: 1.
export HEADROOM_TIER_FLEX_MODELS="${HEADROOM_TIER_FLEX_MODELS:-gpt-5,gpt-4.1,o4-mini}"

# ── 8. plugin tuning (defaults shown; override as needed) ─────────────────────
# skill_search fires on Anthropic w/ >=min skills; tool_search on synthetic-tier
# providers w/ >=min tools; lossless_guard lossy tier is opt-in (kept OFF).
export HEADROOM_SKILL_SEARCH_MIN_SKILLS="${HEADROOM_SKILL_SEARCH_MIN_SKILLS:-8}"
export HEADROOM_TOOL_SEARCH_MIN_TOOLS="${HEADROOM_TOOL_SEARCH_MIN_TOOLS:-5}"
# export HEADROOM_LOSSLESS_GUARD_LOSSY=1   # opt-in irreversible Bash-noise drop

# ── 9. coding profile + mode ───────────────────────────────────────────────────
# savings_profile=coding tunes the pipeline for coding-agent traffic; cache mode
# freezes prior turns to preserve the provider prefix-cache (what coding wants).
export HEADROOM_SAVINGS_PROFILE=coding

# ── 10. run + confirm ──────────────────────────────────────────────────────────
cleanup() { [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "▶ starting proxy on :$PORT  (profile=coding, mode=cache, all 5 extensions)…"
headroom proxy --port "$PORT" --mode cache --proxy-extension '*' > "$LOG" 2>&1 &
PROXY_PID=$!

# wait for readiness (no foreground sleep on this harness)
curl -s --retry 40 --retry-delay 1 --retry-all-errors --max-time 60 \
  "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || true

echo
echo "══════════════════ CONFIRMATION ══════════════════"
echo "── extensions loaded (from $LOG) ──"
grep -iE "Extensions:|license accepted|installed \(" "$LOG" | sed 's/^/  /' || true

echo "── Modal Kompress endpoint reachable? ──"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$HEADROOM_KOMPRESS_ENDPOINT" || echo "unreachable")
echo "  $HEADROOM_KOMPRESS_ENDPOINT -> HTTP $code (any response = up; offload runs on real traffic)"

echo "── /stats surfaces (empty until traffic flows) ──"
curl -s --max-time 5 "http://127.0.0.1:$PORT/stats" | "$PY" -c '
import sys,json
d=json.load(sys.stdin)
print("  extension_savings   :", d.get("extension_savings"))
print("  by_layer            :", list(d.get("savings",{}).get("by_layer",{})))
print("  tokens_saved_by_strat:", d.get("tokens_saved_by_strategy"))
print("  otel.enabled        :", d.get("otel",{}).get("enabled"))
print("  langfuse.enabled    :", d.get("langfuse",{}).get("enabled"))
' 2>/dev/null || echo "  (stats not ready)"

cat <<EOF

── where each effect shows up ──
  lossless_guard  -> dashboard (compression layer)  + /stats.tokens_saved_by_strategy
  skill_search    -> /stats.extension_savings  (NOT dashboard) — Anthropic client, >=8 skills
  tool_search     -> /stats.extension_savings  (NOT dashboard) — synthetic-tier client, >=5 tools
  observability   -> Langfuse UI (spans + gen_ai.usage.cost)   — send x-headroom-org/user/session
  tier_router     -> service_tier on the wire / provider bill  (no token delta)

── drive traffic (two clients — they exercise different plugins) ──
  Claude Code : ANTHROPIC_BASE_URL=http://localhost:$PORT claude        # lossless_guard + skill_search
  OpenAI/opencode: OPENAI_BASE_URL=http://localhost:$PORT/v1 <client>    # tool_search
  Dashboard   : headroom dashboard   (http://127.0.0.1:$PORT/dashboard)
  Raw stats   : curl -s localhost:$PORT/stats | python3 -m json.tool

Proxy is running (pid $PROXY_PID). Ctrl-C to stop. Logs: $LOG
═══════════════════════════════════════════════════
EOF

wait "$PROXY_PID" || true
