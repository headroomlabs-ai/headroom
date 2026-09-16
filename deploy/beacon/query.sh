#!/usr/bin/env bash
# Query the telemetry corpus in R2 with DuckDB.
#
#   ./query.sh                 # fleet summary
#   ./query.sh sessions        # one row per session (deduped)
#   ./query.sh "SELECT ..."    # your own SQL against the corpus
#
# Setup, once:
#   brew install duckdb
#   Cloudflare > R2 > API > Create Account API Token (Object Read only,
#   scoped to headroom-telemetry), then put the values in ~/env.txt
#   (or any file named by HEADROOM_ENV_FILE):
#
#     R2_ACCOUNT_ID=...
#     R2_ACCESS_KEY_ID=...
#     R2_SECRET_ACCESS_KEY=...
#
# R2_ACCOUNT_TOKEN is Cloudflare's REST-API token and is NOT used here — the
# S3 protocol wants the access-key pair.
set -euo pipefail

BUCKET="${R2_BUCKET:-headroom-telemetry}"
[[ "$BUCKET" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid R2_BUCKET value — must contain only alphanumeric, hyphen, or underscore characters" >&2; exit 1; }
_repo_env="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
ENV_FILE="${HEADROOM_ENV_FILE:-$HOME/env.txt}"
[ -f "$ENV_FILE" ] || ENV_FILE="$_repo_env"

[ -f "$ENV_FILE" ] || { echo "no env file (~/env.txt or $_repo_env) — see this script's header" >&2; exit 1; }
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY; do
  [ -n "${!v:-}" ] || { echo "$v not set in $ENV_FILE" >&2; exit 1; }
done
command -v duckdb >/dev/null || { echo "duckdb not installed: brew install duckdb" >&2; exit 1; }

# Credentials go in via a heredoc on stdin, never on the command line, so they
# stay out of `ps` and shell history.
SECRET="
INSTALL httpfs; LOAD httpfs;
CREATE OR REPLACE SECRET r2corpus (
    TYPE r2,
    KEY_ID '${R2_ACCESS_KEY_ID}',
    SECRET '${R2_SECRET_ACCESS_KEY}',
    ACCOUNT_ID '${R2_ACCOUNT_ID}'
);
"

# The corpus is heartbeats: a session reports every 5 minutes with CUMULATIVE
# totals under one id. So the row with the highest seq per (install, session) is
# the whole session — never SUM across heartbeats, you would count each session
# once per report.
DEDUPE="
CREATE OR REPLACE TEMP VIEW sessions AS
SELECT * FROM read_ndjson('r2://${BUCKET}/sessions/**/*.json', union_by_name = true)
QUALIFY row_number() OVER (
    PARTITION BY resource['headroom.install_id'], session.id
    ORDER BY session.seq DESC
) = 1;
"

case "${1:-summary}" in
  summary)
    # Fleet rates come from summing raw counts. Averaging the per-session
    # rates.*_pct fields would weight a 10-token session equal to a 1M one.
    QUERY="
    SELECT count(*)                                                    AS sessions,
           count(DISTINCT resource['headroom.install_id'])             AS installs,
           sum(session.turns)                                          AS turns,
           sum(tokens.saved)                                           AS tokens_saved,
           sum(tokens.tool_saved)                                      AS tool_tokens_saved,
           round(sum(tokens.attempted) * 100.0
                 / nullif(sum(tokens.original), 0), 2)                 AS eligible_pct,
           round(sum(tokens.saved) * 100.0
                 / nullif(sum(tokens.attempted), 0), 2)                AS yield_pct,
           round(sum(tokens.saved) * 100.0
                 / nullif(sum(tokens.original), 0), 2)                 AS saved_pct,
           -- saved_pct/yield_pct above are context-compression only, because
           -- tool_saved never lands in original/attempted. This is the
           -- dashboard headline (server.py `savings_percent`): tool-schema
           -- savings on BOTH sides, since deferred schemas were attempted work
           -- that succeeded whole. On a tool-heavy fleet the two differ several-
           -- fold, so say which one you are quoting.
           round(sum(tokens.saved + tokens.tool_saved) * 100.0
                 / nullif(sum(tokens.original + tokens.tool_saved), 0), 2)
                                                                       AS all_layers_pct,
           sum(failures)                                               AS failures
    FROM sessions;"
    ;;
  sessions)
    QUERY="
    SELECT resource['headroom.install_id'][1:8] AS install,
           session.id, session.seq, session.turns, session.duration_s,
           tokens.original, tokens.attempted, tokens.saved,
           rates.saved_pct, rates.eligible_pct, rates.yield_pct,
           providers, models, skips
    FROM sessions
    ORDER BY session.duration_s DESC
    LIMIT 50;"
    ;;
  *) QUERY="$1" ;;
esac

printf '%s\n%s\n%s\n' "$SECRET" "$DEDUPE" "$QUERY" | duckdb -box
