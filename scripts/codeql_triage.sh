#!/usr/bin/env bash
# Weekly CodeQL triage issue, run by .github/workflows/codeql-triage.yml.
#
# The issue is public, so it carries only counts of open critical/high alerts
# and a link to the code scanning page (visible to authorized roles only):
# never an alert's rule, file, line or number.
#
# Idempotent per ISO week. The week key is the ISO week of "today" (UTC), so a
# re-run of Monday's job and a manual dispatch later the same week both find
# the issue already opened for that week, including a closed one, and reuse it
# instead of opening another. A reused issue is never edited, reopened or
# retitled, so its closure stands. Each issue body and update comment records
# the counts in a hidden marker. A re-run comments with the new counts when
# they differ from the last recorded ones, or when an open alert changed after
# that record (its updated_at moves when it is created, reopened or otherwise
# changed, so one alert closing while another reopens is still reported);
# otherwise it does nothing. The marker holds only the counts, and the time of
# the record is the public timestamp of the issue or comment carrying it, so
# nothing in the issue identifies an alert. The marker in the issue body is
# trusted (the issue is found by its week marker and the security label, which
# only collaborators can set); in comments only those by TRIAGE_BOT_LOGIN
# count, so a reader cannot silence an update by commenting.
#
# Env: REPO (owner/name), TRIAGE_ASSIGNEE (login). TRIAGE_BOT_LOGIN (default
# github-actions[bot], the GITHUB_TOKEN identity) is the author whose comments
# are trusted. TRIAGE_TODAY (YYYY-MM-DD) overrides the date for tests. Needs
# gh, jq and python3.
set -euo pipefail

: "${REPO:?REPO is required}"
: "${TRIAGE_ASSIGNEE:?TRIAGE_ASSIGNEE is required}"
today="${TRIAGE_TODAY:-$(date -u +%Y-%m-%d)}"
bot="${TRIAGE_BOT_LOGIN:-github-actions[bot]}"

# ISO week key (e.g. 2026-W41) and that week's Monday. Python's isocalendar
# handles the year boundary (2027-01-01 is in 2026-W53) on any platform's date.
read -r week_key week_start < <(python3 -c '
import datetime, sys
d = datetime.date.fromisoformat(sys.argv[1])
y, w, wd = d.isocalendar()
print(f"{y}-W{w:02d}", d - datetime.timedelta(days=wd - 1))
' "$today")
marker="<!-- codeql-triage:${week_key} -->"

# jq programs live in quoted heredocs: they are literal jq, not shell, so
# nothing in them is expanded and no single-quoted '$' or backtick is left for
# ShellCheck (SC2016) to misread.
alert_rows_jq=$(cat <<'JQ'
.[] | select(.rule.security_severity_level == "critical" or .rule.security_severity_level == "high")
    | "\(.rule.security_severity_level) \(.updated_at // .created_at)"
JQ
)
# Input: the issue object, then the comment pages. Output: "<state> <time>" of
# the last marker in the issue body or a bot comment, or nothing.
last_state_jq=$(cat <<'JQ'
.[0] as $issue
| ([$issue] + ((.[1:] | add // []) | map(select(.user.login == $bot))))
| map({at: .created_at, s: ((.body // "") | capture("<!-- codeql-triage-state:(?<s>c[0-9]+-h[0-9]+) -->") | .s)})
| last // empty
| "\(.s) \(.at)"
JQ
)
week_issue_jq=$(cat <<'JQ'
.[] | select(.pull_request | not) | select((.body // "") | contains($marker)) | .number
JQ
)

# One "<severity> <updated_at>" row per open critical/high alert. Kept local:
# only the counts are written to the issue.
alerts="$(gh api --paginate -X GET "repos/$REPO/code-scanning/alerts" -f state=open -f per_page=100 \
  | jq -r "$alert_rows_jq")"
critical="$(printf '%s' "$alerts" | grep -c '^critical ' || true)"
high="$(printf '%s' "$alerts" | grep -c '^high ' || true)"
count=$((critical + high))
state="c${critical}-h${high}"
state_marker="<!-- codeql-triage-state:${state} -->"
link="https://github.com/$REPO/security/code-scanning?query=is%3Aopen+severity%3Acritical%2Chigh"

# Listing, not search: the search index lags, so an issue created seconds ago
# by the previous run could be missed. state=all so a closed issue counts.
existing="$(gh api --paginate "repos/$REPO/issues?state=all&labels=security&per_page=100" \
  | jq -r --arg marker "$marker" "$week_issue_jq" | head -n 1)"

if [ -n "$existing" ]; then
  # Read both before parsing, so a failed request stops the run instead of
  # passing for "no marker" and posting a duplicate or skipping an update.
  issue_json="$(gh api "repos/$REPO/issues/$existing")"
  comments_json="$(gh api --paginate "repos/$REPO/issues/$existing/comments")"
  recorded="$(printf '%s\n%s\n' "$issue_json" "$comments_json" \
    | jq -rs --arg bot "$bot" "$last_state_jq")"
  recorded_state="${recorded%% *}"
  recorded_at="${recorded#* }"
  # ISO-8601 UTC timestamps compare correctly as strings.
  changed_since="$(printf '%s\n' "$alerts" \
    | awk -v at="$recorded_at" 'NF == 2 && $2 > at { n++ } END { print n + 0 }')"
  if [ -n "$recorded" ] && [ "$recorded_state" = "$state" ] && [ "$changed_since" -eq 0 ]; then
    echo "Week $week_key already has issue #$existing and the open critical/high alerts are unchanged. Nothing to do."
    exit 0
  fi
  {
    echo "$state_marker"
    echo "Re-run on $today: the open critical/high alerts changed since the last update."
    echo "Now open: critical $critical, high $high. Review them in code scanning: $link"
  } > comment.md
  gh issue comment "$existing" --repo "$REPO" --body-file comment.md
  echo "Week $week_key: posted updated counts to #$existing."
  exit 0
fi

{
  echo "$marker"
  echo "$state_marker"
  echo "Weekly code scanning triage for the week of **$week_start** ($week_key). Owner: @$TRIAGE_ASSIGNEE (tejas@headroomlabs.ai)."
  echo
  echo "For each open alert: fix it (link the PR in the alert) or dismiss it in code scanning with a reason. Close this issue when none is left open."
  echo "Target: critical and high fixed or dismissed within the vulnerability management SLA."
  echo
  echo "### Open critical/high alerts: $count"
  echo
  if [ "$count" -eq 0 ]; then
    echo "None. Close this issue to record the review."
  else
    echo "Critical: $critical, high: $high. Details are in code scanning (authorized roles only): $link"
  fi
  echo
  echo "All open alerts: https://github.com/$REPO/security/code-scanning"
} > body.md
gh issue create --repo "$REPO" \
  --title "CodeQL triage: week of $week_start ($count critical/high open)" \
  --body-file body.md \
  --assignee "$TRIAGE_ASSIGNEE" \
  --label security
