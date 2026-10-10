#!/usr/bin/env bash
# Weekly CodeQL triage issue, run by .github/workflows/codeql-triage.yml.
#
# Idempotent per ISO week. The week key is the ISO week of "today" (UTC), so a
# re-run of Monday's job and a manual dispatch later the same week both find
# the issue already opened for that week, including a closed one, and reuse it
# instead of opening another. A reused issue is never edited, reopened or
# retitled, so ticked checkboxes and its closure stand. Open critical/high
# alerts it does not list unticked (new ones, or ones reopened after being
# ticked) are added as a comment; otherwise the run does nothing.
#
# Env: REPO (owner/name), TRIAGE_ASSIGNEE (login). TRIAGE_TODAY (YYYY-MM-DD)
# overrides the date for tests. Needs gh, jq and python3.
set -euo pipefail

: "${REPO:?REPO is required}"
: "${TRIAGE_ASSIGNEE:?TRIAGE_ASSIGNEE is required}"
today="${TRIAGE_TODAY:-$(date -u +%Y-%m-%d)}"

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
alert_lines_jq=$(cat <<'JQ'
.[] | select(.rule.security_severity_level == "critical" or .rule.security_severity_level == "high")
    | "- [ ] [#\(.number)](\(.html_url)) **\(.rule.security_severity_level)** `\(.rule.id)` in `\(.most_recent_instance.location.path):\(.most_recent_instance.location.start_line)` (open since \(.created_at[:10]))"
JQ
)
week_issue_jq=$(cat <<'JQ'
.[] | select(.pull_request | not) | select((.body // "") | contains($marker)) | .number
JQ
)

alerts="$(gh api --paginate -X GET "repos/$REPO/code-scanning/alerts" -f state=open -f per_page=100 \
  | jq -r "$alert_lines_jq")"

# Listing, not search: the search index lags, so an issue created seconds ago
# by the previous run could be missed. state=all so a closed issue counts.
existing="$(gh api --paginate "repos/$REPO/issues?state=all&labels=security&per_page=100" \
  | jq -r --arg marker "$marker" "$week_issue_jq" | head -n 1)"

if [ -n "$existing" ]; then
  # Alerts already listed, unticked, in the issue or an earlier re-run
  # comment. A ticked alert that is open again (dismissed, then reopened) is
  # reported again rather than hidden behind its old tick.
  known="$(gh api "repos/$REPO/issues/$existing" | jq -r '.body // ""')"
  known="$known"$'\n'"$(gh api --paginate "repos/$REPO/issues/$existing/comments" | jq -r '.[].body // ""')"
  new=""
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    ref="${line#*\[#}"             # after "- [ ] [#"
    ref="- [ ] [#${ref%%\]*}]("    # "- [ ] [#123]("
    case "$known" in *"$ref"*) ;; *) new="${new}${line}"$'\n' ;; esac
  done <<< "$alerts"
  if [ -z "$new" ]; then
    echo "Week $week_key already has issue #$existing and no new critical/high alerts. Nothing to do."
    exit 0
  fi
  {
    echo "Re-run on $today: critical/high alerts that are open and not yet listed here (new, or reopened after being ticked):"
    echo
    printf '%s' "$new"
  } > comment.md
  gh issue comment "$existing" --repo "$REPO" --body-file comment.md
  echo "Week $week_key: added $(printf '%s' "$new" | grep -c '^- ') new alert(s) to #$existing."
  exit 0
fi

count="$(printf '%s' "$alerts" | grep -c '^- ' || true)"
{
  echo "$marker"
  echo "Weekly code scanning triage for the week of **$week_start** ($week_key). Owner: @$TRIAGE_ASSIGNEE (tejas@headroomlabs.ai)."
  echo
  echo "For each alert: fix it (link the PR) or dismiss it in code scanning with a reason. Close this issue when every box is ticked."
  echo "Target: critical and high fixed or dismissed within the vulnerability management SLA."
  echo
  echo "### Open critical/high alerts: $count"
  echo
  if [ "$count" -eq 0 ]; then echo "None. Close this issue to record the review."; else printf '%s\n' "$alerts"; fi
  echo
  echo "All open alerts: https://github.com/$REPO/security/code-scanning"
} > body.md
gh issue create --repo "$REPO" \
  --title "CodeQL triage: week of $week_start ($count critical/high open)" \
  --body-file body.md \
  --assignee "$TRIAGE_ASSIGNEE" \
  --label security
