#!/usr/bin/env python3
"""Deterministic stdio MCP server standing in for one enterprise service.

    python3 mcp_server.py <service> <calls.jsonl>

Exposes realistic, full-size tool schemas (GitHub, Jira, Slack, Datadog, PagerDuty,
Confluence) so a coding agent sees a catalog the size real enterprise setups have.
Every tools/call is appended to calls.jsonl so the benchmark can check the task was done.
Standard library only; speaks newline-delimited JSON-RPC 2.0 (MCP stdio transport).
"""

from __future__ import annotations

import json
import sys

SERVICE = sys.argv[1]
CALL_LOG = sys.argv[2]

# (name, description, {param: (type, description)}, required)
CATALOG = {
    "github": [
        (
            "search_repositories",
            "Search GitHub repositories by keyword, language, topic, stars or owner. Returns repository metadata including default branch, visibility, open issue count and last push time.",
            {
                "query": (
                    "string",
                    "GitHub search syntax, e.g. 'billing language:python org:acme'",
                ),
                "sort": ("string", "stars, forks, updated or best-match"),
                "per_page": ("integer", "Results per page, 1-100"),
                "page": ("integer", "Page number, starting at 1"),
            },
            ["query"],
        ),
        (
            "get_repository",
            "Get full metadata for one repository: description, topics, default branch, branch protection summary, languages, license and permissions of the caller.",
            {
                "owner": ("string", "Repository owner, user or organization"),
                "repo": ("string", "Repository name"),
            },
            ["owner", "repo"],
        ),
        (
            "list_pull_requests",
            "List pull requests in a repository with filters for state, base branch, head branch and author. Includes review state, mergeability and CI status summary.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "state": ("string", "open, closed or all"),
                "base": ("string", "Filter by base branch"),
                "author": ("string", "Filter by author login"),
            },
            ["owner", "repo"],
        ),
        (
            "get_pull_request",
            "Get a pull request including body, commits, changed files with patch hunks, review comments, requested reviewers and status checks.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "number": ("integer", "Pull request number"),
            },
            ["owner", "repo", "number"],
        ),
        (
            "create_pull_request",
            "Open a pull request from a head branch into a base branch. Supports draft pull requests, maintainer edits and an initial set of reviewers and labels.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "title": ("string", "Pull request title"),
                "head": ("string", "Branch with your changes"),
                "base": ("string", "Branch to merge into"),
                "body": ("string", "Markdown description"),
                "draft": ("boolean", "Open as a draft"),
            },
            ["owner", "repo", "title", "head", "base"],
        ),
        (
            "merge_pull_request",
            "Merge a pull request using merge, squash or rebase. Fails if required checks or reviews are missing unless the caller can bypass branch protection.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "number": ("integer", "Pull request number"),
                "method": ("string", "merge, squash or rebase"),
                "commit_title": ("string", "Title for the merge commit"),
            },
            ["owner", "repo", "number"],
        ),
        (
            "list_issues",
            "List issues with filters for labels, assignee, milestone, state and creation date. Pull requests are excluded unless requested.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "labels": ("string", "Comma-separated label names"),
                "assignee": ("string", "Assignee login, 'none' or '*'"),
                "state": ("string", "open, closed or all"),
            },
            ["owner", "repo"],
        ),
        (
            "create_issue",
            "Create an issue with title, body, labels, assignees and milestone.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "title": ("string", "Issue title"),
                "body": ("string", "Markdown body"),
                "labels": ("array", "Label names"),
                "assignees": ("array", "Assignee logins"),
            },
            ["owner", "repo", "title"],
        ),
        (
            "add_issue_comment",
            "Add a comment to an issue or pull request conversation.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "number": ("integer", "Issue or pull request number"),
                "body": ("string", "Markdown comment"),
            },
            ["owner", "repo", "number", "body"],
        ),
        (
            "get_file_contents",
            "Read a file or list a directory at a ref. Returns decoded content for text files and metadata for binaries.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "path": ("string", "Path within the repository"),
                "ref": ("string", "Branch, tag or commit SHA"),
            },
            ["owner", "repo", "path"],
        ),
        (
            "list_commits",
            "List commits on a branch or touching a path, newest first, with author, message and verification status.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "sha": ("string", "Branch or commit to start from"),
                "path": ("string", "Only commits touching this path"),
                "since": ("string", "ISO 8601 timestamp"),
            },
            ["owner", "repo"],
        ),
        (
            "get_workflow_runs",
            "List GitHub Actions workflow runs with status, conclusion, triggering event, branch and timing.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "workflow": ("string", "Workflow file name or ID"),
                "branch": ("string", "Filter by branch"),
                "status": ("string", "queued, in_progress, completed, failure, success"),
            },
            ["owner", "repo"],
        ),
        (
            "get_job_logs",
            "Download the log of one workflow job, optionally only the failing step.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "job_id": ("integer", "Workflow job ID"),
                "failed_only": ("boolean", "Return only the failing step"),
            },
            ["owner", "repo", "job_id"],
        ),
        (
            "create_branch",
            "Create a branch from an existing ref.",
            {
                "owner": ("string", "Repository owner"),
                "repo": ("string", "Repository name"),
                "branch": ("string", "New branch name"),
                "from_ref": ("string", "Branch, tag or SHA to branch from"),
            },
            ["owner", "repo", "branch"],
        ),
        (
            "search_code",
            "Search code across repositories with GitHub code search syntax. Returns file paths, repository and matching fragments.",
            {"query": ("string", "Code search query"), "per_page": ("integer", "Results per page")},
            ["query"],
        ),
    ],
    "jira": [
        (
            "search_issues",
            "Search Jira issues with JQL. Returns key, summary, status, assignee, priority, labels and updated time for each match.",
            {
                "jql": ("string", "JQL query, e.g. 'project = BENCH AND status = Open'"),
                "max_results": ("integer", "Maximum issues to return"),
                "fields": ("array", "Fields to include"),
            },
            ["jql"],
        ),
        (
            "get_issue",
            "Get one Jira issue with description, comments, status history, linked issues, components, fix versions and custom fields.",
            {
                "issue_key": ("string", "Issue key, e.g. BENCH-42"),
                "expand": ("string", "Extra sections such as changelog or renderedFields"),
            },
            ["issue_key"],
        ),
        (
            "create_issue",
            "Create a Jira issue in a project with type, summary, description, priority, labels and assignee.",
            {
                "project": ("string", "Project key"),
                "issue_type": ("string", "Bug, Task, Story or Epic"),
                "summary": ("string", "One-line summary"),
                "description": ("string", "Description in Jira markdown"),
                "priority": ("string", "Highest, High, Medium, Low"),
                "labels": ("array", "Labels"),
            },
            ["project", "issue_type", "summary"],
        ),
        (
            "update_issue",
            "Update fields on an issue: summary, description, labels, priority, assignee or custom fields.",
            {"issue_key": ("string", "Issue key"), "fields": ("object", "Field name to new value")},
            ["issue_key", "fields"],
        ),
        (
            "add_comment",
            "Add a comment to a Jira issue. Supports Jira markdown and @mentions.",
            {
                "issue_key": ("string", "Issue key, e.g. BENCH-42"),
                "body": ("string", "Comment text"),
            },
            ["issue_key", "body"],
        ),
        (
            "transition_issue",
            "Move an issue through its workflow, for example To Do -> In Progress -> Done, with an optional resolution and comment.",
            {
                "issue_key": ("string", "Issue key"),
                "transition": ("string", "Transition name or ID"),
                "resolution": ("string", "Resolution when closing"),
                "comment": ("string", "Comment to add with the transition"),
            },
            ["issue_key", "transition"],
        ),
        (
            "assign_issue",
            "Assign an issue to a user, or unassign it.",
            {
                "issue_key": ("string", "Issue key"),
                "assignee": ("string", "Account ID, or null to unassign"),
            },
            ["issue_key"],
        ),
        (
            "list_sprints",
            "List sprints on a board with state and dates.",
            {
                "board_id": ("integer", "Agile board ID"),
                "state": ("string", "active, future or closed"),
            },
            ["board_id"],
        ),
        (
            "get_sprint_issues",
            "List issues in a sprint with status and story points.",
            {"sprint_id": ("integer", "Sprint ID"), "jql": ("string", "Extra JQL filter")},
            ["sprint_id"],
        ),
        (
            "link_issues",
            "Link two issues with a relationship such as blocks, relates to or duplicates.",
            {
                "inward_issue": ("string", "Inward issue key"),
                "outward_issue": ("string", "Outward issue key"),
                "link_type": ("string", "Link type name"),
            },
            ["inward_issue", "outward_issue", "link_type"],
        ),
        (
            "add_worklog",
            "Log time spent on an issue.",
            {
                "issue_key": ("string", "Issue key"),
                "time_spent": ("string", "Duration such as 2h 30m"),
                "comment": ("string", "Work description"),
            },
            ["issue_key", "time_spent"],
        ),
        (
            "get_project",
            "Get a project's issue types, components, versions and workflow scheme.",
            {"project": ("string", "Project key")},
            ["project"],
        ),
    ],
    "slack": [
        (
            "post_message",
            "Post a message to a Slack channel or thread. Supports mrkdwn formatting, blocks and unfurl settings.",
            {
                "channel": ("string", "Channel name like #eng-alerts or channel ID"),
                "text": ("string", "Message text in mrkdwn"),
                "thread_ts": ("string", "Reply in this thread"),
            },
            ["channel", "text"],
        ),
        (
            "list_channels",
            "List public and private channels the bot can see, with member counts and topics.",
            {
                "types": ("string", "public_channel, private_channel, mpim, im"),
                "limit": ("integer", "Maximum channels"),
            },
            [],
        ),
        (
            "get_channel_history",
            "Read recent messages from a channel, newest first.",
            {
                "channel": ("string", "Channel name or ID"),
                "limit": ("integer", "Maximum messages"),
                "oldest": ("string", "Only messages after this timestamp"),
            },
            ["channel"],
        ),
        (
            "get_thread_replies",
            "Read every reply in a message thread.",
            {
                "channel": ("string", "Channel ID"),
                "thread_ts": ("string", "Parent message timestamp"),
            },
            ["channel", "thread_ts"],
        ),
        (
            "search_messages",
            "Search messages across the workspace with Slack search modifiers such as in:, from: and before:.",
            {"query": ("string", "Search query"), "count": ("integer", "Results to return")},
            ["query"],
        ),
        (
            "add_reaction",
            "Add an emoji reaction to a message.",
            {
                "channel": ("string", "Channel ID"),
                "timestamp": ("string", "Message timestamp"),
                "name": ("string", "Emoji name without colons"),
            },
            ["channel", "timestamp", "name"],
        ),
        (
            "get_user_profile",
            "Get a user's profile, title, time zone and status.",
            {"user": ("string", "User ID or email")},
            ["user"],
        ),
        (
            "set_channel_topic",
            "Set a channel's topic.",
            {"channel": ("string", "Channel ID"), "topic": ("string", "New topic")},
            ["channel", "topic"],
        ),
        (
            "upload_file",
            "Upload a file or snippet to one or more channels.",
            {
                "channels": ("array", "Channel IDs"),
                "content": ("string", "File content"),
                "filename": ("string", "File name"),
                "title": ("string", "Title"),
            },
            ["channels", "content"],
        ),
        (
            "schedule_message",
            "Schedule a message for later delivery.",
            {
                "channel": ("string", "Channel ID"),
                "text": ("string", "Message text"),
                "post_at": ("integer", "Unix timestamp"),
            },
            ["channel", "text", "post_at"],
        ),
    ],
    "datadog": [
        (
            "query_metrics",
            "Query a metric time series with Datadog query syntax over a time range. Returns points, unit and aggregation.",
            {
                "query": ("string", "e.g. avg:billing.api.latency{env:prod} by {endpoint}"),
                "from_ts": ("integer", "Start, Unix seconds"),
                "to_ts": ("integer", "End, Unix seconds"),
            },
            ["query", "from_ts", "to_ts"],
        ),
        (
            "search_logs",
            "Search logs with Datadog log query syntax, newest first, including attributes and tags.",
            {
                "query": ("string", "Log query, e.g. service:billing status:error"),
                "from_time": ("string", "Relative or absolute start, e.g. now-1h"),
                "limit": ("integer", "Maximum log events"),
            },
            ["query"],
        ),
        (
            "list_monitors",
            "List monitors with state, query, thresholds, tags and notification targets.",
            {
                "name": ("string", "Filter by name"),
                "tags": ("string", "Comma-separated tags"),
                "group_states": ("string", "alert, warn, no data, ok"),
            },
            [],
        ),
        (
            "get_monitor",
            "Get one monitor's definition, current state per group and recent state changes.",
            {"monitor_id": ("integer", "Monitor ID")},
            ["monitor_id"],
        ),
        (
            "mute_monitor",
            "Mute a monitor, optionally for one scope and until a time.",
            {
                "monitor_id": ("integer", "Monitor ID"),
                "scope": ("string", "Scope to mute"),
                "end": ("integer", "Unix time to unmute"),
            },
            ["monitor_id"],
        ),
        (
            "list_dashboards",
            "List dashboards with title, author and URL.",
            {"filter": ("string", "Title filter")},
            [],
        ),
        (
            "get_apm_traces",
            "Search APM traces by service, resource and duration, with span breakdown.",
            {
                "service": ("string", "Service name"),
                "resource": ("string", "Resource name"),
                "min_duration_ms": ("integer", "Minimum duration"),
            },
            ["service"],
        ),
        (
            "list_incidents",
            "List incidents with severity, state, commander and timeline summary.",
            {
                "state": ("string", "active, stable or resolved"),
                "severity": ("string", "SEV-1 to SEV-5"),
            },
            [],
        ),
        (
            "create_event",
            "Post an event to the event stream, for example a deploy marker.",
            {
                "title": ("string", "Event title"),
                "text": ("string", "Event body"),
                "tags": ("array", "Tags"),
            },
            ["title", "text"],
        ),
        (
            "get_service_dependencies",
            "Get upstream and downstream dependencies of a service from APM.",
            {"service": ("string", "Service name"), "env": ("string", "Environment")},
            ["service"],
        ),
    ],
    "pagerduty": [
        (
            "list_incidents",
            "List PagerDuty incidents with urgency, status, service and assignments.",
            {
                "statuses": ("array", "triggered, acknowledged, resolved"),
                "service_ids": ("array", "Service IDs"),
                "since": ("string", "ISO 8601 start"),
            },
            [],
        ),
        (
            "get_incident",
            "Get one incident with timeline, alerts, notes and responders.",
            {"incident_id": ("string", "Incident ID")},
            ["incident_id"],
        ),
        (
            "acknowledge_incident",
            "Acknowledge an incident on behalf of the caller.",
            {"incident_id": ("string", "Incident ID")},
            ["incident_id"],
        ),
        (
            "resolve_incident",
            "Resolve an incident with an optional resolution note.",
            {"incident_id": ("string", "Incident ID"), "note": ("string", "Resolution note")},
            ["incident_id"],
        ),
        (
            "list_oncalls",
            "List who is on call for schedules and escalation policies.",
            {
                "schedule_ids": ("array", "Schedule IDs"),
                "since": ("string", "Start time"),
                "until": ("string", "End time"),
            },
            [],
        ),
        (
            "add_incident_note",
            "Add a note to an incident timeline.",
            {"incident_id": ("string", "Incident ID"), "content": ("string", "Note text")},
            ["incident_id", "content"],
        ),
        (
            "list_services",
            "List services with escalation policy and integrations.",
            {"query": ("string", "Name filter")},
            [],
        ),
        (
            "create_incident",
            "Trigger a new incident on a service.",
            {
                "service_id": ("string", "Service ID"),
                "title": ("string", "Incident title"),
                "urgency": ("string", "high or low"),
                "body": ("string", "Details"),
            },
            ["service_id", "title"],
        ),
    ],
    "confluence": [
        (
            "search_pages",
            "Search Confluence pages with CQL, returning title, space, excerpt and last editor.",
            {
                "cql": ("string", "CQL query, e.g. space = ENG and text ~ 'discount'"),
                "limit": ("integer", "Maximum results"),
            },
            ["cql"],
        ),
        (
            "get_page",
            "Get a page body in storage format with version, ancestors and labels.",
            {
                "page_id": ("string", "Page ID"),
                "expand": ("string", "body.storage, version, ancestors"),
            },
            ["page_id"],
        ),
        (
            "create_page",
            "Create a page in a space, optionally under a parent page.",
            {
                "space": ("string", "Space key"),
                "title": ("string", "Page title"),
                "body": ("string", "Body in storage format"),
                "parent_id": ("string", "Parent page ID"),
            },
            ["space", "title", "body"],
        ),
        (
            "update_page",
            "Update a page body; requires the next version number.",
            {
                "page_id": ("string", "Page ID"),
                "title": ("string", "Title"),
                "body": ("string", "New body"),
                "version": ("integer", "Next version number"),
            },
            ["page_id", "body", "version"],
        ),
        (
            "add_page_comment",
            "Add a footer comment to a page.",
            {"page_id": ("string", "Page ID"), "body": ("string", "Comment body")},
            ["page_id", "body"],
        ),
        (
            "list_spaces",
            "List spaces with key, name, type and homepage.",
            {"type": ("string", "global or personal"), "limit": ("integer", "Maximum spaces")},
            [],
        ),
        (
            "get_page_children",
            "List child pages of a page.",
            {"page_id": ("string", "Page ID")},
            ["page_id"],
        ),
        (
            "add_label",
            "Add labels to a page.",
            {"page_id": ("string", "Page ID"), "labels": ("array", "Labels")},
            ["page_id", "labels"],
        ),
    ],
}

ISSUE = {
    "key": "BENCH-42",
    "summary": "Discount applied twice at checkout",
    "status": "Open",
    "priority": "High",
    "description": (
        "Customers with a 10% coupon are charged 81% of the price instead of 90%. "
        "The bug is in shop/pricing.py, function apply_discount. "
        "tests/test_pricing.py reproduces it. When fixed, comment on this issue with a one-line "
        "summary of the fix and post a short note to the #eng-alerts Slack channel."
    ),
    "comments": [],
}


def schema(params: dict, required: list) -> dict:
    props = {}
    for name, (typ, desc) in params.items():
        p = {"type": typ, "description": desc}
        if typ == "array":
            p["items"] = {"type": "string"}
        props[name] = p
    return {"type": "object", "properties": props, "required": required}


TOOLS = [
    {"name": n, "description": d, "inputSchema": schema(p, r)} for n, d, p, r in CATALOG[SERVICE]
]


def call(name: str, args: dict) -> str:
    with open(CALL_LOG, "a") as fh:
        fh.write(json.dumps({"service": SERVICE, "tool": name, "args": args}) + "\n")
    if SERVICE == "jira" and name == "get_issue":
        return json.dumps(ISSUE if args.get("issue_key") == "BENCH-42" else {"error": "not found"})
    if SERVICE == "jira" and name == "search_issues":
        return json.dumps(
            {"issues": [{k: ISSUE[k] for k in ("key", "summary", "status", "priority")}]}
        )
    return json.dumps({"ok": True, "service": SERVICE, "tool": name})


def main() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # notification
            continue
        if method == "initialize":
            result = {
                "protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVICE, "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            p = msg.get("params", {})
            result = {
                "content": [{"type": "text", "text": call(p.get("name"), p.get("arguments") or {})}]
            }
        elif method == "ping":
            result = {}
        else:
            sys.stdout.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": mid,
                        "error": {"code": -32601, "message": f"unknown method {method}"},
                    }
                )
                + "\n"
            )
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
