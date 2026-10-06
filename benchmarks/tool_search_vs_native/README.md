# Claude Code tool search: native vs Headroom

A live A/B benchmark. Each run is a fresh, isolated Claude Code session with its own
throwaway `HOME`, a new copy of `fixture/`, and six fake enterprise MCP servers
(`mcp_server.py`: GitHub, Jira, Slack, Datadog, PagerDuty and Confluence, 63 tools).
Every session does the same task:

1. Read Jira BENCH-42.
2. Fix the pricing bug.
3. Run the tests.
4. Search the Datadog logs.
5. Comment on the ticket.
6. Post to Slack.

Usage is read from Claude Code's own transcript, which records what the API billed,
subagents included. A run passes only if the tests pass and all four MCP actions
happened.

```bash
python3 benchmarks/tool_search_vs_native/run.py --arms native,hr_client,hr_server,hr_hot --reps 3 --warmup
python3 benchmarks/tool_search_vs_native/run.py ... --context-kb 200   # lookups deep in a long session
```

The script needs `ANTHROPIC_API_KEY` (or `~/env.txt`) and the repo's `.venv`. A full run
of four arms with three reps and a warmup costs about $1 short and $5 long, on Sonnet 5.5.

| Arm | Setup |
|---|---|
| `native` | Claude Code straight to api.anthropic.com, with its own ToolSearch (on by default). |
| `eager` | Claude Code straight to Anthropic with `ENABLE_TOOL_SEARCH=false`, so nothing is deferred. |
| `hr_client` | Claude Code through Headroom with `ENABLE_TOOL_SEARCH=true` (what `headroom wrap` sets today). |
| `hr_server` | Claude Code through Headroom with `ENABLE_TOOL_SEARCH=false`. Headroom defers tools on Anthropic's side. |
| `hr_hot` | As `hr_server`, but the MCP tools a developer uses routinely stay loaded (`HEADROOM_TOOL_SEARCH_CORE_TOOLS`). |

## Results (2026-10-06, Claude Code 2.1.290, Sonnet 5.5, n=3 after warmup, median)

| | native | hr_client | hr_server | hr_hot |
|---|---|---|---|---|
| short task, before fixes | $0.059 | $0.062 | $0.081 | $0.049 |
| long task, before fixes | $0.283 | $0.323 | $0.350 | $0.426 |
| long task, **after fixes** | $0.284 | **$0.284** | — | **$0.255 (−10%)** |
| short task, **after fixes** | $0.061 | $0.062 | — | **$0.046 (−25%)** |

The "after" rows held in every run: on the long task `hr_hot` cost $0.254–0.256 and
`native` $0.282–0.284.

### What the runs showed

- **A tool lookup costs a re-read of the whole conversation.** That holds for Claude
  Code's ToolSearch, which takes an extra turn, and for Anthropic's server-side search,
  which takes an extra sampling pass inside the same call (usage sums the iterations). A
  schema that stays loaded costs about 0.1× its size per turn. So keeping a routinely
  used tool loaded is far cheaper than looking it up.
- **Server-side search without names searches blindly.** Claude Code tells the model the
  deferred tool names, so it loads everything it needs in one `select:`. Headroom's
  server-side deferral gives no names, and the model ran 2–5 regex searches. Each search
  pulled in about 4.7K tokens of schemas, uncached.
- **Bug: cache mode broke when Claude Code re-serialized its last reply.** Headroom
  records the model's raw reply as the end of the previous turn. Claude Code sends that
  reply back edited: it strips a leading `cd <cwd> &&` from Bash commands and drops
  `caller`. The history then read as diverged, Headroom forwarded the raw originals over
  its earlier compressed messages, and the whole conversation was re-written to cache:
  66.8K tokens on every long `hr_hot` run. Fixed in `headroom/cache/prefix_tracker.py`
  (`_client_rewrote_reply_delta`). Set `HEADROOM_DEBUG_PREFIX_MISMATCH=<dir>` to log and
  dump where a history diverges.
- **Bug: compressing small MCP results cost turns.** The Jira ticket (a few hundred
  tokens) was compressed and lost its description, so the model called
  `headroom_retrieve` and then re-fetched the ticket: two extra turns, about +15% on the
  long task. MCP results under `HEADROOM_MCP_RESULT_MIN_CHARS` (default 4000) are now
  forwarded verbatim.

The fake MCP schemas are terse compared with real servers such as GitHub's official one,
so absolute savings here understate a real catalog.
