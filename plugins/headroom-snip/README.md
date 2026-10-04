# headroom-snip

A Claude Code mod that shows what Headroom does to each request, live. When a
request goes through the proxy, a pair of scissors runs along a bar the size of
the original prompt, cuts it down to what was actually sent, and the token
count drops as it goes:

```
✂ headroom 21k → 16k   tok ━━━━━━━━━━━━━━━━━━━━━━━✂⡀⠄⠂⠁⢀⠈⠐⠠ −81%
✂ headroom 21k → 7.6k  tok ━━━━━━━━━━━✂⠈⠐⠠⡀⠄⠂⠁⢀⠈⠐⠠⡀⠄⠂······ −81%
✂ headroom 21k → 4.1k  tok ━━━━━━·························· −81%
  JSON crush ×2 · Kompress text · cache align   session 184k saved (62%) over 23 req ≈ 307 pages  details  hide
```

## What it shows

- **The band above the prompt.** The latest request: original → sent tokens,
  the snip animation, and the percentage cut. Below that, which compressors did
  the cutting (`JSON crush`, `code AST`, `Kompress text`, `log squash`,
  `cache align`, ...) and the session total. A request sent unchanged says why,
  for example `kept: user message, recent code` or `nothing worth a snip`.
- **`/headroom`.** Opens a pane with the session's requests: one bar per
  request, what was cut and what was protected, how long compression took, the
  biggest snip so far, and an all-time total.
  `/headroom hide` and `/headroom show` toggle the band.
- **The status line.** `✂ 184k tok saved · 62%`.
- **Milestone toasts** at 10k, 50k, 100k, 250k, 500k, 1M and 5M tokens saved.
- **No proxy?** If nothing answers, the band says so and suggests
  `headroom wrap claude`.

## How it works

While a turn runs, the mod polls the proxy's loopback `GET /stats?cached=1`
once a second. It reads the `recent_requests` tail (per-request token counts
and `transforms_applied`), and keeps polling for a few seconds after the turn
ends. Requests the proxy served before the session started count as history:
they aren't animated or added to the totals. Nothing leaves your machine.

It finds the proxy in this order:

1. `HEADROOM_PROXY_URL`
2. `ANTHROPIC_BASE_URL`, when it points at localhost (as `headroom wrap claude` sets it)
3. `http://127.0.0.1:8787`

## Install

```bash
claude plugin marketplace add headroomlabs-ai/headroom
claude plugin install headroom-snip@headroom-marketplace
```

Or, from a checkout: `claude --plugin-dir plugins/headroom-snip`.

## Develop

```bash
claude plugin validate plugins/headroom-snip
claude plugin test plugins/headroom-snip      # tests/snip.test.tsx
tsc -p plugins/headroom-snip                  # after Claude Code has loaded it once
```

`hooks/snip.ts` holds the pure parts (parsing, transform labels, the animation
frames) and `hooks/register.tsx` holds the hooks and drawing.
