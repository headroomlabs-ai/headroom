# Output Token Reduction — User Guide

A plain-English guide to cutting the tokens the model **writes back**.

## Why this exists

Headroom normally shrinks the prompt you **send**. This feature shrinks what the
model **returns**. That matters because:

- Output tokens cost **5× more** than input on Opus-class models.
- A lot of model output is waste: "Great, let me help with that…" intros,
  re-printing code you already showed it, restating tool results, and long
  internal "thinking" even on trivial steps.

You don't change any code. It runs in the Headroom proxy.

## Turn it on

```bash
export HEADROOM_OUTPUT_SHAPER=1     # off by default
headroom proxy --port 8787
```

> **If a proxy is already running** (e.g. `headroom wrap claude` attaches to one
> on port 8787 instead of starting a fresh one), it reads this switch from the
> environment it was launched with — so exporting it afterwards wouldn't reach
> it. `headroom wrap` handles this for you: it hot-syncs your current output
> settings to the running proxy (loopback `POST /admin/runtime-env`), applied
> immediately with no restart. Set the variables before you run `wrap`. Because
> one proxy is shared by every session attached to it, these settings are global
> — the most recent explicit value wins.

That's it. Two things now happen on every request:

1. **Verbosity steering** — a short "be terse, don't restate context" instruction
   is added to the **end** of the system prompt. (The end, so your prompt cache
   still works.)
2. **Effort routing** — if a turn is just the model continuing after a tool ran
   (e.g. it read a file and there were no errors), Headroom turns the model's
   "thinking effort" down for that one turn. Real questions and error-handling
   turns keep full effort.

## The verbosity dial (levels 0–4)

| Level | What the model is told | Good for |
|------:|------------------------|----------|
| 0 | (off) | disable steering |
| 1 | Skip the intro/outro chit-chat | people who read everything |
| 2 | Also: don't restate code/output already on screen | **default** — safe |
| 3 | Also: conclusions only, skip the reasoning | people who skim |
| 4 | Bare minimum, fragments OK | maximum savings, terse |

Set it by hand if you want:

```bash
export HEADROOM_VERBOSITY_LEVEL=3
```

Or — better — let Headroom learn it from your habits (next section).

## Let Headroom pick the level for you

People rarely *say* "be brief." They *show* it: they interrupt long answers, or
reply so fast they couldn't have read the whole thing. `headroom learn
--verbosity` reads your past sessions and picks a level from those signals.

```bash
# Preview what it found (doesn't change anything)
headroom learn --verbosity

# Save it — the proxy uses this level from now on
headroom learn --verbosity --apply
```

Example output:

```
Verbosity — headroom
  Interrupts:  29  (11% of turns)        ← push-back signal
  Fast-skips:  31 / 119 long answers (26% unread)   ← strongest signal
  >> Recommended verbosity level: 3 (confidence: high)
```

Add `--llm-judge` to have an LLM double-check the level (needs an API key).

## See how much you saved

Here's the honest part. We **can't directly measure** output savings — we never
see what the model *would* have written without our nudge. So Headroom reports an
**estimate with a confidence range**, never a fake exact number:

```bash
headroom output-savings
```

```
Output-token reduction
  Method:    ESTIMATED (synthetic control)
  Requests:  1,240 shaped
  Saved:     ~410,000 output tokens
  Reduction: 28.0%   (95% CI 24.1% … 31.9%)
```

- **ESTIMATED** = compared against a baseline of your past (unshaped) sessions.
- **MEASURED** = the gold standard, if you opt into a holdout (below).

### Want a *measured* number?

Leave a slice of traffic unshaped as a control group:

```bash
export HEADROOM_OUTPUT_HOLDOUT=0.1     # 10% of conversations stay unshaped
```

Now `headroom output-savings` compares shaped vs unshaped directly and reports a
**measured** reduction. The trade-off: you give up the savings on that 10%.

## On the dashboard

Open `http://localhost:8787/dashboard`. Next to the input-compression card
you'll see an **Output Tokens Saved** card showing the token count, the percent,
a `measured`/`estimated` badge, and the confidence range.

## FAQ

**Will this make answers worse?**
At level 2 (default), no — in our tests the model finds the same bugs and writes
the same fixes; it just stops re-printing code and skipping the "let me…" intro.
Levels 3–4 are terser by design; that's why learning the level per user matters.

**Does it break prompt caching?**
No. The steering text is added at the *end* of the system prompt and is
byte-stable, so your cached prefix is untouched.

**Is it safe with extended thinking / tool loops?**
Yes. It never disables thinking outright (that can error), it only lowers effort
on routine turns, and it never adds settings the model doesn't support.

**How do I turn it off?**
Unset `HEADROOM_OUTPUT_SHAPER` (or set it to `0`) and restart the proxy. You can
also send `x-headroom-bypass: true` on a request to skip it for that call.

---

Deep dive (design + the counterfactual math): [`proposals/output-token-reduction.md`](proposals/output-token-reduction.md)
