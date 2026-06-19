# Loop weighting in Headroom Learn + RTK-loop eval

**Status:** proposed (branch `purva/rtk-loop-evals`).
**Context:** Tejas asked for (1) an eval that reproduces the RTK loop, runs it
through Headroom Learn, and checks the generated rule prevents re-triggering,
and (2) a change so Headroom Learn gives loops more weight.

## The gap

Before this change, `headroom learn` ranked every recommendation by a single
LLM-guessed `estimated_tokens_saved`, with a flat hardcoded `confidence`
(`0.9`/`0.7`). It had **no notion of a loop**. Two consequences:

1. **RTK re-fetch loops were invisible.** RTK truncates a shell command's
   output (`grep foo` → `grep foo | head -50`, see `docs/rtk-architecture.md`).
   When the truncation drops what the agent needed, the agent re-runs a
   *variant* to fetch more. **Those calls succeed** (`is_error=False`), so the
   analyzer's failure-oriented path ignored them — and `analyze()` even
   early-returned when a session had no failures and no events.

2. **Even when surfaced, a loop ranked no higher than a one-off.** A pattern
   that wastes 5,000 tokens by repeating 6× was ranked the same as a one-time
   200-token mistake, because ranking trusted the LLM's per-rule guess.

## The change

A new module `headroom/learn/loops.py`:

- **`detect_loops(sessions)`** groups tool calls within a session by a
  *canonical signature* that collapses RTK re-fetch variants (it strips
  pagination/limit fragments — `head -N`, `-n N`, `LIMIT N`, … — and bare
  integers), then flags any signature repeated `>= 3×`. It classifies each as
  an `error-loop` or an `rtk-refetch-loop` and computes **measured** wasted
  tokens (error loops waste every call; re-fetch loops credit the first,
  legitimate call and count the N−1 redundant re-fetches).
- **`format_loops_for_digest(loops)`** prepends a `=== Detected Loops (HIGHEST
  PRIORITY) ===` block to the LLM digest, with each loop's measured waste.
- **`apply_loop_weighting(recs, loops)`** raises a matching recommendation's
  `estimated_tokens_saved` to at least the loop's measured waste and tags it
  `is_loop_guardrail=True`. Because measured loop waste aggregates many
  repetitions, this reliably lifts loop guardrails above one-offs **without
  trusting the LLM** to have weighted them.

Wiring in `analyzer.py`: loops are detected up front (so a no-failure re-fetch
loop is now a first-class reason to analyze, fixing the early-return), surfaced
in the digest, the system prompt makes loops the #1 priority, and weighting +
re-sort run after parsing.

### Why measured-waste weighting (vs. relying on the LLM's estimate)

The LLM's `estimated_tokens_saved` is a free-form guess, not grounded in the
transcript, so ranking on it alone is unreliable. Deriving the weight from
*observed repetition* — the real output bytes summed across the repeated calls
— is deterministic and auditable: the boost equals waste we actually counted.

Honest caveat on the current implementation: we do BOTH — the digest also tells
the model the measured waste and asks it to rank loops first. In real-LLM runs
that prompt hint is doing much of the work (the model echoes the measured
figure), while the post-hoc `apply_loop_weighting` boost is fuzzy-match-based
and does not always fire. Making the measured-waste boost the deterministic,
load-bearing mechanism — independent of the model's wording — is tracked as
follow-up.

## The eval

`benchmarks/rtk_loop_learn_eval.py` (CI wrapper: `tests/test_learn/
test_rtk_loop_eval.py`). Two phases:

- **Phase 1 — trigger + learn:** reproduce the RTK re-fetch loop, run the
  analyzer, and score the guardrail: produced? ranked first? names the command?
  prescribes a fix? does its savings estimate reflect measured waste?
- **Phase 2 — guardrail holds:** inject that guardrail as a prior pattern, feed
  a session where the agent *followed* it (one full-output fetch, no loop), and
  assert no new loop guardrail is re-emitted — i.e. once the rule exists and is
  honored, the loop does not re-trigger.

Runs deterministically in CI (stubbed analyzer LLM) and against a real LLM with
`--real` — via an API key or an installed CLI backend (`HEADROOM_LEARN_CLI=claude`).

```
$ python benchmarks/rtk_loop_learn_eval.py
  [PASS] loop_detected          (1 loop(s), ~5,005 tok wasted)
  [PASS] guardrail_produced
  [PASS] ranked_first
  [PASS] names_command
  [PASS] prescribes_fix
  [PASS] weight_reflects_waste
  [PASS] guardrail_holds
  RESULT: PASS — loop caught, guardrail ranked first, and it holds.
```

### Real-LLM run (claude CLI backend)

Running `--real` against the actual analyzer model proved the weighting works
end-to-end *and* caught an over-brittle check. The model produced this rule,
ranked **first** with the measured 5,005-token weight:

> **Commands** — When grepping logs (or any large file), never loop with
> increasing `| head -N` limits — tool output is capped at ~4 KB regardless of
> N, so repeated attempts return identical bytes. Instead: redirect to a temp
> file (`grep ... > /tmp/out.txt`) then read it, or use `grep -c` first…

That rule is *more general* than the fixture's — it identifies the looping
command (`grep` + `head -N`) without echoing the incidental search string. An
early `names_command` check required the literal "TimeoutError" and wrongly
failed; the real run exposed it, and the check now verifies the rule names the
looping command, not an incidental literal. This is exactly why the governance
treats real output — not mocks — as proof.

## Honest limitations / open questions for review

- **Phase 2 is a non-recurrence check, not a live agent.** It proves the
  guardrail is *adequate* (names the command, prescribes the fix) and that a
  guarded, non-looping session produces no new rule. It does **not** run a real
  agent that obeys the rule end-to-end — that needs a live agent harness and is
  the natural next step if we want a stronger claim.
- **Loop signature is heuristic.** The pagination-stripping regex covers the
  common RTK truncation shapes (`head`/`tail`/`-n`/`LIMIT`/`OFFSET`); exotic
  truncations may not collapse to one signature. Easy to extend as we see real
  transcripts.
- **`min_occurrences = 3`** treats a single retry as not-yet-a-loop. If we have
  data showing 2× re-fetches are already worth a rule, lower it.
- **Matching rules to loops is fuzzy** (token overlap between the rule text and
  the looped command). A structured loop→rule id from the LLM would be tighter
  but adds prompt/parse surface.
