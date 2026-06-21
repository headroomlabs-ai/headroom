# Output Token Reduction

**Branches:** `feat/output-token-reduction` (Phase 1) → `feat/verbosity-learning-and-counterfactual` (Phase 2).
**Status:** Phase 1 (output shaper) and Phase 2 (`learn --verbosity`, AIMD controller, counterfactual estimator, dashboard) both implemented + tested. Runtime AIMD signal-capture is staged (controller built/tested, live emission off by default).

See **§7** for the counterfactual measurement methodology (how we honestly report a number we can't directly observe).

---

## 1. The problem in one line

Headroom's entire transform pipeline compresses what goes **into** the model.
Nothing today touches what comes **out**. But output tokens are billed at
5× input on Opus-class models ($25 vs $5 per MTok on `claude-opus-4-8`), and in
agentic coding loops a large fraction of the bill is output: thinking tokens,
restated code, ceremony ("Great, let me…"), and full-file rewrites where a
10-line edit would do.

**The constraint that shapes everything:** the proxy never generates output
tokens — the model does. Once a token is streamed it is already billed. So
every output-token lever is **request-side**: change what we ask for, cap what
we allow, or avoid the generation entirely. There is no post-hoc lever.

That gives three lever families plus a learning loop:

| Lever | Mechanism | Status |
|---|---|---|
| **Verbosity steering** | Append a terse-style instruction to the system-prompt tail | ✅ built |
| **Effort routing** | Lower `output_config.effort` on mechanical turns | ✅ built |
| **Thinking-budget clamp** | Clamp legacy `thinking.budget_tokens` on mechanical turns | ✅ built |
| **Per-user learned level** | Mine past sessions for the right verbosity per user (`learn --verbosity`) | ✅ built |
| **Counterfactual estimator** | Honestly estimate output tokens saved + dashboard surfacing | ✅ built |
| **Runtime AIMD auto-tune** | Adjust level live from interrupt / skip signals | 🟡 controller built/tested; live signal emission off by default |

---

## 2. Phase 1 — the output shaper (built)

### 2.1 What it is

`headroom/proxy/output_shaper.py` — a request-body rewriter invoked in
`handle_anthropic_messages` after every other body mutation (so the turn
classifier sees the final message list) and gated behind the same
`x-headroom-bypass` header as compression. Opt-in via `HEADROOM_OUTPUT_SHAPER=1`.

### 2.2 Lever A — verbosity steering

A deterministic instruction block is appended to the **tail** of the system
prompt. Five levels, cumulative:

- **L0** — off (touch nothing).
- **L1** — no ceremony: skip preamble/postamble, don't announce what you're about to do.
- **L2** — L1 + no echo: never restate code/diffs/tool output already in context; reference by path:line; don't narrate tool results. **(default)**
- **L3** — L2 + conclusions only, omit rationale unless asked, prefer smallest edit.
- **L4** — caveman: fragments, minimum tokens, nothing but the answer.

**Why the tail, not the head.** Prompt caching is a prefix match — any byte
change ahead of a `cache_control` breakpoint invalidates everything after it.
Prepending steering text would bust the provider prefix cache and cost more
than it saves. Appending after the last system block leaves the cached prefix
byte-identical; only the small, byte-stable steering block is reprocessed. The
steering text is frozen per level and applied idempotently (sentinel-tagged),
so repeated requests keep an identical prefix and a level change replaces the
block in place rather than stacking.

### 2.3 Lever B — effort routing

In an agentic loop, most API calls are **mechanical continuations**: the last
message is a clean `tool_result` (a file read, a passing test) and the model is
just resuming. Harnesses like Claude Code pin `output_config.effort` at `xhigh`
for *every* turn, including these — and effort drives thinking depth, which
bills as output. The router lowers effort to `low` on mechanical turns only.

Turn classification is **purely structural** — no content regexes, no keyword
lists (per the project's no-hardcoded-patterns rule):

| Last user message contains… | Classification | Action |
|---|---|---|
| Any text / image / document block | `NEW_USER_ASK` | leave effort alone |
| Only `tool_result`, none `is_error` | `MECHANICAL_CONTINUATION` | lower effort → `low` |
| Any `tool_result` with `is_error: true` | `ERROR_CONTINUATION` | leave effort alone (model must reason about the failure) |
| Anything else | `UNKNOWN` | leave alone |

### 2.4 Lever C — legacy thinking-budget clamp

On older models still sending `thinking: {type: "enabled", budget_tokens: N}`,
the router clamps `N` to the API floor (1024) on mechanical turns. The `type`
field is **never** toggled.

### 2.5 Safety rules (each prevents a concrete failure)

1. **Never inject `output_config.effort` where the client didn't send it.**
   Models without effort support return 400 on it. Lowering an
   already-present value is always valid — its presence proves the target
   model accepts the param.
2. **Never toggle `thinking.type`.** Disabling thinking while history carries
   thinking blocks 400s on some models, and the toggle busts the messages
   cache tier (per the caching invalidation hierarchy).
3. **Byte-stable, idempotent steering** — repeated requests keep an identical
   prefix; cache stays warm.
4. **Respect `x-headroom-bypass`** — sub-agent calls that opt out of
   compression also opt out of shaping.

### 2.6 Configuration

| Env var | Default | Meaning |
|---|---|---|
| `HEADROOM_OUTPUT_SHAPER` | off | master switch (`1`/`true`/`yes`) |
| `HEADROOM_VERBOSITY_LEVEL` | `2` | 0–4 (clamped) |
| `HEADROOM_EFFORT_ROUTER` | on | set `0` to disable effort routing |
| `HEADROOM_MECHANICAL_EFFORT` | `low` | floor effort for mechanical turns |

### 2.7 Tests

`tests/test_output_shaper.py` — 34 tests, all passing. Covers turn
classification (every block-type path), cache-safe steering (string→block
conversion, append-after-`cache_control`, idempotency, level change),
effort routing (lower / never-inject / untouched-on-non-mechanical /
configurable target), legacy budget clamp, and the env gate. Ruff + mypy clean.

---

## 3. Live before/after results

Measured against `claude-opus-4-8` via `scripts/eval_output_shaper.py`,
comparing the exact body a client sends vs. the body the proxy forwards.
`usage.output_tokens` includes thinking.

### 3.1 Verbosity steering — complex code-review ask

Prompt: "review this TTLCache, find every bug, show fixes."

| Condition | Mean output tokens | Reduction |
|---|---|---|
| Baseline | ~1,800–1,930 | — |
| L2 (no ceremony, no echo) | ~1,180–1,470 | **−22% to −39%** |
| L3 (conclusions only) | ~670 | **−63%** |

**Quality check — same bugs found, only redundancy removed.** The baseline
opens with a title + "Let me go through them," finds the bugs, **re-prints the
entire fixed class**, then adds a **summary table restating all six bugs**,
then a trailing notes section — the same information appears ~2.5×. L2 opens
directly with the findings, gives the same fixed code block once, and stops.
Both correctly identify the no-locking race, `popitem(last=True)` evicting the
wrong end, and the mutate-during-iteration `RuntimeError`. Nothing of substance
is lost at L2; L3 additionally drops rationale prose (only for users who don't
read explanations).

### 3.2 Effort routing — agentic mechanical continuation

Transcript ending in a clean `tool_result`, `effort: xhigh` the way Claude Code
sends it.

| Condition | Mean output tokens | Reduction |
|---|---|---|
| Baseline (xhigh) | ~1,120 | — |
| Shaped (routed to low) | 793 | **−29%** |

### 3.3 Honest framing

Caveman (L4) would get ~60–70% but degrades the experience. The taxonomy-driven
default (L2 + effort routing on mechanical turns) gets a realistic **25–40%**
with the user barely noticing. The learning loop (Phase 2) finds where each
user sits between those poles.

---

## 4. Phase 2 — learning the right level per user

### 4.1 Why this is necessary

The right verbosity is **per-user**, not global. A fixed `HEADROOM_VERBOSITY_LEVEL`
is a guess. We can do better: mine the user's own past sessions to infer what
they actually tolerate — exactly the philosophy of `headroom learn`, which
already reads `~/.claude/projects/*.jsonl` and turns history into learned
context.

### 4.2 The key insight (validated on real data)

I prototyped the signal extraction and ran it over **24 real sessions** for
this project. The finding that shapes the whole design:

| Signal | This user's data | Carries signal? |
|---|---|---|
| Explicit "be brief" keywords | **1** of 210 human msgs | ❌ almost none |
| Explicit "explain more" keywords | **1** of 210 | ❌ almost none |
| **Interruptions** (user cuts Claude off) | **29** (~1 per 7 turns) | ✅ strong |
| **Fast-skips** (reply <30s after a >250-word answer) | **15 of 100 long outputs** | ✅ strong |
| Long-output frequency | 100 outputs >250 words | ✅ context |

**Users almost never *say* what verbosity they want — they show it
behaviorally.** Keyword matching (the obvious approach) is nearly empty. The
behavioral signals are rich. The strongest is **fast-skip**: a reply arriving
faster than the answer could have been read is a direct measurement of
"generated tokens nobody consumed," computable from timestamps + lengths
already in the JSONL.

For this user — 29 interrupts, 15% unread-long-output rate, zero "explain more"
requests — the data reads as a clear **L2, arguably L3** user.

### 4.3 Design — `headroom learn --verbosity`

Slots into the existing `learn` architecture. `headroom/learn/plugins/claude.py`
already parses the JSONL; `analyzer.py` already does cheap-extraction → digest →
LLM-returns-JSON. We add a verbosity analysis path:

**Step 1 — structural pass (pure Python, no patterns).** Per session, compute:
- **interrupt rate** — `[Request interrupted by user…]` markers per human turn
- **fast-skip rate** — human reply latency vs. preceding assistant output length
- **long-output frequency** — share of assistant texts over N words
- **echo ratio** — n-gram overlap between assistant output and prior context (restated code)
- reply-latency distribution vs. output length

All mechanical, all from data already on disk. (Prototype:
`scripts/verbosity_scan.py`, validated above.)

**Step 2 — LLM judgment pass (mirrors today's `learn`).** Feed a digest of
human messages + the structural stats to the analyzer LLM with one question:
*"Does this user read long explanations, and when they push back, is it for
more detail or less?"* This replaces brittle keyword lists with judgment —
consistent with the no-hardcoded-patterns rule. Returns a level + confidence +
rationale as JSON.

**Step 3 — output (this is what `--verbosity` produces).** See §4.4.

**Step 4 — runtime AIMD auto-tune.** The offline pass sets the *starting* level;
a live loop tracks drift. An interrupt or fast-skip nudges the level up one; an
"explain"/"why" follow-up drops it back. Hysteresis (require 2–3 consistent
signals before moving) prevents oscillation. Like TCP congestion control: probe
toward terser, back off on a "too terse" signal.

### 4.4 What `--verbosity` outputs — and how it helps

The command produces three concrete artifacts:

**(a) A human-readable report (stdout):**
```
Verbosity analysis — /Users/tcms/demo/headroom (24 sessions, 210 turns)

  Interrupts:        29  (1 per 7.2 turns)   ← strong "too much" signal
  Fast-skips:        15 / 100 long outputs   ← 15% of long answers unread
  Explicit brevity:   1   explicit verbose:  1   ← behavioral, not stated

  LLM read:  "User interrupts frequently and rarely reads long
             explanations; pushes back for less, never more detail."

  Recommended verbosity level: 2  (confidence: high)
  Estimated output-token reduction at L2: ~25–35%
```

**(b) A persisted setting** written to `~/.headroom/` (alongside the existing
savings tracker) — per-project verbosity level + confidence:
```json
{"project": "/Users/tcms/demo/headroom",
 "verbosity_level": 2, "confidence": "high",
 "signals": {"interrupt_rate": 0.138, "fast_skip_rate": 0.15},
 "learned_at": "2026-06-12T…"}
```

**(c) The shaper reads it as its default.** `OutputShaperSettings.from_env()`
gains a fallback: if `HEADROOM_VERBOSITY_LEVEL` is unset, load the learned
per-project level instead of the hardcoded `2`. So **the output of `--verbosity`
directly becomes the live verbosity the proxy applies** — no manual tuning.

**How it helps, concretely:**
1. **Removes the guess.** Today you set `HEADROOM_VERBOSITY_LEVEL=2` by hand.
   After `learn --verbosity`, the level is derived from *your* behavior — a
   heavy-interrupter gets L3, a "read everything" user gets L1.
2. **Per-project, not global.** Your exploratory side-project and your
   production repo can carry different levels.
3. **Justified, not magic.** The report shows the signals and the LLM's read,
   so the recommendation is auditable (matches the dashboard philosophy of
   showing directional data, not opaque scores).
4. **Seeds the runtime loop.** The learned level is the AIMD starting point;
   live signals refine it without re-running the offline pass.

### 4.5 Mapping signals → level (initial heuristic, LLM-refined)

| Interrupt rate | Fast-skip rate | "explain more" present | → Level |
|---|---|---|---|
| low | low | yes | 1 |
| low–med | low–med | no | 2 |
| high | high | no | 3 |
| very high | very high | no | 4 (offer, don't auto-apply) |

The LLM judgment pass adjusts this — the table is the prior, not the verdict.

---

## 5. Files

| File | Status | Purpose |
|---|---|---|
| `headroom/proxy/output_shaper.py` | ✅ built | the shaper (steering + effort routing) |
| `headroom/proxy/handlers/anthropic.py` | ✅ wired | invoke shaper after body mutations |
| `tests/test_output_shaper.py` | ✅ 34 passing | unit coverage |
| `scripts/eval_output_shaper.py` | ✅ built | live before/after eval |
| `scripts/verbosity_scan.py` | 🔬 prototype | session-mining signal extraction |
| `headroom/learn/plugins/claude.py` (+ analyzer) | 🔜 extend | `--verbosity` analysis path |
| `~/.headroom/verbosity.json` | 🔜 | persisted per-project learned level |

---

## 6. Roadmap

1. ✅ **Measure** — live eval establishes the realistic ceiling and baseline.
2. ✅ **Effort router** — biggest win per unit risk, fully mechanical, invisible.
3. ✅ **Verbosity ladder at fixed L2** — safe default, cache-safe tail injection.
4. 🔜 **`learn --verbosity`** — derive the per-user starting level from sessions.
5. 🔜 **Runtime AIMD auto-tune** — refine the level live from interrupt/skip signals.
6. 🔭 **Waste taxonomy on the dashboard** — echo ratio, ceremony ratio,
   full-file-rewrite detection, as token counts (no dollar estimates).
7. 🔭 **Budget whispering** — tell the model its token budget per turn-type,
   sized from the historical output distribution in SQLite.

---

## 7. Counterfactual measurement — how we show a % we can't directly observe

This is the hard part, and it deserves its own section.

### 7.1 Why output savings are not directly measurable

Input compression is a **pure function**: Headroom takes a request, shrinks it,
and can count `tokens_before` and `tokens_after` — both are observable on the
same request. Output is different. When the shaper makes a request terser, the
model emits N output tokens. We **never observe** what it *would* have emitted
without the steering. Only one side of the counterfactual ever happens. So a
flat "we saved 30%" is a guess dressed as a fact.

The design rule that follows: **never report a single number as if it were
measured.** Separate what is genuinely measured from what is estimated, label
each, and always attach uncertainty.

### 7.2 Three tiers of honesty

**Tier 1 — Estimated (synthetic control).** Build a per-stratum baseline of
*unshaped* output tokens from session history that predates the shaper
(`learn --verbosity` does this in the same pass that picks the level). For each
shaped request, the expected unshaped output is the baseline mean for that
request's stratum. Aggregate estimate:

```
tokens_saved = Σ over shaped requests ( baseline_mean[stratum] − observed_output )
```

Summed as **signed** deltas — never clamped per-request. Clamping each delta at
zero would throw away the cases where a shaped response happened to be *longer*
by chance, biasing the total upward. Over many requests the noise averages out;
the systematic effect remains. Reported with a propagated 95% CI (see §7.5) and
always labelled "estimated."

**Tier 2 — Measured (A/B holdout).** Set `HEADROOM_OUTPUT_HOLDOUT=0.1` and 10%
of conversations are deliberately left **unshaped** as a control arm. Within
each stratum, `mean(control) − mean(treatment)` is an **unbiased causal
estimate** of the per-request saving. This is the only number we call
"measured." It self-corrects: if steering doesn't actually help on some
workload, the holdout reveals it. The cost is real (you forgo savings on 10% of
traffic), so it's opt-in; default holdout is 0 (estimate-only).

**Tier 3 — Direct waste (no counterfactual at all).** Echo ratio — the n-gram
overlap between a response and the context it was given — is a property of a
single response. "32% of this output restated context already on screen" needs
no counterfactual; it's a measured fact about output we *did* see, and it's
exactly what the shaper targets. Surfaced in `learn --verbosity` as
`mean_echo_ratio`.

### 7.3 Stratification — comparing like with like

You can't compare a "fix this typo" response to "design a caching layer." The
estimator buckets every request by features observable **before** the response:

```
stratum = model_family | turn_kind | input_token_bucket | has_tools
        = e.g. "opus | mechanical_continuation | xl | tools"
```

Coarse on purpose (~25–50 strata) so per-stratum baselines stay dense. The live
proxy computes the stratum the exact same way the offline baseline does, so
treatment requests line up with their baseline. Unseen strata fall back
hierarchically (drop `has_tools`, then the bucket, …, then the global mean).

### 7.4 The two constraints that happen to align

Holdout assignment is **conversation-stable** — a whole conversation is either
treatment or control, decided by hashing a conversation-stable key (model +
first user message). This matters for two independent reasons that point the
same way:

1. **Measurement validity** — mixing shaped and unshaped turns within one
   conversation would contaminate the comparison (the history itself differs).
2. **Cache safety** — flipping a conversation's verbosity mid-stream changes the
   system-prompt tail, which busts the provider prefix cache.

So the same rule (assign per conversation, never per turn) is forced by both
the statistics and the caching. Nice when constraints agree.

### 7.5 The confidence interval

For the estimated tier, uncertainty comes from two sources, both propagated:

```
Var(tokens_saved) ≈ Σ_s [ n_s · σ²_observed,s  +  n_s² · σ²_baseline,s / m_s ]
                         └─ spread of shaped outputs ─┘   └─ finite-baseline error ─┘
```

where `n_s` is treatment count in stratum `s` and `m_s` the baseline sample
count. For the measured tier it's the standard difference-of-means variance
`σ²_c/n_c + σ²_t/n_t` per stratum. The 95% band is `point ± 1.96·√Var`, surfaced
everywhere the number is (CLI and dashboard), so the reader sees the precision,
not just a point estimate.

Output-token counts are right-skewed (a few huge responses). Means are still the
right statistic for *totals* (you're billed on the sum), but the CI widens
honestly when a stratum is dominated by a few large responses — which is the
correct signal that the estimate is soft there.

### 7.6 How it flows end to end

```
learn --verbosity --apply
   ├─ writes verbosity.json         (the level the shaper applies)
   └─ seeds output_savings.json     (the per-stratum baseline = synthetic control)

proxy request (HEADROOM_OUTPUT_SHAPER=1)
   ├─ assign_arm(conversation)      → treatment | control (holdout)
   ├─ stratum_key(request features)
   ├─ treatment: shape body; control: leave unshaped
   └─ tag (arm, stratum) onto transforms_applied   ← rides existing plumbing

response completes → emit_request_outcome (one funnel, all paths)
   └─ recorder.record(arm, stratum, output_tokens)  → output_savings.json

headroom output-savings   /   dashboard "Output Tokens Saved" card
   └─ best_estimate(): measured if a holdout exists, else estimated; with CI
```

The recording rides the existing `transforms_applied` label channel, so it
works for streaming, non-streaming, and backend paths with no change to
`RequestOutcome` or its construction sites.

### 7.7 What the user sees

- **CLI:** `headroom output-savings` →
  `Reduction: 31.7%  (95% CI 27.7% … 35.7%)  [MEASURED, 400 shaped requests]`
- **Dashboard:** an "Output Tokens Saved" hero card next to input compression —
  token count, percent, a `measured`/`estimated` badge, and the CI band.
- **No dollar estimates** on the output card (per project convention) — token
  counts and directional percentages only.

### 7.8 Honest limitations

- Estimated-tier accuracy depends on the baseline matching current workload; if
  your tasks drift, re-run `learn --verbosity` or turn on a small holdout.
- The baseline must come from *unshaped* history. If you learn from sessions
  where the shaper was already active, the baseline is contaminated — the live
  holdout is the clean path forward.
- Runtime AIMD upward-ratcheting is gated off by default: we can reliably detect
  "too much output" (fast-skip timing, stream cancellation) but not yet "too
  little" at runtime without content heuristics, so auto-escalation stays
  behind `HEADROOM_VERBOSITY_AUTOTUNE` until both directions are trustworthy.
