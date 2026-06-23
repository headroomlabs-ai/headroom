# agent-evals

End-to-end accuracy A/B framework for **Headroom**: run trusted coding-agent benchmarks
**WITH vs WITHOUT** Headroom's context-compression proxy and produce a statistically
defensible verdict — *does compression preserve what the agent can solve, and how much
does it save?*

This is a self-contained nested project inside the `headroom` repo. It consumes Headroom
only as the system-under-test (via `base_url`); it is **not** part of the `headroom-ai`
wheel and is **not** wired into headroom's `make ci-precheck`.

## The clean A/B (why a proxy helps)

Headroom sits in the request path, so the only variable between arms is `base_url`:

| Arm | base_url | Headroom mode | Isolates |
|-----|----------|---------------|----------|
| `A0_DIRECT` | provider API | none | native agent score |
| `A1_PASSTHROUGH` | `localhost:N` | `--no-optimize` | proxy-hop cost only |
| `B_HEADROOM` | `localhost:M` | `--mode token` | compression cost (vs A1) |

Headline accuracy claim = **B vs A1**. `A1 vs A0` is the transparency sanity check (≈0).

## Method (in one line)

Agentic evals are noisy (single-run pass@1 swings several points even at temp 0), so we run
**paired, multi-run, non-inferiority** experiments: same tasks through every arm, K runs each,
and a TOST equivalence test on accuracy (win = savings up, accuracy within margin δ).

## Phases

- **Phase 0** (this PR) — foundation: 3-arm abstraction, resumable orchestrator, per-task
  savings capture, proxy-transparency check. Runnable with no benchmark deps and ~no spend.
- **Phase 1** — Aider Polyglot end-to-end (first real accuracy + savings scorecard).
- **Phase 2** — SWE-bench Verified via OpenHands (the headline), per-transform ablation.

See the design spec for the full architecture.

## Develop

```bash
cd agent-evals
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev,stats]"
make gate           # ruff + mypy + pytest  (the agent-evals push gate)
agent-evals show-config
```

Live tests (spawn a real proxy / hit upstreams) are `-m live`/`-m real_llm` and are skipped
unless provider keys are present in your environment. Keys come from your shell/`.env`; they
are never read from or written to source.
