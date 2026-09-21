# Jev-backed semantic model routing

Optional policy for Headroom's proxy [ModelRouter](../headroom/proxy/model_router.py)
(issue [#1706](https://github.com/headroomlabs-ai/headroom/issues/1706) mechanism;
proposal [#3690](https://github.com/headroomlabs-ai/headroom/issues/3690)).

Uses TypeSafe **Jev** (System One) to classify truncated user text into
`economy` / `standard` / `frontier`, then rewrites the outgoing model when
confidence clears a threshold.

Canonical docs: [TypeSafe Introduction](https://docs.typesafe.ai/introduction).
Patterns: [intent routing](https://docs.typesafe.ai/patterns/intent-routing),
[confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing).

## Why

Static `HEADROOM_MODEL_ROUTES` rules see token counts and tool presence, not
task meaning. Jev adds a calibrated Choice + confidence so trivial turns can
use cheaper models and hard turns keep frontier models — complementary to
local compression (which stays on-machine).

## Privacy

- **Off by default.**
- Only truncated **user text** is sent to TypeSafe for the routing decision
  (not full tool dumps).
- Enabling this feature means routing state leaves your machine for that
  classification call. Compression itself remains local.

## Configuration

```bash
export HEADROOM_JEV_MODEL_ROUTER=1
export TYPESAFE_API_KEY=...   # never commit
export HEADROOM_JEV_TIER_MODELS='{"economy":"gpt-mini","standard":"gpt","frontier":"gpt-pro"}'
# optional:
export HEADROOM_JEV_CONFIDENCE_MIN=0.6
export HEADROOM_JEV_TIMEOUT_S=2
```

Behavior:

1. If Jev is enabled and returns a **matched** decision (including
   low-confidence keep), that decision wins.
2. If Jev is disabled, errors, or has empty user text (**unmatched**), the
   existing heuristic `ModelRouter` runs as before.
3. Fail-open: timeouts / bad keys never take the proxy down.

## Demo

```bash
python examples/jev_model_router/demo.py
# live:
TYPESAFE_API_KEY=... python examples/jev_model_router/demo.py --live
```

## Non-goals

- Calling Jev from ContentRouter / SmartCrusher on tool outputs
- Auto-selecting `AgentSavingsProfile`
- Replacing local compression
