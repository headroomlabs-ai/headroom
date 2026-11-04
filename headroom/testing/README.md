# Headroom Testing Harness

`headroom.testing` is the fluent scenario contract for local simulations,
deployment planning, and `headroom-bench` handoff.

## Scenario

```python
from headroom.testing import Headroom, ScenarioTask

scenario = (
    Headroom.WithBedrock(region="us-east-1", profile="bench")
    .OnAppleSilicon()
    .WithCompression(mode="cache", kompress=False, savings_profile="coding")
    .WithCCR(enabled=True, inject_tool=False, inject_marker=True)
    .WithMemory(enabled=True, mode="tool", top_k=4)
    .Configure(lambda c: setattr(c, "default_mode", "optimize"))
    .Build()
)

task = ScenarioTask(
    task_id="smoke",
    messages=[{"role": "user", "content": "Summarize this payload."}],
)

report = scenario.orchestrate([task])
assert report.passed
```

## Suite

```python
suite = (
    Headroom.Suite("phase-1")
    .Add(Headroom.WithOpenAI().named("openai-cache").WithCompression(mode="cache"))
    .Add(
        Headroom.WithBedrock(region="us-east-1")
        .named("bedrock-token")
        .WithCompression(mode="token")
    )
)

suite.write_manifest_bundle("headroom-testing-bundle.json", provider="openai", port_start=19000)
suite.write_agent_evals_manifests(
    "agent-evals-manifests",
    benchmark="mini_swebench",
    benchmark_ref="mini@abc123",
    provider="openai",
)
```

## Contract

Every scenario builds real `HeadroomConfig` and `ProxyConfig` objects. The harness
exports the complete dataclass constructor surface, including compatibility
`InitVar` fields, through:

- `scenario.audit_contract()`
- `scenario.deployment_plan()`
- `scenario.bench_manifest_fragment()`
- `scenario.agent_evals_manifest()`
- `suite.manifest_bundle()`
- `suite.agent_evals_manifests()`

Phase-1 validation is local and no-key: `simulate` and `orchestrate` run the SDK
transform pipeline without calling upstream providers. Live proxy deployment is
available through `scenario.deploy_local(...)`; deployment plans carry the full
proxy config in `HEADROOM_PROXY_CONFIG_JSON`.

## headroom-bench

`agent_evals_manifest(...)` emits the same top-level field names as
`agent_evals.models.RunManifest`, including `arms` entries shaped like
`ArmSpec`. The harness keeps this adapter dependency-free: `headroom-ai` can
write bench-native JSON without importing `agent-evals`, while `headroom-bench`
can validate the artifact with its own pydantic model.
