# Detached Mode

Detached mode makes local-state degradation explicit for container, CI, serverless, read-only, and load-balanced deployments.

## Configuration

```bash
HEADROOM_STATELESS=true headroom proxy
headroom proxy --stateless
headroom proxy --stateless --detached-profile strict
```

`HEADROOM_STATELESS=true` implies detached operation. It disables avoidable filesystem writes and forces affected local-state features into disabled or memory-only behavior.

`HEADROOM_DETACHED_PROFILE` controls startup policy:

| Profile | Behavior |
|---|---|
| `lenient` | Default. Start and log the capability matrix. |
| `strict` | Refuse startup if an explicitly enabled required feature would degrade. |
| `silent` | Start without startup capability logs. |

## Capability Matrix

The proxy exposes the resolved matrix at:

```bash
curl http://127.0.0.1:8787/capabilities
```

The same payload is embedded in `/health` and `/stats`.

Features report:

- `local_state_dependency`: `none`, `optional`, or `required`
- `state`: `full`, `degraded`, or `disabled`
- `degradation_mode`: `full`, `memory-only`, `remote-backed`, or `disabled`
- `reason`: operator-readable explanation

## Prometheus

`/metrics` includes:

```text
headroom_feature_enabled{feature="compression",state="full",degradation="full",dependency="none"} 1
```

Alert when a feature expected to be `full` reports `degraded` or `disabled`.

CCR capability reporting uses the initialized store: the normal SQLite default reports `full`, while explicit memory storage or fallback after a backend initialization failure reports `degraded`. Stateless mode selects memory storage. Successfully loaded third-party adapters report `custom`; their persistence guarantees belong to the adapter. Persistent memory is disabled for every backend under `--stateless`, and strict mode refuses startup when memory was explicitly requested.

The proxy initializes CCR during capability resolution at startup. Configure its backend and TTL environment variables, or install an explicit process-wide store, before calling `create_app()`. Later calls to the singleton getter retain the initialized store and its options.

TOIN reporting also follows the initialized backend: an unavailable adapter cannot report remote persistence, and in-memory observations remain enabled with degraded durability. Configure TOIN before startup. Learning is actually disabled when local state is unavailable in lenient or silent mode; strict mode rejects explicitly requested learning in that situation.
