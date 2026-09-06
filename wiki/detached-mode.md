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
