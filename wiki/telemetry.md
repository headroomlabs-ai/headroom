# Telemetry Surfaces

Headroom exposes several observability surfaces. This page is the operator-facing
inventory for what each surface emits, whether it leaves the host, and how to
disable or export it.

For the runtime view of your current environment, run:

```bash
headroom telemetry list
headroom telemetry list --json
```

## Surface Inventory

| Surface | Default | Leaves host by default | Prompt content | How to observe | Retention |
|---|---:|---:|---:|---|---|
| Anonymous beacon | On | Yes | No | Proxy startup log | Headroom service controlled |
| Prometheus | On | No | No | `GET /metrics` | Scraper controlled |
| OTEL metrics | Off | Only when enabled | No | OTEL collector or console exporter | Exporter controlled |
| Langfuse tracing | Off | Only when enabled | No | Langfuse dashboard | Langfuse controlled |
| Savings tracker | On | No | No | `${HEADROOM_WORKSPACE_DIR}/proxy_savings.json` | 365 days / 5000 points by default |
| Dashboard | On | No | No | `GET /dashboard`, `GET /stats` | Memory + savings tracker |
| TOIN | On | No for filesystem backend | No | `${HEADROOM_WORKSPACE_DIR}/toin.json` | Backend controlled |
| Proxy log | On | No | No by default | `${HEADROOM_WORKSPACE_DIR}/logs/proxy.log` | Until log rotation or cleanup |
| Active sessions | On | No unless clustered mode is enabled | No | `${HEADROOM_WORKSPACE_DIR}/sessions` | Stale manifests ignored after runtime TTL |

## Controls

Anonymous aggregate telemetry is controlled by `HEADROOM_TELEMETRY`.

```bash
HEADROOM_TELEMETRY=off headroom proxy
headroom proxy --no-telemetry
```

OTEL metrics are independent from anonymous telemetry:

```bash
HEADROOM_OTEL_METRICS_ENABLED=true
HEADROOM_OTEL_METRICS_EXPORTER=otlp_http
HEADROOM_OTEL_METRICS_ENDPOINT=http://127.0.0.1:4318/v1/metrics
HEADROOM_OTEL_SERVICE_NAME=headroom-proxy
```

Langfuse trace export is opt-in:

```bash
HEADROOM_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

TOIN persistence can be disabled for local-only in-memory operation:

```bash
HEADROOM_TOIN_BACKEND=none
```

## Clustered Sessions

Headroom writes a per-process active-session manifest under
`${HEADROOM_WORKSPACE_DIR}/sessions/<session-id>/session.json`. These manifests
contain session identifiers, heartbeat timestamps, agent type, PID, and aggregate
counters only. They do not include prompts, responses, tool output, or request
payloads.

Enable filesystem-backed cluster mirroring with:

```bash
HEADROOM_CLUSTER_ENABLED=true
HEADROOM_CLUSTER_ID=team-gamma
HEADROOM_CLUSTER_DIR=/mnt/shared/headroom-cluster
headroom proxy
```

When enabled, Headroom mirrors the local manifest to:

```text
${HEADROOM_CLUSTER_DIR}/${HEADROOM_CLUSTER_ID}/sessions/<instance-id>/<session-id>/session.json
```

The `/stats` response includes:

```json
{
  "active_sessions": {
    "current": {},
    "local": [],
    "local_summary": {}
  },
  "cluster": {
    "enabled": true,
    "cluster_id": "team-gamma",
    "active_sessions": [],
    "summary": {}
  }
}
```

Use `headroom telemetry list --json` to verify whether active-session mirroring
is local-only or cluster-enabled in the current environment.
