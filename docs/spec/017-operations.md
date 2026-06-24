# 017. Operations

**Status:** done

## Health Endpoints

### `GET /health`

Basic health check. Returns 200 if process is running.

```bash
curl http://localhost:8787/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### `GET /livez`

Liveness check. Returns 200 if process is alive.

```bash
curl http://localhost:8787/livez
```

---

### `GET /readyz`

Readiness check. Returns 200 if ready to serve traffic.

```bash
curl http://localhost:8787/readyz
```

**Response:**
```json
{
  "ready": true,
  "checks": {
    "database": true,
    "cache": true,
    "provider": true
  }
}
```

---

## Logs

### Log Locations

| Installation | Location |
|-------------|----------|
| Docker | `docker logs headroom` |
| Native | `~/.headroom/logs/` |
| Systemd | `journalctl -u headroom` |

### Log Levels

Set via CLI flag or `RUST_LOG` env var for the Rust proxy:
```bash
# Python proxy
headroom proxy --log-level debug

# Rust proxy
RUST_LOG=debug headroom-proxy --upstream http://...
```

---

## Metrics

### Prometheus

**Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'headroom'
    static_configs:
      - targets: ['localhost:8787']
    metrics_path: '/metrics'
```

---

## Upgrade Procedure

### Docker

```bash
docker pull headroom-ai/headroom:latest
docker-compose down
docker-compose up -d
```

### Native

```bash
pip install --upgrade headroom-ai
# Restart headroom service
```

### Embedded

```bash
pip install --upgrade headroom-ai
# Restart application
```

---

## Rollback

### Docker

```bash
docker-compose down
docker tag headroom-ai/headroom:latest headroom-ai/headroom:rollback
# Edit docker-compose.yml to use rollback tag
docker-compose up -d
```

---

## Monitoring

### Key Metrics to Watch

1. **Request rate** — requests per second
2. **Error rate** — 4xx + 5xx / total
3. **Savings rate** — average savings percentage
4. **Latency** — p50, p95, p99
5. **Cache hit rate** — hits / total

**Prometheus queries:**
```promql
# Request rate
rate(headroom_requests_total[5m])

# Error rate
rate(headroom_errors_total[5m]) / rate(headroom_requests_total[5m])

# Average savings
rate(headroom_tokens_original[5m] - headroom_tokens_compressed[5m]) / rate(headroom_tokens_original[5m])

# Cache hit rate
rate(headroom_cache_hits_total[5m]) / (rate(headroom_cache_hits_total[5m]) + rate(headroom_cache_misses_total[5m]))
```

---

## Runbook

| Symptom | Cause | Solution |
|---------|-------|----------|
| "Connection refused" | Proxy not running | Start it with `headroom proxy` |
| "Cache miss on every request" | Cache disabled | Start without `--no-cache` |
| "No savings shown" | Database locked | Check file permissions |
| "Provider timeout" | Network issue | Check firewall/proxy |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial operations document |
