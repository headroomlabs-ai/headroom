# Pi / OMP local fleet runbook

Local-install only. Diagnose one machine, then pin or remove `headroom-pi`.

## Symptoms

- `/headroom status` shows a last error after later savings.
- `headroom doctor` fails `metrics` or a native extension pin.
- Packed-host or live compression looks like a rejection for a no-op.

## Diagnose

```bash
/headroom status
/headroom stats
headroom doctor --json
curl -sS -D - http://127.0.0.1:8787/metrics | head
```

Expect:

- footer `Headroom saved N tokens this session` after accepted savings, or `Headroom online` before any
- `/reload` and `/resume` keep that session total; `/new` resets it
- last error `none` after a later accepted compression
- no-ops counted as skipped, not rejected
- `/metrics` 200, or 500 with `# scrape_error ...` and `headroom_metrics_scrape_errors_total`

## Recover

1. Restart the local proxy if `/metrics` or `/livez` is down.
2. If the pin drifted: reinstall the released CLI and rerun `headroom init -g pi` / `headroom init -g omp`.
3. To leave the host: `headroom init -g remove pi` and/or `headroom init -g remove omp`.

Do not install `latest`. Durable OMP must not edit `models.yml`.
