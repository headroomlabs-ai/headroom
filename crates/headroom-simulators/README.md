# Headroom Simulators

`headroom-simulators` is a local, deterministic upstream for Headroom proxy tests.
It serves provider-shaped responses without contacting real LLM APIs.

```bash
cargo run -p headroom-simulators -- --listen 127.0.0.1:8789
```

Optional JSON config:

```json
{
  "stubs": [
    {
      "name": "exact chat smoke",
      "method": "POST",
      "path": "/v1/chat/completions",
      "body_contains": "ping",
      "response": {
        "status": 200,
        "headers": {"x-simulator": "configured"},
        "json": {"ok": true}
      }
    }
  ]
}
```

Point the proxy at `http://127.0.0.1:8789` to run local or CI validations.
