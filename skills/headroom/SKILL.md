---
name: headroom
description: Use when working with the local Headroom Codex integration, proxy health, MCP compression/retrieve tools, persistent runtime profile, or Headroom configuration.
---

# Headroom For Codex

Headroom should be installed so the `headroom` command is available on `PATH` and configured for Codex through the persistent `init-user` profile on port `8787`.

Use Headroom alongside Cognitive Context Manager. CCM remains the default project-memory and working-context system; Headroom handles local proxy routing, compression, CCR retrieve tools, and proxy/session stats.

Useful checks:

```bash
headroom install status --profile init-user
curl -fsS http://127.0.0.1:8787/readyz
curl -fsS http://127.0.0.1:8787/health
```

If the proxy is stopped, start it with:

```bash
cd /tmp && headroom install agent ensure --profile init-user
```

Avoid running the Headroom CLI from a source checkout when starting the proxy unless the checkout has a compiled `headroom._core`; otherwise Python may import the checkout instead of the installed wheel. A neutral cwd such as `/tmp` uses the installed package with the Rust extension.
