# DeepSeek Harness (dsh)

Headroom wraps [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(`dsh`) so its DeepSeek chat-completions traffic is compressed by the local proxy
before being forwarded upstream to DeepSeek.

## Wrap

```bash
# Start the proxy and launch dsh (web profile)
headroom wrap dsh

# One-shot task with the headless profile
headroom wrap dsh --profile headless "your task"

# Explicit launcher override (e.g. pnpm)
headroom wrap dsh --command "pnpm dsh"
```

`headroom wrap dsh` starts the local proxy and sets `DEEPSEEK_BASE_URL` to
`http://127.0.0.1:{port}/v1`. dsh appends `/chat/completions` directly to its
`baseURL` (no `/v1`), so pointing it at the proxy's `/v1` prefix lands on the
proxy's existing OpenAI-compatible `/v1/chat/completions` route.

### Profiles

- `web` (default) — interactive web session.
- `headless` — `dsh --profile headless <task>`, for one-shot tasks.

### Command resolution

The launcher resolves in order: an explicit `--command`, then `dsh` on `PATH`,
then `pnpm dsh`. If none resolve it fails with an install hint
(`npm i -g @deepseek-ai/dsh`).

## Unwrap

`headroom wrap dsh` only sets the launch environment — it makes no durable
config changes. `headroom unwrap dsh` therefore just stops the local proxy:

```bash
headroom unwrap dsh              # stop the proxy
headroom unwrap dsh --no-stop-proxy   # leave the proxy running
```

## Authentication

`headroom wrap dsh` passes `DEEPSEEK_API_KEY` through to dsh, and the proxy
forwards the `Authorization` header verbatim — no extra login is required.

## DeepSeek upstream (`--deepseek-api-url`)

The proxy routes DeepSeek traffic upstream to `https://api.deepseek.com` by
default. Override it with `--deepseek-api-url` (or the `DEEPSEEK_TARGET_API_URL`
env var):

```bash
headroom wrap dsh --deepseek-api-url https://your-gateway.example.com
```

## baseURL precedence caveat

dsh resolves its effective `baseURL` from `config.baseURL` when set, then
`$DEEPSEEK_BASE_URL`, then DeepSeek's public API (`config` is the merged settings
/ `cordis.yml` section). A `baseURL` configured in dsh settings or `cordis.yml`
therefore **overrides** `$DEEPSEEK_BASE_URL` and would silently bypass the proxy.
If dsh is reaching DeepSeek directly, check your dsh settings / `cordis.yml` for
a `baseURL` entry.
