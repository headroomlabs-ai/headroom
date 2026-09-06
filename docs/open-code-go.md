# Using Headroom with OpenCode Go

Compress ALL 22+ open-source coding models through Headroom — $5 for your first
month, then $10/month.

## How it works

```
OpenCode → Headroom Proxy (:8788) → OpenCode Go API → Kimi K3, Grok, GLM...
OpenCode → Headroom Proxy (:8789) → OpenCode Go API → MiniMax M3, Qwen3.7...
              ↑ compresses input
              + shapes output
```

Two proxy instances route to a single Go subscription. OpenAI-format models
(Kimi, Grok, GLM, DeepSeek via Go) go through one proxy; Anthropic-format
models (MiniMax, Qwen) go through the other. All get compressed.

OpenCode Go is a flat-rate subscription — you get 22+ open-source coding
models with generous usage limits. See [OpenCode Go
docs](https://opencode.ai/docs/go) for the full details and official pricing.

## Pricing & limits

OpenCode Go includes the following usage limits:

- 5 hour limit — $12 of usage
- Weekly limit — $30 of usage
- Monthly limit — $60 of usage

Limits are defined in dollar value, so cheaper models allow for more requests.
The table below shows estimated request counts per model (from official Go
docs — check the link above for updates):

| Model | per 5 hour | per week | per month |
|---|---|---|---|
| Grok 4.5 | 120 | 300 | 600 |
| GLM 5.2 / 5.1 | 880 | 2,150 | 4,300 |
| Kimi K3 | 110 | 250 | 490 |
| Kimi K2.7 Code | 1,350 | 3,380 | 6,750 |
| Kimi K2.6 | 1,150 | 2,880 | 5,750 |
| MiMo V2.5 | 30,100 | 75,200 | 150,400 |
| MiMo V2.5 Pro | 3,250 | 8,150 | 16,300 |
| MiniMax M3 | 3,200 | 8,000 | 16,000 |
| MiniMax M2.7 | 3,400 | 8,500 | 17,000 |
| Qwen3.7 Max | 950 | 2,390 | 4,770 |
| Qwen3.7 Plus | 4,300 | 10,800 | 21,600 |
| Qwen3.6 Plus | 3,300 | 8,200 | 16,300 |
| DeepSeek V4 Pro | 3,450 | 8,550 | 17,150 |
| DeepSeek V4 Flash | 31,650 | 79,050 | 158,150 |
| Hy3 | 4,300 | 10,750 | 21,500 |

Per-token pricing (per 1M tokens):

| Model | Input | Output | Cached Read | Usage Bucket |
|---|---|---|---|---|
| Grok 4.5 | $2.00 | $6.00 | $0.30 | $15 |
| GLM 5.2 | $1.40 | $4.40 | $0.26 | $60 |
| Kimi K3 | $3.00 | $15.00 | $0.30 | $15 |
| Kimi K2.7 Code | $0.95 | $4.00 | $0.19 | $60 |
| DeepSeek V4 Pro | $0.435 | $0.87 | $0.003625 | $15 |
| DeepSeek V4 Flash | $0.14 | $0.28 | $0.0028 | $60 |
| MiniMax M3 | $0.30 | $1.20 | $0.06 | $60 |
| MiMo V2.5 | $0.14 | $0.28 | $0.0028 | $60 |
| Qwen3.7 Plus | $0.40 | $1.60 | $0.04 | $60 |
| Hy3 | $0.14 | $0.58 | $0.035 | $60 |

(A representative sample. Full pricing at
[opencode.ai/docs/go](https://opencode.ai/docs/go).)

---

## 1. Subscribe to OpenCode Go

1. Sign in to [OpenCode Zen](https://opencode.ai/zen) and subscribe to Go
2. Copy your API key from the console
3. Store it:
```bash
export OPENCODE_GO_API_KEY="sk-your-go-key-here"
```

---

## 2. Start the proxies

**Proxy A — OpenAI-format models** (Kimi K3, Grok, GLM, Hy3, DeepSeek via Go, etc.)

```bash
HEADROOM_OUTPUT_SHAPER=1 HEADROOM_VERBOSITY_LEVEL=2 \
headroom proxy --port 8788 --openai-api-url https://opencode.ai/zen/go/v1
```

**Proxy B — Anthropic-format models** (MiniMax M3/M2.7, Qwen3.7 Max/Plus, etc.)

```bash
HEADROOM_OUTPUT_SHAPER=1 HEADROOM_VERBOSITY_LEVEL=2 \
headroom proxy --port 8789 --anthropic-api-url https://opencode.ai/zen/go/v1
```

Both proxies route to the same Go API — one Go subscription, one API key, all
models compressed.

Verify both are running:

```bash
curl http://127.0.0.1:8788/health  # → "status": "healthy"
curl http://127.0.0.1:8789/health  # → "status": "healthy"
```

---

## 3. Configure OpenCode

**Note:** If you have an existing `~/.config/opencode/opencode.json`, merge the
provider sections below into it. Keep one config file — mixing `.json` and
`.jsonc` can cause conflicts.

Edit `~/.config/opencode/opencode.json`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "headroom-go/kimi-k3",
  "provider": {
    "headroom-go": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Headroom → Go",
      "options": {
        "baseURL": "http://127.0.0.1:8788/v1",
        "apiKey": "sk-your-go-key"
      },
      "models": {
        "deepseek-v4-pro": { "name": "DeepSeek V4 Pro (Go)" },
        "deepseek-v4-flash": { "name": "DeepSeek V4 Flash (Go)" },
        "grok-4.5": { "name": "Grok 4.5" },
        "kimi-k3": { "name": "Kimi K3" },
        "kimi-k2.7-code": { "name": "Kimi K2.7 Code" },
        "kimi-k2.6": { "name": "Kimi K2.6" },
        "kimi-k2.5": { "name": "Kimi K2.5" },
        "glm-5.2": { "name": "GLM 5.2" },
        "glm-5.1": { "name": "GLM 5.1" },
        "glm-5": { "name": "GLM 5" },
        "mimo-v2.5": { "name": "MiMo V2.5" },
        "mimo-v2.5-pro": { "name": "MiMo V2.5 Pro" },
        "mimo-v2-pro": { "name": "MiMo V2 Pro" },
        "mimo-v2-omni": { "name": "MiMo V2 Omni" },
        "hy3": { "name": "Hy3" },
        "hy3-preview": { "name": "Hy3 Preview" }
      }
    },
    "headroom-go-anthropic": {
      "npm": "@ai-sdk/anthropic",
      "name": "Headroom → Go (AN)",
      "options": {
        "baseURL": "http://127.0.0.1:8789/v1",
        "apiKey": "sk-your-go-key"
      },
      "models": {
        "minimax-m3": { "name": "MiniMax M3" },
        "minimax-m2.7": { "name": "MiniMax M2.7" },
        "minimax-m2.5": { "name": "MiniMax M2.5" },
        "qwen3.7-max": { "name": "Qwen3.7 Max" },
        "qwen3.7-plus": { "name": "Qwen3.7 Plus" },
        "qwen3.6-plus": { "name": "Qwen3.6 Plus" },
        "qwen3.5-plus": { "name": "Qwen3.5 Plus" }
      }
    }
  },
  "mcp": {
    "headroom": {
      "type": "local",
      "command": ["headroom", "mcp", "serve"],
      "enabled": true
    }
  }
}
```

**Important:** Only include model IDs that appear in the proxy's `/v1/models`
response. You can verify with:

```bash
curl -s http://127.0.0.1:8788/v1/models \
  -H "Authorization: Bearer sk-your-go-key" | jq '.data[].id'
```

---

## 4. Which proxy for which model?

| Models | Format | Provider | Port |
|---|---|---|---|
| Kimi K3, K2.7, K2.6, K2.5 | OpenAI | headroom-go | 8788 |
| Grok 4.5 | OpenAI | headroom-go | 8788 |
| GLM 5.2, 5.1, 5 | OpenAI | headroom-go | 8788 |
| DeepSeek V4 Pro, V4 Flash | OpenAI | headroom-go | 8788 |
| MiMo V2.5, V2.5 Pro, V2 Pro, V2 Omni | OpenAI | headroom-go | 8788 |
| Hy3, Hy3 Preview | OpenAI | headroom-go | 8788 |
| MiniMax M3, M2.7, M2.5 | Anthropic | headroom-go-anthropic | 8789 |
| Qwen 3.7 Max, 3.7 Plus, 3.6 Plus, 3.5 Plus | Anthropic | headroom-go-anthropic | 8789 |

---

## 5. Start OpenCode

```bash
opencode
```

Run `/models` — you should see two providers:
- **Headroom → Go** with 16 models
- **Headroom → Go (AN)** with 7 models

Switch with `/model headroom-go/kimi-k3` or `/model headroom-go-anthropic/minimax-m3`.

---

## 6. Verify compression

Each proxy has its own stats:

```bash
curl http://127.0.0.1:8788/stats | python3 -m json.tool | grep -A5 tokens
curl http://127.0.0.1:8789/stats | python3 -m json.tool | grep -A5 tokens
```

Or check the dashboards at [http://127.0.0.1:8788/dashboard](http://127.0.0.1:8788/dashboard)
and [http://127.0.0.1:8789/dashboard](http://127.0.0.1:8789/dashboard).

---

## 7. Adding direct DeepSeek API

If you also want DeepSeek's direct API (separate API key, per-token pricing,
no Go usage caps), add a third proxy and provider. See the companion guide:
**[open-code-deepseek.md](open-code-deepseek.md)**.

The three-proxy full setup gives you 25+ compressed models across DeepSeek
direct + OpenCode Go.

---

## Common issues

### "Authentication Fails"

The `apiKey` is missing or wrong. Make sure your Go API key is set under
`options.apiKey` in your OpenCode config.

### Models don't appear under "Headroom → Go"

1. Verify the proxy is running: `curl http://127.0.0.1:8788/health`
2. Check exposed models: `curl http://127.0.0.1:8788/v1/models -H "Authorization: Bearer sk-your-key"`
3. Make sure config model IDs match the proxy's model list exactly
4. Restart OpenCode after config changes
5. Don't mix `.json` and `.jsonc` config files

### Anthropic models fail with "invalid model"

You're sending an Anthropic-format model through the OpenAI proxy (or vice
versa). Check the [model-to-proxy reference](#4-which-proxy-for-which-model)
above.

### "headroom" command not found

`uv tool install` puts binaries in `~/.local/bin/`. Add it to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Output shaping shows no savings initially

Output savings need a learned baseline. After a few sessions, run:

```bash
headroom learn --verbosity --apply
```

---

## What's NOT in this guide

- **Claude or GPT models** — use [OpenCode Zen](https://opencode.ai/docs/zen) for those
- **`headroom wrap`** — do not use it; it overrides your config
- **Direct DeepSeek API** — see the [DeepSeek guide](open-code-deepseek.md) for that setup
- **Deprecated model names** — `deepseek-chat` and `deepseek-reasoner` are compatibility
  aliases; use `deepseek-v4-pro` and `deepseek-v4-flash` instead
- **Kompress (ML compression)** — requires extra dependencies; SmartCrusher handles
  the majority of use cases
