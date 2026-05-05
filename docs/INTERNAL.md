# Headroom Internal — orq Setup

Orq's internal deployment of the Headroom GPU fork. All coding agent traffic routes through this proxy.

## Server

The proxy runs as a Docker container on the GPU host.

| | |
|---|---|
| Host | `100.77.242.54` (Tailscale) |
| Proxy port | `8788` → container `8787` |
| Container name | `headroom-internal-test` |
| Data volume | `headroom-internal-data` |
| Image | `headroom-internal:gpu` |

The container starts with `--memory`, `--code-graph`, and `--log-messages` enabled.

---

## Using the proxy locally

**Prereq: Tailscale must be connected.** The GPU host is only reachable via Tailscale (`100.77.242.54`). If you're on the same local network you can also use `10.213.31.36` directly (skip the VPN hop), but `aurl gpu` handles both via Tailscale by default.

### 1. Add `aurl` to `~/.zshrc`

```bash
# Toggle Headroom proxy: aurl [gpu|local|off]
#   gpu   -> GPU host via Tailscale (recommended)
#   local -> GPU host via local network IP (same network only)
#   off   -> unset, go direct to Anthropic/OpenAI
aurl() {
    case "$1" in
        gpu|"")
            export ANTHROPIC_BASE_URL=http://100.77.242.54:8788
            export OPENAI_BASE_URL=http://100.77.242.54:8788/v1
            export HEADROOM_PROXY_URL=http://100.77.242.54:8788
            ;;
        local)
            export ANTHROPIC_BASE_URL=http://10.213.31.36:8788
            export OPENAI_BASE_URL=http://10.213.31.36:8788/v1
            export HEADROOM_PROXY_URL=http://10.213.31.36:8788
            ;;
        off)
            unset ANTHROPIC_BASE_URL
            unset OPENAI_BASE_URL
            unset HEADROOM_PROXY_URL
            ;;
        *)
            echo "usage: aurl [gpu|local|off]"
            return 1
            ;;
    esac
    echo "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-<unset>}"
    echo "OPENAI_BASE_URL=${OPENAI_BASE_URL:-<unset>}"
    echo "HEADROOM_PROXY_URL=${HEADROOM_PROXY_URL:-<unset>}"
}
aurl gpu  # default to GPU host on shell startup
```

Then `source ~/.zshrc`. Shell defaults to `aurl gpu` on every new session.

### 2. Install the MCP server

```bash
aurl gpu
pip install "headroom-ai[mcp]"
headroom mcp install
```

`headroom mcp install` reads `HEADROOM_PROXY_URL` from the environment set by `aurl` — no hardcoded URLs. To switch targets later, run `aurl <target>` and reinstall.

Verify:

```bash
headroom mcp status
```

Expected:

```
MCP SDK:        ✓ Installed
Claude Config:  ✓ Configured
Proxy URL:      http://100.77.242.54:8788
Proxy Status:   ✓ Running at http://100.77.242.54:8788
```

### 3. Launch Claude through the proxy

```bash
aurl gpu && claude
```

Subagents spawned inside Claude Code inherit `ANTHROPIC_BASE_URL` automatically — no extra config needed.

---

## Deploy

### Prerequisites

On a fresh GPU host, install the NVIDIA container toolkit first (one-time):

```bash
scp install-nvidia-toolkit.sh bauke@100.77.242.54:~
ssh bauke@100.77.242.54 'bash ~/install-nvidia-toolkit.sh'
```

### Build the image

On the GPU host:

```bash
git clone https://github.com/orq-ai/headroom-internal.git
cd headroom-internal
DOCKER_BUILDKIT=1 docker build -f Dockerfile.gpu -t headroom-internal:gpu .
```

Build takes ~10–20 minutes (Rust compilation + HuggingFace weight download). The image bakes in the Kompress-base model weights so it never reaches out at runtime.

### Start the container

```bash
bash run-headroom-internal.sh
```

The script stops any existing container with the same name, starts a fresh one, and waits for `/readyz`. Environment overrides:

```bash
PORT=8789 bash run-headroom-internal.sh          # different port
GPUS=none bash run-headroom-internal.sh          # CPU-only
LOG_LEVEL=DEBUG bash run-headroom-internal.sh    # verbose logging
IMAGE=headroom-internal:gpu-v2 bash run-headroom-internal.sh
```

Container restarts automatically (`--restart=unless-stopped`) on host reboot.

---

## Stats and health

All endpoints are on the proxy host at port `8788`.

| Endpoint | What it shows |
|---|---|
| `GET /readyz` | Health check — returns 200 when ready |
| `GET /stats` | Overall proxy stats (requests, tokens saved, compression ratio) |
| `GET /v1/retrieve/stats` | CCR retrieval cache stats |
| `GET /v1/toin/stats` | Token input/output stats |

Quick check:

```bash
curl http://100.77.242.54:8788/readyz
curl http://100.77.242.54:8788/stats
```

Community dashboard (aggregated public stats): https://headroomlabs.ai/dashboard

---

## What's enabled in this deployment

| Feature | Flag | Effect |
|---|---|---|
| CCR | always on | Compresses tool outputs; agents retrieve originals via `headroom_retrieve` |
| Memory | `--memory` | Cross-agent memory store (shared between Claude Code sessions) |
| Code graph | `--code-graph` | Codebase intelligence index for better code compression |
| Message logging | `--log-messages` | Logs full request/response for debugging |
| Kompress-base | baked in image | GPU-accelerated ML text compression |
| SmartCrusher | baked in image | Rust extension for JSON compression |
