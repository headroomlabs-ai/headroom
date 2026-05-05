*Headroom proxy setup — route your Claude/Codex traffic through our GPU host* :gpu:

Repo: <https://github.com/orq-ai/headroom-internal>

*What it does:* All LLM traffic from coding agents routes through a compression proxy. Tool outputs, logs, RAG chunks get compressed before hitting the model — same answers, fewer tokens.

*Prereq: Tailscale must be connected.* The GPU host is only reachable via Tailscale (`100.77.242.54`). If you're on the same local network you can also use `10.213.31.36` directly (skip the VPN hop), but `aurl gpu` handles both via Tailscale by default.

---

*1. Add `aurl` to your `~/.zshrc`*

```
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

Then `source ~/.zshrc`.

---

*2. Launch Claude through the proxy*

```
aurl gpu && claude
```

Subagents inherit `ANTHROPIC_BASE_URL` automatically. Ping me with questions.
