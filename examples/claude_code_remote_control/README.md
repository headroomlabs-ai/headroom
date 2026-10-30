# Claude Code Remote Control + Headroom (HTTPS proxy route)

Claude Code v2.1.196+ disables Remote Control (`/remote-control`, `/rc`: a local
session mirrored to claude.ai/code and the Claude mobile apps) as soon as
`ANTHROPIC_BASE_URL` names a host other than `api.anthropic.com`. The check is
client-side and compares the host only ([issue #1779](https://github.com/headroomlabs-ai/headroom/issues/1779)),
so the usual Headroom setup (`ANTHROPIC_BASE_URL=http://127.0.0.1:8787`) always
loses Remote Control.

This recipe keeps both. `ANTHROPIC_BASE_URL` stays unset, so the gate passes, and
Claude Code reaches Headroom through `HTTPS_PROXY` instead — the
[enterprise proxy path](https://code.claude.com/docs/en/network-config) Claude Code
supports, with a local CA trusted through `NODE_EXTRA_CA_CERTS`:

```text
Claude Code --HTTPS_PROXY--> mitmdump 127.0.0.1:8790
    api.anthropic.com  /v1/messages*  -> Headroom 127.0.0.1:8787 -> api.anthropic.com
    api.anthropic.com  other paths    -> api.anthropic.com (unchanged)
    any other host (claude.ai, OAuth) -> raw CONNECT tunnel, never decrypted
```

Only `api.anthropic.com` is decrypted (`--allow-hosts`). The Remote Control
bridge talks to `claude.ai`, which stays an untouched tunnel, and subscription
auth is unchanged: Claude Code still believes it talks to the first-party API.

## Setup (macOS / Linux)

1. Install [mitmproxy](https://mitmproxy.org/) and generate its CA once:

   ```bash
   brew install mitmproxy            # or: pipx install mitmproxy
   mitmdump --listen-host 127.0.0.1 --listen-port 8790 -q & sleep 4; kill %1
   ls ~/.mitmproxy/mitmproxy-ca-cert.pem
   ```

2. Run the proxy with this addon (keep it running, e.g. as a LaunchAgent or a
   systemd user unit with restart-on-failure):

   ```bash
   mitmdump --listen-host 127.0.0.1 --listen-port 8790 \
     --allow-hosts '^api\.anthropic\.com(:443)?$' \
     -s examples/claude_code_remote_control/route_to_headroom.py
   ```

   Set `HEADROOM_RC_UPSTREAM=host:port` if Headroom does not listen on
   `127.0.0.1:8787`.

3. Point Claude Code at it in `~/.claude/settings.json` (user settings: Claude
   Code ignores a `true` `remoteControlAtStartup` in project settings):

   ```json
   {
     "env": {
       "HTTPS_PROXY": "http://127.0.0.1:8790",
       "NO_PROXY": "localhost,127.0.0.1,::1",
       "NODE_EXTRA_CA_CERTS": "/Users/you/.mitmproxy/mitmproxy-ca-cert.pem"
     },
     "remoteControlAtStartup": true
   }
   ```

   and **remove `ANTHROPIC_BASE_URL`** from that `env` block and from your shell.
   `remoteControlAtStartup` is optional: without it, run `/remote-control` when needed.

4. Start Headroom **without** the proxy variables. If `headroom init claude`
   installed hooks that start the proxy (`headroom init hook ensure`), prefix their
   commands so the Headroom process does not inherit them:

   ```text
   env -u HTTPS_PROXY -u HTTP_PROXY -u NODE_EXTRA_CA_CERTS /path/to/headroom init hook ensure ...
   ```

   Headroom's upstream client honours `HTTPS_PROXY`. Inheriting it would send its
   own calls back through mitmproxy; the addon's hop marker catches that and sends
   them to Anthropic, but it is a second line of defense, not a reason to skip this.

## Verify

1. New `claude` session: `/status` shows `Proxy: http://127.0.0.1:8790`.
2. Ask anything: the answer streams in, and the mitmdump output shows one
   `127.0.0.1:8787` request per turn (no repeated lines).
3. `/remote-control` exists; the session appears in the Claude mobile app (Code
   tab) and answers messages sent from the phone.
4. `headroom savings` (or the dashboard) counts the requests.

## Caveats

- Every process launched by Claude Code (Bash tool, MCP servers) inherits
  `HTTPS_PROXY`. Non-Anthropic hosts are tunnelled untouched, but if mitmdump is
  down, Claude Code loses network access: keep it supervised.
- `headroom init claude` or a Headroom update may write `ANTHROPIC_BASE_URL` back;
  Remote Control then disappears again.
- On a successful first pass the `x-rc-proxy-hop` marker reaches Anthropic through
  Headroom (Headroom only strips `x-headroom-*`); it is an inert extra header.

## Rollback

Put `ANTHROPIC_BASE_URL` back, remove the three variables and
`remoteControlAtStartup`, remove the `env -u` hook prefix, stop mitmdump.
`rm -rf ~/.mitmproxy` drops the CA.
