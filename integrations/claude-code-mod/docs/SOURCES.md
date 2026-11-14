# Reviewed primary sources

Reviewed October 5, 2026. Implementation is original; these URLs establish the
contracts and compatibility assumptions. Native runtime validation remains required.

- Official mod tutorial (published October 1, 2026): https://claude.dev/blog/getting-started-with-claude-code-mods/
- Official CLI: https://code.claude.com/docs/en/cli-reference
- Official environment variables: https://code.claude.com/docs/en/env-vars
- Official built-in diff mod: https://github.com/anthropics/claude-code/blob/main/mods/diff/hooks/register.ts
- Official generated API snapshot (header identifies 2.1.277): https://github.com/anthropics/claude-code/blob/main/mods/types/claude-code.d.ts
- Official mod test guidance: https://github.com/anthropics/claude-code/blob/main/mods/README.md
- User-provided mod directory: https://github.com/karanb192/awesome-claude-code-mods

Pinned Headroom review commit:
`a493f559a1a85a5462fc3bf3f363d64c3c9444ce`

- `headroom/proxy/server.py` — existing feed/stats behavior, app.state.proxy, extension/security ordering
- `headroom/proxy/models.py` — RequestLog fields and absence of a populated session_id
- `headroom/proxy/request_logger.py` — bounded metadata/body retention and expensive whole-body copies
- `headroom/proxy/helpers.py` — extract_tags preserves noncredential x-headroom-* tags
- `headroom/proxy/extensions.py` — stable opt-in extension entry point
- `headroom/proxy/loopback_guard.py` — local peer/Host and origin guards
- `headroom/dashboard/templates/dashboard.html` — hosted/local dashboard polling and presentation conventions
- `wiki/metrics.md` — metric accounting and billing caveats
- `.claude-plugin/marketplace.json` — existing `headroom` startup-hooks plugin name

Pinned source base:
https://github.com/headroomlabs-ai/headroom/tree/a493f559a1a85a5462fc3bf3f363d64c3c9444ce
