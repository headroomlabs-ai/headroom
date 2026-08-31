# Wrap Target Registry

The wrap target registry (`headroom/providers/wrap_registry.py`) describes env-var wrap
targets as data: the binary to launch, which environment variables point at the local
proxy, upstream defaults, and proxy routing declarations. `cli/wrap.py` generates one
`headroom wrap <name>` command per entry. Tools that need imperative setup (settings
files, token exchange, MCP registrars — claude, codex, opencode, copilot) keep bespoke
commands; config entries cannot shadow them.

This page records the design decisions and the failure modes that produced them.

## Registry seams (v1)

Each of these exists because of an observed production failure, not speculation:

- **URL style is a contract.** `openai_v1` ends in `/v1` (the client appends
  `/chat/completions`); `anthropic` is a bare origin (the client appends `/v1/messages`);
  `bare_origin` is for tools that append their own full path prefix — IBM Bob appends
  `/inference/v1/chat/completions`, so handing it a `/v1` base doubles the prefix.
- **Upstream URL round-trip.** Bob's `openai_api_url` must carry the `/inference/v1`
  suffix because the proxy's `_normalize_api_url` strips `/v1` and `handle_openai_chat`
  re-appends `/v1/chat/completions`; the two compose back into the path IBM serves.
- **Origin passthrough.** Tools that build full gateway paths themselves
  (`/inference/v1/model/info`, `/admin/v1/profile`) declare
  `origin_passthrough_prefixes`; the catch-all forwards those inbound paths verbatim to
  the upstream origin. Joining them onto the base URL's path produces doubled or
  misrooted URLs that 403 at the gateway edge (#3360).
- **Response-key stripping.** `origin_passthrough_strip_json_keys` removes declared keys
  from passthrough responses. Bob 2.0.1 rewrites its gateway host from `region_domain`
  in the `/admin/v1/profile` response while keeping the proxied port; with the key
  stripped it falls back to its configured gateway URL (the proxy).
- **`default_mode`.** A target may prefer a proxy mode (Bob prefers `token`: it bills
  flat per token, so compression converts 1:1 into dollars). The generated command
  exports it as `HEADROOM_MODE` only when the user has not set one.
- **`default_args`.** CLI arguments prepended to every launch of the tool, before any
  per-invocation args (`headroom wrap bob -- ...`) — prepended so invocation args win
  under the usual last-flag-wins rule. Configurable per target in `wrap_targets.json`,
  e.g. `{"bob": {"default_args": ["--auto-approve"]}}`.

## User configuration: `wrap_targets.json` (v2)

`~/.headroom/config/wrap_targets.json` overlays the code defaults per field. Headroom
never creates this file (config-dir contract: read-mostly, user-owned). Example:

```json
{
  "version": 1,
  "targets": {
    "bob": { "default_mode": "cache" },
    "mytool": {
      "binaries": ["mytool"],
      "install_hint": "pip install mytool",
      "env_vars": [{ "key": "OPENAI_BASE_URL", "style": "openai_v1" }]
    }
  }
}
```

Design rules, in the order they were decided:

- **Precedence is unchanged product-wide:** explicit CLI flag > environment variable >
  this file > code default. The file only replaces code defaults, so every existing
  env-wins check downstream (e.g. `HEADROOM_MODE` over `default_mode`) is untouched by
  construction. The alternative (file-over-env) was evaluated and rejected: the
  inversion would matter for almost no fields, would split Headroom into two precedence
  regimes, and would break export-driven workflows like
  `HEADROOM_MODE=cache headroom wrap bob` A/B runs.
- **`"version": 1` is required.** A wrong or missing version rejects the whole file with
  a loud warning instead of silently misapplying it under a future format change.
- **Per-target atomic resolution.** Any invalid field skips that target's entire
  overlay; the built-in stays in force. A half-applied target (one override live,
  another silently reverted) is the worst failure mode — it reproduces exactly the class
  of mid-session routing breakage the strip rules exist to prevent.
- **New targets are data-only.** Behavior-crossing fields (`extra_chat_routes`,
  `origin_passthrough_prefixes`, `origin_passthrough_strip_json_keys`) are accepted only
  as overrides of built-in targets. File-defined routes and rewrites affect all proxy
  traffic; per-target scoping is future work.
- **Fail-open.** A corrupt or invalid file degrades to exactly the built-in registry,
  with warnings surfaced by the validation front doors below.
- **Performance.** The load is stat-guarded (no file: one `os.stat`, and the resolved
  registry is the built-in dict by identity); resolution is cached per process; origin
  passthrough consults a precomputed host→rules index instead of scanning all targets
  per request.

Every field is described by a descriptor carrying an *effect class* — `data`,
`launch_env`, `upstream`, `mode`, `proxy_route`, `proxy_rewrite` — so validation output
says what changing the field actually does. A test pins descriptor parity with the
dataclass (names and round-trip types).

## Validation and staleness

- `headroom wrap targets` lists every effective target with its source (built-in,
  overridden with fields, config-defined, or skipped with errors) and exits non-zero on
  problems — a pre-flight check after editing the file.
- `headroom doctor` includes the same validation as a pass/warn check.
- The proxy's `/health` config block reports `mode` (boot-time proxy mode) and
  `wrap_targets_config_hash` (fingerprint of the overlay as loaded). On proxy reuse,
  `headroom wrap` warns when the on-disk file differs from what the proxy loaded, or
  when the session's requested mode differs from the proxy's — warning-only, because
  other clients may be attached to a shared proxy.

## Running multiple harnesses

A shared proxy supports concurrent wrapped clients (per-PID markers in
`~/.headroom/clients/<port>/`; the proxy is only stopped when the last client exits),
and savings attribution stays separate via `agent_type` and the `/p/<project>` prefix.
Two settings are proxy-wide and fixed at boot, which constrains layouts:

- **Mode.** One proxy runs one mode. A token-mode target joining a cache-mode proxy is
  reused as-is (with a warning, see above).
- **OpenAI-family upstream.** One `openai_api_url` per proxy. A target with a custom
  gateway (Bob → IBM) joining a proxy started without it triggers a proxy restart and
  repoints the OpenAI-family upstream for every attached client.

Rule of thumb: harnesses with the same mode and standard upstreams share a port;
a target with a custom upstream or different mode gets its own `--port` (and thus its
own proxy).
