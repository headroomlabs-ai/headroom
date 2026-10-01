# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest release on PyPI | :white_check_mark: |
| Any earlier release | :x:              |

Headroom releases frequently and only the most recent release is supported. Security
fixes ship in a new release rather than being backported to earlier ones, so upgrading is
the remediation path. Check what you are running with `headroom --version`, and compare it
against [the current release](https://pypi.org/project/headroom-ai/).

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please DO NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email us at: **security@headroomlabs.ai**

Include the following information:
- Type of vulnerability (e.g., injection, data exposure, authentication bypass)
- Full path of the affected source file(s)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact assessment

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: We will assess the vulnerability and determine its severity
3. **Updates**: We will keep you informed of our progress
4. **Resolution**: We aim to resolve critical issues within 7 days
5. **Credit**: With your permission, we will credit you in the security advisory

### Security Best Practices for Users

When using Headroom:

1. **API Keys**: Never commit API keys. Use environment variables.
2. **Proxy Exposure**: Don't expose the proxy server to the public internet without authentication
3. **Log Files**: Headroom always writes an operational log to `~/.headroom/logs/proxy.log`.
   The opt-in `--log-file` request log, and especially `--log-messages`, additionally write
   request and response *content* to disk. Be aware that both may contain sensitive information.
4. **Budget Limits**: Set budget limits to prevent unexpected costs
5. **Workspace directory**: `~/.headroom` holds both credential material and cached
   request content. Treat it as sensitive — especially on shared or multi-user hosts.
   Note that it is not the only location: with memory enabled, extracted memories default
   to a project-local `.headroom/` directory beside the code you ran the agent in. See
   [Security Model](https://headroom-docs.vercel.app/docs/security-model) for exactly what
   is written where.

### Scope

The following are in scope for security reports:
- Headroom Python package (`pip install headroom-ai`)
- Headroom proxy server
- Official integrations (LangChain, Agno, Strands, LiteLLM, Vercel AI SDK, Anthropic/OpenAI SDK wrappers, MCP)

The following are out of scope:
- Third-party integrations not maintained by us
- Issues in dependencies (report these to the upstream project)
- Social engineering attacks

## Security Posture

Headroom is a local proxy that sits between your agents and your LLM providers, so
it necessarily handles both your traffic and your credentials. The
[Security Model](https://headroom-docs.vercel.app/docs/security-model) documents this in
full — what is written to disk, with what permissions and retention, what leaves the
host, and what the local admin surface does and does not allow. The summary:

- **Provider API keys are forwarded, not persisted.** Keys supplied via environment
  variables or request headers are used to authenticate the upstream call and are not
  written to disk or emitted to logs. The one exception is deliberate: the
  `ANTHROPIC_TARGET_API_HEADERS` / `OPENAI_TARGET_API_HEADERS` header maps, if you set
  them through the dashboard settings GUI, are persisted to `~/.headroom/settings.json`
  in plaintext — and a header map is where a gateway key typically goes.
- **Some credentials *are* stored, by design.** `headroom copilot-auth login` persists a
  GitHub Copilot OAuth refresh token to `~/.headroom/copilot_auth.json` (relocatable via
  `$HEADROOM_COPILOT_AUTH_FILE`). It is plaintext JSON, written with `0600` permissions
  on a best-effort basis.
- **Cached request content is written to disk.** CCR stores pre-compression originals —
  tool outputs, file contents, retrieved chunks — in a local SQLite database
  (`~/.headroom/ccr_store.db`, `0600`) so they can be retrieved after compression. Entries
  are plaintext and are not encrypted at rest. Set `HEADROOM_CCR_BACKEND=memory` if your
  deployment cannot accept disk persistence.
- **Anonymous session summaries are uploaded by default.** The `HEADROOM_BEACON` telemetry
  beacon is opt-*out*: it POSTs content-free session counters, a random per-install UUID,
  version, OS, and architecture to Headroom Labs. No prompts, completions, file contents,
  or paths are included. Disable with `HEADROOM_BEACON=off`, `DO_NOT_TRACK=1`, or
  `HEADROOM_OFFLINE=1`. This is separate from `HEADROOM_TELEMETRY`, which is opt-in and
  stays on the machine.
- **Passthrough mode**: Sensitive content passes through unchanged by default.
- **Operational logs are written unconditionally.** `~/.headroom/logs/proxy.log`
  (10 MB × 5 rotations) is always on and also carries the admin audit stream. The
  request log (`--log-file`) and full message logging (`--log-messages`) are opt-in and
  write request/response content to a path you choose.
- **Local control surfaces are loopback-gated.** The `/admin/*`, `/debug/*`,
  `/v1/retrieve*`, `/v1/telemetry*`, `/v1/toin/*`, and `/settings*` routes require a
  loopback client address *and* a loopback `Host:` header, plus a same-origin check on
  the mutating ones. This is a local-trust model: any process on the same host is
  trusted. Setting `HEADROOM_PROXY_TRUSTED_DASHBOARD_CLIENT_CIDRS` deliberately widens
  the settings and stats surface beyond loopback.

Thank you for helping keep Headroom and its users safe!

## Unpatched optional dependency advisories (reviewed 2026-09-10)

The following public upstream advisories remain unresolved. They are included in
`uv.lock` through optional extras; the presence of a package in that universal
lockfile does not mean it is installed with every Headroom installation.

### CrewAI / ChromaDB

The `crewai` extra brings in ChromaDB through CrewAI. The locked ChromaDB 1.1.1
and the latest published version, 1.5.9, are affected by:

- [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c):
  pre-authentication code injection through model repository configuration.
- [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf):
  code injection through model repository configuration with `trust_remote_code`.
- [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr):
  cross-tenant access to collection data.
- [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr):
  missing resource-scope checks in `SimpleRBACAuthorizationProvider`.

There is no published patched release. The upstream authorization fix
[chroma-core/chroma#7602](https://github.com/chroma-core/chroma/pull/7602)
is still open. Upgrading CrewAI alone also retains ChromaDB. Headroom's CrewAI
integration wraps tools; it does not start a ChromaDB server or configure its
authorization. Deployments that separately expose ChromaDB must not rely on its
affected authorization for tenant isolation. Keep it inaccessible to untrusted
clients and do not allow untrusted model repository or `trust_remote_code`
configuration. These exposure restrictions are mitigations, not upstream fixes.

### Voice training / Accelerate

The `voice-train` extra includes Accelerate (locked at 1.12.0).
[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79)
describes path traversal and denial of service through unvalidated `weight_map`
entries in sharded checkpoint indexes. Use only trusted checkpoints, including
their index files and referenced shards, in training environments.

The advisory currently lists versions through 1.14.0, but an upgrade to 1.15.0
is not a verified fix: its
[checkpoint loader](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/modeling.py#L1941-L1944)
still joins index values to the checkpoint directory without containment or
file-type validation. The proposed fixes
[#4070](https://github.com/huggingface/accelerate/pull/4070) and
[#4138](https://github.com/huggingface/accelerate/pull/4138) were closed without
merging; the latter also explicitly leaves the named-pipe denial of service
unfixed. Keep the alert open until a released fix covers both cases.

Dependabot ignores only the reviewed unpatched ranges (ChromaDB through 1.5.9
and Accelerate through 1.15.0). Later releases remain eligible for review. These
update exceptions do not remediate the advisories or dismiss vulnerability alerts.
