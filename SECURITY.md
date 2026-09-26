# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.38.x (latest) | :white_check_mark: |
| < 0.38.x | :x:              |

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
3. **Log Files**: Be aware that request logs may contain sensitive information
4. **Budget Limits**: Set budget limits to prevent unexpected costs

### Scope

The following are in scope for security reports:
- Headroom Python package (`pip install headroom-ai`)
- Headroom proxy server
- Official integrations (LangChain, Agno, Strands, LiteLLM, Vercel AI SDK, Anthropic/OpenAI SDK wrappers, MCP)

The following are out of scope:
- Third-party integrations not maintained by us
- Issues in dependencies (report these to the upstream project)
- Social engineering attacks

## Security Features

Headroom includes several security features:

- **Credential handling**: In interactive use (`headroom proxy`, `headroom
  wrap`) provider API keys are read from the process environment and forwarded
  upstream; that path writes no credential to disk. Two paths do persist
  credentials, both deliberately:
  - `headroom copilot login` stores the GitHub Copilot OAuth refresh token in
    `~/.headroom/copilot_auth.json`, created with mode `0600`.
  - A persistent install stores every value passed to `headroom install --env
    KEY=VALUE`, verbatim and in cleartext. Supervisors (launchd, systemd, Task
    Scheduler, cron) start the proxy from a bare environment, so `--env` is the
    only channel by which a provider API key reaches a supervised proxy — and
    if you pass one, it is written to
    `~/.headroom/deploy/<profile>/manifest.json` (mode `0600`) and `export`ed in
    cleartext by the generated runner scripts `run-headroom.sh` /
    `ensure-headroom.sh` (mode `0700`) in the same directory (mode `0700`).
    The installer sets those modes and then reads them back: where the
    platform enforces POSIX permission bits it refuses to leave behind a runner
    script it could not restrict, and warns if the profile directory could not
    be narrowed. On Windows the bits are advisory — access is governed by ACLs —
    so the modes above are requested but not enforced there, and protection
    comes from the account's own profile directory instead. Where they do hold,
    the files are owner-only, not encrypted: anything that can read the account
    — the user themself, root, a home-directory backup or sync — can read the
    key, and rotating it requires re-running `headroom install apply`.
    With the Docker runtime the same values are additionally passed as `docker
    run --env NAME=VALUE`, which places them in the container process's command
    line and so in `ps` output for other users on the host. Prefer a
    supervisor-native secret facility (systemd `LoadCredential=` /
    `EnvironmentFile=` on an owner-only file, a macOS keychain helper) over
    `--env` for long-lived provider keys.

  Request and response bodies may be written to the runtime log when logging is
  enabled, so treat that log as sensitive.
- **Passthrough mode**: Sensitive content passes through unchanged by default
- **Input validation**: All inputs are validated before processing
- **Safe defaults**: Security-conscious defaults out of the box

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
