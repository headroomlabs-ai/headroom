# Security Policy

## Supported Versions

Headroom is pre-1.0 and follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
while the major version is `0`, any release may contain breaking changes and only the
newest line receives fixes.

| Version | Supported |
| ------- | --------- |
| Latest published minor release (`0.x`) on [PyPI](https://pypi.org/project/headroom-ai/) and [npm](https://www.npmjs.com/package/headroom-ai) | :white_check_mark: |
| Any earlier release | :x: — please upgrade to the latest |

Security fixes ship in a new patch or minor release; they are not backported to older
lines. If you cannot upgrade, contact us at the address below to discuss options.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please
report it privately and give us a chance to fix it before any public disclosure.

**Please DO NOT open a public GitHub issue, discussion, or pull request for a security
vulnerability.**

### How to Report

Use whichever channel you prefer:

1. **GitHub Private Vulnerability Reporting (preferred)** — open a private report at
   <https://github.com/headroomlabs-ai/headroom/security/advisories/new>. This keeps the
   discussion, fix, and advisory in one place.
2. **Email** — <security@headroomlabs.ai>. For sensitive details or exploit code, please
   encrypt with our PGP key (fingerprint and key published at
   <https://headroomlabs.ai/.well-known/security.txt>).

Please include as much of the following as you can:

- Type of vulnerability (e.g. injection, SSRF, data exposure, authentication bypass)
- The affected version or commit SHA, and the install surface (PyPI package, npm package,
  proxy server, a specific integration)
- Full path of the affected source file(s) and the relevant code
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code, if possible
- Impact assessment and any suggested remediation

### What to Expect

| Stage | Target |
| ----- | ------ |
| Acknowledgement of your report | within 48 hours |
| Initial severity assessment (CVSS + affected surface) | within 5 business days |
| Progress updates | at least weekly until resolved |
| Fix released for a critical issue | we aim for within 7 days of triage |
| Public advisory (GHSA) + CVE, if warranted | on or before the disclosure date |

We follow **coordinated disclosure**: we ask that you keep the report private until a fix
is released or **90 days** have passed since your report, whichever comes first. If a fix
is taking longer we will tell you why and agree a revised date with you.

With your permission we will credit you in the published
[security advisory](https://github.com/headroomlabs-ai/headroom/security/advisories).

### Safe Harbor

We will not pursue or support legal action against anyone who, in good faith:

- reports a vulnerability through one of the channels above,
- makes a reasonable effort to avoid privacy violations, data destruction, and service
  disruption,
- only interacts with accounts and data they own or have explicit permission to test, and
- does not exploit the issue beyond the minimum needed to demonstrate it, and does not
  disclose it publicly before the coordinated date.

Out of bounds: denial-of-service testing, spam, social engineering of Headroom staff or
users, physical attacks, and accessing or exfiltrating data that is not yours.

## Scope

**In scope:**

- The `headroom-ai` Python package (`pip install headroom-ai`) and the `headroom` CLI
- The `headroom-ai` npm package (TypeScript SDK)
- The Headroom proxy server and the `headroom install` deployment runners
- Official integrations maintained in this repository: LangChain, Agno, Strands, LiteLLM,
  Vercel AI SDK, the Anthropic / OpenAI SDK wrappers, and the MCP server

**Out of scope:**

- Third-party integrations or forks not maintained by us
- Vulnerabilities solely in third-party dependencies — report those upstream — **unless**
  Headroom's use of the dependency is what introduces the exposure
- Findings that require an already-compromised host, physical access, or a
  man-in-the-middle position on the operator's own network
- Missing hardening headers or best-practice suggestions with no demonstrated impact
- Social engineering and spam

## Security Best Practices for Operators

- **API keys**: pass credentials via environment variables or a secrets manager; never
  commit them. Headroom redacts known key patterns from its logs and does not persist
  them, but treat any host running the proxy as sensitive.
- **Proxy exposure**: do not expose the proxy to untrusted networks without an
  authentication layer in front of it. It is designed to run alongside your agent, not as
  a public endpoint.
- **Logs**: request and response bodies can be written to logs depending on your log
  level and flags. Store logs accordingly and scrub them before sharing.
- **Budget limits**: set budget limits to bound the blast radius of a misconfigured or
  abused client.

## Security Design

- **Credential handling**: API keys are redacted from logs by pattern and are not written
  to disk by Headroom itself.
- **Passthrough by default**: message content is forwarded unchanged unless a
  transformation is explicitly enabled.
- **Input validation**: requests are validated against the provider schema before
  processing.
- **Fail open, not wide**: malformed routing and compression rules are skipped rather
  than silently broadened.

Thank you for helping keep Headroom and its users safe.
