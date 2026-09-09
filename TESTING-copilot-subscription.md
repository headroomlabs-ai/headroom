# Testing: GitHub Copilot subscription mode (`headroom wrap copilot --subscription`)

This feature has live coverage on macOS and Windows. Additional Linux secret-store
coverage is still useful (see [Status](#status)). If you have a GitHub Copilot
subscription and 10 minutes, please run one of the flows below and
[file a report](https://github.com/headroomlabs-ai/headroom/issues/new?template=copilot-subscription-test-report.md).

> ⚠️ This reads your Copilot login token and routes your
> Copilot CLI traffic through a local Headroom proxy. Only run it if you're
> comfortable with that. The branch is open for inspection.

## What it does (and what "subscription" means here)

Normally `headroom wrap copilot` is **BYOK** — you bring an Anthropic/OpenAI API
key and pay that vendor. `--subscription` is different: it lets you use the
**Copilot seat you already pay GitHub for**, with **no separate API key**, while
still routing through Headroom so your context gets compressed.

Mechanically: the Copilot CLI's only interposition hook is its provider-override
(the "BYOK transport"), so Headroom uses that knob but supplies **your
subscription token** and points back at **GitHub's own Copilot API**. So the CLI
may print "BYOK" and require an explicit `--model`, but you are **not** paying a
third party — it's your subscription, just compressed. (Proof it's working: the
proxy forwards to GitHub's Copilot API — `https://api.githubcopilot.com` by
default — with your token.)

## API host & Enterprise / data-residency

Headroom routes wrapped Copilot traffic to GitHub's **generic public host**,
`https://api.githubcopilot.com`, for both `--subscription` and the implicit
OAuth path. That host serves the full model set (including newer models on the
responses API) and matches the routing that worked before 0.23.

Headroom deliberately does **not** auto-select a per-account host from
`/copilot_internal/user`. That endpoint advertises a segmented host (e.g.
`api.individual.githubcopilot.com`) that does **not** serve newer models on the
responses API and is not the host the official Copilot client routes with — using
it regressed `headroom wrap copilot` after 0.22.4
([#610](https://github.com/headroomlabs-ai/headroom/issues/610)).

**Enterprise / data-residency:** if your organization is provisioned on a
dedicated Copilot API host (GitHub Enterprise Cloud with data residency, or an
egress proxy), pin it explicitly — the override flows through both
`--subscription` and OAuth, and onward through the proxy to the upstream request:

```bash
export GITHUB_COPILOT_API_URL=https://api.<your-host>.githubcopilot.com
headroom wrap copilot --subscription -- --model gpt-5.4
```

If you operate such an environment and would like Headroom to **auto-detect** the
correct host instead of pinning it, please [open an issue](https://github.com/headroomlabs-ai/headroom/issues/new) —
the intended path is to resolve it from GitHub's token-exchange endpoint (the
source the official Copilot client uses), and we'd want to validate it against a
real enterprise tenant.

## Status

| Platform | Mechanism (compress + forward) | Token **auto-discovery** from the OS secret store |
|----------|:---:|:---:|
| macOS (Keychain) | ✅ verified | ✅ verified (`copilot-cli`) |
| Linux (`secret-tool`/libsecret) | ✅ expected | ❓ **needs testing** |
| Windows (Headroom device auth) | ✅ verified | ✅ verified |
| Windows (Copilot CLI credential reuse) | ✅ verified after auth | ❌ Copilot CLI 1.0.81 does not expose the legacy Credential Manager schema |
| Any OS via `GITHUB_COPILOT_TOKEN` env var | ✅ verified by tests | n/a (bypasses discovery) |

The two things we want to learn:
1. **Does it work end to end on your OS?**
2. **Does it find your Copilot token automatically**, or do you have to set
   `GITHUB_COPILOT_TOKEN`? If it can't find it, we need the **storage schema**
   (see each flow) so we can fix auto-discovery.

## Prerequisites (all platforms)

1. A **GitHub Copilot subscription**.
2. The **GitHub Copilot CLI**: `npm install -g @github/copilot`
3. **Log in once**: run `copilot`, complete the device-code login in your
   browser, then type `/exit`.

---

## Linux — the flow we most need (tests auto-discovery)

Auto-discovery only works with a **host-native** install (a container can't read
your host secret store). Linux has prebuilt wheels, so:

```bash
pipx install --pip-args='--pre' headroom-ai     # or: pip install --pre headroom-ai
# (no separate API key needed — that's the point)
headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with exactly: HEADROOM_OK"
```

- **If it prints `HEADROOM_OK`** → auto-discovery works on your Linux. 🎉 Report success.
- **If it errors with "no reusable bearer token"** → discovery missed your token. Please grab the **schema** so we can fix it (redact the secret), then confirm the mechanism works via the env var:
  ```bash
  secret-tool search --all 2>/dev/null | sed -E 's/^secret = .*/secret = <redacted>/'
  # then retry, supplying the token explicitly:
  GITHUB_COPILOT_TOKEN='<your-token>' headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with: HEADROOM_OK"
  ```
  Report the `attribute.*` lines from `secret-tool` and whether the env-var retry worked.

---

## Windows

For a source checkout with Python and Rust installed, build the current tree with
the proxy extra and authorize Headroom's dedicated OAuth app:

```powershell
uv sync --extra proxy --extra dev
uv run --no-sync headroom copilot-auth login
uv run --no-sync python e2e/copilot_live.py --vscode-extension `
  --model gpt-5-mini --model gpt-5.5 `
  --model gpt-5.6-luna --model gpt-5.6-sol --model gpt-5.6-terra
```

The live suite uses the official Copilot CLI, exercises subscription wrapping,
sends requests through an isolated VS Code proxy configuration, and optionally
drives the installed VS Code extension through `code chat`. It snapshots and
restores real VS Code settings byte-for-byte, deliberately occupies the requested
port to verify fallback-port propagation, and checks every selected model in both
the Copilot response and Headroom's traffic accounting. The Docker-native wrap
suite additionally captures an A-to-B-to-A model sequence at its mock upstream,
proving the outbound request bodies change without stale model state. Neither
suite reads or prints token values.

Packaged-install alternatives:

**A. Mechanism test (easiest — Docker Desktop or WSL2):**
```powershell
$env:HEADROOM_DOCKER_IMAGE = "ghcr.io/headroomlabs-ai/headroom:<branch-tag>"   # ask the maintainer for the tag
# run the Docker-native installer (scripts/install.ps1), then:
$env:GITHUB_COPILOT_TOKEN = "<your-token>"
headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with: HEADROOM_OK"
```
Report whether it prints `HEADROOM_OK`.

**B. Native auto-discovery schema:** after `copilot` login, check whether the
installed Copilot CLI exposes a reusable Windows credential target:
```cmd
cmd /c "cmdkey /list"
```
Report only a Copilot-related `Target:` line (it shows the target name, not the
secret). Copilot CLI 1.0.81 did not expose such a target in live Windows testing,
so use `headroom copilot-auth login` when native reuse is unavailable.

> A native Windows wheel is still tracked separately; source builds can run the
> full Windows authentication and routing matrix today.

---

## macOS (already proven — a second data point still helps)

```bash
pipx install --pip-args='--pre' headroom-ai
headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with exactly: HEADROOM_OK"
```
Schema, for reference: Keychain generic password, service `copilot-cli`
(`security find-generic-password -s copilot-cli -w`).

---

## What to report

Please open a
[Copilot subscription test report](https://github.com/headroomlabs-ai/headroom/issues/new?template=copilot-subscription-test-report.md)
with:

- **OS + version** and **how you installed** (pipx/pip wheel, Docker, source).
- Was plain `copilot` logged in?
- Did `wrap copilot --subscription` print **`HEADROOM_OK`**? Paste any error.
- Did it work **without** setting `GITHUB_COPILOT_TOKEN` (auto-discovery), or
  only **with** it?
- The **storage schema** if discovery failed (`secret-tool search --all` /
  `cmdkey /list`), with the secret redacted.
