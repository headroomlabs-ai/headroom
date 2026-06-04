# Testing: GitHub Copilot subscription mode (`headroom wrap copilot --subscription`)

This is an **experimental** feature and we need help verifying it on **Linux and
Windows**. It already works on macOS; the cross-platform gap is small and
specific (see [Status](#status)). If you have a GitHub Copilot subscription and
10 minutes, please run one of the flows below and
[file a report](https://github.com/chopratejas/headroom/issues/new?template=copilot-subscription-test-report.md).

> ⚠️ This is experimental, and it reads your Copilot login token + routes your
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
proxy forwards to `https://api.*.githubcopilot.com` with your token.)

## Status

| Platform | Mechanism (compress + forward) | Token **auto-discovery** from the OS secret store |
|----------|:---:|:---:|
| macOS (Keychain) | ✅ verified | ✅ verified (`copilot-cli`) |
| Linux (`secret-tool`/libsecret) | ✅ expected | ❓ **needs testing** |
| Windows (Credential Manager) | ✅ expected | ❓ **needs testing** |
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

There is **no native Windows wheel yet**, so pick one:

**A. Mechanism test (easiest — Docker Desktop or WSL2):**
```powershell
$env:HEADROOM_DOCKER_IMAGE = "ghcr.io/chopratejas/headroom:<branch-tag>"   # ask the maintainer for the tag
# run the Docker-native installer (scripts/install.ps1), then:
$env:GITHUB_COPILOT_TOKEN = "<your-token>"
headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with: HEADROOM_OK"
```
Report whether it prints `HEADROOM_OK`.

**B. Native auto-discovery schema (even without a working install):** after
`copilot` login, tell us where Windows stored the token:
```cmd
cmd /c "cmdkey /list"
```
Report the `Target:` line that looks Copilot-related (it shows the target name,
not the secret). That single fact lets us make native Windows discovery work.

> Native Windows auto-discovery becomes fully testable once we add a Windows
> wheel to the build matrix — tracked separately.

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
[Copilot subscription test report](https://github.com/chopratejas/headroom/issues/new?template=copilot-subscription-test-report.md)
with:

- **OS + version** and **how you installed** (pipx/pip wheel, Docker, source).
- Was plain `copilot` logged in?
- Did `wrap copilot --subscription` print **`HEADROOM_OK`**? Paste any error.
- Did it work **without** setting `GITHUB_COPILOT_TOKEN` (auto-discovery), or
  only **with** it?
- The **storage schema** if discovery failed (`secret-tool search --all` /
  `cmdkey /list`), with the secret redacted.
