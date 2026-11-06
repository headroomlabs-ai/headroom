# OpenCode 2.x Anthropic Messages transport proof

This records the actual OpenCode runtime boundary requested in
[PR #4594's review](https://github.com/headroomlabs-ai/headroom/pull/4594#pullrequestreview-5745390972).
The OpenCode process, its Anthropic adapter, built plugin, and Headroom proxy are real.
Only the upstream API is a controlled local fixture. No live Copilot credentials are used.

## Environment

Recorded on **2026-11-04 at 04:23 UTC**, against Headroom source
`8766328a6c20909520b0a7ac7b5d487900316084`:

- Linux 6.18.57, x86_64; Python 3.12.14; Node 24.22.1; npm 11.9.0.
- Official OpenCode **2.2.5**, from `@opencode/cli-linux-x64-baseline@2.2.5`.
- `headroom-opencode` 0.42.0; Headroom 0.42.1-dev from this checkout.
- Built `dist-standalone/index.js`, byte-identical to the checked-in wheel bundle.
  SHA-256: `de88ab2120df79ad67fad3f7a06f8618ea093707cfcf1180921b1ae42e7b1cea`.
- Real provider adapter: `@opencode/ai/providers/anthropic`, model `claude-sonnet-4-6`.
  This is a configured Copilot-shaped provider, not OpenCode's live Copilot discovery flow.
- **Python-only degraded proxy mode**, explicitly observed in startup logs because
  this environment has no `headroom._core`. `HEADROOM_REQUIRE_RUST_CORE=false`
  permits that mode; `--no-optimize --no-cache` limits the experiment to transport.

## Exact commands / steps

From the repository root, prepare the source proxy and build the plugin:

```sh
uv sync --frozen --extra proxy --no-install-project
npm --prefix plugins/opencode ci
npm --prefix plugins/opencode test
npm --prefix plugins/opencode run typecheck
npm --prefix plugins/opencode run build
npm --prefix plugins/opencode run build:standalone
cmp plugins/opencode/dist-standalone/index.js headroom/providers/opencode/_dist/index.js
```

Use an official OpenCode 2.x executable. For the Linux x86_64 run above, the binary
was installed without executing a package install script:

```sh
npm install --prefix /tmp/opencode-runtime --omit=optional --ignore-scripts \
  --no-save --no-audit --no-fund @opencode/cli-linux-x64-baseline@2.2.5
OPENCODE_BIN=/tmp/opencode-runtime/node_modules/@opencode/cli-linux-x64-baseline/bin/opencode
.venv/bin/python plugins/opencode/scripts/runtime-proof.py \
  --opencode "$OPENCODE_BIN" --output /tmp/headroom-opencode-proof
```

The installation/output directories are selectable; `--output` must be new.
The script requires POSIX process groups and the `127.0.0.2` loopback alias.
It starts a new real proxy and a strict-path SSE fixture for each scenario, then runs:

```sh
opencode run --standalone --print-logs --log-level debug --format json \
  'Reply with HEADROOM_RUNTIME_OK.'
```

Each run uses isolated HOME/XDG directories, a minimal environment, disabled model
metadata fetching, and synthetic bearer/API-key values. This isolates default
credential discovery, not filesystem access. The upstream binds only to `127.0.0.2`,
which exercises model rewriting without the plugin's `localhost`/`127.0.0.1`
exclusion. That address is explicitly allowlisted in the test proxy. Ports are
allocated dynamically. OpenCode loads the built standalone **directory**, whose
`index.js` is its plugin entrypoint. The script does not call `headroomSetup` itself,
replace the OpenCode context, or send the model request with a manual `fetch`.

## Observed result

All four scenarios exited successfully:

| Scenario | Plugin | Proxy POSTs | Upstream POSTs | Upstream path |
| --- | --- | --- | --- | --- |
| Baseline | disabled | 0 | 2 | `/messages` |
| Copilot-shaped root URL | enabled | 2 | 2 | `/v1/messages` |
| `/v1` gateway | enabled | 2 | 2 | `/v1/messages` |
| Prefixed gateway | enabled | 0 | 2 | `/api/coding/paas/v4/messages` |

OpenCode made a title request and a user-turn request in each case. Every upstream
request retained the synthetic `Authorization` and `x-api-key`, the expected model,
`stream: true`, and `anthropic-version: 2023-06-01`. None contained `x-headroom-*`.
Each user turn received `HEADROOM_RUNTIME_OK` as an OpenCode text event, followed by
`step_finish`. The fixture flushes individual SSE events rather than returning a
JSON completion.

The proxied root-URL run logged these events (irrelevant fields omitted):

```text
04:23:10.316Z msg="loading plugin" entrypoint=file:///<checkout>/plugins/opencode/dist-standalone/index.js
04:23:10.975  event=proxy_inbound_request method=POST path=/v1/messages
             user-agent=opencode/latest/2.2.5/cli
             x-headroom-base-url=http://127.0.0.2:41341
             x-headroom-original-path=/messages x-headroom-project=copilot-shaped
04:23:10.979  event=outbound_headers forwarder=anthropic_messages stripped_count=3
04:23:10.998  event=outbound_request forwarder=streaming method=POST
             path=http://127.0.0.2:41341/v1/messages body_mutated=false source=passthrough
```

The script verifies proxy and upstream counts, exact upstream paths, preserved
synthetic auth, stripped internal headers, plugin loading, and a completed streamed
client reply. A fixture request to an unexpected endpoint receives 404. The root
URL case deliberately records the existing proxy contract of forwarding to
`/v1/messages`; it does not establish which endpoints a live Copilot account accepts.
The prefix exclusion is exercised by the real Anthropic adapter with the full
prefix intact and no model POST reaching Headroom.

Machine-readable results are in [runtime-proof-result.json](scripts/runtime-proof-result.json).
A rerun saves generated config, proxy/OpenCode logs, upstream observations and its
summary under the selected output directory. Plugin tests also passed **64/64**,
as did typecheck, both builds, bundle comparison, and Ruff checks for this script.

## Not tested

- Live `api.githubcopilot.com`, account discovery, subscription authentication,
  token refresh, Copilot-specific headers, or actual model inference.
- A live z.ai gateway or its provider-specific adapter. The prefix control uses
  the same real Anthropic adapter against a local path fixture.
- Rust-core behavior, compression/optimization, cache behavior, or production rollout.
- HTTPS/TLS, HTTP/2, rate limits, retries, or streaming latency/backpressure.
- OpenCode versions other than 2.2.5. The script rejects non-2.x versions and records
  the actual Rust-core startup mode, which may differ on another machine.
