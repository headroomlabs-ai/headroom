# Auth Modes

Headroom classifies every inbound request into one of three **auth modes** at request entry. The mode drives every downstream compression, cache, and header policy decision. Detection is a pure function of HTTP headers — no I/O, no allocation beyond a single `to_lowercase` of the User-Agent, runs in <10us per call.

The classifier ships in two equivalent implementations:

- **Rust:** `crates/headroom-core/src/auth_mode.rs` (`classify`)
- **Python:** `headroom/proxy/auth_mode.py` (`classify_auth_mode`)

Both are byte-for-byte identical on every header set covered by the parity test suite (`crates/headroom-core/tests/auth_mode.rs` + `tests/test_auth_mode.py`).

## The three modes

### `Payg` — pay-as-you-go API key

| Signal | Examples |
|---|---|
| `Authorization: Bearer sk-ant-api*` | Anthropic PAYG key |
| `Authorization: Bearer sk-*` (excluding `sk-ant-oat-`) | OpenAI PAYG key |
| `x-api-key: ...` | Anthropic API key style |
| `x-goog-api-key: ...` | Google Gemini key |

**Compression policy:** aggressive. The caller pays per token; compression directly saves them money. Cache hit-rate matters financially (Anthropic 1.25-2× cache write, 0.10× cache read). Live-zone compression, CCR, type-aware compressors all turn on. This is the OSS default.

### `OAuth` — OAuth bearer / IAM-signed

| Signal | Examples |
|---|---|
| `Authorization: Bearer sk-ant-oat-*` | Claude Pro / Max OAuth |
| `Authorization: Bearer <jwt>` (3-segment) | Codex / Cursor / Copilot OAuth |
| `Authorization: AWS4-HMAC-SHA256 ...` | Bedrock SigV4 |
| Any other non-`Bearer` Authorization scheme | Vertex ADC, custom proxies |

**Compression policy:** passthrough-prefer. Per-token cost is opaque (subscription) or zero from the caller's POV (IAM-bound usage); compression value is **extending effective context within rate-limit / quota windows**, not saving money. Cache safety is paramount because OAuth scopes pin to `(account, model, session)` and beta-header drift can break OAuth-issued scopes. **No auto-`cache_control`, no auto-`prompt_cache_key`, no lossy compressors.** Lossless-only.

### `Subscription` — UX-bound CLI / IDE

| Signal | UA prefix |
|---|---|
| Claude Code | `claude-code/` |
| Claude CLI | `claude-cli/` |
| Codex CLI | `codex-cli/` |
| Cursor | `cursor/` |
| Claude VS Code | `claude-vscode/` |
| GitHub Copilot | `github-copilot/` |
| Anthropic CLI | `anthropic-cli/` |
| Antigravity | `antigravity/` |

**Compression policy:** stealth. Provider rate-limits by request count; programmatic-fingerprint detection means Headroom MUST look like the upstream agent. Same compression policy as `OAuth` **plus**:

- Preserve `accept-encoding` byte-for-byte.
- Never inject `X-Headroom-*` headers on upstream-bound requests.
- Never mutate `User-Agent`.
- Skip `X-Forwarded-*` headers on upstream-bound requests (see Phase F PR-F4).

The Subscription UA wins over any bearer token shape — a Claude Code session that happens to carry a `sk-ant-oat-*` token is still a subscription client, never OAuth.

## Detection signals — full decision order

The classifier evaluates these in order; the **first** matching rule wins.

1. **User-Agent contains a `SUBSCRIPTION_UA_PREFIXES` entry** → `Subscription`.
2. **`Authorization: Bearer sk-ant-oat-*`** → `OAuth`.
3. **`Authorization: Bearer sk-ant-api*` or `Bearer sk-*`** → `Payg`.
4. **`Authorization: Bearer <jwt>`** (3 dot-separated segments) → `OAuth`.
5. **`Authorization` present but not `Bearer ...`** (e.g., `AWS4-HMAC-SHA256`) → `OAuth`.
6. **`x-api-key` present** → `Payg`.
7. **`x-goog-api-key` present** → `Payg`.
8. **Default** → `Payg`.

The subscription-UA check is intentionally first because the same OAuth token shape appears in both Claude Pro (web) and Claude Code (CLI), and only the User-Agent disambiguates them.

The "default to PAYG" rule is intentionally conservative: misclassifying a non-PAYG client as PAYG over-compresses, which only costs us a re-run; under-compressing a PAYG client leaves money on the table, which is worse for the OSS-default user.

## How to extend

### Adding a new subscription CLI

Add the UA prefix to **both** files — they must agree:

- `crates/headroom-core/src/auth_mode.rs` → `SUBSCRIPTION_UA_PREFIXES`
- `headroom/proxy/auth_mode.py` → `SUBSCRIPTION_UA_PREFIXES`

Then add a parametrised parity test case in **both**:

- `crates/headroom-core/tests/auth_mode.rs` → add a `#[test] fn ..._ua_classified_subscription`.
- `tests/test_auth_mode.py` → covered automatically by the existing `test_every_subscription_prefix_classified_subscription` parametrised test.

### Adding a new OAuth token shape

Add the prefix check **before** the `sk-` PAYG branch in both `classify` and `classify_auth_mode`. Order matters: any token shape that's a strict prefix of `sk-` must be checked first.

### Making the prefix list user-configurable

The list lives in a `const` so a future Phase F follow-up PR can swap it for a configurable source (env var, TOML config, CLI flag) without touching the function body. The recommended path:

1. Read the list from `Config::subscription_ua_prefixes` (Rust) / `headroom.config.Settings.subscription_ua_prefixes` (Python).
2. Default to the current static list if unset.
3. Pass the list as a parameter to `classify` / `classify_auth_mode`.

The classifier itself is a pure function — adding a parameter is a localized change.

## Performance

| Path | Per-call latency (p50) |
|---|---|
| Rust empty headers | ~20 ns |
| Rust PAYG (Bearer prefix match) | ~50 ns |
| Rust Subscription (UA lowercase + scan) | ~600 ns |
| Python empty headers | ~3 us |
| Python Subscription | ~12 us |

All paths are well under the <10us Rust budget and the <100us Python budget asserted by the test suite.

## Where the auth-mode value flows

After classification, the value is stored on the request object so downstream code reads it without re-classifying:

- **Rust:** `req.extensions_mut().insert(auth_mode)`. Read with `req.extensions().get::<headroom_core::auth_mode::AuthMode>()`.
- **Python:** `request.state.auth_mode`. Read with `request.state.auth_mode`.

A structured log line (`event = auth_mode_classified`) fires once per request at request entry; the value is logged as `auth_mode = "payg" | "oauth" | "subscription"`.

## Phase F roadmap

PR-F1 (this PR) lands the helper. The rest of Phase F wires it into specific policy gates:

- **PR-F2:** per-mode compression policy gates (auto-`cache_control`, `prompt_cache_key`, lossy compressors).
- **PR-F3:** TOIN per-tenant aggregation key includes `(auth_mode, model_family, structure_hash)`.
- **PR-F4:** `X-Forwarded-*` skipped on Subscription mode.

See `REALIGNMENT/08-phase-F-auth-mode.md` for the full phase plan.
