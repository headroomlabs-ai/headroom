# Validation record

## Sidebar 0.1.2 / companion 0.1.1 — October 7, 2026

- JavaScript: 56 passed, including proxy-confirmed controls, legacy-companion
  upgrade guidance, stale window responses and session-switch isolation.
- Companion: 72 passed, 1 skipped (POSIX executable fixture on Windows).
  Covers strict control payloads, origin guards, capacity bounds, reset retention,
  and window totals beyond the display limit. Includes real Headroom guards/logger.
- Repository regressions: 189 passed across marketplace/versioning, extension
  ordering and loopback gating.
- Real proxy E2E: passed with capture on and off. The local provider receives
  unchanged content while paused, compressed content after resume, and compressed
  requests from another conversation throughout. Reset preserves inspectable logs.
  Response caching is disabled in this test so every probe reaches the provider.
- Native Claude test kit: 6 passed, including POST controls, time-window selection
  and rendered percentage bars. Companion wheel 0.1.1 builds; Ruff and whitespace
  checks pass.

The new controls and bars also have authenticated live-session evidence on Claude
2.1.293 / Haiku 5.5: a fictional todo SPA was built in a disposable worktree through
Headroom. Four retained successful requests recorded 17,687 tokens removed / 3.34%
weighted reduction. Native keyboard pause/resume changed the companion's confirmed
state. The current [screenshots](screenshots/README.md) and cropped TUI text come
from that session's ConPTY output. Images were pushed and fetched back for hash
verification before removing the temporary session, fixtures, credentials copy,
capture runtime and worktree. Raw transcripts and account/workspace details are
not published. Desktop live UI and native mouse input remain unverified. Control
state and reset markers are in-memory and clear on proxy restart.

## Sidebar 0.1.1 — October 6, 2026

Executed on Windows, October 6, 2026, against published Headroom 0.40.0 and the
repository proxy source based on main `855390d6110e0bbb199d9b97025baac16c2f5cf3`.

| Gate | Result | Boundary |
|---|---|---|
| JavaScript unit suite | 43 passed | Shipped hooks, protocol host fixture, lifecycle, stale responses, `/clear` state loss, backward chunk paging and polling/inspection race |
| Companion Python suite | 61 passed, 1 skipped | Accounting including provider clamping/replay debt, filtering, bounded previews, retention, exact-origin guards and launcher |
| Real Headroom logger/security gate | Passed | Actual RequestLogger/RequestLog and loopback/origin guards; no substituted Headroom modules |
| HTTP bridge smoke | Passed | Actual local HTTP, synthetic request records, shipped hooks |
| Real proxy E2E, capture on/off | Passed | Actual proxy pipeline, local provider stub, actual retained records and shipped hooks; 18,583 tokens saved in the synthetic fixture |
| Native Claude validation | Passed on 2.1.290 and 2.1.291 | Native hook validator |
| Native Claude runtime tests | 5 passed | Official Claude test-kit execution; includes sidebar startup tool invocation and resume guidance |
| Authenticated native sessions | Passed | Actual Anthropic-backed Windows Claude tool task; 22,801 saved tokens / 9.3%, matching proxy records |
| Companion wheel | Built/installed | Universal wheel and console-script/extension entry points |
| Runtime dependency check | Passed | 99 compatible installed dependencies |
| Repository marketplace/extension contracts | 29 passed | Mirrored marketplace entries and extension middleware security ordering |
| Windows launcher | Passed | Doctor/restart exits 0; forced startup timeout leaves no listener; existing services remain running |

The Windows skip is the POSIX executable fixture in `test_launcher.py`; native
Windows launching was exercised separately. The real-Headroom logger/guard test
passed both in the suite and independently through the native gate. Guard-module
fixtures are isolated by pytest's monkeypatch cleanup. A provider 429 in native testing was recorded as
failure, not successful savings.

## Reproduce

From `integrations/claude-code-mod`, in an environment containing Headroom's proxy
extra, this companion, pytest, Node 20+, and (for the native gate) Claude 2.1.287+:

```powershell
python -m pip install ./companion pytest
npm test
python -m pytest -q -c companion/pyproject.toml companion/tests
python -m pytest -q -c companion/pyproject.toml companion/tests/test_real_headroom.py
python scripts/run-http-smoke.py
python scripts/run-proxy-e2e.py
python scripts/run-proxy-e2e.py --no-capture
python scripts/build-wheel.py
python scripts/validate-native.py
```

The real proxy E2E spawns the checkout's proxy with a local Anthropic-compatible
provider stub, SmartCrusher, and model downloads disabled. It sends two UUID-tagged
requests, requires positive compression, checks the provider received compressed
tool output without internal headers, rejects cross-session detail reads, and
checks the sidebar's actual HTTP metrics, original/compressed/diff inspection,
close behavior and session-change safety. The capture-off run requires metrics
to work without exposing message text. No provider credentials are needed.

## Native acceptance performed

October 6 follow-up exercised interactive Claude 2.1.291 in a separate repository,
`G:\git\3dprint4me`, through the native Windows terminal. A direct Claude launch
showed an unlinked setup pane; `1`/`2` changed tabs and `q` closed it. Resuming the
same UUID through the dedicated launcher reached `LIVE`. A no-tools prompt
completed through Headroom, with 252,353 → 229,071 input tokens and 23,282 saved
(9.23%), matching the session-scoped companion record. Page Down exposed the
request row and Tab/Enter opened its message inspector. `/clear` removed prior
telemetry and the final installed hooks correctly displayed the changed-session
diagnostic. Native runtime tests additionally activate tab, refresh, close,
original/compressed/diff, and message-navigation buttons. Regression tests cover
restored periodic refresh when resuming the original linked session after state
reset, and accurate offline diagnostics when the first companion read fails.
The final follow-up passed 43 JavaScript tests, four native runtime tests, and
real-proxy E2E with capture both enabled and disabled. Native mouse input remains
unverified in this terminal host.

- Two independent sessions displayed separate request histories. Cross-session
  detail requests returned 404.
- Native tool-output task completed correctly; savings matched companion metrics.
- Original/compressed/diff inspector and keyboard text paging worked.
- At 80x30, native PageDown exposed content below the visible pane.
- Proxy outage displayed offline/stale status; restart changed epoch and reset
  retained counters. Capture off preserved metrics and disabled detail.
- Exact UUID resume worked. `/clear` discarded native plugin state; the recovery
  fix displayed an unlinked/relaunch notice. `/headroom` reopened that pane in
  that historical native run. The current Sidebar command is
  `/headroom-sidebar`; `/headroom` belongs to Snip when both are installed.
- Windows proxy startup failure cleanup was tested after forced timeout; doctor
  and capture-enabled restart returned successfully after detaching console handles.

Native mouse input was unavailable in the ConPTY test host. Claude Desktop live
UI and every platform/Claude version combination have not been exercised. CI
covers Linux/Windows protocol-host E2E, not authenticated native Claude sessions.
Capture and native session artifacts contain conversation text and are deliberately
excluded from this repository. Hosted checks are evaluated on the PR's exact head.

CI loads checkout Python source with the unchanged Rust extension from the
published 0.40.0 wheel; this PR does not modify native code.

## Sidebar 0.1.3 visual hierarchy pass

58 JavaScript tests, six native Claude test-kit tests, and 26 version/manifest regressions passed. Version verification and `git diff --check` passed. An independent review found no material issues. Real isolated Haiku 5.5 todo work produced eight successful retained requests, 39,524 → 39,406 tokens (118 removed, 0.3%). Native pause/resume was confirmed against the companion. Current screenshots show the framed savings card, spaced sections and request rows, and collapsed secondary details. Strict empty MCP configuration and fictional fixtures kept private project data out of the session.
