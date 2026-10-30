# AGENTS.md

## If you landed here to use Headroom in an app

The PyPI package is `headroom-ai`; the import and the CLI are `headroom`.

```bash
pip install headroom-ai
```

```python
from headroom import compress

result = compress(messages, model="gpt-4o")  # OpenAI or Anthropic message dicts
# send result.messages instead of messages; log result.tokens_before / tokens_after / tokens_saved
```

In-process, no API key, no model download. `HEADROOM_OFFLINE=1` makes the run
zero-network (no update check, no model or pricing-table downloads, no beacon).
Verify without a provider key:

```bash
HEADROOM_OFFLINE=1 python -c "from headroom import compress; log='\n'.join(f'INFO step {i} ok' for i in range(3000))+'\nFATAL disk full'; r=compress([{'role':'user','content':'What failed?'},{'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function','function':{'name':'logs','arguments':'{}'}}]},{'role':'tool','tool_call_id':'c1','content':log}], model='gpt-4o'); print(r.tokens_before, '->', r.tokens_after, 'FATAL kept:', 'FATAL' in str(r.messages))"
```

- FastAPI example: https://docs.headroomlabs.ai/docs/fastapi
- Full docs index for agents: https://docs.headroomlabs.ai/llms.txt (hand-curated subset: `llms.txt` in this repo)
- Coding agents (Claude Code, Codex, Cursor, Copilot): `uv tool install --python 3.13 "headroom-ai[all]"` then `headroom wrap claude`
- `headroom.integrations.asgi.CompressionMiddleware` is for apps that *serve* `/v1/chat/completions`-style routes, not for an app that *calls* a provider; use `compress()` for that
- Canonical repo: https://github.com/headroomlabs-ai/headroom. `github.com/chopratejas/headroom` redirects here; there is no `headroom-sdk` org. The PyPI distribution named `headroom` is a different project.

## If you are changing this repo

- Layout: Python package in `headroom/`, Rust core and PyO3 extension in `crates/` (built by maturin), TypeScript SDK in `sdk/typescript/`, docs site (Next.js + Fumadocs) in `docs/content/docs/`, agent plugins in `plugins/`, examples in `examples/`.
- Setup: `uv sync --extra dev` and `make install-git-hooks` (pre-commit, commitlint, pre-push). Python checks: `uv run ruff check .`, `uv run ruff format .`, `uv run pytest`. Rust checks: `make ci-precheck-rust`. All gates: `make ci-precheck`. See `CONTRIBUTING.md`.
- Commits and PR titles are Conventional Commits (`feat:`, `fix:`, `docs:`, ...). Never edit `CHANGELOG.md`; release-please writes it from PR titles.
- Every external PR needs a "Real behavior proof" section (setup, exact commands, observed output, what was not tested). Bug fixes need a test that fails before the fix.
- One docs site: `docs/`, deployed by Vercel to docs.headroomlabs.ai. `wiki/` and the `gh-pages` branch are not published; do not add content there.
- Public API examples in `README.md`, `llms.txt`, `AGENTS.md` and `docs/` must use names that exist in `headroom/compress.py` (`compress`, `CompressResult.messages`, `.tokens_before`, `.tokens_after`, `.tokens_saved`, `.compression_ratio`, `.transforms_applied`).
- Savings claims anywhere in the repo must match the measured table under "Proof" in `README.md`: 21-57% on the four agent scenarios, 90%+ on repeated JSON arrays and log lines, little on prose.
- Version-carrying files (`pyproject.toml`, plugin and marketplace manifests, `sdk/typescript/package.json`, `server.json`) are kept in step by `scripts/version-sync.py` and gated by `scripts/verify-versions.py`; add any new one to both.
