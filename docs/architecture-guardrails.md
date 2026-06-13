# Headroom Architectural Guardrails

Headroom has repo-specific invariants that generic linting cannot infer. The
guardrail runner in `scripts/headroom_guardrails.py` captures those contracts as
small static checks so repeated review findings become pre-commit and CI
failures.

Run it locally with:

```bash
python scripts/headroom_guardrails.py
```

Current rules cover:

- preserving provider-specific message metadata during backend reconstruction;
- rejecting positional restoration from optimized messages back to originals;
- keeping CCR marker hashes aligned with stored cache keys;
- requiring `INPUT_COMPRESSED` events to expose `original_messages`;
- keeping proxy CORS defaults scoped to the configured localhost port;
- ensuring CI and pre-commit run the guardrail runner;
- requiring Rust crates to opt into the workspace lint policy.

When review finds a repeatable bug class, add a focused `Rule` implementation
and a test in `scripts/tests/test_headroom_guardrails.py`.
