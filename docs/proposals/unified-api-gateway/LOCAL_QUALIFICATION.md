# Unified API Gateway Local Qualification

Status: draft, local evidence only. No paid or live-provider request was run.

## Qualified source artifact

- Qualified source commit: `10481bda908a0c24e3f26e325fcbae93ce5e12a1`
- Base commit: `5ff4ea1ef948563304c9e8f4b9ccff0e2ae3aedd`
- Wheel: `headroom_ai-0.38.0-cp310-abi3-win_amd64.whl`
- Wheel SHA-256: `16b95caf23c1116431e80bb8330f36000af62c06e8c00aabc666a70a9e1297cc`
- Platform: Windows amd64, CPython 3.13; wheel ABI is CPython 3.10+ abi3

## Provenance digests

| Input | SHA-256 |
| --- | --- |
| `CAPABILITIES.json` | `86dd768ffdfda376a82a9f5322570132d30b303bfc618d750c31378ba9858dd4` |
| `gateway-config.schema.json` | `92bc28b71bf715146f470ac798fd82ad8392bf313eb910a64e7b522bb030607c` |
| `requirements.json` | `525f45db304bb66a950f9595120786e82ff3fedba223d07c3924966195ec028c` |
| `scenarios.json` | `bd2ac7b16d3801be5b2cf6a8ee519c124d783d7e3150bd012299b1209a8bdab7` |
| `gateway.api-keys.json` | `49af99ccc621f9c16d21f7822df6bff1ed98063277a9f96613ec8f8b0b78189f` |
| `gateway.cloud-identities.json` | `3c65facde5c5150083bfc497e4158e90114a29df134de18d4ce171ac555a6ebf` |
| `gateway.private-upstream.json` | `e120b0b72481f1b67cbc9322fcea19d6d42b3a19c41e41d6d7967bec2b2d7a72` |
| `protocol_fixtures.py` | `59641ff6b765bc38aa105b34a5c9c54344cf55bfc5126471f1d011fd2c593928` |

## Commands and observed results

- `.\.venv\Scripts\python.exe -m pytest tests/unified_gateway -q`: 819 passed,
  0 failed, 0 skipped.
- `.\.venv\Scripts\python.exe -m pytest
  tests/unified_gateway/test_artifact_verifier.py -q`: 1 passed.
- `.\.venv\Scripts\python.exe -m ruff check headroom/proxy/gateway headroom/cli
  headroom/proxy headroom/providers tests/unified_gateway
  scripts/verify_unified_gateway_artifact.py`: passed.
- `.\.venv\Scripts\python.exe -m ruff format --check headroom/proxy/gateway
  headroom/cli headroom/proxy headroom/providers tests/unified_gateway
  scripts/verify_unified_gateway_artifact.py`: passed.
- `.\.venv\Scripts\python.exe -m mypy headroom/proxy/gateway`: passed.
- `cargo fmt --check`: passed.
- `cargo test --workspace`: passed; the primary crate reported 935 passed,
  0 failed, and 1 ignored, and all remaining workspace crates/doc tests passed.
- `uv build --wheel --out-dir dist/qualification-10481bda9`: passed.
- `.\.venv\Scripts\python.exe scripts/verify_unified_gateway_artifact.py --wheel
  dist\qualification-10481bda9\headroom_ai-0.38.0-cp310-abi3-win_amd64.whl
  --expected-sha256
  16b95caf23c1116431e80bb8330f36000af62c06e8c00aabc666a70a9e1297cc`:
  passed. The verifier created a clean environment,
  installed this exact wheel with its declared `proxy` dependencies, ran the installed
  CLI outside the source tree, and imported `GatewayRuntime` under isolated mode.
- Hosted CI run `35945634525` at source `44934b26a`: 20/20 jobs passed. The PR
  check rollup reported 58 successful checks, 10 skipped, 0 failed, and 0 pending. The only source
  delta in `10481bda9` corrects artifact-verifier isolation; final hosted CI is required
  again on the qualification commit.

## Requirement-by-requirement audit

- The R01-R19 ledger contains 19 requirements and the T001-T100 manifest contains
  100 scenarios; all scenario requirement references resolve. Status totals are
  80 `proved_locally`, 14 `unavailable_negatively_tested`, and 6
  `external_not_run`. The gateway suite's manifest checks passed as part of the
  819-test gate.
- R01-R04 and R06-R14: proved locally by gateway unit, integration, and real-process
  checks, except cells explicitly classified unavailable and negatively tested.
- R05: local credential-source, lease, and broker behavior is proved locally; T025,
  long-uptime refresh using real cloud SDK credentials, remains `external_not_run`.
- R15: local workload-identity construction/failure and unavailable identity behavior
  are proved. T071 (real desktop credential stores), T073 (managed-enterprise source
  precedence), and T075 (locked real keychain behavior) remain `external_not_run`.
- R16: OpenAI and Anthropic SDK parsers plus Gemini wire shape exercised through real
  processes. Paid/live-provider behavior remains `external_not_run`.
- R17: static checks, Rust checks, local wheel build, digest verification, clean
  installation, and installed-artifact smoke passed. T081-T082, which bind official
  Gate A source and Gate B retained release bytes, remain `external_not_run`.
- R18-R19: unavailable native identities were negatively tested and hostile OAuth
  device-flow behavior was tested locally. No subscription adapter is enabled.

## Explicit boundaries

The following were not run: paid/live-provider calls, provider entitlement checks,
real credential expiry/refresh, OS keychain persistence, official retained release
qualification, release publication, or production activation. This report does not
approve merge, auto-merge, release, or production rollout. Human merge approval remains
required.
