# Unified API Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a secure, provider-neutral gateway profile to the existing Python
proxy, satisfying proposal requirements R01–R19 while preserving legacy behavior.

**Architecture:** The existing FastAPI application remains the only listener and
enforcement authority. An immutable `headroom.proxy.gateway` runtime authenticates
the caller, authorizes a route, reserves budget/concurrency, leases an
audience-bound credential, and supplies a request-scoped dispatch context to the
existing native forwarders or explicitly qualified protocol adapters.

**Tech Stack:** Python 3.10+, FastAPI/Starlette, Pydantic v2, HTTPX, Click, pytest,
pytest-asyncio, real Uvicorn process tests, OpenAI/Anthropic/Gemini SDKs, Maturin.

**Spec:**
`docs/superpowers/specs/2026-09-22-unified-api-gateway-design.md`

## Global Constraints

- Start from `origin/main` commit
  `94206e265203acfd72a3b939e9a964e29175ad50`.
- Keep the existing proxy behavior byte-for-byte compatible when `--gateway` is
  absent.
- Gateway v1 is Python-only, loopback-only, single-worker, and transform-off.
- Do not read secrets, resolve DNS, refresh credentials, or contact providers in
  `--check-config` mode.
- Never forward a Headroom client token or accept a caller-selected upstream URL
  for a managed credential.
- Reject unsupported capabilities before credential acquisition or upstream I/O.
- Do not enable any native subscription adapter without its provider admission
  dossier and approved live evidence.
- Every production behavior begins with an observed failing test and follows
  red-green-refactor.
- Keep the PR draft. Do not merge, release, tag, or run paid/live provider tests.
- The unchanged baseline failure
  `tests/test_bundled_tools_savings.py::test_ast_grep_slice_saves_tokens` is tracked
  separately and is not a gateway success criterion.

## Requirement ownership

| Requirement | Owning task |
|---|---|
| R01 | Task 2: operating-profile compatibility |
| R02 | Task 2: authoritative pure forwarding |
| R03 | Task 3: client authentication and grants |
| R04 | Task 4: credential-bound egress |
| R05 | Task 4: credential lifecycle and refresh |
| R06 | Task 5: model aliases and capability catalog |
| R07 | Task 6: qualified protocol translation |
| R08 | Task 5: native incremental streaming |
| R09 | Task 7: resource and WebSocket ownership |
| R10 | Task 8: account selection and retries |
| R11 | Task 8: atomic budget/concurrency admission |
| R12 | Task 9: privacy and truthful observability |
| R13 | Task 3: browser/input/network guards |
| R14 | Task 9: reload, ownership, and shutdown |
| R15 | Task 4: system-identity qualification |
| R16 | Task 11: SDK and legacy regressions |
| R17 | Task 12: source and artifact qualification |
| R18 | Task 10: provider admission |
| R19 | Task 10: hostile-flow-safe OAuth machinery |

Scenario batches are owned as follows: Task 2 covers T001 through T010; Task 3
covers T011 through T015 and T061 through T065; Task 4 covers T016 through T025
and T071 through T075; Task 5 covers T026 through T030 and T036 through T040;
Task 6 covers T031 through T035; Task 7 covers T041 through T045; Task 8 covers
T046 through T055; Task 9 covers T056 through T060 and T066 through T070; Task 10
covers T086 through T100; Tasks 11 and 12 close T076 through T085 and validate the
complete executable coverage map.

---

### Task 1: Retain and validate the proposal contract

**Files:**

- Create: `docs/proposals/unified-api-gateway/README.md`
- Create: `docs/proposals/unified-api-gateway/gateway-config.schema.json`
- Create: `docs/proposals/unified-api-gateway/requirements.json`
- Create: `docs/proposals/unified-api-gateway/scenarios.json`
- Create: `docs/proposals/unified-api-gateway/examples/gateway.api-keys.json`
- Create: `docs/proposals/unified-api-gateway/examples/gateway.cloud-identities.json`
- Create: `docs/proposals/unified-api-gateway/examples/gateway.private-upstream.json`
- Create: `tests/unified_gateway/test_proposal_contract.py`

**Interfaces:**

- Produces the normative v1 schema and stable T001–T100 scenario IDs used by all
  later tasks.
- Corrects the bundle defect by declaring `routes[].upstream_path_prefix` and
  `routes[].ingress_protocols` while retaining `additionalProperties: false`.

- [ ] **Step 1: Add a failing contract test**

```python
def test_all_examples_validate_against_runtime_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for example in sorted(EXAMPLES.glob("*.json")):
        assert list(validator.iter_errors(json.loads(example.read_text()))) == []


def test_every_requirement_has_five_stable_scenarios() -> None:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))["scenarios"]
    by_requirement = Counter(item["requirement"] for item in scenarios)
    assert by_requirement == {f"R{index:02d}": 5 for index in range(1, 19)} | {"R19": 10}
```

- [ ] **Step 2: Verify the tests fail because proposal artifacts are absent**

Run:
`python -m pytest tests/unified_gateway/test_proposal_contract.py -q`

Expected: FAIL because the retained schema and ledgers do not exist.

- [ ] **Step 3: Copy the reviewed artifacts and repair only the two missing schema properties**

The route property definitions are:

```json
"upstream_path_prefix": {
  "type": "string",
  "minLength": 1,
  "pattern": "^/"
},
"ingress_protocols": {
  "type": "array",
  "items": {
    "enum": [
      "openai-chat", "openai-responses", "anthropic-messages",
      "gemini-generate", "vertex-generate", "bedrock-invoke"
    ]
  },
  "minItems": 1,
  "uniqueItems": true
}
```

- [ ] **Step 4: Run the contract test and proposal validator**

Run:

```console
python -m pytest tests/unified_gateway/test_proposal_contract.py -q
python C:\Users\jd\Downloads\headroom-unified-proxy-proposal-2026-09-21\validation\run_checks.py
```

If the archive is not extracted, run its validator from a temporary extraction
outside the repository. Expected: contract tests pass; bundle validator result is
recorded without treating it as feature evidence.

- [ ] **Step 5: Commit**

```console
git add docs/proposals/unified-api-gateway tests/unified_gateway/test_proposal_contract.py
git commit -m "docs: retain unified gateway contract"
```

### Task 2: Parse gateway configuration and enforce the pure profile

**Files:**

- Create: `headroom/proxy/gateway/__init__.py`
- Create: `headroom/proxy/gateway/config.py`
- Create: `headroom/proxy/gateway/errors.py`
- Modify: `headroom/proxy/models.py`
- Modify: `headroom/cli/proxy.py`
- Modify: `headroom/proxy/server.py`
- Create: `tests/unified_gateway/test_config.py`
- Create: `tests/unified_gateway/test_cli_profile.py`
- Create: `tests/unified_gateway/test_pure_profile.py`

**Interfaces:**

- Produces `GatewayConfigSnapshot.load(path: Path) -> GatewayConfigSnapshot`.
- Produces `GatewayConfigSnapshot.redacted_dict() -> dict[str, object]`.
- Adds `ProxyConfig.gateway: GatewayConfigSnapshot | None`.
- Adds CLI flags `--gateway`, `--gateway-config PATH`, and `--check-config`.
- Later tasks consume immutable `PrincipalConfig`, `CredentialConfig`, and
  `RouteConfig` Pydantic models from this module.

- [ ] **Step 1: Write failing offline-validation tests for T001–T005**

```python
def test_check_config_has_no_secret_or_network_side_effects(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-must-not-be-read")
    monkeypatch.setattr(socket, "getaddrinfo", fail_if_called)
    result = runner.invoke(
        proxy, ["--gateway", "--gateway-config", str(config_path), "--check-config"]
    )
    assert result.exit_code == 0
    assert "configuration valid" in result.output.lower()
    assert "sentinel-must-not-be-read" not in result.output


@pytest.mark.parametrize("flag", ["--memory", "--code-graph", "--compress-user-messages"])
def test_gateway_rejects_transforming_flags(
    runner: CliRunner, config_path: Path, flag: str
) -> None:
    result = runner.invoke(proxy, ["--gateway", "--gateway-config", str(config_path), flag])
    assert result.exit_code != 0
    assert "incompatible with gateway pure mode" in result.output
```

- [ ] **Step 2: Run the focused tests and observe missing flags/models**

Run:
`python -m pytest tests/unified_gateway/test_config.py tests/unified_gateway/test_cli_profile.py tests/unified_gateway/test_pure_profile.py -q`

Expected: FAIL on unknown CLI options/imports.

- [ ] **Step 3: Implement immutable Pydantic configuration models**

```python
class GatewayConfigSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    runtime: RuntimeConfig
    transforms: TransformConfig
    privacy: PrivacyConfig
    client_auth: ClientAuthConfig
    credentials: tuple[CredentialConfig, ...]
    routes: tuple[RouteConfig, ...]
    transport: TransportConfig = TransportConfig()

    @classmethod
    def load(cls, path: Path) -> "GatewayConfigSnapshot":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def redacted_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")
```

Validation must reject duplicate IDs/models, missing references, provider/source
mismatches, non-loopback/remote/multi-worker runtime, transform modes other than
`off`, enabled routes with disabled credentials, and invalid origin/path pairs.

- [ ] **Step 4: Wire CLI construction without resolving credentials**

Load the snapshot before `ProxyConfig` creation. `--check-config` prints the
redacted configuration digest and exits. Normal gateway startup sets
`no_optimize=True`, `cache_enabled=False`, `memory_enabled=False`,
`traffic_learning_enabled=False`, `ccr_inject_tool=False`, and suppresses eager
model initialization.

- [ ] **Step 5: Run focused and legacy profile tests**

Run:

```console
python -m pytest tests/unified_gateway/test_config.py tests/unified_gateway/test_cli_profile.py tests/unified_gateway/test_pure_profile.py -q
python -m pytest tests/test_lossless_mode.py tests/test_proxy_disable_kompress.py tests/test_kompress_preload_deferral.py -q
ruff check headroom/proxy/gateway headroom/cli/proxy.py headroom/proxy/models.py tests/unified_gateway
```

Expected: all pass, with no warmup/compression probe calls in gateway mode.

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/proxy/models.py headroom/cli/proxy.py headroom/proxy/server.py tests/unified_gateway
git commit -m "feat(proxy): add pure gateway profile"
```

### Task 3: Authenticate principals and enforce route grants

**Files:**

- Create: `headroom/proxy/gateway/auth.py`
- Create: `headroom/proxy/gateway/context.py`
- Create: `headroom/proxy/gateway/middleware.py`
- Modify: `headroom/proxy/server.py`
- Modify: `headroom/providers/proxy_routes.py`
- Create: `tests/unified_gateway/test_auth.py`
- Create: `tests/unified_gateway/test_authorization.py`
- Create: `tests/unified_gateway/test_browser_guards.py`

**Interfaces:**

- Produces `GatewayPrincipal(id, scopes, routes)`.
- Produces `GatewayRequestContext(principal, route, ingress_protocol,
  request_id, snapshot_generation)` on `request.state.gateway`.
- Produces `GatewayAuthenticator.authenticate(headers) -> GatewayPrincipal`.
- Produces `GatewayAuthorizer.authorize(principal, protocol, public_model,
  capability) -> RouteConfig`.

- [ ] **Step 1: Write failing T011–T015 and T061–T065 tests**

```python
def test_missing_client_token_has_zero_downstream_side_effects(gateway_client, spies) -> None:
    response = gateway_client.post("/v1/responses", json={"model": "public-gpt", "input": "hi"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "gateway_auth_required"
    assert spies.credential_acquisitions == 0
    assert spies.upstream_requests == 0


def test_conflicting_protocol_credentials_are_rejected(gateway_client, spies) -> None:
    response = gateway_client.post(
        "/v1/messages",
        headers={"authorization": "Bearer client-a", "x-api-key": "client-b"},
        json={"model": "public-claude", "messages": []},
    )
    assert response.status_code == 401
    assert spies.upstream_requests == 0
```

Include literal cases for wrong secret, missing scope, ungranted route, unknown
model, query token, hostile Origin, forged Forwarded/Host, and every WebSocket
generation frame.

- [ ] **Step 2: Verify failures occur before auth middleware exists**

Run:
`python -m pytest tests/unified_gateway/test_auth.py tests/unified_gateway/test_authorization.py tests/unified_gateway/test_browser_guards.py -q`

- [ ] **Step 3: Implement constant-time auth and request-scoped authorization**

Use `hmac.compare_digest`, resolve only configured `env:` principal refs at service
startup, accept the normal protocol header slots, require all supplied credential
values to agree, strip query credentials, and return protocol-shaped errors. The
middleware handles HTTP authentication; WebSocket handlers invoke the same
authenticator for the handshake and each generation turn.

- [ ] **Step 4: Register explicit gateway protocols rather than a secret-bearing catch-all**

Add a route-to-protocol map for OpenAI Chat/Responses, Anthropic Messages, Gemini,
Vertex, and Bedrock. Managed credentials must never reach the existing generic
`/{path:path}` passthrough.

- [ ] **Step 5: Run auth and existing hardening regressions**

Run:

```console
python -m pytest tests/unified_gateway/test_auth.py tests/unified_gateway/test_authorization.py tests/unified_gateway/test_browser_guards.py -q
python -m pytest tests/test_header_isolation.py tests/test_proxy_hardening.py tests/test_proxy_cors.py tests/test_forwarded_headers.py -q
ruff check headroom/proxy/gateway headroom/providers/proxy_routes.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/proxy/server.py headroom/providers/proxy_routes.py tests/unified_gateway
git commit -m "feat(proxy): enforce gateway caller grants"
```

### Task 4: Broker opaque credentials and restrict egress

**Files:**

- Create: `headroom/proxy/gateway/credentials.py`
- Create: `headroom/proxy/gateway/credential_sources/__init__.py`
- Create: `headroom/proxy/gateway/credential_sources/environment.py`
- Create: `headroom/proxy/gateway/credential_sources/gcp.py`
- Create: `headroom/proxy/gateway/credential_sources/aws.py`
- Create: `headroom/proxy/gateway/egress.py`
- Modify: `headroom/proxy/helpers.py`
- Modify: `headroom/proxy/passthrough.py`
- Modify: `headroom/proxy/handlers/anthropic.py`
- Modify: `headroom/proxy/handlers/openai.py`
- Create: `tests/unified_gateway/test_credentials.py`
- Create: `tests/unified_gateway/test_refresh.py`
- Create: `tests/unified_gateway/test_egress.py`

**Interfaces:**

- Produces immutable `CredentialLease(id, provider, account_ref, audience,
  expires_at, generation, secret: SecretHandle)`.
- Produces `CredentialSource.acquire(route, now) -> CredentialLease` and
  `invalidate(lease, reason) -> None`.
- Produces `CredentialBroker.acquire(route, account_ref=None) -> CredentialLease`.
- Produces `EgressPolicy.authorize(lease, url, resolved_addresses) -> None`.
- Existing forwarders consume a request-scoped lease through
  `GatewayRequestContext.upstream_headers(url)`; they never read gateway secrets
  directly.

- [ ] **Step 1: Write failing T016–T025 and T071–T075 tests**

```python
async def test_client_and_provider_keys_never_cross(gateway_process, fake_upstream) -> None:
    response = await gateway_process.openai.responses.create(model="public-gpt", input="hello")
    assert response.output_text == "fixture"
    sent = fake_upstream.only_request
    assert sent.headers["authorization"] == "Bearer upstream-sentinel"
    assert "client-sentinel" not in sent.headers.values()


async def test_redirect_cannot_move_managed_credential(fake_transport, lease, policy) -> None:
    fake_transport.redirect("https://api.openai.com/v1/responses", "https://evil.example/steal")
    with pytest.raises(GatewayEgressDenied):
        await send_managed_request(lease, policy, fake_transport)
    assert fake_transport.requests_to("https://evil.example") == []
```

Cover origin, port, path prefix, project/region, DNS rebinding, metadata/link-local
addresses, TLS hostname verification, disabled credentials, missing ADC/AWS chain,
expiry, concurrent refresh, stale refresh publication, revocation, and redaction.

- [ ] **Step 2: Observe failures before broker/egress modules exist**

Run:
`python -m pytest tests/unified_gateway/test_credentials.py tests/unified_gateway/test_refresh.py tests/unified_gateway/test_egress.py -q`

- [ ] **Step 3: Implement secret handles, leases, and source adapters**

Environment source reads only the named variable. GCP and AWS adapters wrap their
official SDK credential chains and expose expiry/generation metadata without
serializing secrets. A per-account async single-flight publishes refresh results
only when its generation is current.

- [ ] **Step 4: Enforce final-destination policy at the forwarding seam**

Remove all client auth header variants before merging lease headers. Disable
redirect following for managed requests. Validate the final origin and path before
attaching credentials and validate resolved addresses on each connection for
private compatible routes.

- [ ] **Step 5: Run credential, header, and trust tests**

Run:

```console
python -m pytest tests/unified_gateway/test_credentials.py tests/unified_gateway/test_refresh.py tests/unified_gateway/test_egress.py -q
python -m pytest tests/test_header_isolation.py tests/test_upstream_credential_scoping.py tests/test_upstream_guard.py -q
ruff check headroom/proxy/gateway headroom/proxy/helpers.py headroom/proxy/handlers tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/proxy/helpers.py headroom/proxy/passthrough.py headroom/proxy/handlers tests/unified_gateway
git commit -m "feat(proxy): broker gateway credentials"
```

### Task 5: Build the capability registry and native dispatch

**Files:**

- Create: `headroom/proxy/gateway/models.py`
- Create: `headroom/proxy/gateway/dispatch.py`
- Modify: `headroom/providers/proxy_routes.py`
- Modify: `headroom/proxy/server.py`
- Create: `tests/unified_gateway/test_model_catalog.py`
- Create: `tests/unified_gateway/test_native_dispatch.py`
- Create: `tests/unified_gateway/test_native_bytes.py`
- Create: `tests/unified_gateway/fixtures/native/`

**Interfaces:**

- Produces `Capability` enum and immutable `RouteCapabilities`.
- Produces `ModelRegistry.visible_routes(principal) -> tuple[PublishedRoute, ...]`.
- Produces `GatewayDispatcher.resolve(context, body) -> DispatchPlan` where
  `DispatchPlan.contract` is `strict-native`, `routed-native`, or `translated`.
- Existing provider handlers receive the plan through request state and preserve
  original bytes for strict-native dispatch.

- [ ] **Step 1: Write failing T026–T030 and native T036–T040 tests**

```python
def test_model_catalog_contains_only_granted_enabled_routes(gateway_client) -> None:
    response = gateway_client.get("/v1/models", headers=client_headers("principal-a"))
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["public-gpt"]


@pytest.mark.parametrize("fixture", strict_native_fixtures())
def test_strict_native_request_and_response_entities_are_exact(fixture, gateway_process) -> None:
    result = fixture.send(gateway_process)
    assert result.upstream_request_body == fixture.request_bytes
    assert result.client_response_body == fixture.response_bytes
```

Fixtures contain unknown fields, Unicode, whitespace, numeric forms, tool schemas,
parallel tool IDs, and signed/opaque content.

- [ ] **Step 2: Verify catalog and byte tests fail**

Run:
`python -m pytest tests/unified_gateway/test_model_catalog.py tests/unified_gateway/test_native_dispatch.py tests/unified_gateway/test_native_bytes.py -q`

- [ ] **Step 3: Implement registry and dispatch plans**

Resolve public models only after principal auth. Enforce declared ingress protocol,
native protocol, body contract, and capability. For strict-native routes reuse the
raw body already retained by `body_forwarding`; for routed-native routes apply only
the declared model/endpoint patch and record its mutation reason.

- [ ] **Step 4: Integrate OpenAI Chat/Responses, Anthropic Messages, Gemini, Vertex, and Bedrock routes**

Each route calls the shared admission/dispatch path before its existing handler.
Unknown or incompatible models return protocol-shaped errors before broker access.
`/v1/models` and protocol-specific metadata routes use the principal-filtered
registry rather than upstream discovery.

- [ ] **Step 5: Run native and legacy byte-fidelity suites**

Run:

```console
python -m pytest tests/unified_gateway/test_model_catalog.py tests/unified_gateway/test_native_dispatch.py tests/unified_gateway/test_native_bytes.py -q
python -m pytest tests/test_proxy_byte_faithful_forwarding.py tests/test_codex_responses_passthrough_bytes.py tests/test_provider_proxy_routes.py -q
ruff check headroom/proxy/gateway headroom/providers/proxy_routes.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/providers/proxy_routes.py headroom/proxy/server.py tests/unified_gateway
git commit -m "feat(proxy): dispatch native gateway routes"
```

### Task 6: Add qualified protocol translation

**Files:**

- Create: `headroom/proxy/gateway/protocols/__init__.py`
- Create: `headroom/proxy/gateway/protocols/content.py`
- Create: `headroom/proxy/gateway/protocols/openai.py`
- Create: `headroom/proxy/gateway/protocols/anthropic.py`
- Create: `headroom/proxy/gateway/protocols/gemini.py`
- Create: `headroom/proxy/gateway/protocols/events.py`
- Modify: `headroom/proxy/gateway/dispatch.py`
- Create: `tests/unified_gateway/test_translation.py`
- Create: `tests/unified_gateway/test_translation_streaming.py`
- Create: `tests/unified_gateway/fixtures/translation/`

**Interfaces:**

- Produces typed `Conversation`, `ContentBlock`, `ToolDefinition`, `ToolCall`,
  `ToolResult`, `StructuredOutput`, and `StreamEvent` models.
- Produces directed `decode_openai`, `decode_anthropic`, `decode_gemini`,
  `encode_openai`, `encode_anthropic`, `encode_gemini`, and explicit event
  translators for each enabled source/target pair.
- `GatewayDispatcher` selects a directed adapter only when the route declares the
  requested capability set.

- [ ] **Step 1: Write failing T031–T035 semantic-oracle tests**

```python
@pytest.mark.parametrize("case", directed_translation_cases())
def test_directed_translation_matches_independent_literal_oracle(case) -> None:
    translated = translate(case.source_protocol, case.target_protocol, case.source)
    assert translated == case.expected_target


@pytest.mark.parametrize("feature", ["signed_thinking", "provider_tool", "unknown_role"])
def test_unrepresentable_feature_is_rejected_before_upstream(
    feature, gateway_client, spies
) -> None:
    response = gateway_client.post(
        "/v1/messages", json=fixture_for(feature), headers=client_headers()
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "gateway_capability_unavailable"
    assert spies.upstream_requests == 0
```

Include text, system messages, multimodal input, parallel client-owned tools,
structured output, refusals, finish reasons, usage units, fragmented tool arguments,
and event ordering. Expectations are literal fixtures, not round trips.

- [ ] **Step 2: Observe missing adapter failures**

Run:
`python -m pytest tests/unified_gateway/test_translation.py tests/unified_gateway/test_translation_streaming.py -q`

- [ ] **Step 3: Implement the typed content/event boundary and directed adapters**

Keep adapters independent. Decode validates source semantics; capability negotiation
rejects loss before credential acquisition; encode maps only qualified semantics.
Round-trip tests are supplemental and never replace literal directed oracles.

- [ ] **Step 4: Integrate translation into dispatch and streaming**

Translated requests use reconstructed JSON and are labeled `translated` in
request context. Stream event adapters preserve order and incremental tool
arguments; they never buffer an entire upstream response to simulate streaming.

- [ ] **Step 5: Run translation and existing backend tests**

Run:

```console
python -m pytest tests/unified_gateway/test_translation.py tests/unified_gateway/test_translation_streaming.py -q
python -m pytest tests/test_google_multimodal.py tests/test_openai_streaming_backend.py tests/test_vertex_claude_compression.py -q
ruff check headroom/proxy/gateway/protocols tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway/protocols headroom/proxy/gateway/dispatch.py tests/unified_gateway
git commit -m "feat(proxy): translate qualified gateway protocols"
```

### Task 7: Bind stateful resources and WebSocket turns

**Files:**

- Create: `headroom/proxy/gateway/resources.py`
- Modify: `headroom/proxy/gateway/middleware.py`
- Modify: `headroom/providers/proxy_routes.py`
- Modify: `headroom/proxy/handlers/openai.py`
- Modify: `headroom/providers/codex/live.py`
- Create: `tests/unified_gateway/test_resources.py`
- Create: `tests/unified_gateway/test_websocket.py`
- Create: `tests/unified_gateway/test_cancellation.py`

**Interfaces:**

- Produces `ResourceBinding(provider_id, principal_id, route_id, account_ref,
  adapter, expires_at)`.
- Produces async `ResourceRegistry.bind`, `authorize`, `delete`, and `expire`.
- WebSocket session state carries one principal/snapshot and binds every accepted
  response to the selected account; each `response.create` repeats authorization
  and admission.

- [ ] **Step 1: Write failing T041–T045 tests**

```python
async def test_cross_principal_response_lookup_never_reaches_upstream(
    gateway_process, fake_upstream
) -> None:
    response_id = await create_response(gateway_process, principal="a")
    result = await get_response(gateway_process, response_id, principal="b")
    assert result.status_code == 404
    assert fake_upstream.lookup_count == 0


async def test_every_websocket_generation_turn_is_reauthorized(ws_client, grants) -> None:
    await ws_client.send_json(response_create("public-gpt"))
    assert await ws_client.receive_json() == fixture_created_event()
    grants.revoke("principal-a", "route-a")
    await ws_client.send_json(response_create("public-gpt"))
    assert (await ws_client.receive_json())["error"]["code"] == "gateway_route_forbidden"
```

Cover response create/get/cancel/delete, subpaths, expired bindings, socket affinity,
revocation, accepted-operation ambiguity, disconnect cleanup, and frame limits.

- [ ] **Step 2: Observe failures against unbound existing routes**

Run:
`python -m pytest tests/unified_gateway/test_resources.py tests/unified_gateway/test_websocket.py tests/unified_gateway/test_cancellation.py -q`

- [ ] **Step 3: Implement bounded ownership registry and route guards**

Store metadata only; never persist prompts. Stateless mode advertises no durable
continuation. Missing/expired bindings return a local not-found/unavailable result
without probing another account.

- [ ] **Step 4: Enforce WebSocket auth/admission and cancellation cleanup**

Authenticate the handshake, authorize every generation frame, preserve account
affinity, cap incremental input/output queues, propagate disconnect cancellation,
and release all leases/reservations within the configured cleanup bound.

- [ ] **Step 5: Run stateful and legacy WebSocket suites**

Run:

```console
python -m pytest tests/unified_gateway/test_resources.py tests/unified_gateway/test_websocket.py tests/unified_gateway/test_cancellation.py -q
python -m pytest tests/test_codex_live.py tests/test_proxy_codex_route_aliases.py tests/test_graceful_shutdown.py -q
ruff check headroom/proxy/gateway headroom/providers/codex headroom/providers/proxy_routes.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/providers/proxy_routes.py headroom/proxy/handlers/openai.py headroom/providers/codex/live.py tests/unified_gateway
git commit -m "feat(proxy): secure gateway stateful operations"
```

### Task 8: Add account routing, safe retries, and atomic admission

**Files:**

- Create: `headroom/proxy/gateway/routing.py`
- Create: `headroom/proxy/gateway/admission.py`
- Modify: `headroom/proxy/gateway/context.py`
- Modify: `headroom/proxy/helpers.py`
- Modify: `headroom/proxy/cost.py`
- Modify: `headroom/proxy/server.py`
- Create: `tests/unified_gateway/test_routing.py`
- Create: `tests/unified_gateway/test_retries.py`
- Create: `tests/unified_gateway/test_admission.py`

**Interfaces:**

- Produces `AccountRouter.select(route, principal, resource_binding=None) -> AccountSelection`.
- Produces `RetryDecision.classify(failure, exposure, provider_contract) -> bool`.
- Produces async `AdmissionController.reserve(request) -> AdmissionReservation`.
- `AdmissionReservation.finalize(usage)` and `.release()` are idempotent and used
  through an async context manager.

- [ ] **Step 1: Write failing T046–T055 tests**

```python
async def test_atomic_budget_reservation_allows_only_one_boundary_request(controller) -> None:
    results = await asyncio.gather(
        controller.try_reserve(estimated_cost=0.75),
        controller.try_reserve(estimated_cost=0.75),
    )
    assert sorted(result.allowed for result in results) == [False, True]


@pytest.mark.parametrize("exposure", ["ambiguous_acceptance", "first_byte", "tool_event"])
def test_exposed_or_ambiguous_operation_is_never_retried(exposure) -> None:
    assert RetryDecision.classify(transport_failure(), exposure, fixture_contract()) is False
```

Cover priority/weighted selection, scoped round-robin, sticky resources, account
cooldown, `Retry-After`, cancellation, per-attempt authorization/accounting,
WebSocket admission, strict-budget unknown cost, queue bounds, and reservation leak
checks.

- [ ] **Step 2: Verify routing/admission tests fail**

Run:
`python -m pytest tests/unified_gateway/test_routing.py tests/unified_gateway/test_retries.py tests/unified_gateway/test_admission.py -q`

- [ ] **Step 3: Implement selection and cooldown without policy evasion**

Candidates are filtered by grant, capability, source availability, billing,
residency, and affinity. Cooldowns are scoped to the provider quota key. The
router never switches owners for stateful work or rotates accounts after ambiguous
acceptance/output.

- [ ] **Step 4: Implement atomic reservations and integrate every generation path**

Reserve concurrency and estimated budget before credential acquisition. Finalize
against known upstream usage; retain `unknown` rather than inventing cost. Use the
same controller for HTTP, translated calls, fallback attempts, batch items, and
each WebSocket generation turn.

- [ ] **Step 5: Run focused and legacy budget/retry suites**

Run:

```console
python -m pytest tests/unified_gateway/test_routing.py tests/unified_gateway/test_retries.py tests/unified_gateway/test_admission.py -q
python -m pytest tests/test_proxy_budget.py tests/test_proxy_retry_429.py tests/test_h2_stream_reset_retry.py tests/test_upstream_write_timeout.py -q
ruff check headroom/proxy/gateway headroom/proxy/cost.py headroom/proxy/helpers.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/proxy/cost.py headroom/proxy/helpers.py headroom/proxy/server.py tests/unified_gateway
git commit -m "feat(proxy): add atomic gateway admission"
```

### Task 9: Implement secure control, reload, shutdown, and observability

**Files:**

- Create: `headroom/proxy/gateway/control.py`
- Create: `headroom/proxy/gateway/observability.py`
- Create: `headroom/proxy/gateway/runtime.py`
- Modify: `headroom/proxy/server.py`
- Modify: `headroom/proxy/prometheus_metrics.py`
- Modify: `headroom/proxy/persistent_metrics.py`
- Modify: `headroom/proxy/request_logger.py`
- Create: `tests/unified_gateway/test_control.py`
- Create: `tests/unified_gateway/test_reload.py`
- Create: `tests/unified_gateway/test_privacy.py`
- Create: `tests/unified_gateway/test_shutdown.py`

**Interfaces:**

- Produces `GatewayRuntime(snapshot, broker, registry, admission, resources)` with
  monotonic snapshot generation.
- Produces `GatewayRuntime.reload(path) -> ReloadResult` using validate-then-publish.
- Produces `GatewayRuntime.status(principal) -> RedactedGatewayStatus`.
- Registers gateway `GET /readyz` response exactly containing local service/profile
  identity plus bounded non-secret readiness fields.

- [ ] **Step 1: Write failing T056–T070 tests**

```python
def test_gateway_readyz_is_local_and_non_secret(gateway_client, spies) -> None:
    response = gateway_client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["service"] == "headroom"
    assert response.json()["profile"] == "gateway"
    assert "sentinel" not in response.text
    assert spies.credential_acquisitions == 0
    assert spies.upstream_requests == 0


async def test_invalid_reload_keeps_previous_snapshot(runtime, invalid_path) -> None:
    generation = runtime.generation
    result = await runtime.reload(invalid_path)
    assert result.applied is False
    assert runtime.generation == generation
    assert runtime.snapshot.routes[0].public_model == "public-gpt"
```

Add sentinel scans across errors/logs/metrics/traces/status, inference-vs-admin scope,
immutable in-flight snapshots, revocation, no runtime env mutation, unrelated
listener ownership, drain/cancel, and no beacon/payload persistence.

- [ ] **Step 2: Observe control/operations failures**

Run:
`python -m pytest tests/unified_gateway/test_control.py tests/unified_gateway/test_reload.py tests/unified_gateway/test_privacy.py tests/unified_gateway/test_shutdown.py -q`

- [ ] **Step 3: Implement runtime snapshots and scoped local control**

Publish validated snapshots atomically. Status/control requires a separate admin
scope and loopback origin. Readiness checks configuration only. Disable the
settings/runtime-env mutation endpoints under gateway profile unless an endpoint
is explicitly on the audited non-security allowlist.

- [ ] **Step 4: Add bounded, content-free gateway telemetry**

Labels are limited to ingress protocol, configured route class, adapter, credential
source kind, failure origin, retry reason, and terminal result. Pseudonymous account
references appear only in authenticated local status and are never secret-derived.

- [ ] **Step 5: Run control and observability regressions**

Run:

```console
python -m pytest tests/unified_gateway/test_control.py tests/unified_gateway/test_reload.py tests/unified_gateway/test_privacy.py tests/unified_gateway/test_shutdown.py -q
python -m pytest tests/test_proxy_health.py tests/test_proxy_healthchecks.py tests/test_observability_metrics.py tests/test_observability_tracing.py tests/test_graceful_shutdown.py tests/test_runtime_env.py -q
ruff check headroom/proxy/gateway headroom/proxy/server.py headroom/proxy/prometheus_metrics.py headroom/proxy/persistent_metrics.py headroom/proxy/request_logger.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/proxy/server.py headroom/proxy/prometheus_metrics.py headroom/proxy/persistent_metrics.py headroom/proxy/request_logger.py tests/unified_gateway
git commit -m "feat(proxy): operate gateway securely"
```

### Task 10: Add closed-by-default provider admission and OAuth machinery

**Files:**

- Create: `headroom/proxy/gateway/admission_catalog.py`
- Create: `headroom/proxy/gateway/oauth.py`
- Create: `headroom/cli/gateway_auth.py`
- Modify: `headroom/cli/__init__.py`
- Create: `tests/unified_gateway/test_provider_admission.py`
- Create: `tests/unified_gateway/test_oauth.py`
- Create: `tests/unified_gateway/test_device_flow.py`

**Interfaces:**

- Produces `ProviderAdmissionCatalog.check(provider, identity_kind,
  product_class, audience) -> AdmissionDecision`.
- Produces `BrowserAuthorizationTransaction` with PKCE S256, one-time state,
  issuer/account constraints, exact callback policy, deadline, and consume-once
  semantics.
- Produces `DeviceAuthorizationTransaction.poll(cancel_event) -> OAuthCredential`.
- CLI `headroom auth discover/status/login` reports unavailable adapters honestly;
  `login` cannot begin without an admitted provider flow.

- [ ] **Step 1: Write failing T086–T100 tests**

```python
@pytest.mark.parametrize(
    "provider",
    ["claude-subscription", "antigravity", "codex-native", "gemini-cli", "muse", "davin"],
)
def test_unadmitted_native_provider_is_not_advertised(provider, catalog) -> None:
    decision = catalog.check(provider, "native", "subscription", "inference")
    assert decision.admitted is False
    assert decision.public_capabilities == ()


def test_callback_state_is_consumed_once(transaction) -> None:
    credential = transaction.consume(valid_callback())
    assert credential.account_ref == "approved-account"
    with pytest.raises(OAuthReplayRejected):
        transaction.consume(valid_callback())
```

Cover metadata-only discovery, explicit import consent, restricted subscriptions,
agent-runtime rejection, no identifying-header impersonation, PKCE/state mismatch,
lookalike redirects, issuer/account mismatch, callback replay, port occupation,
device pending/slowdown/denial/expiry, and cancellation cleanup.

- [ ] **Step 2: Observe admission/OAuth failures**

Run:
`python -m pytest tests/unified_gateway/test_provider_admission.py tests/unified_gateway/test_oauth.py tests/unified_gateway/test_device_flow.py -q`

- [ ] **Step 3: Implement explicit admission catalog and unavailable CLI results**

API/workload identities for configured OpenAI, Anthropic API, Gemini API, Vertex,
Bedrock, and compatible origins are admitted by exact identity kind. All native
subscription entries are unavailable in this PR. Discovery returns metadata only.

- [ ] **Step 4: Implement reusable hostile-flow-safe OAuth transactions**

Generate state/verifier with `secrets`, derive S256 challenge, bind callback before
opening a browser, validate issuer/redirect/account/scope, consume state once, stop
listeners on every terminal path, and never store a credential without an admitted
adapter and explicit operator activation.

- [ ] **Step 5: Run focused and existing provider-auth tests**

Run:

```console
python -m pytest tests/unified_gateway/test_provider_admission.py tests/unified_gateway/test_oauth.py tests/unified_gateway/test_device_flow.py -q
python -m pytest tests/test_oauth_bearer_routing.py tests/test_copilot_auth.py tests/test_provider_codex_runtime.py -q
ruff check headroom/proxy/gateway headroom/cli/gateway_auth.py tests/unified_gateway
```

- [ ] **Step 6: Commit**

```console
git add headroom/proxy/gateway headroom/cli tests/unified_gateway
git commit -m "feat(auth): gate gateway provider identities"
```

### Task 11: Build real-process and SDK acceptance coverage

**Files:**

- Create: `tests/unified_gateway/process/__init__.py`
- Create: `tests/unified_gateway/process/harness.py`
- Create: `tests/unified_gateway/process/fake_upstream.py`
- Create: `tests/unified_gateway/process/fake_identity.py`
- Create: `tests/unified_gateway/process/test_cli_gateway.py`
- Create: `tests/unified_gateway/process/test_sdk_openai.py`
- Create: `tests/unified_gateway/process/test_sdk_anthropic.py`
- Create: `tests/unified_gateway/process/test_sdk_gemini.py`
- Create: `tests/unified_gateway/process/test_stream_faults.py`
- Create: `tests/unified_gateway/process/test_no_egress.py`
- Create: `tests/unified_gateway/scenario_coverage.py`
- Create: `tests/unified_gateway/test_scenario_coverage.py`

**Interfaces:**

- Produces `GatewayProcess` that launches the real `headroom` CLI with a private
  HOME, explicit environment allowlist, fake services, and egress deny-by-default.
- Produces a coverage manifest mapping every T001–T100 ID to one or more concrete
  test node IDs and an evidence level.

- [ ] **Step 1: Write failing process-oracle and coverage tests**

```python
def test_every_scenario_has_an_executable_or_explicit_external_cell() -> None:
    coverage = load_scenario_coverage()
    assert set(coverage) == {f"T{index:03d}" for index in range(1, 101)}
    assert all(cell.status in {"local_test", "external_not_run"} for cell in coverage.values())
    assert all(cell.test_nodes for cell in coverage.values() if cell.status == "local_test")


async def test_openai_sdk_stream_is_incremental(gateway_process, barrier_upstream) -> None:
    stream = await gateway_process.openai.responses.create(
        model="public-gpt", input="hello", stream=True
    )
    first = await anext(stream)
    assert first == fixture_first_event()
    assert barrier_upstream.completion_released is False
```

- [ ] **Step 2: Observe failures because the process harness/coverage map is absent**

Run:
`python -m pytest tests/unified_gateway/process tests/unified_gateway/test_scenario_coverage.py -q`

- [ ] **Step 3: Implement isolated real-process fixtures**

Launch fake upstream/identity sockets first, then the actual CLI. Pass only explicit
variables; exclude developer provider secrets. Block all non-fixture egress. Add
fault controls for delayed first byte, mid-frame stall, malformed terminal event,
fragmented UTF-8/tool arguments, abrupt EOF, redirect, TLS failure, backpressure,
ambiguous acceptance, and WebSocket disconnect.

- [ ] **Step 4: Exercise maintained SDK serializers and parsers**

Use OpenAI, Anthropic, and Google Gemini SDKs through their ordinary base URL and
API-key slots. Assert native/translated content, tools, errors, streaming, usage,
and cancellation through real sockets. Do not count constructor-only tests.

- [ ] **Step 5: Populate and validate T001–T100 coverage**

Only provider entitlement, real expiry, OS keychain, and official retained release
cells may be `external_not_run`; unavailable-capability rejection remains a local
test. Validate every mapped node with `pytest --collect-only` so stale names fail.

- [ ] **Step 6: Run process/SDK acceptance**

Run:

```console
python -m pytest tests/unified_gateway/process tests/unified_gateway/test_scenario_coverage.py -q
ruff check tests/unified_gateway/process tests/unified_gateway/scenario_coverage.py
```

- [ ] **Step 7: Commit**

```console
git add tests/unified_gateway
git commit -m "test(proxy): qualify unified gateway locally"
```

### Task 12: Qualify the locally retained artifact and prepare the draft PR

**Files:**

- Create: `scripts/verify_unified_gateway_artifact.py`
- Create: `docs/proposals/unified-api-gateway/LOCAL_QUALIFICATION.md`
- Create: `docs/proposals/unified-api-gateway/CAPABILITIES.json`
- Create: `docs/proposals/unified-api-gateway/PR_BODY.md`
- Modify: `README.md`
- Modify: `llms.txt`
- Create: `tests/unified_gateway/test_artifact_verifier.py`

**Interfaces:**

- Artifact verifier accepts `--wheel PATH --expected-sha256 DIGEST`, creates a
  clean temporary environment, installs exactly that wheel, launches its
  `headroom` entry point, and runs the retained smoke matrix.
- Qualification report identifies source SHA, wheel SHA-256, config/fixture
  digests, commands, pass/fail/skip counts, external `not_run` cells, and the known
  baseline failure.

- [ ] **Step 1: Write a failing artifact-verifier integration test**

```python
def test_artifact_verifier_rejects_a_digest_mismatch(built_wheel: Path) -> None:
    result = run_verifier(built_wheel, expected_sha256="0" * 64)
    assert result.returncode != 0
    assert "wheel digest mismatch" in result.stderr.lower()
```

- [ ] **Step 2: Observe failure because the verifier does not exist**

Run:
`python -m pytest tests/unified_gateway/test_artifact_verifier.py -q`

- [ ] **Step 3: Implement the verifier and user-facing documentation**

The verifier must use the installed wheel outside the source tree and refuse a
digest mismatch. Document configuration, client usage, supported identities,
unavailable native identities, pure-mode behavior, error codes, and external
qualification boundaries.

- [ ] **Step 4: Run focused static and test gates**

Run:

```console
python -m pytest tests/unified_gateway -q
ruff check headroom/proxy/gateway headroom/cli headroom/proxy headroom/providers tests/unified_gateway scripts/verify_unified_gateway_artifact.py
ruff format --check headroom/proxy/gateway headroom/cli headroom/proxy headroom/providers tests/unified_gateway scripts/verify_unified_gateway_artifact.py
python -m mypy headroom/proxy/gateway
cargo fmt --check
cargo test --workspace
```

- [ ] **Step 5: Run the full Python regression suite and classify only the recorded baseline failure**

Run:
`python -m pytest -q`

Expected: gateway tests and all non-baseline tests pass. Re-run the known
`ast-grep` failure separately and record its unchanged signature. Any other failure
must be fixed or explicitly demonstrated on the baseline commit before proceeding.

- [ ] **Step 6: Build and verify the exact local wheel**

Run:

```console
$gatewayWheel = Get-ChildItem dist\*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gatewayDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $gatewayWheel.FullName).Hash.ToLowerInvariant()
python scripts/verify_unified_gateway_artifact.py --wheel $gatewayWheel.FullName --expected-sha256 $gatewayDigest
```

Expected: build succeeds and the installed-artifact smoke matrix passes against the
exact digest.

- [ ] **Step 7: Complete a requirement-by-requirement audit**

Read the spec, requirements ledger, T001–T100 coverage manifest, capability matrix,
test output, artifact report, and final diff. Mark each item `proved locally`,
`external_not_run`, or `unavailable and negatively tested`; no requirement may be
missing or inferred from unrelated tests.

- [ ] **Step 8: Commit qualification artifacts**

```console
git add scripts/verify_unified_gateway_artifact.py docs/proposals/unified-api-gateway README.md llms.txt tests/unified_gateway/test_artifact_verifier.py
git commit -m "docs: qualify unified gateway draft"
```

- [ ] **Step 9: Push and open a draft pull request without merging**

```console
git push -u origin feat/unified-api-gateway
gh pr create --draft --base main --head feat/unified-api-gateway --title "feat(proxy): add unified API gateway" --body-file docs/proposals/unified-api-gateway/PR_BODY.md
gh pr view --json number,url,isDraft,headRefOid,baseRefName
```

The PR body must state the exact source/wheel digests, commands and counts, known
baseline failure, unsupported native providers, external live/final-release gates,
and `Human merge approval required: yes`. Do not enable auto-merge or merge the PR.
