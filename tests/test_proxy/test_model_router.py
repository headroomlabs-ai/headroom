"""Tests for cost-aware model routing (issue #1706)."""

from __future__ import annotations

from headroom.proxy.model_router import (
    ModelDecision,
    ModelRoute,
    ModelRouter,
    ModelRouterConfig,
    estimate_input_tokens,
)

# ---------------------------------------------------------------------------
# ModelRoute.matches
# ---------------------------------------------------------------------------


def test_route_matches_on_max_tokens_and_no_tools() -> None:
    route = ModelRoute(to_model="cheap", max_input_tokens=4000, require_no_tools=True)
    assert route.matches(model="strong", input_tokens=1000, has_tools=False)
    # too many tokens
    assert not route.matches(model="strong", input_tokens=5000, has_tools=False)
    # tools present
    assert not route.matches(model="strong", input_tokens=1000, has_tools=True)


def test_route_min_tokens() -> None:
    route = ModelRoute(to_model="strong", min_input_tokens=10000)
    assert route.matches(model="cheap", input_tokens=20000, has_tools=True)
    assert not route.matches(model="cheap", input_tokens=5000, has_tools=True)


def test_route_from_models_restriction() -> None:
    route = ModelRoute(to_model="cheap", from_models=("gpt-5.5", "gpt-5.4"))
    assert route.matches(model="gpt-5.5", input_tokens=1, has_tools=False)
    assert not route.matches(model="claude-sonnet-4-6", input_tokens=1, has_tools=False)


def test_route_matches_even_for_same_model() -> None:
    # A same-model rule still MATCHES (strict first-match-wins); it is a no-op
    # that short-circuits later rules, enabling explicit exemption rules.
    route = ModelRoute(to_model="cheap")
    assert route.matches(model="cheap", input_tokens=1, has_tools=False)


# ---------------------------------------------------------------------------
# ModelRouter.select
# ---------------------------------------------------------------------------


def _router(*routes: ModelRoute, enabled: bool = True) -> ModelRouter:
    return ModelRouter(ModelRouterConfig(enabled=enabled, routes=tuple(routes)))


def test_disabled_router_is_passthrough() -> None:
    router = _router(ModelRoute(to_model="cheap", max_input_tokens=10_000), enabled=False)
    d = router.select(model="strong", input_tokens=10, has_tools=False)
    assert not d.matched and not d.changed
    assert d.routed_model == "strong"


def test_first_matching_rule_wins() -> None:
    router = _router(
        ModelRoute(to_model="nano", max_input_tokens=2000, name="tiny"),
        ModelRoute(to_model="mini", max_input_tokens=8000, name="small"),
    )
    d = router.select(model="gpt-5.5", input_tokens=1500, has_tools=False)
    assert d.changed and d.routed_model == "nano" and d.rule_name == "tiny"

    d2 = router.select(model="gpt-5.5", input_tokens=5000, has_tools=False)
    assert d2.changed and d2.routed_model == "mini" and d2.rule_name == "small"


def test_exemption_rule_short_circuits_later_rules() -> None:
    # An explicit same-model rule wins first and stops a later downgrade rule.
    router = _router(
        ModelRoute(to_model="keep", from_models=("keep",), name="exempt"),
        ModelRoute(to_model="cheap", max_input_tokens=10_000, name="downgrade"),
    )
    d = router.select(model="keep", input_tokens=100, has_tools=False)
    assert d.matched and not d.changed
    assert d.routed_model == "keep" and d.rule_name == "exempt"


def test_no_rule_matches_is_passthrough() -> None:
    router = _router(ModelRoute(to_model="mini", max_input_tokens=1000))
    d = router.select(model="gpt-5.5", input_tokens=50_000, has_tools=True)
    assert not d.matched and not d.changed and d.routed_model == "gpt-5.5"
    assert d.reason == "no rule matched"


def test_empty_source_model_is_passthrough() -> None:
    router = _router(ModelRoute(to_model="mini"))
    d = router.select(model="", input_tokens=10, has_tools=False)
    assert not d.matched and d.routed_model == ""


def test_enabled_requires_routes() -> None:
    assert not ModelRouter(ModelRouterConfig(enabled=True, routes=())).enabled


# ---------------------------------------------------------------------------
# ModelDecision
# ---------------------------------------------------------------------------


def test_decision_changed_only_when_model_differs() -> None:
    assert ModelDecision("a", "b", matched=True, reason="x").changed
    assert not ModelDecision("a", "a", matched=True, reason="x").changed
    assert not ModelDecision("a", "b", matched=False, reason="x").changed


# ---------------------------------------------------------------------------
# ModelRouterConfig.from_env (fail-open parsing)
# ---------------------------------------------------------------------------


def test_from_env_disabled_by_default() -> None:
    cfg = ModelRouterConfig.from_env(None, None)
    assert not cfg.enabled and cfg.routes == ()


def test_from_env_parses_routes() -> None:
    routes = (
        '[{"name":"small","max_input_tokens":4000,"require_no_tools":true,'
        '"to_model":"gpt-5.4-mini","from_models":["gpt-5.5"]}]'
    )
    cfg = ModelRouterConfig.from_env("true", routes)
    assert cfg.enabled
    assert len(cfg.routes) == 1
    r = cfg.routes[0]
    assert r.to_model == "gpt-5.4-mini"
    assert r.max_input_tokens == 4000
    assert r.require_no_tools is True
    assert r.from_models == ("gpt-5.5",)


def test_from_env_enabled_but_no_routes_disables() -> None:
    cfg = ModelRouterConfig.from_env("true", None)
    assert not cfg.enabled


def test_from_env_malformed_json_fails_open() -> None:
    cfg = ModelRouterConfig.from_env("true", "{not json")
    assert not cfg.enabled and cfg.routes == ()


def test_from_env_non_array_json_ignored() -> None:
    cfg = ModelRouterConfig.from_env("true", '{"to_model":"x"}')
    assert cfg.routes == ()


def test_from_env_skips_bad_entries_keeps_good() -> None:
    routes = '[{"no_to_model":true}, {"to_model":"mini","max_input_tokens":"3000"}]'
    cfg = ModelRouterConfig.from_env("1", routes)
    assert len(cfg.routes) == 1
    assert cfg.routes[0].to_model == "mini"
    # numeric string coerced
    assert cfg.routes[0].max_input_tokens == 3000


def test_from_env_malformed_int_skips_route() -> None:
    # A bool or non-numeric token bound must fail open (skip the route), never
    # silently widen to "no cap".
    assert (
        ModelRouterConfig.from_env("yes", '[{"to_model":"m","max_input_tokens":true}]').routes == ()
    )
    assert (
        ModelRouterConfig.from_env("yes", '[{"to_model":"m","min_input_tokens":"abc"}]').routes
        == ()
    )


def test_from_env_malformed_require_no_tools_skips_route() -> None:
    # A string "false" must not be coerced to True.
    cfg = ModelRouterConfig.from_env("yes", '[{"to_model":"m","require_no_tools":"false"}]')
    assert cfg.routes == ()


def test_from_env_malformed_from_models_skips_route() -> None:
    assert (
        ModelRouterConfig.from_env("yes", '[{"to_model":"m","from_models":"gpt-5.5"}]').routes == ()
    )
    assert ModelRouterConfig.from_env("yes", '[{"to_model":"m","from_models":[1,2]}]').routes == ()


def test_from_env_negative_bound_skips_route() -> None:
    # A negative bound would match everything; it must fail open (skip the route).
    assert (
        ModelRouterConfig.from_env("yes", '[{"to_model":"m","min_input_tokens":-1}]').routes == ()
    )
    assert (
        ModelRouterConfig.from_env("yes", '[{"to_model":"m","max_input_tokens":-5}]').routes == ()
    )


def test_from_env_unknown_key_skips_route() -> None:
    # A misspelled condition key must not be silently ignored (which would widen
    # the rule to match everything).
    assert ModelRouterConfig.from_env("yes", '[{"to_model":"m","max_input_token":5}]').routes == ()
    assert ModelRouterConfig.from_env("yes", '[{"to_model":"m","typo":true}]').routes == ()


def test_from_env_valid_bool_and_ints_kept() -> None:
    cfg = ModelRouterConfig.from_env(
        "yes",
        '[{"to_model":"m","require_no_tools":false,"max_input_tokens":10,"min_input_tokens":0}]',
    )
    assert len(cfg.routes) == 1
    r = cfg.routes[0]
    assert r.require_no_tools is False and r.max_input_tokens == 10 and r.min_input_tokens == 0


def test_from_env_various_truthy_values() -> None:
    for v in ("1", "true", "YES", "on", "enabled"):
        assert ModelRouterConfig.from_env(v, '[{"to_model":"m"}]').enabled, v
    for v in ("0", "false", "", "off", None):
        assert not ModelRouterConfig.from_env(v, '[{"to_model":"m"}]').enabled


# ---------------------------------------------------------------------------
# estimate_input_tokens
# ---------------------------------------------------------------------------


def test_estimate_input_tokens_basic() -> None:
    messages = [{"role": "user", "content": "a" * 400}]
    assert estimate_input_tokens(messages) == 100


def test_estimate_input_tokens_includes_tools() -> None:
    with_tools = estimate_input_tokens([{"content": "x" * 40}], tools=[{"name": "y" * 40}])
    without = estimate_input_tokens([{"content": "x" * 40}])
    assert with_tools > without


def test_estimate_input_tokens_never_raises() -> None:
    assert estimate_input_tokens(None) == 0
    assert estimate_input_tokens("not a list") == 0
    assert estimate_input_tokens([123, {"content": "ok"}]) >= 0


def test_estimate_input_tokens_counts_system_string() -> None:
    # A large top-level system prompt must not be ignored.
    small = estimate_input_tokens([{"content": "hi"}])
    with_system = estimate_input_tokens([{"content": "hi"}], system="s" * 4000)
    assert with_system >= small + 900


def test_estimate_input_tokens_counts_system_blocks() -> None:
    blocks = [{"type": "text", "text": "x" * 4000}]
    assert estimate_input_tokens([{"content": "hi"}], system=blocks) > 100
