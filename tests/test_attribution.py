"""Attribution labels: resolution, configuration, request binding, OTel export."""

from __future__ import annotations

import pytest

from headroom.proxy import attribution
from headroom.proxy.attribution import (
    AttributionRequest,
    bind_label,
    format_labels,
    get_current_attribution,
    merge_attribution,
    normalize_attribution,
    parse_dimension_spec,
    parse_labels,
    register_dimension,
    resolve_attribution,
    sanitize_dimension_name,
    set_current_attribution,
    unregister_dimension,
)

CLI_UA = "claude-cli/2.1.0 (external, cli)"


def _req(headers=None, **kwargs) -> AttributionRequest:
    return AttributionRequest(headers={k.lower(): v for k, v in (headers or {}).items()}, **kwargs)


@pytest.fixture(autouse=True)
def _clean_context(monkeypatch):
    monkeypatch.delenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, raising=False)
    set_current_attribution(None)
    yield
    set_current_attribution(None)


def test_default_dimensions_are_the_builtins():
    assert attribution.default_dimensions()[:3] == ("project", "client", "tenant")


def test_resolves_builtins_from_headers():
    labels = resolve_attribution(
        _req(
            {
                "User-Agent": CLI_UA,
                "X-Headroom-Project": "web%20app",
                "X-Headroom-Tenant-ID": "acme",
            },
            peer="192.168.0.12",
        )
    )
    assert labels == {"project": "web app", "client": "claude-code", "tenant": "acme"}


def test_absent_signals_leave_dimensions_out_rather_than_placeholder():
    labels = resolve_attribution(_req({}, peer=None))
    assert "project" not in labels
    assert "client" not in labels
    assert "tenant" not in labels


def test_path_prefix_project_is_used_when_no_header():
    labels = resolve_attribution(_req({}, path_project="aider-proj"), ["project"])
    assert labels == {"project": "aider-proj"}


def test_unregistered_dimension_reads_its_own_header():
    labels = resolve_attribution(_req({"X-Headroom-Cost-Center": "42"}), ["cost_center"])
    assert labels == {"cost_center": "42"}


def test_labels_header_supplies_listed_dimensions_and_direct_header_wins():
    request = _req({"X-Headroom-Labels": "team=infra; owner=sam", "X-Headroom-Owner": "kim"})
    assert resolve_attribution(request, ["team", "owner", "other"]) == {
        "team": "infra",
        "owner": "kim",
    }


def test_labels_header_names_unlisted_dimensions_only_with_open_token():
    request = _req({"X-Headroom-Labels": "team=infra, squad=blue"})
    assert resolve_attribution(request, ["team"]) == {"team": "infra"}
    assert resolve_attribution(request, ["labels"]) == {"team": "infra", "squad": "blue"}


def test_open_labels_cannot_shadow_a_registered_dimension():
    request = _req({"X-Headroom-Labels": "host=spoofed", "X-Headroom-Host": "real"})
    assert resolve_attribution(request, ["host", "labels"]) == {"host": "real"}


def test_open_labels_name_registered_dimensions_that_are_not_selected():
    request = _req({"X-Headroom-Labels": "project=web; team=infra"})
    assert resolve_attribution(request, ["labels"]) == {"project": "web", "team": "infra"}


def test_a_selected_resolver_has_the_last_word_over_open_labels():
    request = _req({"X-Headroom-Labels": "client=spoofed; tenant=spoofed", "User-Agent": CLI_UA})
    # ``client`` resolves; ``tenant`` is selected and finds nothing, which stands.
    assert resolve_attribution(request, ["client", "tenant", "labels"]) == {"client": "claude-code"}


def test_open_labels_never_readmit_a_removed_dimension(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels,-Project")
    request = _req({"X-Headroom-Labels": "project=web; team=infra"})
    assert resolve_attribution(request) == {"team": "infra"}
    assert attribution.excluded_dimensions() == frozenset({"project"})


def test_a_removed_dimension_is_not_bound_by_a_handler_either(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels,-team")
    bind_label("team", "infra")
    bind_label("squad", "blue")
    assert dict(get_current_attribution()) == {"squad": "blue"}


def test_open_labels_keep_an_extension_dimension_reserved():
    register_dimension("user", lambda r: "from-identity", default=False)
    register_dimension("region", lambda r: None, default=False, client_settable=True)
    try:
        request = _req({"X-Headroom-Labels": "user=mallory; region=eu"})
        # Not selected: only the client-settable name may come from the client.
        assert resolve_attribution(request, ["labels"]) == {"region": "eu"}
        # Selected: each resolver has the last word.
        assert resolve_attribution(request, ["user", "region", "labels"]) == {
            "user": "from-identity"
        }
    finally:
        unregister_dimension("user")
        unregister_dimension("region")


def test_spec_words_are_never_dimension_names():
    request = _req({"X-Headroom-Labels": "default=a; none=b; labels=c; team=infra"})
    assert resolve_attribution(request, ["labels"]) == {"team": "infra"}
    for word in ("default", "none", "labels"):
        with pytest.raises(ValueError):
            register_dimension(word, lambda r: "x")


def test_labels_header_is_filtered_before_it_is_capped(monkeypatch):
    filler = "; ".join(f"d{i}=x" for i in range(attribution.MAX_DIMENSIONS))
    request = _req({"X-Headroom-Labels": f"{filler}; team=infra"})
    # A listed dimension is found wherever it sits in the header.
    assert resolve_attribution(request, ["team"]) == {"team": "infra"}
    # Names that are filtered out do not use up the cap.
    removed = ",".join(f"-d{i}" for i in range(attribution.MAX_DIMENSIONS))
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, f"labels,{removed}")
    assert resolve_attribution(request) == {"team": "infra"}


def test_credential_headers_never_become_labels():
    request = _req({"X-Headroom-Proxy-Token": "secret", "X-Headroom-Api-Key": "k"})
    assert resolve_attribution(request, ["proxy-token", "api-key", "labels"]) == {}


def test_failing_resolver_is_skipped(monkeypatch):
    def boom(_request):
        raise RuntimeError("broken extension")

    register_dimension("flaky", boom)
    try:
        labels = resolve_attribution(_req({"User-Agent": CLI_UA}), ["flaky", "client"])
        assert labels == {"client": "claude-code"}
    finally:
        unregister_dimension("flaky")


def test_register_requires_replace_to_override():
    register_dimension("team_x", lambda r: "a")
    try:
        with pytest.raises(ValueError):
            register_dimension("team_x", lambda r: "b")
        register_dimension("team_x", lambda r: "b", replace=True)
        assert resolve_attribution(_req({}), ["team_x"]) == {"team_x": "b"}
    finally:
        unregister_dimension("team_x")


def test_extension_can_replace_a_builtin_resolver():
    register_dimension("client", lambda r: "from-identity", replace=True)
    try:
        assert resolve_attribution(_req({"X-Client": "codex"}), ["client"]) == {
            "client": "from-identity"
        }
        # What a handler knows about the harness no longer overrides it.
        set_current_attribution({"client": "from-identity"})
        attribution.bind_builtin_label("client", "codex")
        assert dict(get_current_attribution()) == {"client": "from-identity"}
    finally:
        unregister_dimension("client")
    # Unregistering a replaced built-in restores it.
    assert "client" in attribution.default_dimensions()
    assert resolve_attribution(_req({"X-Client": "codex"}), ["client"]) == {"client": "codex"}
    attribution.bind_builtin_label("client", "codex")
    assert dict(get_current_attribution()) == {"client": "codex"}


def test_a_handler_label_is_bound_only_for_an_enabled_dimension(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels")
    attribution.bind_builtin_label("project", "web")
    assert dict(get_current_attribution()) == {}


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, ("a", "b")),
        ("", ("a", "b")),
        ("default,team", ("a", "b", "team")),
        ("default,-a", ("b",)),
        ("team,a,team", ("team", "a")),
        ("none", ()),
        ("none,team", ("team",)),
        ("Bad Name!,ok", ("ok",)),
    ],
)
def test_parse_dimension_spec(spec, expected):
    assert parse_dimension_spec(spec, ("a", "b")) == expected


def test_spec_is_capped():
    spec = ",".join(f"d{i}" for i in range(40))
    assert len(parse_dimension_spec(spec, ())) == attribution.MAX_DIMENSIONS


def test_a_bad_spec_warns_once_not_per_request(monkeypatch, caplog):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "team,not a name!")
    attribution._parsed_spec.cache_clear()
    with caplog.at_level("WARNING", logger=attribution.__name__):
        for _ in range(3):
            assert attribution.enabled_dimensions() == ("team",)
    assert len([r for r in caplog.records if "not a name!" in r.getMessage()]) == 1


def test_env_selects_dimensions(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "host,team")
    labels = resolve_attribution(_req({"X-Headroom-Host": "a", "X-Headroom-Team": "t"}))
    assert labels == {"host": "a", "team": "t"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Host", "host"), ("tag.team", "tag.team"), ("a b", None), ("", None), ("x" * 33, None)],
)
def test_sanitize_dimension_name(name, expected):
    assert sanitize_dimension_name(name) == expected


def test_label_sanitizing_and_normalize():
    assert normalize_attribution({"Host": " box\x00 ", "bad key": "x", "empty": "  "}) == {
        "host": "box"
    }
    assert normalize_attribution("not a mapping") == {}
    long = normalize_attribution({"a": "y" * 500})
    assert len(long["a"]) == attribution.LABEL_MAX_LENGTH


def test_parse_and_format_labels_roundtrip():
    labels = parse_labels(" team = infra ;owner=sam,broken, =x")
    assert labels == {"team": "infra", "owner": "sam"}
    assert parse_labels(format_labels(labels)) == labels
    assert parse_labels(None) == {}
    # No escaping in the header: a label with a separator is left out, not cut.
    assert format_labels({"team": "design, ops", "owner": "sam"}) == "owner=sam"


def test_bind_label_updates_context_copy_on_write():
    set_current_attribution({"host": "a"})
    before = get_current_attribution()
    bind_label("client", "codex")
    bind_label("host", None)
    assert dict(get_current_attribution()) == {"client": "codex"}
    assert dict(before) == {"host": "a"}
    with pytest.raises(TypeError):
        get_current_attribution()["x"] = "y"  # type: ignore[index]


def test_merge_attribution_later_layers_win():
    assert merge_attribution({"a": "1", "b": "2"}, None, {"b": "3"}) == {"a": "1", "b": "3"}


def test_otel_provider_exports_selected_dimensions(monkeypatch):
    registered = []
    import headroom.observability.metrics as metrics

    monkeypatch.setattr(
        metrics, "register_otel_metric_attribute_provider", lambda p: registered.append(p) or p
    )
    monkeypatch.setattr(
        metrics, "unregister_otel_metric_attribute_provider", lambda p: registered.remove(p)
    )
    try:
        assert attribution.install_otel_attribution({}) is False
        assert registered == []
        assert attribution.install_otel_attribution({"HEADROOM_OTEL_ATTRIBUTION": "host"}) is True
        set_current_attribution({"host": "studio", "project": "web"})
        assert registered[0]() == {"headroom.attribution.host": "studio"}
        attribution.install_otel_attribution({"HEADROOM_OTEL_ATTRIBUTION": "all"})
        assert len(registered) == 1
        assert registered[0]() == {
            "headroom.attribution.host": "studio",
            "headroom.attribution.project": "web",
        }
    finally:
        attribution.install_otel_attribution({})
    assert registered == []


def test_a_header_label_is_percent_decoded_exactly_once(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "team,squad")
    request = _req({"X-Headroom-Team": "a%252541", "X-Headroom-Labels": "squad=b%252542"})
    labels = resolve_attribution(request, ["team", "squad"])
    assert labels == {"team": "a%2541", "squad": "b%2542"}
    # Every later stage leaves the resolved label alone.
    assert normalize_attribution(labels) == labels
    assert merge_attribution(labels, labels) == labels
    set_current_attribution({})
    bind_label("team", labels["team"])
    assert get_current_attribution()["team"] == "a%2541"


def test_spec_words_cannot_be_bound_or_merged_as_dimensions(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels")
    for word in ("labels", "default", "none"):
        bind_label(word, "x")
    assert dict(get_current_attribution()) == {}
    assert normalize_attribution({"labels": "x", "default": "y", "team": "infra"}) == {
        "team": "infra"
    }
    assert resolve_attribution(_req({"X-Headroom-Default": "x"}), ["default", "none"]) == {}


def test_merge_keeps_the_later_layers_when_over_the_cap():
    crowd = {f"d{i}": "x" for i in range(attribution.MAX_DIMENSIONS)}
    merged = merge_attribution(crowd, {"client": "codex", "project": "web"})
    assert len(merged) == attribution.MAX_DIMENSIONS
    assert (merged["client"], merged["project"]) == ("codex", "web")


def test_bind_label_stops_at_the_cap(monkeypatch):
    monkeypatch.setenv(attribution.ATTRIBUTION_DIMENSIONS_ENV_VAR, "labels")
    set_current_attribution({f"d{i}": "x" for i in range(attribution.MAX_DIMENSIONS)})
    bind_label("d0", "y")
    bind_label("extra", "z")
    labels = get_current_attribution()
    assert len(labels) == attribution.MAX_DIMENSIONS
    assert labels["d0"] == "y" and "extra" not in labels


def test_only_the_start_of_the_labels_header_is_read():
    padding = "x" * attribution.LABELS_HEADER_MAX_LENGTH
    request = _req({"X-Headroom-Labels": f"team=infra; pad={padding}; late=1"})
    assert resolve_attribution(request, ["team", "late"]) == {"team": "infra"}
