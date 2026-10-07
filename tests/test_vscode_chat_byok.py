"""VS Code Copilot Chat BYOK integration (`wrap vscode-chat`).

The proxy already serves the three API shapes VS Code's Custom Endpoint provider
can speak, so this layer is config generation plus file surgery on two files the
user also edits by hand. That makes the destructive-edit properties the important
ones: never clobber another provider, never claim an entry we did not write.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from headroom.providers.copilot import VSCODE_MODEL_ID_PREFIX
from headroom.providers.copilot.vscode_chat import (
    BYOK_ENABLED_SETTING,
    HEADROOM_PROVIDER_NAME,
    build_model_entries,
    build_provider_block,
    byok_entitlement_enabled,
    configure_chat_models,
    disable_byok_setting,
    enable_byok_setting,
    remove_chat_models,
)

FIXTURE = Path(__file__).parent / "fixtures" / "copilot_models" / "models_list.json"
BASE = "http://127.0.0.1:8787/p/proj"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def by_catalog_id(entries: list[dict]) -> dict[str, dict]:
    """Index entries by the *Copilot* model id, undoing the registration prefix.

    Tests care about which catalog model an entry describes; the prefix is a
    picker-visibility concern covered on its own below.
    """
    return {e["id"].removeprefix(VSCODE_MODEL_ID_PREFIX): e for e in entries}


# ---------------------------------------------------------------------------
# Entitlement preflight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("tid=abc;client_byok=1;chat=1:tok", True),
        ("tid=abc;client_byok=0;chat=1:tok", False),
        ("tid=abc;chat=1:tok", None),  # claim absent => unknown, not "denied"
        ("garbage", None),
    ],
)
def test_byok_entitlement_is_read_from_the_token(token: str, expected: bool | None) -> None:
    """VS Code hides the whole Custom Endpoint feature when the org disables BYOK.

    Distinguishing "denied" from "unknown" matters: denied should refuse at
    launch, unknown should proceed with a note rather than block a working setup.
    """
    assert byok_entitlement_enabled(token) is expected


# ---------------------------------------------------------------------------
# Model entry generation
# ---------------------------------------------------------------------------


def test_entries_cover_every_selectable_chat_model(payload: dict) -> None:
    entries = build_model_entries(payload, BASE)
    assert len(entries) == 22
    ids = set(by_catalog_id(entries))
    assert "claude-opus-4.8" in ids
    assert "gpt-5.4" in ids


def test_non_chat_and_unselectable_models_are_excluded(payload: dict) -> None:
    """Embeddings in a chat picker would be user-visible nonsense."""
    ids = set(by_catalog_id(build_model_entries(payload, BASE)))
    assert "text-embedding-3-small" not in ids
    assert "trajectory-compaction" not in ids  # picker-disabled
    assert "gpt-4o" not in ids  # picker-disabled


def test_registered_ids_never_collide_with_copilots_own(payload: dict) -> None:
    """The whole reason the prefix exists.

    VS Code's picker keys a model on its bare id and ignores the contributing
    provider, so an entry registered as ``claude-opus-5`` is treated as the same
    model as Copilot's native one and rendered once -- as the native entry. That
    silently removed the Headroom twin of every recently-used or GitHub-featured
    model, i.e. exactly the ones a user selects most.
    """
    catalog_ids = {m["id"] for m in payload["data"]}
    registered = {e["id"] for e in build_model_entries(payload, BASE)}
    assert registered.isdisjoint(catalog_ids)
    assert all(i.startswith(VSCODE_MODEL_ID_PREFIX) for i in registered)


def test_prefixed_ids_survive_the_round_trip_to_a_real_model(payload: dict) -> None:
    """A registered id must resolve back to the catalog id Copilot accepts.

    Asserted against the proxy's own resolver rather than by re-implementing the
    strip here: the two must not be able to drift apart.
    """
    from headroom.proxy.handlers.openai import resolve_copilot_model_id

    for entry in build_model_entries(payload, BASE):
        resolved = resolve_copilot_model_id(
            entry["id"], upstream_base_url="https://api.githubcopilot.com", cards=None
        )
        assert resolved == entry["id"].removeprefix(VSCODE_MODEL_ID_PREFIX)


def test_prefix_is_only_stripped_for_copilot_upstreams() -> None:
    """The strip is Copilot-gated, so no other upstream sees a rewritten model."""
    from headroom.proxy.handlers.openai import resolve_copilot_model_id

    prefixed = f"{VSCODE_MODEL_ID_PREFIX}claude-opus-5"
    assert (
        resolve_copilot_model_id(prefixed, upstream_base_url="https://api.openai.com", cards=None)
        == prefixed
    )
    # A bare prefix names no model; guessing one would send junk upstream.
    assert (
        resolve_copilot_model_id(
            VSCODE_MODEL_ID_PREFIX, upstream_base_url="https://api.githubcopilot.com", cards=None
        )
        == VSCODE_MODEL_ID_PREFIX
    )


def test_api_type_never_selects_messages(payload: dict) -> None:
    """The Anthropic wire rejects VS Code's placeholder key, so it is avoided.

    Claude models are reachable on chat-completions, so nothing is lost.
    """
    types = {e["apiType"] for e in build_model_entries(payload, BASE)}
    assert "messages" not in types
    assert types <= {"chat-completions", "responses"}


def test_responses_only_models_get_the_responses_wire(payload: dict) -> None:
    entries = by_catalog_id(build_model_entries(payload, BASE))
    assert entries["mai-code-1-flash-picker"]["apiType"] == "responses"
    assert entries["mai-code-1-flash-picker"]["url"].endswith("/v1/responses")
    assert entries["claude-opus-4.8"]["apiType"] == "chat-completions"
    assert entries["claude-opus-4.8"]["url"].endswith("/v1/chat/completions")


def test_capabilities_come_from_the_payload_not_hardcoded(payload: dict) -> None:
    """`toolCalling: false` hides a model from agent mode, so it must be real.

    Asserted by mutating the payload rather than by naming a model that happens to
    lack a capability today: every selectable model in the current live catalog
    advertises both tool calling and vision, so a fixture-value assertion would
    pass even if the fields were hardcoded.
    """
    entries = by_catalog_id(build_model_entries(payload, BASE))
    assert entries["claude-opus-4.8"]["toolCalling"] is True
    assert entries["claude-opus-4.8"]["vision"] is True
    assert entries["claude-opus-4.8"]["maxOutputTokens"] == 64000

    import copy

    mutated = copy.deepcopy(payload)
    for model in mutated["data"]:
        if model.get("id") == "claude-opus-4.8":
            model["capabilities"]["supports"]["tool_calls"] = False
            model["capabilities"]["supports"]["vision"] = False
            model["capabilities"]["limits"]["max_output_tokens"] = 1234
    changed = by_catalog_id(build_model_entries(mutated, BASE))
    assert changed["claude-opus-4.8"]["toolCalling"] is False
    assert changed["claude-opus-4.8"]["vision"] is False
    assert changed["claude-opus-4.8"]["maxOutputTokens"] == 1234


def test_urls_keep_the_project_prefix(payload: dict) -> None:
    """Per-project savings attribution rides in the base URL."""
    for entry in build_model_entries(payload, BASE):
        assert entry["url"].startswith("http://127.0.0.1:8787/p/proj/")


def test_empty_payload_yields_no_entries() -> None:
    assert build_model_entries({}, BASE) == []
    assert build_model_entries({"data": "nonsense"}, BASE) == []


# ---------------------------------------------------------------------------
# chatLanguageModels.json surgery
# ---------------------------------------------------------------------------


def _block(payload: dict, n: int = 2) -> dict:
    return build_provider_block(build_model_entries(payload, BASE)[:n])


def test_provider_is_added_to_an_empty_config(tmp_path: Path, payload: dict) -> None:
    p = tmp_path / "chatLanguageModels.json"
    assert configure_chat_models(p, _block(payload)) == "added"
    written = json.loads(p.read_text(encoding="utf-8"))
    assert len(written) == 1
    assert written[0]["vendor"] == "customendpoint"
    assert written[0]["name"] == HEADROOM_PROVIDER_NAME


def test_other_providers_are_preserved(tmp_path: Path, payload: dict) -> None:
    """The file is shared: a user's own providers must survive untouched."""
    p = tmp_path / "chatLanguageModels.json"
    mine = {"name": "My Ollama", "vendor": "customendpoint", "models": [{"id": "llama"}]}
    p.write_text(json.dumps([mine]), encoding="utf-8")
    configure_chat_models(p, _block(payload))
    written = json.loads(p.read_text(encoding="utf-8"))
    assert mine in written
    assert len(written) == 2


def test_refresh_replaces_in_place_without_duplicating(tmp_path: Path, payload: dict) -> None:
    p = tmp_path / "chatLanguageModels.json"
    configure_chat_models(p, _block(payload, 2))
    assert configure_chat_models(p, _block(payload, 3)) == "updated"
    written = json.loads(p.read_text(encoding="utf-8"))
    assert len(written) == 1, "refresh appended instead of replacing"
    assert len(written[0]["models"]) == 3


def test_a_same_named_provider_we_did_not_write_is_never_replaced(
    tmp_path: Path, payload: dict
) -> None:
    """Ownership is proven by digest, not by name.

    A user may copy our entry and edit it; silently overwriting their edits would
    be the same class of bug that marker-based ownership caused elsewhere.
    """
    p = tmp_path / "chatLanguageModels.json"
    theirs = {"name": HEADROOM_PROVIDER_NAME, "vendor": "customendpoint", "models": [{"id": "x"}]}
    p.write_text(json.dumps([theirs]), encoding="utf-8")
    with pytest.raises(click.ClickException, match="did not write"):
        configure_chat_models(p, _block(payload))
    assert json.loads(p.read_text(encoding="utf-8")) == [theirs]


def test_remove_takes_only_our_entry(tmp_path: Path, payload: dict) -> None:
    p = tmp_path / "chatLanguageModels.json"
    mine = {"name": "My Ollama", "vendor": "customendpoint", "models": []}
    p.write_text(json.dumps([mine]), encoding="utf-8")
    configure_chat_models(p, _block(payload))
    assert remove_chat_models(p) is True
    assert json.loads(p.read_text(encoding="utf-8")) == [mine]


def test_remove_is_a_noop_when_nothing_is_ours(tmp_path: Path) -> None:
    p = tmp_path / "chatLanguageModels.json"
    theirs = [{"name": "Someone else", "vendor": "customendpoint", "models": []}]
    p.write_text(json.dumps(theirs), encoding="utf-8")
    assert remove_chat_models(p) is False
    assert json.loads(p.read_text(encoding="utf-8")) == theirs


def test_malformed_config_is_refused_not_overwritten(tmp_path: Path, payload: dict) -> None:
    p = tmp_path / "chatLanguageModels.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(click.ClickException, match="Could not parse"):
        configure_chat_models(p, _block(payload))
    assert p.read_text(encoding="utf-8") == "{not json"


def test_json_object_instead_of_array_is_refused(tmp_path: Path, payload: dict) -> None:
    p = tmp_path / "chatLanguageModels.json"
    p.write_text('{"providers": []}', encoding="utf-8")
    with pytest.raises(click.ClickException, match="JSON array"):
        configure_chat_models(p, _block(payload))


def test_initial_add_is_never_written_without_its_ownership_record(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording ownership happens before the file is written, not after.

    With no record, a later run can never prove this entry is Headroom's:
    ``configure_chat_models`` would see the same provider name without a digest
    and raise the conflicting-provider error, and ``remove_chat_models`` would
    report nothing to remove -- both refresh and unwrap stuck until a hand edit.
    Refusing the write itself, before it happens, is the only thing that
    prevents that stuck state from ever existing.
    """
    from headroom.providers.copilot import vscode_chat as vscode_chat_module

    p = tmp_path / "chatLanguageModels.json"
    mine = {"name": "My Ollama", "vendor": "customendpoint", "models": []}
    original = json.dumps([mine])
    p.write_text(original, encoding="utf-8")

    # A file where the record's parent directory needs to be created makes
    # `mkdir` raise, simulating an unwritable workspace directory.
    blocker = tmp_path / "unwritable"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        vscode_chat_module, "_provenance_path", lambda path: blocker / "record.json"
    )

    with pytest.raises(click.ClickException, match="Could not create an ownership record"):
        configure_chat_models(p, _block(payload))

    assert p.read_text(encoding="utf-8") == original, (
        "chatLanguageModels.json was written even though its ownership record failed"
    )
    assert not (blocker / "record.json").exists(), "a record survived a failed write"


def test_refresh_is_never_written_without_its_ownership_record(
    tmp_path: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refresh that cannot re-prove ownership must leave the old entry provable.

    The record is keyed on the file path alone, not on the block's digest, so a
    refresh overwrites the *same* record the original add created. If the
    refresh's write then fails, restoring that record -- not just clearing it --
    is what keeps the still-current-on-disk (unrefreshed) entry removable and
    refreshable on the next run.
    """
    from headroom import fsutil
    from headroom.providers.copilot.vscode_chat import _read_provenance

    p = tmp_path / "chatLanguageModels.json"
    configure_chat_models(p, _block(payload, 2))
    original = p.read_text(encoding="utf-8")
    expected_before = _read_provenance(p)
    assert expected_before is not None

    real_write_text = fsutil.write_text

    def _fail_on_config_write(path: Path, content: str) -> None:
        if Path(path) == p:
            raise OSError("synthetic disk-full on the config write")
        real_write_text(path, content)

    monkeypatch.setattr(fsutil, "write_text", _fail_on_config_write)

    with pytest.raises(click.ClickException, match="Could not write"):
        configure_chat_models(p, _block(payload, 3))

    assert p.read_text(encoding="utf-8") == original, (
        "chatLanguageModels.json changed even though the refresh write failed"
    )
    assert _read_provenance(p) == expected_before, (
        "the ownership record no longer matches the entry actually on disk"
    )


# ---------------------------------------------------------------------------
# settings.json: the 1.132 visibility gate
# ---------------------------------------------------------------------------


def test_byok_setting_is_added_and_removed(tmp_path: Path) -> None:
    """Without this, 1.132+ silently hides every Custom Endpoint model."""
    p = tmp_path / "settings.json"
    p.write_text('{\n\t"editor.fontSize": 14\n}\n', encoding="utf-8")
    assert enable_byok_setting(p) == "added"
    text = p.read_text(encoding="utf-8")
    assert BYOK_ENABLED_SETTING in text
    assert '"editor.fontSize": 14' in text
    assert json.loads(_strip(text))[BYOK_ENABLED_SETTING] is True

    assert disable_byok_setting(p) is True
    after = p.read_text(encoding="utf-8")
    assert BYOK_ENABLED_SETTING not in after
    assert '"editor.fontSize": 14' in after


def test_a_user_false_value_is_reported_distinctly(tmp_path: Path) -> None:
    """`false` is the one value that silently yields zero visible models.

    It is also the setting's default and it is not in the Settings UI, so
    reporting it as "already configured" would send someone hunting for models
    that cannot appear. The user's value is still never overwritten.
    """
    p = tmp_path / "settings.json"
    p.write_text(f'{{\n\t"{BYOK_ENABLED_SETTING}": false\n}}\n', encoding="utf-8")
    assert enable_byok_setting(p) == "set to false by user"
    assert "false" in p.read_text(encoding="utf-8"), "the user's value was overwritten"


def test_byok_setting_respects_a_user_true_value(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text(f'{{\n\t"{BYOK_ENABLED_SETTING}": true\n}}\n', encoding="utf-8")
    assert enable_byok_setting(p) == "already set by user"


def test_byok_setting_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "settings.json"
    p.write_text("{}\n", encoding="utf-8")
    enable_byok_setting(p)
    assert enable_byok_setting(p) == "already set"
    assert p.read_text(encoding="utf-8").count(BYOK_ENABLED_SETTING) == 1


def _strip(text: str) -> str:
    import re

    from headroom.providers.copilot.vscode import _strip_jsonc_comments

    return re.sub(r",\s*([}\]])", r"\1", _strip_jsonc_comments(text))


def test_api_key_is_an_inert_literal_not_an_input_prompt(payload: dict) -> None:
    """An ``${input:...}`` variable can prompt the user to type a key.

    The proxy substitutes the real Copilot credential itself, so the value is
    never read — but a user who pastes a live key in response to that prompt would
    be putting a credential somewhere it serves no purpose. A fixed inert literal
    removes the prompt path entirely.
    """
    from headroom.providers.copilot.vscode_chat import PLACEHOLDER_API_KEY

    block = build_provider_block(build_model_entries(payload, BASE))
    assert block["apiKey"] == PLACEHOLDER_API_KEY
    assert "${input:" not in block["apiKey"]
    assert "unused" in PLACEHOLDER_API_KEY


def test_generated_config_contains_no_credential_material(payload: dict, tmp_path: Path) -> None:
    """Nothing token-shaped may reach a file the user might share."""
    p = tmp_path / "chatLanguageModels.json"
    configure_chat_models(p, build_provider_block(build_model_entries(payload, BASE)))
    text = p.read_text(encoding="utf-8")
    for pattern in ("gho_", "ghs_", "ghu_", "github_pat_", "sk-", "Bearer ", "tid=", "sku="):
        assert pattern not in text, f"{pattern!r} leaked into the generated config"


def test_responses_models_declare_zero_data_retention(payload: dict) -> None:
    """Without this, every /responses model 400s on first use.

    VS Code sets `store: !zeroDataRetentionEnabled` on each /responses request.
    Measured against the live API: `store: true` returns
    `400 store is not supported`, while `store: false` and an absent `store`
    both return 200 — so it is the `true` case that must be prevented.
    """
    entries = build_model_entries(payload, BASE)
    responses = [e for e in entries if e["apiType"] == "responses"]
    assert responses, "fixture should contain responses-only models"
    for entry in responses:
        assert entry.get("zeroDataRetentionEnabled") is True, entry["id"]
    for entry in (e for e in entries if e["apiType"] == "chat-completions"):
        assert "zeroDataRetentionEnabled" not in entry, "only the responses wire needs it"


def test_forced_store_false_is_lost_unless_the_body_is_marked_mutated() -> None:
    """Pins the forwarder contract the `store` rewrite depends on.

    The forwarder replays `original_body_bytes` verbatim for an unmutated body,
    so mutating `payload` in place is not enough on its own — the handler must
    also mark the body mutated or the rewrite is silently discarded and the
    request 400s.

    Scope: this constrains the forwarder, not the handler's call site. The
    end-to-end property (a `store: true` request actually reaching Copilot as
    `false`) is verified against the live API, not here.
    """
    from headroom.proxy.body_forwarding import select_outbound_body
    from headroom.proxy.handlers.openai import _ensure_chatgpt_responses_store_false

    body = {"model": "gpt-5.5", "store": True}
    original = json.dumps(body).encode()
    assert _ensure_chatgpt_responses_store_false(
        body, is_chatgpt_auth=False, is_copilot_upstream=True
    )
    assert body["store"] is False

    unmarked = select_outbound_body(
        body=body, original_body_bytes=original, body_mutated=False, forwarder_mode="canonical"
    )
    assert json.loads(unmarked.content)["store"] is True  # the rewrite is lost

    marked = select_outbound_body(
        body=body, original_body_bytes=original, body_mutated=True, forwarder_mode="canonical"
    )
    assert json.loads(marked.content)["store"] is False


def test_store_rewrite_is_scoped_to_the_upstreams_that_need_it() -> None:
    """A plain OpenAI/custom upstream must keep whatever the client sent."""
    from headroom.proxy.handlers.openai import _ensure_chatgpt_responses_store_false

    untouched = {"model": "gpt-5.5", "store": True}
    assert not _ensure_chatgpt_responses_store_false(
        untouched, is_chatgpt_auth=False, is_copilot_upstream=False
    )
    assert untouched["store"] is True


# ---------------------------------------------------------------------------
# Sharing one proxy between the Copilot CLI and VS Code
# ---------------------------------------------------------------------------


def test_same_account_shares_a_proxy_but_a_different_one_never_does() -> None:
    """`wrap copilot --native` and `wrap vscode-chat` are one user, one proxy.

    Copilot seeds are per-session, so any running proxy used to force a wrap
    session onto its own port — which split a single account's two surfaces
    across two proxies, two dashboards and two halves of the savings. Sharing is
    now allowed, but *only* on a proven identity match.
    """
    from headroom.cli.wrap import _proxy_serves_same_copilot_seed
    from headroom.copilot_auth import token_fingerprint

    mine = "gho_my_oauth_token"
    running = {"copilot_token_fingerprint": token_fingerprint(mine)}

    assert _proxy_serves_same_copilot_seed(running, mine) is True
    assert _proxy_serves_same_copilot_seed(running, "gho_someone_elses_token") is False


@pytest.mark.parametrize(
    ("running_config", "token", "why"),
    [
        (None, "gho_tok", "no config could be read"),
        ({}, "gho_tok", "proxy predates the fingerprint field"),
        ({"copilot_token_fingerprint": None}, "gho_tok", "proxy has no seed"),
        ({"copilot_token_fingerprint": ""}, "gho_tok", "empty fingerprint"),
        ({"copilot_token_fingerprint": 123}, "gho_tok", "non-string fingerprint"),
        ({"copilot_token_fingerprint": "sha256:abc"}, None, "no oauth token to compare"),
        ({"copilot_token_fingerprint": "sha256:abc"}, "", "empty oauth token"),
    ],
)
def test_unknown_identity_never_counts_as_a_match(
    running_config: dict | None, token: str | None, why: str
) -> None:
    """Fails closed: anything short of a proven match keeps sessions isolated.

    Guessing wrong here would send one account's traffic upstream under
    another's credential, so "unknown" must behave exactly like "different".
    """
    from headroom.cli.wrap import _proxy_serves_same_copilot_seed

    assert _proxy_serves_same_copilot_seed(running_config, token) is False, why


# ---------------------------------------------------------------------------
# CAPI routing — the mechanism that also covers agent-chosen models
# ---------------------------------------------------------------------------


def test_deferral_is_skipped_when_any_name_cannot_be_a_namespace() -> None:
    """Injecting the search tool namespaces the WHOLE array, not just deferred tools.

    In that mode a function name is read as ``namespace.function``: it may not
    contain a second "." and may not occupy a namespace reserved for a built-in
    tool type. VS Code Copilot Chat names tools ``file_search.file_search``,
    which breaks both rules, and the upstream rejects the entire request --
    every tool lost, whether or not that tool was the one deferred.

    So leaving the offending tool merely resident is not enough; the mode itself
    is what validates. The optimisation has to be skipped wholesale, which keeps
    the request byte-identical to what the client would have sent alone.
    """
    from headroom.proxy.helpers import inject_tool_search_deferral_openai

    def tool(name: str) -> dict:
        return {"type": "function", "name": name, "parameters": {"type": "object"}}

    plain = [tool(f"vscode_tool_{i}") for i in range(20)]

    # Untouched: returning the caller's own list is the signal that nothing was
    # injected, so the request goes upstream exactly as the client built it.
    for offender in ("file_search.file_search", "mcp.server.do_thing", "file_search"):
        tools = [*plain, tool(offender)]
        assert inject_tool_search_deferral_openai(tools, "gpt-5.5", client=None) is tools, offender

    # Still applies when every name can legally be a namespace.
    out = inject_tool_search_deferral_openai(plain, "gpt-5.5", client=None)
    assert out is not plain
    assert any(t.get("type") == "tool_search" for t in out if isinstance(t, dict))
    assert any(t.get("defer_loading") for t in out if isinstance(t, dict))


def test_tool_search_deferral_is_scoped_to_a_real_anthropic_upstream() -> None:
    """Deferring tools against Copilot silently disarms the client's whole toolset.

    Anthropic's tool-search protocol is first-party only. Both `wrap copilot
    --native` and the VS Code CAPI redirect point this wire at the Copilot host
    so Claude models work there, and such requests still arrive as provider
    "anthropic" -- so gating on the route name alone fired the deferral against
    an API that does not implement it.

    The damage is not partial: the core-tool allowlist is Claude Code's names,
    which match none of VS Code's, so every tool was marked `defer_loading` and
    the agent correctly reported it had no subagent tool to call.
    """
    from headroom.proxy.handlers.anthropic import _is_anthropic_upstream

    assert _is_anthropic_upstream("https://api.anthropic.com") is True
    assert _is_anthropic_upstream(None) is True
    # The configurations this bug was reported from:
    assert _is_anthropic_upstream("https://api.githubcopilot.com") is False
    assert _is_anthropic_upstream("https://copilot-api.enterprise.ghe.com") is False


def test_capi_override_round_trips_and_leaves_other_settings_alone(tmp_path: Path) -> None:
    """Redirecting Copilot Chat's own API is what makes subagents compressed.

    BYOK only ever covered models a *human* picked from the picker, so a model
    the agent chose for a subagent silently ran on Copilot's uncompressed
    endpoint. Pointing the CAPI at the proxy covers both, because every endpoint
    the extension uses is derived from that one base URL.
    """
    from headroom.providers.copilot.vscode_chat import (
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    original = '{\n\t"editor.fontSize": 14,\n\t"telemetry.telemetryLevel": "off"\n}\n'
    settings.write_text(original, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:8970/p/proj") == "added"
    written = settings.read_text(encoding="utf-8")
    assert f'"{CAPI_OVERRIDE_SETTING}": "http://127.0.0.1:8970/p/proj"' in written
    assert '"editor.fontSize": 14' in written
    assert '"telemetry.telemetryLevel": "off"' in written

    assert enable_capi_override(settings, "http://127.0.0.1:8970/p/proj") == "already set"

    assert disable_capi_override(settings) is True
    # Byte-for-byte restoration: this file is the user's, not ours.
    assert settings.read_text(encoding="utf-8") == original
    assert disable_capi_override(settings) is False


def test_capi_override_never_overwrites_the_users_own_value(tmp_path: Path) -> None:
    """Someone pointing Copilot Chat at their own gateway means it.

    Silently repointing it would send their traffic somewhere they did not
    choose, and removing it on unwrap would break a setup Headroom never owned.
    """
    from headroom.providers.copilot.vscode_chat import (
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    theirs = f'{{\n\t"{CAPI_OVERRIDE_SETTING}": "https://my-gateway.example"\n}}\n'
    settings.write_text(theirs, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:8970") == "already set by user"
    assert settings.read_text(encoding="utf-8") == theirs
    assert disable_capi_override(settings) is False
    assert settings.read_text(encoding="utf-8") == theirs


@pytest.mark.parametrize(
    "seed",
    [
        None,  # absent: a fresh VS Code has no settings.json until you change a setting
        "{}\n",
        "{\n}\n",
        '{\n\t"editor.fontSize": 14,\n}\n',  # a legal JSONC trailing comma
    ],
    ids=["absent", "empty-object", "empty-multiline", "trailing-comma"],
)
def test_unwrap_recovers_from_any_starting_file(tmp_path: Path, seed: str | None) -> None:
    """Unwrap is the recovery path, so it must never be the thing that is stuck.

    Appending into an object with no properties gives the first block no
    separator and the second one a leading comma. Removing the first then
    promoted that comma to the object's first token: invalid JSON, so the write
    was refused -- and because the CLI aborted on the first failing step, nothing
    was removed and every later unwrap failed the same way. VS Code stayed
    pointed at a dead port with no way back but hand-editing.
    """
    from headroom.providers.copilot.vscode_chat import (
        CAPI_OVERRIDE_SETTING,
        disable_byok_setting,
        disable_capi_override,
        enable_byok_setting,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    if seed is not None:
        settings.write_text(seed, encoding="utf-8")

    enable_capi_override(settings, "http://127.0.0.1:8787/p/proj")
    enable_byok_setting(settings)

    # Removal in the order `unwrap` actually uses: routing first.
    assert disable_capi_override(settings) is True
    assert disable_byok_setting(settings) is True

    remaining = settings.read_text(encoding="utf-8")
    assert CAPI_OVERRIDE_SETTING not in remaining
    assert BYOK_ENABLED_SETTING not in remaining
    json.loads(remaining.replace(",\n}", "\n}"))  # still parses


def test_rerunning_with_a_different_url_rewrites_it(tmp_path: Path) -> None:
    """A stale URL is worse than no URL: chat points at a port nothing serves.

    The block outlives the session that wrote it, while the URL can change
    between runs -- the proxy may land on another port, and the `/p/<project>`
    prefix follows the launch directory. Matching on the marker alone left the
    old value in place while the caller reported the new one.
    """
    from headroom.providers.copilot.vscode_chat import enable_capi_override

    settings = tmp_path / "settings.json"
    settings.write_text('{\n\t"editor.fontSize": 14\n}\n', encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:8787/p/a") == "added"
    assert enable_capi_override(settings, "http://127.0.0.1:8787/p/a") == "already set"
    assert enable_capi_override(settings, "http://127.0.0.1:9999/p/b") == "updated"

    written = settings.read_text(encoding="utf-8")
    assert "http://127.0.0.1:9999/p/b" in written
    assert "8787" not in written, "the superseded URL is still in the file"
    assert '"editor.fontSize": 14' in written
    assert written.count("overrideCapiUrl") == 1, "the rewrite duplicated the setting"


def test_a_lone_start_marker_is_never_claimed(tmp_path: Path) -> None:
    """A half-applied Settings Sync merge leaves exactly this shape.

    Marker presence was treated as proof of ownership, so the update path ran
    remove-then-append: remove found no closing marker and did nothing, append
    saw the start marker and did nothing, both return values were discarded, and
    the caller reported "updated". The stale URL stayed live while the CLI told
    the user the new proxy was installed.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_START,
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    original = (
        "{\n\t"
        + _CAPI_MARKER_START
        + '\n\t"'
        + CAPI_OVERRIDE_SETTING
        + '": "http://127.0.0.1:1111"\n}\n'
    )
    settings.write_text(original, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:9999/p/new") == "not ours"
    assert settings.read_text(encoding="utf-8") == original, "an unowned block was rewritten"

    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == original


def test_a_user_authored_marker_pair_is_never_touched(tmp_path: Path) -> None:
    """Marker text is not proof: anyone can paste it, and Settings Sync copies it.

    Deleting between a pair we did not write removes whatever the user put
    there — verified before the fix: `editor.fontSize` and `my.setting` were
    both silently destroyed and the command reported success.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_END,
        _CAPI_MARKER_START,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    theirs = (
        "{\n\t"
        + _CAPI_MARKER_START
        + '\n\t"editor.fontSize": 14,\n\t"my.setting": true\n\t'
        + _CAPI_MARKER_END
        + "\n}\n"
    )
    settings.write_text(theirs, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:9999") == "not ours"
    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == theirs, "user settings were destroyed"


@pytest.mark.parametrize(
    ("shape", "body"),
    [
        # Our key sharing a physical line with the user's. A line-based check
        # passed this, so the whole line -- their settings included -- was
        # rewritten or deleted.
        ("keys sharing our line", '"{key}": "http://mine:1", "editor.fontSize": 14'),
        # A trailing comment is content we would have destroyed.
        ("trailing comment", '"{key}": "http://mine:1" // corporate gateway'),
        # Two keys, ours first.
        ("two keys", '"{key}": "http://mine:1",\n\t"my.setting": true'),
    ],
    ids=["keys-sharing-a-line", "trailing-comment", "two-keys"],
)
def test_a_block_holding_anything_else_is_refused(tmp_path: Path, shape: str, body: str) -> None:
    """None of these shapes are ours without a record, same as any other body.

    These predate the digest-only rule and were originally written against a
    line-based adoption check that accepted a line also carrying the user's
    settings, or a trailing comment -- both then rewritten or deleted, the exact
    loss the marker-ownership work exists to prevent. There is no adoption path
    left to check the body against, but the shapes stay: an unrecorded block is
    "not ours" regardless of what it holds, and these are still valid, if now
    redundant with that broader rule, regression coverage for what caused real
    damage before.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_END,
        _CAPI_MARKER_START,
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    original = (
        "{\n\t"
        + _CAPI_MARKER_START
        + "\n\t"
        + body.format(key=CAPI_OVERRIDE_SETTING)
        + "\n\t"
        + _CAPI_MARKER_END
        + "\n}\n"
    )
    settings.write_text(original, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:9999") == "not ours", shape
    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == original, f"{shape}: user content was destroyed"


def test_marker_text_inside_a_string_value_is_not_a_block(tmp_path: Path) -> None:
    """The PR comment said "quoted" markers, and quoting is the dangerous case.

    A substring scan finds our marker inside a setting's *value* and treats
    everything between the two occurrences as our block -- so a user storing the
    marker text as data lost every setting in between.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_END,
        _CAPI_MARKER_START,
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    original = (
        "{\n"
        f'\t"myext.header": "{_CAPI_MARKER_START}",\n'
        f'\t"{CAPI_OVERRIDE_SETTING}": "https://my-own-gateway:1234",\n'
        f'\t"myext.footer": "{_CAPI_MARKER_END}"\n'
        "}\n"
    )
    settings.write_text(original, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:9999") == "not ours"
    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == original


def test_a_recorded_digest_is_authoritative(tmp_path: Path) -> None:
    """Once we have a record, a mismatch refuses -- there is no fallback to fall back to.

    A pair the digest doesn't match is edited or foreign either way, so letting
    anything else -- shape, key, prior ownership -- override a mismatched record
    would make the digest decorative: an edited block would be silently
    overwritten, which is precisely what the reviewer asked us to fail closed
    on.
    """
    from headroom.providers.copilot.vscode_chat import (
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    settings.write_text('{\n\t"editor.fontSize": 14\n}\n', encoding="utf-8")
    assert enable_capi_override(settings, "http://127.0.0.1:8970") == "added"

    edited = settings.read_text(encoding="utf-8").replace("8970", "1234")
    settings.write_text(edited, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:8970") == "not ours"
    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == edited


def test_byok_markers_are_not_claimed_by_the_routing_key(tmp_path: Path) -> None:
    """A BYOK pair holding the routing key is still just an unrecorded pair.

    This predates the digest-only rule, when accepting either key for either
    pair let a BYOK-marked block be claimed on the strength of the routing key:
    `enable_byok_setting` then reported "already set" when the setting was
    absent, leaving every model hidden. The per-marker key check that caught it
    is gone -- a mismatched or absent record now refuses every unrecorded pair
    regardless of which key it holds, which subsumes it -- but the shape stays
    as regression coverage for that failure.
    """
    from headroom.providers.copilot.vscode_chat import (
        _BYOK_MARKER_END,
        _BYOK_MARKER_START,
        CAPI_OVERRIDE_SETTING,
        disable_byok_setting,
        enable_byok_setting,
    )

    settings = tmp_path / "settings.json"
    original = (
        "{\n\t"
        + _BYOK_MARKER_START
        + f'\n\t"{CAPI_OVERRIDE_SETTING}": "http://x:1"\n\t'
        + _BYOK_MARKER_END
        + "\n}\n"
    )
    settings.write_text(original, encoding="utf-8")

    assert enable_byok_setting(settings) == "not ours"
    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_byok_setting(settings)
    assert settings.read_text(encoding="utf-8") == original


def test_an_unrecorded_block_predating_provenance_is_never_adopted(tmp_path: Path) -> None:
    """No record means "not ours", even for a block Headroom itself once wrote.

    A digest-only proof locks out every install that predates provenance: their
    block is genuinely ours but unrecorded. Adopting it structurally instead --
    on the strength of the body being exactly our one setting -- is unsafe,
    because that shape is indistinguishable from a user's own pair (see
    ``test_an_exact_single_key_user_authored_pair_is_never_adopted`` below).
    Migrating a pre-provenance block therefore needs an explicit user-confirmed
    step or independent evidence, neither of which exists here, so the safe
    default is to leave the bytes alone and report "not ours".
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_START,
        _settings_provenance_path,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    settings.write_text('{\n\t"editor.fontSize": 14\n}\n', encoding="utf-8")
    assert enable_capi_override(settings, "http://127.0.0.1:1234") == "added"

    # A block written before provenance existed: our bytes, no record of them.
    _settings_provenance_path(settings, _CAPI_MARKER_START).unlink(missing_ok=True)
    unrecorded = settings.read_text(encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:8970") == "not ours"
    assert settings.read_text(encoding="utf-8") == unrecorded, "an unrecorded block was rewritten"

    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == unrecorded, "an unrecorded block was deleted"


def test_an_exact_single_key_user_authored_pair_is_never_adopted(tmp_path: Path) -> None:
    """The indistinguishable shape: a user-authored pair holding only our key.

    `_owns_settings_block()` used to structurally adopt any unrecorded, intact
    pair whose body was exactly ``{overrideCapiUrl: value}`` -- exactly what a
    user gets by copying a Headroom block into a new profile, or writing one by
    hand from documentation. `enable_capi_override` then rewrote their gateway
    and reported "updated"; `disable_capi_override` deleted their setting and
    reported success. Neither this pair's shape nor its single key is proof of
    who wrote it, so absent a recorded digest it must be refused and the bytes
    preserved -- covering both the enable and disable paths, as requested.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_END,
        _CAPI_MARKER_START,
        CAPI_OVERRIDE_SETTING,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    theirs = (
        "{\n\t"
        + _CAPI_MARKER_START
        + f'\n\t"{CAPI_OVERRIDE_SETTING}": "https://my-own-gateway:1234"\n\t'
        + _CAPI_MARKER_END
        + "\n}\n"
    )
    settings.write_text(theirs, encoding="utf-8")

    assert enable_capi_override(settings, "http://127.0.0.1:9999") == "not ours"
    assert settings.read_text(encoding="utf-8") == theirs, "a user-authored pair was rewritten"

    with pytest.raises(click.ClickException, match="cannot prove it wrote"):
        disable_capi_override(settings)
    assert settings.read_text(encoding="utf-8") == theirs, "a user-authored pair was deleted"


def test_a_new_block_is_never_written_without_its_ownership_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording ownership happens before the block is written, not after.

    With no fallback left to adopt an unrecorded pair, losing the record for a
    block Headroom itself just wrote would leave it stuck forever -- proven ours
    at the moment of writing, provable by nothing the moment after. Writing
    settings.json first and recording second (the original order) could not
    prevent that: only refusing the write itself, before it happens, can.
    """
    from headroom.providers.copilot import vscode_chat as vscode_chat_module
    from headroom.providers.copilot.vscode_chat import enable_capi_override

    settings = tmp_path / "settings.json"
    original = '{\n\t"editor.fontSize": 14\n}\n'
    settings.write_text(original, encoding="utf-8")

    # A file where the record's parent directory needs to be created makes
    # `mkdir` raise, simulating an unwritable workspace directory.
    blocker = tmp_path / "unwritable"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(
        vscode_chat_module,
        "_settings_provenance_path",
        lambda path, start: blocker / "record.json",
    )

    with pytest.raises(click.ClickException, match="Could not create an ownership record"):
        enable_capi_override(settings, "http://127.0.0.1:9999")

    assert settings.read_text(encoding="utf-8") == original, (
        "settings.json was written even though its ownership record failed"
    )


def test_duplicate_blocks_are_refused_not_half_removed(tmp_path: Path) -> None:
    """Removing one of two copies reports success while the setting stays live.

    Settings Sync merges and profile copies can duplicate a block, and a partial
    removal is the worst outcome: the user is told Copilot Chat was restored
    while it is still routed at a proxy that is about to stop.
    """
    from headroom.providers.copilot.vscode_chat import (
        _CAPI_MARKER_END,
        _CAPI_MARKER_START,
        disable_capi_override,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    enable_capi_override(settings, "http://127.0.0.1:8787")
    raw = settings.read_text(encoding="utf-8")
    block = raw[raw.find(_CAPI_MARKER_START) : raw.find(_CAPI_MARKER_END) + len(_CAPI_MARKER_END)]
    settings.write_text(raw.replace(block, f"{block}\n\t{block}"), encoding="utf-8")

    with pytest.raises(click.ClickException, match="copies of the Headroom block"):
        disable_capi_override(settings)


def test_capi_and_byok_blocks_are_independent(tmp_path: Path) -> None:
    """Both can be present; removing one must not disturb the other."""
    from headroom.providers.copilot.vscode_chat import (
        disable_byok_setting,
        disable_capi_override,
        enable_byok_setting,
        enable_capi_override,
    )

    settings = tmp_path / "settings.json"
    settings.write_text('{\n\t"editor.fontSize": 14\n}\n', encoding="utf-8")
    assert enable_capi_override(settings, "http://127.0.0.1:8970") == "added"
    assert enable_byok_setting(settings) == "added"

    assert disable_capi_override(settings) is True
    after = settings.read_text(encoding="utf-8")
    assert BYOK_ENABLED_SETTING in after, "removing CAPI routing took the BYOK block with it"
    assert '"editor.fontSize": 14' in after

    assert disable_byok_setting(settings) is True
    assert settings.read_text(encoding="utf-8") == '{\n\t"editor.fontSize": 14\n}\n'


def test_anthropic_key_is_never_forwarded_to_another_vendor() -> None:
    """The backstop that makes one shared proxy safe for Claude Code.

    The Anthropic handler forwards the client's own ``x-api-key`` unchanged, and
    the shared proxy's ``/v1/messages`` upstream is the Copilot host so the
    Copilot CLI can drive Claude models. Without this check, a Claude Code
    request that lost its upstream pin would hand the user's Anthropic key to
    GitHub (reproduced live: Copilot answers ``missing required Authorization
    header``, having already received it).
    """
    from headroom.proxy.handlers.anthropic import _is_anthropic_upstream

    assert _is_anthropic_upstream("https://api.anthropic.com") is True
    assert _is_anthropic_upstream(None) is True  # unset means the default
    assert _is_anthropic_upstream("https://api.githubcopilot.com") is False
    # Host-based, so a lookalike cannot smuggle the real host into a path or
    # userinfo segment and be mistaken for Anthropic.
    assert _is_anthropic_upstream("https://evil.example.com/api.anthropic.com") is False
    assert _is_anthropic_upstream("https://api.anthropic.com.evil.example") is False
    assert _is_anthropic_upstream("https://api.anthropic.com@evil.example") is False


def test_claude_upstream_pin_only_fires_when_the_proxy_points_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin Claude Code's own upstream, but only when sharing needs it.

    Injecting the header unconditionally would put a redundant override on every
    ordinary single-client session; never injecting it makes the shared proxy
    unusable for Claude Code. It must key off what the running proxy actually
    reports.
    """
    import headroom.cli.wrap as wrap_mod

    def fake_proxy(config: dict | None):
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda port: {"config": config})
        monkeypatch.setattr(wrap_mod, "_proxy_health_config", lambda payload: config)

    # Shared proxy pinned at Copilot -> pin this client back to Anthropic.
    fake_proxy({"anthropic_api_url": "https://api.githubcopilot.com"})
    env: dict[str, str] = {}
    assert wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970) == "https://api.anthropic.com"
    assert "X-Headroom-Base-Url: https://api.anthropic.com" in env["ANTHROPIC_CUSTOM_HEADERS"]

    # An ordinary Anthropic-pointed proxy needs no override.
    fake_proxy({"anthropic_api_url": "https://api.anthropic.com"})
    env = {}
    assert wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970) is None
    assert env == {}

    # No proxy running, nothing to share with.
    fake_proxy(None)
    env = {}
    assert wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970) is None
    assert env == {}

    # A user's own override always wins, and existing headers are preserved.
    fake_proxy({"anthropic_api_url": "https://api.githubcopilot.com"})
    env = {"ANTHROPIC_CUSTOM_HEADERS": "X-Headroom-Base-Url: https://my-gateway.example"}
    assert wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970) is None
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Headroom-Base-Url: https://my-gateway.example"

    fake_proxy({"anthropic_api_url": "https://api.githubcopilot.com"})
    env = {"ANTHROPIC_CUSTOM_HEADERS": "X-Headroom-Project: demo"}
    assert wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970) == "https://api.anthropic.com"
    assert "X-Headroom-Project: demo" in env["ANTHROPIC_CUSTOM_HEADERS"]
    assert "X-Headroom-Base-Url: https://api.anthropic.com" in env["ANTHROPIC_CUSTOM_HEADERS"]


def test_claude_upstream_pin_targets_the_upstream_this_session_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user's own gateway (#1358) must never be bypassed by the shared-proxy pin."""
    import headroom.cli.wrap as wrap_mod

    gateway = "https://litellm.example.internal/anthropic"

    def fake_proxy(config: dict | None):
        monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda port: {"config": config})
        monkeypatch.setattr(wrap_mod, "_proxy_health_config", lambda payload: config)

    # The proxy already forwards to the gateway this session asked for.
    fake_proxy({"anthropic_api_url": gateway})
    env: dict[str, str] = {}
    assert (
        wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970, requested_upstream=gateway)
        is None
    )
    assert env == {}

    # A Copilot-pinned shared proxy: pin to the gateway, not to api.anthropic.com.
    fake_proxy({"anthropic_api_url": "https://api.githubcopilot.com"})
    env = {}
    assert (
        wrap_mod._apply_anthropic_upstream_pin_env(env, port=8970, requested_upstream=gateway)
        == gateway
    )
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == f"X-Headroom-Base-Url: {gateway}"


def test_schema_required_fields_are_always_present(payload: dict) -> None:
    """A model missing these yields NaN token limits rather than a visible error."""
    for entry in build_model_entries(payload, BASE):
        for field in ("id", "name", "url", "apiType", "toolCalling", "vision"):
            assert field in entry, f"{entry.get('id')} missing {field}"
        assert isinstance(entry["maxInputTokens"], int) and entry["maxInputTokens"] > 0
        assert isinstance(entry["maxOutputTokens"], int) and entry["maxOutputTokens"] > 0


def test_required_limits_survive_a_catalog_without_them(payload: dict) -> None:
    """Emit defaults rather than omitting a schema-required field."""
    import copy

    mutated = copy.deepcopy(payload)
    for model in mutated["data"]:
        if model.get("id") == "claude-opus-4.8":
            model["capabilities"]["limits"] = {}
    entry = by_catalog_id(build_model_entries(mutated, BASE))["claude-opus-4.8"]
    assert entry["maxInputTokens"] > 0
    assert entry["maxOutputTokens"] > 0


def test_context_window_is_written_from_the_catalog(payload: dict) -> None:
    """VS Code otherwise derives it as input+output, which understates some models."""
    entries = by_catalog_id(build_model_entries(payload, BASE))
    assert entries["gpt-5-mini"]["contextWindow"] == 264000
