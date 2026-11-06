from __future__ import annotations

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.proxy.gateway.admission_catalog import ProviderAdmissionCatalog


@pytest.mark.parametrize(
    "provider",
    [
        "claude-subscription",
        "antigravity",
        "codex-native",
        "gemini-cli",
        "muse",
        "davin",
    ],
)
def test_unadmitted_native_provider_is_not_advertised(provider: str) -> None:
    decision = ProviderAdmissionCatalog().check(provider, "native", "subscription", "inference")
    assert decision.admitted is False
    assert decision.public_capabilities == ()


@pytest.mark.parametrize(
    ("provider", "identity_kind"),
    [
        ("openai", "api-key"),
        ("anthropic", "api-key"),
        ("gemini", "api-key"),
        ("vertex", "workload"),
        ("bedrock", "workload"),
        ("compatible", "api-key"),
    ],
)
def test_only_explicit_public_api_and_workload_identities_are_admitted(
    provider: str, identity_kind: str
) -> None:
    decision = ProviderAdmissionCatalog().check(provider, identity_kind, "public-api", "inference")
    assert decision.admitted is True
    assert decision.public_capabilities == ("inference",)


def test_identifying_header_impersonation_is_closed() -> None:
    decision = ProviderAdmissionCatalog().check(
        "openai", "identifying-header", "public-api", "inference"
    )
    assert decision.admitted is False
    assert decision.reason == "identity_not_admitted"


def test_auth_discovery_is_metadata_only_and_login_refuses_unadmitted_flow() -> None:
    runner = CliRunner()
    discovered = runner.invoke(main, ["auth", "discover"])
    assert discovered.exit_code == 0
    assert "openai api-key" in discovered.output
    assert "claude-subscription" not in discovered.output

    login = runner.invoke(
        main,
        ["auth", "login", "--provider", "claude-subscription", "--identity-kind", "native"],
    )
    assert login.exit_code != 0
    assert "not admitted" in login.output
