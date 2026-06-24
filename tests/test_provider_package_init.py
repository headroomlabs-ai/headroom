from __future__ import annotations

import importlib
import types

import pytest

import headroom.providers as providers
from headroom.install.models import ManagedMutation
from headroom.providers import install_registry


def test_providers_package_resolves_exports_lazily_and_caches_them(monkeypatch) -> None:
    module = importlib.reload(providers)
    module.__dict__.pop("OpenAIProvider", None)
    sentinel = object()
    import_calls: list[str] = []

    def fake_import_module(name: str):
        import_calls.append(name)
        return types.SimpleNamespace(OpenAIProvider=sentinel)

    monkeypatch.setattr(module, "import_module", fake_import_module)

    try:
        assert module.OpenAIProvider is sentinel
        assert module.OpenAIProvider is sentinel
        assert import_calls == ["headroom.providers.openai"]
        assert "OpenAIProvider" in module.__dir__()
    finally:
        module.__dict__.pop("OpenAIProvider", None)


def test_providers_package_rejects_missing_and_dunder_path_attributes() -> None:
    module = importlib.reload(providers)

    with pytest.raises(AttributeError, match="__path__"):
        module.__getattr__("__path__")

    with pytest.raises(AttributeError, match="does_not_exist"):
        module.__getattribute__("does_not_exist")


def test_install_registry_build_install_target_envs_uses_known_targets_only(monkeypatch) -> None:
    monkeypatch.setattr(
        install_registry,
        "_ENV_BUILDERS",
        {
            "claude": lambda *, port, backend: {"CLAUDE_PORT": f"{port}:{backend}"},
            "cursor": lambda *, port, backend: {"CURSOR_PORT": f"{port}:{backend}"},
        },
    )

    envs = install_registry.build_install_target_envs(
        port=8787,
        backend="anthropic",
        targets=["claude", "unknown", "cursor"],
    )

    assert envs == {
        "claude": {"CLAUDE_PORT": "8787:anthropic"},
        "cursor": {"CURSOR_PORT": "8787:anthropic"},
    }


def test_install_registry_apply_provider_scope_mutations_skips_missing_and_none(
    monkeypatch,
) -> None:
    manifest = types.SimpleNamespace(targets=["claude", "codex", "unknown"])
    codex_mutation = ManagedMutation(target="codex", kind="toml-block")

    monkeypatch.setattr(
        install_registry,
        "_PROVIDER_SCOPE_HANDLERS",
        {
            "claude": (lambda _manifest: None, lambda mutation, manifest: None),
            "codex": (lambda _manifest: codex_mutation, lambda mutation, manifest: None),
        },
    )

    mutations = install_registry.apply_provider_scope_mutations(manifest)

    assert mutations == [codex_mutation]


def test_install_registry_revert_provider_scope_mutation_dispatches_known_targets(
    monkeypatch,
) -> None:
    manifest = types.SimpleNamespace()
    mutation = ManagedMutation(target="codex", kind="toml-block")
    recorded: list[tuple[ManagedMutation, object]] = []

    monkeypatch.setattr(
        install_registry,
        "_PROVIDER_SCOPE_HANDLERS",
        {
            "codex": (
                lambda _manifest: None,
                lambda incoming_mutation, incoming_manifest: recorded.append(
                    (incoming_mutation, incoming_manifest)
                ),
            )
        },
    )

    install_registry.revert_provider_scope_mutation(manifest, mutation)
    install_registry.revert_provider_scope_mutation(
        manifest, ManagedMutation(target="unknown", kind="noop")
    )

    assert recorded == [(mutation, manifest)]
