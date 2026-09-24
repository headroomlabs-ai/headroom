"""A deployment must pick up managed env vars added after it was installed.

``tool_envs`` is computed once, by ``build_manifest`` during ``headroom
install``, and stored on disk. Every lifecycle command after that re-applies the
STORED map, so anything added to a provider's install env later never reaches an
existing deployment -- not on ``start``, not on ``restart``, not on an upgrade.

The case that motivated this: ``ENABLE_TOOL_SEARCH`` joined Claude's install env
on 2026-06-19 (GH #746), because pointing Claude Code at a custom
``ANTHROPIC_BASE_URL`` makes it stop deferring MCP tool schemas and inline all of
them. A deployment installed before that date keeps getting the base URL written
without the mitigation -- the expensive half of the change, indefinitely.
"""

from __future__ import annotations

from headroom.cli.install import _reconcile_tool_envs, pending_tool_envs
from headroom.install.models import DeploymentManifest
from headroom.providers.claude import TOOL_SEARCH_DEFAULT, TOOL_SEARCH_ENV

CLAUDE = "claude"


def _manifest(tool_envs: dict[str, dict[str, str]]) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="local",
        runtime_kind="native",
        supervisor_kind="none",
        scope="provider",
        provider_mode="claude",
        targets=[CLAUDE],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        tool_envs=tool_envs,
    )


def test_pre_746_manifest_gains_the_tool_search_var() -> None:
    """The regression this exists for: base URL set, deferral never enabled."""
    manifest = _manifest({CLAUDE: {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}})

    added = _reconcile_tool_envs(manifest)

    assert TOOL_SEARCH_ENV in manifest.tool_envs[CLAUDE]
    assert manifest.tool_envs[CLAUDE][TOOL_SEARCH_ENV] == TOOL_SEARCH_DEFAULT
    assert added == {CLAUDE: [TOOL_SEARCH_ENV]}


def test_existing_values_are_never_overwritten() -> None:
    """A stored value may be deliberate; healing an omission is the smaller claim."""
    manifest = _manifest(
        {
            CLAUDE: {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9999",
                TOOL_SEARCH_ENV: "false",
            }
        }
    )

    added = _reconcile_tool_envs(manifest)

    assert manifest.tool_envs[CLAUDE]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9999"
    assert manifest.tool_envs[CLAUDE][TOOL_SEARCH_ENV] == "false"
    assert added == {}


def test_current_manifest_is_unchanged() -> None:
    """No churn, and nothing printed, for a deployment that is already correct."""
    manifest = _manifest({})
    _reconcile_tool_envs(manifest)
    current = {k: dict(v) for k, v in manifest.tool_envs.items()}

    manifest2 = _manifest(current)
    added = _reconcile_tool_envs(manifest2)

    assert added == {}
    assert manifest2.tool_envs == current


def test_reconcile_is_idempotent() -> None:
    manifest = _manifest({CLAUDE: {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}})

    first = _reconcile_tool_envs(manifest)
    second = _reconcile_tool_envs(manifest)

    assert first == {CLAUDE: [TOOL_SEARCH_ENV]}
    assert second == {}


def test_a_target_absent_from_the_manifest_is_populated() -> None:
    """An older manifest may predate the target entirely, not just one key."""
    manifest = _manifest({})

    added = _reconcile_tool_envs(manifest)

    assert CLAUDE in manifest.tool_envs
    assert TOOL_SEARCH_ENV in manifest.tool_envs[CLAUDE]
    assert added.get(CLAUDE)


def test_untargeted_tools_are_left_alone() -> None:
    """Reconciliation only covers targets this deployment actually manages."""
    manifest = _manifest({"someone-elses-tool": {"CUSTOM": "1"}})

    _reconcile_tool_envs(manifest)

    assert manifest.tool_envs["someone-elses-tool"] == {"CUSTOM": "1"}


def test_the_rebuilt_base_url_uses_the_manifest_port() -> None:
    """Rebuilding must not silently re-point a deployment at a default port."""
    manifest = _manifest({})
    manifest.port = 9123

    _reconcile_tool_envs(manifest)

    assert "9123" in manifest.tool_envs[CLAUDE]["ANTHROPIC_BASE_URL"]


def test_a_non_dict_entry_does_not_crash_a_lifecycle_command() -> None:
    """A hand-edited manifest must not turn ``install restart`` into a traceback.

    ``load_manifest`` does not type-check ``tool_envs``, so a string here
    reached ``stored.update(...)`` and raised AttributeError from a path with
    no handler.
    """
    manifest = _manifest({CLAUDE: "http://127.0.0.1:8787"})  # type: ignore[dict-item]

    added = _reconcile_tool_envs(manifest)

    assert isinstance(manifest.tool_envs[CLAUDE], dict)
    assert TOOL_SEARCH_ENV in manifest.tool_envs[CLAUDE]
    assert added.get(CLAUDE)


def test_a_target_with_no_managed_env_is_not_written() -> None:
    """No silent manifest churn: a no-op reconcile must not rewrite the file."""
    manifest = _manifest({})
    manifest.targets = [CLAUDE, "opencode"]

    _reconcile_tool_envs(manifest)

    assert "opencode" not in manifest.tool_envs


def test_pending_does_not_mutate() -> None:
    """The lifecycle commands ask before deciding; asking must be free."""
    manifest = _manifest({})
    before = {k: dict(v) for k, v in manifest.tool_envs.items()}

    pending = pending_tool_envs(manifest)

    assert pending  # there IS something to do
    assert manifest.tool_envs == before  # ...but nothing was done


def test_pending_is_empty_once_reconciled() -> None:
    manifest = _manifest({})
    _reconcile_tool_envs(manifest)

    assert pending_tool_envs(manifest) == {}
