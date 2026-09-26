"""A-10: the `sandbox` extra promised isolation it never provided, and
SECURITY.md drifted four minor releases behind the code.

Both are documentation defects with a security consequence: a reader who
believes either one makes a deployment decision on a false premise. These tests
pin the rename's compatibility alias and make the stale version table — the part
that silently rots — a build failure instead of a reviewer's job.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import tomllib

_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_sandbox_alias_still_resolves_to_the_lean_profile() -> None:
    """The old name must keep working for one release.

    `sandbox` was renamed to `lean`; anything already installing the old extra
    (our own python-314-wheels CI job did) must not break on upgrade.
    """
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "lean" in extras, "the renamed extra is missing"
    assert "sandbox" in extras, "the deprecated alias was removed too early"
    assert extras["sandbox"] == ["headroom-ai[lean]"], (
        "the alias must forward to `lean` rather than duplicate its dependency "
        "list, or the two will drift"
    )


def test_lean_profile_stays_torch_free() -> None:
    """The profile's whole purpose is avoiding torch; guard it by name.

    A torch-pulling extra added here would quietly defeat the reason anyone
    selects this profile.
    """
    extras = _pyproject()["project"]["optional-dependencies"]
    (spec,) = extras["lean"]
    selected = set(re.search(r"\[(.*)\]", spec).group(1).split(","))

    torch_pulling = {"ml", "memory", "evals", "image", "voice", "pytorch-mps"}
    assert not (selected & torch_pulling), (
        f"`lean` must stay torch-free; it now selects {sorted(selected & torch_pulling)}"
    )


def test_security_policy_supported_versions_match_the_shipped_version() -> None:
    """SECURITY.md's table sat at 0.27.x while the code shipped 0.38.0.

    A supported-versions table that lags is worse than none: it tells a reporter
    their release is unsupported when it is the current one.
    """
    version = _pyproject()["project"]["version"]
    series = ".".join(version.split(".")[:2]) + ".x"

    text = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert f"| {series} (latest) |" in text, (
        f"SECURITY.md does not list {series} as the supported series "
        f"(pyproject version is {version})"
    )
    assert f"| < {series} |" in text, f"SECURITY.md does not mark < {series} unsupported"


def test_security_policy_does_not_claim_blanket_no_credential_storage() -> None:
    """`headroom copilot login` writes an OAuth refresh token to disk.

    The policy used to say "We never store or log API keys", which is broader
    than the code supports. Keep the honest wording.
    """
    text = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "never store or log API keys" not in text, (
        "the blanket no-credential-storage claim is contradicted by "
        "headroom/copilot_auth.py, which persists the Copilot OAuth refresh token"
    )
    assert "copilot_auth.json" in text, "the one stored credential must stay documented"


# --- The persistent-install credential path -------------------------------
#
# The assertions above are negative string checks: they stop the *old* wording
# coming back, but they would happily pass while the replacement wording is
# itself false (review of PR #3727 — the first correction claimed Headroom
# "does not write [API keys] to disk", which `headroom install --env` falsifies).
#
# So the tests below drive the real install path with a provider key, read what
# actually lands in `~/.headroom/deploy/<profile>/`, and compare it against the
# claims parsed out of SECURITY.md. A change to either side that is not matched
# on the other fails here.

_SECRET = "sk-ant-api03-not-a-real-key"

_MANIFEST_MODE_CLAIM = re.compile(
    r"`~/\.headroom/deploy/<profile>/manifest\.json` \(mode `(0[0-7]{3})`\)"
)
_SCRIPT_MODE_CLAIM = re.compile(
    r"`run-headroom\.sh` / `ensure-headroom\.sh` \(mode `(0[0-7]{3})`\)"
)
_DIR_MODE_CLAIM = re.compile(r"in the same directory \(mode `(0[0-7]{3})`\)")


def _security_policy_prose() -> str:
    """SECURITY.md with its line wrapping flattened, so claims match as written."""
    return re.sub(r"\s+", " ", (_ROOT / "SECURITY.md").read_text(encoding="utf-8"))


def _claimed_mode(pattern: re.Pattern[str]) -> int:
    match = pattern.search(_security_policy_prose())
    assert match is not None, (
        f"SECURITY.md no longer states the mode matched by {pattern.pattern!r}. The "
        "persistent-install credential path must keep documenting where a key "
        "passed to `headroom install --env` is written and how it is protected."
    )
    return int(match.group(1), 8)


def _install_with_a_provider_key(home: Path) -> tuple[Path, Path, Path]:
    """Run the real planner/state/supervisor path with a key in ``--env``.

    Returns the profile directory, its manifest and its foreground runner
    script. Imports are local so the module-level packaging assertions above
    still run if the install package fails to import.
    """
    from headroom.install.paths import manifest_path, profile_root, unix_run_script_path
    from headroom.install.planner import build_manifest
    from headroom.install.state import save_manifest
    from headroom.install.supervisors import render_runner_scripts

    assert Path.home() == home, "the fake HOME monkeypatch did not take effect"

    manifest = build_manifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="passthrough",
        memory_enabled=False,
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        extra_env={"ANTHROPIC_API_KEY": _SECRET},
    )
    save_manifest(manifest)
    render_runner_scripts(manifest)
    return profile_root("default"), manifest_path("default"), unix_run_script_path("default")


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits are advisory on Windows, where access is governed by ACLs",
)
def test_install_env_persists_a_provider_key_exactly_as_the_policy_describes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--env ANTHROPIC_API_KEY=...` really does write the key to disk.

    This is deliberate — supervisors start from a bare environment, so `--env`
    is the only channel that reaches a supervised proxy — but it means the
    policy may not claim keys are never written. Pin both halves: that the key
    lands where SECURITY.md says, and that it lands owner-only.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    profile_dir, manifest_file, run_script = _install_with_a_provider_key(tmp_path)

    # 1. What lands on disk. Both files carry the key in cleartext; the runner
    #    script `export`s it because launchd/systemd hand the process nothing.
    assert manifest_file.exists(), "a persistent install must write a manifest"
    assert _SECRET in manifest_file.read_text(encoding="utf-8"), (
        "the key passed to --env is stored verbatim in manifest.json; if that "
        "stopped being true, SECURITY.md and the --env help must change with it"
    )
    assert run_script.exists(), "a persistent install must write a runner script"
    assert f"export ANTHROPIC_API_KEY={_SECRET}" in run_script.read_text(encoding="utf-8")

    # 2. With what protection — read back out of the policy document, so the
    #    two cannot drift.
    assert manifest_file.stat().st_mode & 0o777 == _claimed_mode(_MANIFEST_MODE_CLAIM)
    assert run_script.stat().st_mode & 0o777 == _claimed_mode(_SCRIPT_MODE_CLAIM)
    assert profile_dir.stat().st_mode & 0o777 == _claimed_mode(_DIR_MODE_CLAIM)

    # 3. And, concretely: nothing in that directory is readable by group or
    #    other, whatever the umask was.
    for path in [profile_dir, *profile_dir.iterdir()]:
        assert not path.stat().st_mode & 0o077, f"{path.name} is readable beyond its owner"


def test_security_policy_discloses_the_persistent_install_credential_path() -> None:
    """The policy must name the `--env` path, not just the Copilot token."""
    prose = _security_policy_prose()

    assert "does not write them to disk" not in prose, (
        "`headroom install --env ANTHROPIC_API_KEY=...` writes the key to "
        "~/.headroom/deploy/<profile>/, so a blanket 'not written to disk' "
        "claim is false"
    )
    assert "headroom install --env KEY=VALUE" in prose, (
        "the supported way to hand a supervised proxy a provider key must be "
        "documented as a credential-persistence path"
    )
    assert "~/.headroom/deploy/<profile>/manifest.json" in prose
    assert "owner-only, not encrypted" in prose, (
        "owner-only file modes are not encryption; the policy must say so "
        "rather than let a reader infer the key is protected at rest"
    )
