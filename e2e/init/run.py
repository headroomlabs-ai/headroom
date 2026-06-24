"""Docker e2e cases for ``headroom init``.

Every case is described declaratively with :class:`Case` from
``e2e/_lib/harness.py``. Three groups run in order:

1. **existing sequence**: preserves the original scenario that exercised
   ``headroom init claude`` (local) -> ``init -g copilot`` (global) ->
   ``init codex`` (local), sharing scratch state so manifest-merge is
   exercised end-to-end.
2. **bare ``init -g`` detection**: verifies the UX regression from #245
   stays fixed — both "no shims found" (friendly error, exit 1) and
   "all shims found" (exit 0, all four agents configured).
3. **per-subcommand**: one case per ``init -g <agent>`` with only that
   agent's shim on PATH, so the explicit path is covered independently.

The fourth group covers ``--verbose`` output going to stderr.

Run directly: ``python e2e/init/run.py`` (inside the Docker image built
from ``e2e/init/Dockerfile``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

# Add repo root to sys.path so the harness import works whether the file is
# invoked as ``python e2e/init/run.py`` or ``python -m e2e.init.run``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from e2e._lib import (  # noqa: E402
    Case,
    CaseContext,
    run_case_sequence,
    run_cases,
)
from headroom.cli import init as init_cli  # noqa: E402

# ----- helpers reused across cases --------------------------------------------

# Docker image builds the workspace at /workspace; the marketplace source
# falls back to that repo checkout when a local marketplace manifest is found.
REPO_ROOT_IN_CONTAINER = Path("/workspace")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _expect_hook_command(command: str, profile: str) -> None:
    if "init hook ensure" not in command:
        raise AssertionError(f"missing 'init hook ensure' in: {command}")
    if f"--profile {profile}" not in command:
        raise AssertionError(f"missing '--profile {profile}' in: {command}")


def _read_manifest(home: Path, profile: str) -> dict[str, object]:
    path = home / ".headroom" / "deploy" / profile / "manifest.json"
    if not path.exists():
        raise AssertionError(f"Expected manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_codex_hooks_feature(config: str) -> None:
    parsed = tomllib.loads(config)
    features = parsed.get("features")
    if not isinstance(features, dict) or features.get("hooks") is not True:
        raise AssertionError("Codex config should enable hooks")
    if "codex_hooks" in features:
        raise AssertionError("Codex config should not keep deprecated codex_hooks")


# ----- existing-flow assertions (ported verbatim from the old run.py) ---------


def _verify_claude_local(ctx: CaseContext) -> None:
    settings_path = ctx.project / ".claude" / "settings.local.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if settings["env"]["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:9011":
        raise AssertionError(
            f"Claude local settings should point at port 9011, got "
            f"{settings['env']['ANTHROPIC_BASE_URL']!r}"
        )
    session_start = settings["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    pre_tool = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    profile = init_cli._local_profile(ctx.project)
    _expect_hook_command(session_start, profile)
    _expect_hook_command(pre_tool, profile)

    manifest = _read_manifest(ctx.home, profile)
    if "claude" not in manifest["targets"]:
        raise AssertionError(
            f"Claude init should register the claude target, got {manifest['targets']}"
        )

    claude_calls = [
        record["argv"] for record in _read_jsonl(ctx.shim_log) if record["tool"] == "claude"
    ]
    # `init` auto-registers the headroom MCP server after the marketplace
    # install (see d9d8972 — keeps `[Retrieve more: hash=…]` markers from
    # being dead pointers for users who never ran `headroom mcp install`).
    # The `-e HEADROOM_PROXY_URL=…` arg is only emitted when the proxy
    # port differs from the 8787 default; this case uses --port 9011.
    expected = [
        ["plugin", "marketplace", "add", str(REPO_ROOT_IN_CONTAINER)],
        ["plugin", "install", "headroom@headroom-marketplace", "--scope", "local"],
        [
            "mcp",
            "add",
            "headroom",
            "-s",
            "user",
            "-e",
            "HEADROOM_PROXY_URL=http://127.0.0.1:9011",
            "--",
            "headroom",
            "mcp",
            "serve",
        ],
    ]
    if claude_calls != expected:
        raise AssertionError(f"Unexpected Claude install commands: {claude_calls}")


def _verify_copilot_global(ctx: CaseContext) -> None:
    config = json.loads((ctx.home / ".copilot" / "config.json").read_text(encoding="utf-8"))
    if "SessionStart" not in config["hooks"]:
        raise AssertionError("Copilot config missing SessionStart hooks")
    if "PreToolUse" not in config["hooks"]:
        raise AssertionError("Copilot config missing PreToolUse hooks")
    session_start = config["hooks"]["SessionStart"][0]["command"]
    _expect_hook_command(session_start, "init-user")

    for shell_file in (ctx.home / ".bashrc", ctx.home / ".zshrc", ctx.home / ".profile"):
        content = shell_file.read_text(encoding="utf-8")
        for literal in (
            'export COPILOT_PROVIDER_TYPE="openai"',
            'export COPILOT_PROVIDER_BASE_URL="http://127.0.0.1:9005/v1"',
            'export COPILOT_PROVIDER_WIRE_API="completions"',
        ):
            if literal not in content:
                raise AssertionError(f"{shell_file.name} missing {literal!r}")

    copilot_calls = [
        record["argv"] for record in _read_jsonl(ctx.shim_log) if record["tool"] == "copilot"
    ]
    expected = [
        ["plugin", "marketplace", "add", str(REPO_ROOT_IN_CONTAINER)],
        ["plugin", "install", "headroom@headroom-marketplace"],
    ]
    if copilot_calls != expected:
        raise AssertionError(f"Unexpected Copilot install commands: {copilot_calls}")


def _verify_codex_local(ctx: CaseContext) -> None:
    config = (ctx.project / ".codex" / "config.toml").read_text(encoding="utf-8")
    hooks = json.loads((ctx.project / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    profile = init_cli._local_profile(ctx.project)

    if 'base_url = "http://127.0.0.1:9012/v1"' not in config:
        raise AssertionError("Codex config should point at the requested proxy port (9012)")
    if 'env_key = "OPENAI_API_KEY"' in config:
        raise AssertionError("Codex local init should preserve OAuth and never inject env_key")
    # Bug 3 (#406): requires_openai_auth must be absent from headroom provider blocks.
    if "requires_openai_auth" in config:
        raise AssertionError(
            "Codex local init must NOT inject requires_openai_auth into the headroom provider block"
        )
    if "supports_websockets = true" not in config:
        raise AssertionError("Codex local init missing 'supports_websockets = true'")
    if config.count("[features]") != 1:
        raise AssertionError("Codex config should keep a single [features] table")
    _expect_codex_hooks_feature(config)
    command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    _expect_hook_command(command, profile)

    manifest = _read_manifest(ctx.home, profile)
    targets = manifest["targets"]
    if set(targets) != {"claude", "codex"}:
        raise AssertionError(f"Unexpected merged targets: {targets}")


# ----- new cases (issue #245 fix + per-subcommand coverage) -------------------


def _verify_claude_global(ctx: CaseContext) -> None:
    settings = json.loads((ctx.home / ".claude" / "settings.json").read_text(encoding="utf-8"))
    if settings["env"]["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:8787":
        raise AssertionError(
            f"Claude user settings should default to port 8787, got "
            f"{settings['env']['ANTHROPIC_BASE_URL']!r}"
        )
    _expect_hook_command(
        settings["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        init_cli._GLOBAL_PROFILE,
    )


def _verify_codex_global(ctx: CaseContext) -> None:
    config = (ctx.home / ".codex" / "config.toml").read_text(encoding="utf-8")
    if 'base_url = "http://127.0.0.1:8787/v1"' not in config:
        raise AssertionError("Codex user config should point at port 8787 by default")
    if 'env_key = "OPENAI_API_KEY"' in config:
        raise AssertionError("Codex global init should preserve OAuth and never inject env_key")
    # Bug 3 (#406): requires_openai_auth must be absent from headroom provider blocks.
    if "requires_openai_auth" in config:
        raise AssertionError(
            "Codex global init must NOT inject requires_openai_auth into the headroom provider block"
        )
    if "supports_websockets = true" not in config:
        raise AssertionError("Codex global init missing 'supports_websockets = true'")
    _expect_codex_hooks_feature(config)
    hooks = json.loads((ctx.home / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    _expect_hook_command(
        hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"],
        init_cli._GLOBAL_PROFILE,
    )


# ----- case tables ------------------------------------------------------------


def existing_sequence_cases() -> list[Case]:
    """Preserves the original run.py scenario in one shared scratch."""

    return [
        Case(
            name="seq_claude_local",
            argv=["init", "--port", "9011", "claude"],
            shims={"claude": "record-args", "copilot": "record-args"},
            expected_exit=0,
            expected_stdout_contains=["Configured Claude Code (local scope)"],
            extra_assertions=[_verify_claude_local],
        ),
        Case(
            name="seq_copilot_global",
            argv=["init", "-g", "--port", "9005", "--backend", "openai", "copilot"],
            shims={},  # reuse shims from prior case in the sequence
            expected_exit=0,
            expected_stdout_contains=["Configured GitHub Copilot CLI (user scope)"],
            extra_assertions=[_verify_copilot_global],
        ),
        Case(
            name="seq_codex_local",
            argv=["init", "--port", "9012", "codex"],
            shims={},
            expected_exit=0,
            expected_stdout_contains=["Configured Codex (local scope)"],
            extra_assertions=[_verify_codex_local],
        ),
    ]


def bare_init_g_cases() -> list[Case]:
    """Bare ``headroom init -g`` — the direct coverage of issue #245."""

    return [
        Case(
            name="bare_init_g_no_shims",
            argv=["init", "-g"],
            shims={},  # nothing on PATH
            expected_exit=1,
            expected_stderr_contains=[
                # every target should be listed so the user knows what was tried
                "claude",
                "codex",
                "copilot",
                "openclaw",
                # concrete escape hatch — exactly what the user should type next
                "headroom init -g claude",
                # confirm -g itself is still the right flag
                "-g",
            ],
        ),
        Case(
            name="bare_init_g_with_all_shims",
            argv=["init", "-g"],
            shims={
                "claude": "record-args",
                "codex": "noop",
                "copilot": "record-args",
                "openclaw": "noop",
            },
            expected_exit=0,
            expected_stdout_contains=[
                "Configured Claude Code (user scope)",
                "Configured GitHub Copilot CLI (user scope)",
                "Configured Codex (user scope)",
            ],
        ),
    ]


def per_subcommand_cases() -> list[Case]:
    """One case per ``headroom init -g <agent>`` with only that agent's shim."""

    return [
        Case(
            name="init_g_claude_explicit",
            argv=["init", "-g", "claude"],
            shims={"claude": "record-args"},
            expected_exit=0,
            expected_stdout_contains=["Configured Claude Code (user scope)"],
            expected_files=["{home}/.claude/settings.json"],
            extra_assertions=[_verify_claude_global],
        ),
        Case(
            name="init_g_codex_explicit",
            argv=["init", "-g", "codex"],
            shims={"codex": "noop"},
            expected_exit=0,
            expected_stdout_contains=["Configured Codex (user scope)"],
            expected_files=[
                "{home}/.codex/config.toml",
                "{home}/.codex/hooks.json",
            ],
            extra_assertions=[_verify_codex_global],
        ),
        Case(
            name="init_g_copilot_explicit",
            argv=["init", "-g", "copilot"],
            shims={"copilot": "record-args"},
            expected_exit=0,
            expected_stdout_contains=["Configured GitHub Copilot CLI (user scope)"],
            expected_files=["{home}/.copilot/config.json"],
        ),
        # openclaw delegates to `headroom wrap openclaw` which has its own
        # (more expensive) init path and isn't stubbable with a simple shim.
        # We assert it fails fast with a clear error when not installed, and
        # rely on the `bare_init_g_with_all_shims` case (which uses a noop
        # openclaw shim + claude/codex/copilot shims) to cover the success
        # path alongside the other agents.
        Case(
            name="init_g_openclaw_missing",
            argv=["init", "-g", "openclaw"],
            shims={},
            expected_exit=1,
        ),
    ]


def verbose_cases() -> list[Case]:
    """Verbose flag smoke tests — debug lines should appear on stderr."""

    return [
        Case(
            name="init_verbose_no_shims",
            argv=["init", "-v", "-g"],
            shims={},
            expected_exit=1,
            expected_stderr_contains=[
                # A few structural markers from the verbose log. Kept loose so
                # minor wording tweaks don't break the test.
                "detect_init_targets",
                "claude",
                "global_scope=True",
            ],
        ),
    ]


def main() -> None:
    rc = 0
    rc |= run_case_sequence(existing_sequence_cases(), label="existing-sequence")
    rc |= run_cases(bare_init_g_cases())
    rc |= run_cases(per_subcommand_cases())
    rc |= run_cases(verbose_cases())
    if rc != 0:
        raise SystemExit(rc)
    print("[e2e] init e2e completed successfully", flush=True)


if __name__ == "__main__":
    main()
