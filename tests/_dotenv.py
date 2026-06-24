"""Local-only `.env` loader for tests that need provider API keys.

Why this exists: several test modules (compression-summary evals,
query-echo, cost-tracker counterfactual) need real API keys and used to
load the project `.env` at module level via `os.environ.setdefault(...)`.
That ran during pytest collection and *globally* mutated `os.environ`,
which caused unrelated tests (e.g. `test_proxy_passthrough_integration`)
to flip from cleanly skipped to running-live-and-failing — their
`@pytest.mark.skipif(not os.environ.get(...))` guards saw the leaked
key and decided not to skip.

Usage from a test module that needs `.env`:

    from tests._dotenv import load_env_overrides, autouse_apply_env

    _env = load_env_overrides()
    ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or _env.get(
        "ANTHROPIC_API_KEY", ""
    )

    pytestmark = pytest.mark.skipif(
        not ANTHROPIC_KEY,
        reason="ANTHROPIC_API_KEY not set",
    )

    apply_dotenv = autouse_apply_env(_env)

The `apply_dotenv` autouse fixture sets the values via `monkeypatch.setenv`,
which auto-restores at function-scope teardown — no cross-module leak.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def load_env_overrides() -> dict[str, str]:
    """Read the project `.env` file (if present) into a plain dict.

    Returns an empty dict when `.env` is missing — CI runs with real
    secrets in the environment and no `.env`, so the per-test fixture
    becomes a no-op there.
    """
    env_path = Path(__file__).parent.parent / ".env"
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def autouse_apply_env(overrides: dict[str, str]) -> pytest.FixtureFunction:
    """Build an autouse fixture that applies `overrides` for the test
    function and restores at teardown. Skips keys already set in the real
    environment so CI/secret-store values take precedence over `.env`.
    """

    @pytest.fixture(autouse=True)
    def _apply(monkeypatch: pytest.MonkeyPatch) -> None:
        for key, value in overrides.items():
            if not os.environ.get(key):
                monkeypatch.setenv(key, value)

    return _apply


def importorskip_no_env_leak(module_name: str):
    """`pytest.importorskip` substitute that quarantines `os.environ` mutations.

    Why: `litellm` (and other libraries that bundle `python-dotenv`) call
    `dotenv.load_dotenv()` at module import time, which loads the project
    `.env` into the global `os.environ`. When a test module does
    `pytest.importorskip("litellm")` at module-level, that pollution
    happens during pytest's collection phase — and any *later-collected*
    test module whose `@pytest.mark.skipif(not os.environ.get("FOO_API_KEY"))`
    decorator runs after the leak will see the polluted value and stop
    skipping. The proxy-passthrough integration tests stop being safely
    skipped, run live against fake keys, and fail.

    This wrapper snapshots `os.environ`, imports the module, then deletes
    any keys that the import added. The module is fully imported and
    cached in `sys.modules` — its functionality (price tables, model
    metadata) is unaffected. Subsequent `import litellm` calls hit the
    cache and don't re-run the `dotenv.load_dotenv` side-effect.

    Use as a drop-in replacement for `pytest.importorskip` at the top of
    test modules that need litellm or any other dotenv-loading library.
    """
    import importlib

    snapshot = set(os.environ)
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"{module_name} not installed", allow_module_level=True)
    for key in set(os.environ) - snapshot:
        del os.environ[key]
    return mod
