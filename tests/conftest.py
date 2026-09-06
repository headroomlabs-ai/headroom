"""Shared pytest fixtures for Headroom tests."""

# CRITICAL: Must be set before ANY imports that could trigger sentence_transformers
# The Rust tokenizers use parallelism that deadlocks with pytest-asyncio
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests._skip_helpers import external_model_skip_reason


# A live `headroom` dev session exports HEADROOM_* into the shell (and the
# Claude wrap adds ANTHROPIC_CUSTOM_HEADERS). Click `envvar=` options pick
# those up inside CliRunner, so assertions would see the developer's proxy
# config instead of the test's. Scrub them so local runs match CI; tests
# that need a value set it explicitly via monkeypatch or CliRunner env.
@pytest.fixture(autouse=True)
def _skip_proxy_dependency_gate_unless_exercised(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Most CLI tests run without headroom-ai[proxy] extras installed."""
    if request.node.get_closest_marker("proxy_dependency_gate") is not None:
        return
    try:
        from headroom.cli import proxy
    except ModuleNotFoundError:
        # Native-wrapper jobs intentionally install only pytest and exercise the
        # installer scripts without importing Headroom's runtime dependencies.
        return
    monkeypatch.setattr(proxy, "ensure_proxy_dependencies", lambda: None)


@pytest.fixture(autouse=True)
def _scrub_developer_headroom_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("HEADROOM_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)


# The scrub above deletes every HEADROOM_* var — which includes HEADROOM_BEACON,
# and the beacon defaults to ON. So scrubbing for hermeticity is precisely what
# switches it on, and with HEADROOM_TELEMETRY_ENDPOINT scrubbed too it falls back
# to the real production endpoint. Every test that reaches the outcome funnel
# then POSTs a session event for real: observed writing into the live corpus
# during a local run, and CI would do the same on every push.
#
# Depends on the scrub fixture so it is guaranteed to run after it rather than
# relying on declaration order. A test that wants the beacon on just sets the
# var itself — monkeypatch inside the test wins over this.
@pytest.fixture(autouse=True)
def _disable_telemetry_beacon(monkeypatch, _scrub_developer_headroom_env):
    monkeypatch.setenv("HEADROOM_BEACON", "off")


# The MCP install ledger defaults to ``~/.headroom/mcp_installs.json``, so any
# test that registers a server (directly or through `wrap`) writes into the
# developer's REAL ledger — observed adding a live `claude/serena` entry during a
# local run. Since the scrub above deletes HEADROOM_WORKSPACE_DIR, the default is
# always the real home. Redirect the ledger per-test instead: every writer
# (`record_install` / `clear_install` / `headroom_installed_matching`) resolves it
# through this module-global, so one patch covers them all. Patched here rather
# than pointing workspace_dir() at a tmp path, which would break the tests that
# assert the default workspace layout.
@pytest.fixture(autouse=True)
def _isolate_mcp_ledger(monkeypatch, tmp_path_factory):
    # Same guard as _reset_copilot_routing_flag below: the macos/windows-native-
    # wrapper CI jobs install only pytest and drive the installer shell scripts
    # via subprocess, so headroom isn't importable and there is no ledger to
    # redirect. Skip there instead of erroring at setup.
    try:
        from headroom.mcp_registry import ledger
    except ModuleNotFoundError:
        return

    ledger_file = tmp_path_factory.mktemp("mcp-ledger") / "mcp_installs.json"
    monkeypatch.setattr(ledger, "ledger_path", lambda: ledger_file)


# The Copilot "routed to Copilot" flag is a module-global ContextVar that
# build_copilot_upstream_url() sets as a side effect. Unit tests that call that
# builder directly (or otherwise run in the shared root context) would leave it
# set and mislabel a later test's request outcome as "copilot". Reset it around
# every test so build-time side effects can't leak between tests.
@pytest.fixture(autouse=True)
def _reset_copilot_routing_flag():
    # The macos/windows-native-wrapper CI jobs run the installer tests with only
    # pytest installed (no headroom): they drive the installer shell scripts via
    # subprocess, so headroom isn't importable and there's no routing flag to
    # reset. Skip the reset there instead of erroring at setup.
    try:
        from headroom.copilot_auth import reset_request_routed_to_copilot
    except ModuleNotFoundError:
        yield
        return

    reset_request_routed_to_copilot()
    yield
    reset_request_routed_to_copilot()


# `savings_tracker._resolve_litellm_model` is an `lru_cache`d, module-global,
# process-lifetime cache keyed by model name (bounded — see #2860). Many test
# files monkeypatch `savings_tracker.litellm` to a fake with different
# `model_cost`/`cost_per_token` behavior per test, but reuse common model
# names like "gpt-4o" across them. Without a reset, whichever test resolves
# "gpt-4o" first "wins" the cache entry for the rest of the run, and later
# tests silently stop exercising their own fake — a real-not-hypothetical
# order-dependence bug once the cache is process-lifetime instead of per-call.
# Clear before AND after so a test's own within-test resolutions never leak
# in from, or leak out to, a neighboring test either.
@pytest.fixture(autouse=True)
def _reset_litellm_model_resolution_cache():
    try:
        from headroom.proxy.savings_tracker import _resolve_litellm_model
    except ModuleNotFoundError:
        yield
        return

    _resolve_litellm_model.cache_clear()
    yield
    _resolve_litellm_model.cache_clear()


# =============================================================================
# Global test hooks
# =============================================================================


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap test execution to skip transient or offline external model failures.

    This handles model-loading failures that occur when:
    - HuggingFace Hub is slow during model downloads (sentence-transformers)
    - Required HuggingFace model files were not restored into the offline CI cache
    - External embedding APIs timeout
    - Network connectivity issues in CI
    """
    outcome = yield

    if outcome.excinfo is not None:
        exc_type, exc_value, exc_tb = outcome.excinfo
        reason = external_model_skip_reason(exc_value)
        if reason is not None:
            pytest.skip(reason)


@pytest.fixture(autouse=True)
def _null_binary_pins():
    """Null the tools.json SHA-256 pins during tests.

    Installer tests fetch small mock archives, whose digests can't match the
    real published pins. Nulling the pins lets those download/extract mechanics
    tests run (verification then falls back to HTTPS trust); the tests that
    specifically exercise verification set their own pin explicitly. Production
    keeps the real pins (this fixture is test-only) and the tools-hash-refresh
    CI gate guarantees they stay correct.
    """
    try:
        from headroom import binaries
    except Exception:
        # Lean CI environments (e.g. the native-installer jobs) omit heavy deps
        # such as opentelemetry that importing `binaries` pulls in. There are no
        # tool pins to null there, so skip cleanly rather than erroring at setup.
        yield
        return

    saved = [
        (asset, asset.get("sha256"))
        for tool in binaries._registry().get("tools", {}).values()
        for asset in tool.get("assets", {}).values()
    ]
    for asset, _original in saved:
        asset["sha256"] = None
    yield
    for asset, original in saved:
        asset["sha256"] = original


@pytest.fixture(autouse=True)
def _reset_headroom_logger_propagation():
    """Keep `headroom.*` log records flowing to pytest's caplog handler.

    Two sources disable propagation on the headroom logger tree and never
    restore it, which then makes later `caplog`-based assertions flaky in
    full-suite runs (caplog attaches to root, so a `propagate=False` anywhere
    on the chain silently drops the records):

    - ``headroom.proxy.helpers._setup_file_logging`` sets
      ``getLogger("headroom").propagate = False`` on proxy startup.
    - ``benchmarks.claude_session_mode_benchmark._disable_headroom_benchmark_logging``
      (exercised by ``test_claude_session_mode_benchmark``) sets
      ``propagate = False`` + ``CRITICAL`` on ``headroom``, ``headroom.proxy``,
      ``headroom.transforms``, ``headroom.cache`` (and children).

    Resetting only ``"headroom"`` is not enough — a child like
    ``"headroom.proxy"`` left non-propagating blocks the record before it
    reaches root. Reset the whole subtree before every test so capture is
    deterministic regardless of run order.
    """
    import logging as _logging

    for _name in ("headroom", *list(_logging.root.manager.loggerDict)):
        if _name == "headroom" or _name.startswith("headroom."):
            logger = _logging.getLogger(_name)
            logger.disabled = False
            # The benchmark also raises the level to CRITICAL; children
            # inherit it (effective level), so a WARNING would be filtered
            # at the logger before it can propagate to caplog. Reset to
            # NOTSET so the subtree inherits root's level deterministically.
            logger.setLevel(_logging.NOTSET)
            logger.propagate = True
    yield


# =============================================================================
# Sample messages fixtures
# =============================================================================


# Sample messages fixtures
@pytest.fixture
def sample_messages():
    """Basic conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


@pytest.fixture
def sample_messages_with_tools():
    """Conversation with tool calls and responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for user 12345"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search_user", "arguments": '{"user_id": "12345"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"id": "12345", "name": "Alice", "email": "alice@example.com"}',
        },
        {"role": "assistant", "content": "I found user Alice with ID 12345."},
    ]


@pytest.fixture
def sample_tool_output_large():
    """Large tool output for compression testing (100 items)."""
    return json.dumps(
        [
            {
                "id": i,
                "name": f"Item {i}",
                "score": i * 0.1,
                "status": "active" if i % 2 == 0 else "inactive",
            }
            for i in range(100)
        ]
    )


@pytest.fixture
def sample_tool_output_with_errors():
    """Tool output containing error items."""
    items = [{"id": i, "status": "success"} for i in range(20)]
    items[5] = {"id": 5, "status": "error", "message": "Connection refused"}
    items[15] = {"id": 15, "status": "failed", "exception": "TimeoutError"}
    return json.dumps(items)


@pytest.fixture
def sample_system_prompt_with_date():
    """System prompt containing dynamic date."""
    return "You are a helpful assistant. Current date: 2025-01-06. Help the user with their tasks."


@pytest.fixture
def sample_anthropic_messages():
    """Anthropic-style messages with content blocks."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "..."},
                },
            ],
        }
    ]


# Mock client fixtures
@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock = Mock()
    mock.id = "chatcmpl-123"
    mock.model = "gpt-4o"
    mock.usage = Mock()
    mock.usage.prompt_tokens = 100
    mock.usage.completion_tokens = 50
    mock.usage.total_tokens = 150
    mock.choices = [Mock()]
    mock.choices[0].message = Mock()
    mock.choices[0].message.content = "This is a response."
    mock.choices[0].message.role = "assistant"
    mock.choices[0].finish_reason = "stop"
    return mock


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=mock_openai_response)
    return client


# Storage fixtures
@pytest.fixture
def temp_sqlite_db():
    """Temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_jsonl_file():
    """Temporary JSONL file path."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# Provider fixtures
@pytest.fixture
def openai_provider():
    """OpenAI provider instance."""
    from headroom.providers.openai import OpenAIProvider

    return OpenAIProvider()


@pytest.fixture
def openai_tokenizer():
    """OpenAI token counter for gpt-4o."""
    from headroom.providers.openai import OpenAITokenCounter

    return OpenAITokenCounter("gpt-4o")


# Config fixtures
@pytest.fixture
def default_config():
    """Default HeadroomConfig."""
    from headroom.config import HeadroomConfig

    return HeadroomConfig()


@pytest.fixture
def smart_crusher_config():
    """SmartCrusher config for testing."""
    from headroom.config import SmartCrusherConfig

    return SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=3,
        min_tokens_to_crush=0,  # Always crush for tests
    )


# Helper for creating RequestMetrics
@pytest.fixture
def sample_request_metrics():
    """Sample RequestMetrics for storage tests."""
    from headroom.config import RequestMetrics

    return RequestMetrics(
        request_id="test-123",
        timestamp=datetime(2025, 1, 6, 12, 0, 0),
        model="gpt-4o",
        stream=False,
        mode="audit",
        tokens_input_before=1000,
        tokens_input_after=800,
        tokens_output=200,
        block_breakdown={"system": 100, "user": 200, "assistant": 500},
        waste_signals={"json_bloat": 50},
        stable_prefix_hash="abc123",
        cache_alignment_score=85.0,
        cached_tokens=100,
        transforms_applied=["CacheAligner", "SmartCrusher"],
        tool_units_dropped=1,
        turns_dropped=0,
        messages_hash="def456",
    )
