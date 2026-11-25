"""The default pytest run must deselect tests marked ``live`` (they call real
provider APIs with the developer's keys) and ``-m live`` must still select them.

Collection only: nothing here contacts a provider.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_FILE = "tests/test_thinking_signature_scope_live.py"
MIXED_FILE = "tests/test_realignment_live_multi_turn.py"
OFFLINE_TEST = "test_classify_auth_mode_routes_payg_vs_oauth"


def _collect(*args: str, allow_empty: bool = False) -> str:
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "sk-ant-dummy-collect-only",
        "OPENAI_API_KEY": "sk-dummy-collect-only",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # A failed collection prints no test names, which would let the "not in" checks pass.
    # pytest exits 5 when every test is deselected; that is the expected result for a
    # directory whose tests are all live.
    ok = (0, 5) if allow_empty else (0,)
    assert result.returncode in ok, result.stdout + result.stderr
    return result.stdout


def test_default_run_deselects_live_tests() -> None:
    out = _collect(LIVE_FILE, MIXED_FILE)
    assert "test_exact_replay_is_accepted" not in out
    assert "test_anthropic_cache_hit_across_two_turns" not in out


def test_default_run_keeps_the_offline_test_in_a_live_file() -> None:
    assert OFFLINE_TEST in _collect(MIXED_FILE)


def test_dash_m_live_selects_live_tests_again() -> None:
    out = _collect("-m", "live", LIVE_FILE, MIXED_FILE)
    assert "test_exact_replay_is_accepted" in out
    assert "test_anthropic_cache_hit_across_two_turns" in out
    assert OFFLINE_TEST not in out


LIVE_DIR = "tests/test_live"
LIVE_DIR_TESTS = (
    "test_api_accepts_stale_marker_shape",
    "test_model_reads_grouped_format",
    "test_held_read_is_cached_and_not_rewritten",
)


def test_default_run_deselects_the_whole_test_live_directory() -> None:
    out = _collect(LIVE_DIR, allow_empty=True)
    for name in LIVE_DIR_TESTS:
        assert name not in out


def test_dash_m_live_selects_the_test_live_directory_again() -> None:
    out = _collect("-m", "live", LIVE_DIR)
    for name in LIVE_DIR_TESTS:
        assert name in out


# Provider-calling integration files. Each entry is (node-id prefix, why). A prefix that
# is a whole file covers every test in it. The default run must collect none of the
# ``LIVE_PREFIXES`` and ``-m live`` must collect all of them; the ``LOCAL_PREFIXES`` do
# not leave the process (proxy-side validation, mocks), so they stay in the default run.
RESPONSES = "tests/test_proxy_openai_responses_integration.py"
GEMINI_CHAT = "tests/test_proxy_gemini_integration.py"
GEMINI_NATIVE = "tests/test_proxy_gemini_native_integration.py"
VERTEX = "tests/test_proxy_vertex_native_integration.py"
PASSTHROUGH = "tests/test_proxy_passthrough_integration.py"
BATCH = "tests/test_proxy_batch_integration.py"
MEMORY = "tests/test_proxy_memory_integration.py"
STREAMING = "tests/test_openai_streaming_backend.py"

PROVIDER_FILES = (RESPONSES, GEMINI_CHAT, GEMINI_NATIVE, VERTEX)
PROVIDER_FILES += (PASSTHROUGH, BATCH, MEMORY, STREAMING)

LIVE_PREFIXES = (
    # whole files: every test forwards to the provider
    RESPONSES + "::",
    GEMINI_CHAT + "::",
    GEMINI_NATIVE + "::",
    VERTEX + "::",
    # passthrough: models, embeddings, moderations, stats, malformed body
    PASSTHROUGH + "::TestOpenAIModels",
    PASSTHROUGH + "::TestOpenAIEmbeddings",
    PASSTHROUGH + "::TestOpenAIModerations",
    PASSTHROUGH + "::TestGeminiModels",
    PASSTHROUGH + "::TestAnthropicModels",
    PASSTHROUGH + "::TestPassthroughStats",
    PASSTHROUGH + "::TestPassthroughErrorHandling::test_invalid_json_body",
    # no key set still means an unauthenticated request leaves the process
    PASSTHROUGH + "::TestPassthroughErrorHandling::test_missing_auth_header_openai",
    # batch: create-with-compression, list, stats, and the file lookup upstream
    BATCH + "::TestOpenAIBatchCreate::test_batch_create_with_compression",
    BATCH + "::TestOpenAIBatchList",
    BATCH + "::TestAnthropicBatchCreate::test_batch_create_with_compression",
    BATCH + "::TestAnthropicBatchList",
    BATCH + "::TestBatchCompressionStats",
    BATCH + "::TestAnthropicBatchCompressionStats",
    BATCH + "::TestBatchErrorHandling::test_openai_batch_invalid_file_id",
    BATCH + "::TestBatchErrorHandling::test_openai_batch_missing_auth",
    # memory proxy and streaming backend: real Anthropic / OpenAI requests
    MEMORY + "::TestMemoryHeaderValidation",
    MEMORY + "::TestMemoryToolInjection",
    MEMORY + "::TestMemorySaveAndSearch",
    MEMORY + "::TestMemoryUserIsolation",
    MEMORY + "::TestMemoryIdAutoTailAndUpdate",
    STREAMING + "::TestOpenAIStreamingRealAPI",
)

LOCAL_PREFIXES = (
    PASSTHROUGH + "::test_head_root_returns_200_not_405",
    BATCH + "::TestOpenAIBatchCreate::test_batch_create_validation_missing_input_file",
    BATCH + "::TestOpenAIBatchCreate::test_batch_create_validation_missing_endpoint",
    BATCH + "::TestAnthropicBatchCreate::test_batch_create_validation_missing_requests",
    BATCH + "::TestAnthropicBatchCreate::test_batch_create_validation_empty_requests",
    BATCH + "::TestBatchErrorHandling::test_openai_batch_invalid_json",
    BATCH + "::test_copilot_anthropic_batch_rejection_precedes_body_parsing",
    MEMORY + "::TestMemoryStats",
    STREAMING + "::TestOpenAIStreamingMock",
)


@functools.cache
def _node_ids(live: bool) -> tuple[str, ...]:
    # _collect already passes -q; with addopts -v a second -q is what prints node ids
    args = ("-q", "-m", "live") if live else ("-q",)
    out = _collect(*args, *PROVIDER_FILES, allow_empty=True)
    return tuple(line for line in out.splitlines() if "::" in line)


def _selected(ids: tuple[str, ...], prefix: str) -> list[str]:
    return [i for i in ids if i.startswith(prefix)]


def test_default_run_deselects_every_provider_calling_integration_test() -> None:
    ids = _node_ids(live=False)
    leaked = [p for p in LIVE_PREFIXES if _selected(ids, p)]
    assert not leaked, f"collected by default but call a provider: {leaked}"


def test_dash_m_live_selects_every_provider_calling_integration_test() -> None:
    ids = _node_ids(live=True)
    missing = [p for p in LIVE_PREFIXES if not _selected(ids, p)]
    assert not missing, f"not selected by -m live: {missing}"


def test_default_run_keeps_the_local_validation_tests() -> None:
    ids = _node_ids(live=False)
    dropped = [p for p in LOCAL_PREFIXES if not _selected(ids, p)]
    assert not dropped, f"local tests deselected by default: {dropped}"


def test_dash_m_live_leaves_the_local_validation_tests_out() -> None:
    ids = _node_ids(live=True)
    selected = [p for p in LOCAL_PREFIXES if _selected(ids, p)]
    assert not selected, f"local tests wrongly marked live: {selected}"
