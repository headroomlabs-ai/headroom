"""Provider failures must not become matching benchmark answers."""

from __future__ import annotations

import pytest

from headroom.evals.core import CompressionEvaluator, EvalCase, EvalMode, EvalSuite


@pytest.fixture(autouse=True)
def isolated_evaluation_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")


def _case(case_id="failure"):
    return EvalCase(
        id=case_id,
        context='{"answer": 42, "detail": "public synthetic fixture"}',
        query="What is the answer?",
        ground_truth="42",
    )


def _responses(*outcomes):
    values = iter(outcomes)

    def llm(context, query):
        outcome = next(values)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return llm


@pytest.mark.parametrize(
    "outcomes,failed_sides",
    [
        ((TimeoutError("42 timeout"), TimeoutError("42 timeout")), ["original", "compressed"]),
        ((TimeoutError("42 timeout"), "42"), ["original"]),
        (("42", TimeoutError("42 timeout")), ["compressed"]),
        ((ValueError("first"), RuntimeError("second")), ["original", "compressed"]),
    ],
)
def test_failures_are_not_scored_as_matching_answers(outcomes, failed_sides):
    semantic_calls = []

    def similarity(original, compressed):
        semantic_calls.append((original, compressed))
        return 1.0

    evaluator = CompressionEvaluator(
        llm_fn=_responses(*outcomes), semantic_similarity_fn=similarity
    )
    result = evaluator.evaluate_case(_case())
    assert not result.accuracy_preserved
    assert result.exact_match is False
    assert result.f1_score == 0.0
    assert result.semantic_similarity is None
    assert result.contains_ground_truth is ("compressed" not in failed_sides)
    assert semantic_calls == []
    assert result.error is not None
    for side in failed_sides:
        assert f"{side}: " in result.error
    assert result.to_dict()["error"] == result.error


def test_literal_error_text_is_a_successful_answer_when_no_call_raised():
    answer = "ERROR: 42 is the expected diagnostic"
    evaluator = CompressionEvaluator(llm_fn=_responses(answer, answer))
    result = evaluator.evaluate_case(_case())
    assert result.accuracy_preserved
    assert result.f1_score == 1.0
    assert result.exact_match is True
    assert result.contains_ground_truth is True
    assert result.error is None


def test_failed_calls_do_not_inflate_suite_accuracy():
    evaluator = CompressionEvaluator(
        llm_fn=_responses("42", "42", TimeoutError("same"), TimeoutError("same"))
    )
    with pytest.warns(DeprecationWarning, match="evaluate_suite"):
        result = evaluator.evaluate_suite(EvalSuite("mixed", [_case("ok"), _case("failed")]))
    assert result.total_cases == 2
    assert result.passed_cases == 1
    assert result.failed_cases == 1
    assert result.accuracy_preservation_rate == 0.5
    assert result.avg_f1_score == 0.5


@pytest.fixture
def batch_endpoint(monkeypatch, tmp_path):
    """Actual SDK endpoint with controlled errors; no external model traffic."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    pytest.importorskip("anthropic")
    responses = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            outcome = responses.pop(0)
            failed = isinstance(outcome, Exception)
            body = (
                {"type": "error", "error": {"type": "permission_error", "message": str(outcome)}}
                if failed
                else {
                    "id": "msg_fixture",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fixture",
                    "content": [{"type": "text", "text": outcome}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
            if not failed and "chat/completions" in self.path:
                body = {
                    "id": "chat_fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "fixture",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": outcome},
                            "finish_reason": "stop",
                        }
                    ],
                }
            payload = json.dumps(body).encode()
            self.send_response(403 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HEADROOM_OFFLINE", "0")  # Batch constructor guards even loopback.
    monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-fixture")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    try:
        yield responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _batch_case():
    import json

    from headroom.evals.batch_compression_eval import BatchRequest, BatchTestCase, TestCategory

    content = json.dumps([{"id": i, "answer": 42, "payload": "synthetic " * 10} for i in range(80)])
    return BatchTestCase(
        id="batch-failure",
        category=TestCategory.JSON_EXTRACTION,
        request=BatchRequest(
            custom_id="batch-failure", messages=[{"role": "user", "content": content}]
        ),
        ground_truth="42",
        ground_truth_keywords=["42"],
    )


@pytest.mark.parametrize(
    "outcomes,compressed_failed",
    [
        ([TimeoutError("42 timeout"), TimeoutError("42 timeout")], True),
        ([TimeoutError("42 timeout"), "42"], False),
        (["42", TimeoutError("42 timeout")], True),
        ([ValueError("different first"), RuntimeError("distinct second")], True),
    ],
)
def test_batch_sdk_failures_are_not_answers(batch_endpoint, outcomes, compressed_failed):
    from headroom.evals.batch_compression_eval import BatchCompressionEvaluator
    from headroom.transforms.content_router import ContentRouterConfig

    responses, requests = batch_endpoint
    responses.extend(outcomes)
    evaluator = BatchCompressionEvaluator(
        model="claude-fixture",
        router_config=ContentRouterConfig(enable_kompress=False),
        use_semantic_similarity=False,
    )
    try:
        result = evaluator.evaluate_case(_batch_case())
    finally:
        evaluator._llm_client.close()
    assert len(requests) == 2
    assert result.compressed_tokens < result.original_tokens  # Real native compression ran.
    assert not result.accuracy_preserved
    assert result.exact_match is False
    assert result.f1_score == 0.0
    assert result.rouge_l_score == 0.0
    assert result.semantic_similarity is None
    assert result.ground_truth_match is (not compressed_failed)
    assert result.keywords_found == ([] if compressed_failed else ["42"])
    assert result.keywords_missing == (["42"] if compressed_failed else [])
    assert result.error is not None
    assert result.to_dict()["error"] == result.error


def test_batch_literal_error_answer_and_aggregate_are_preserved(batch_endpoint):
    from headroom.evals.batch_compression_eval import BatchCompressionEvaluator
    from headroom.transforms.content_router import ContentRouterConfig

    responses, requests = batch_endpoint
    responses.extend(["ERROR: 42", "ERROR: 42", TimeoutError("42"), TimeoutError("42")])
    evaluator = BatchCompressionEvaluator(
        model="claude-fixture",
        router_config=ContentRouterConfig(enable_kompress=False),
        use_semantic_similarity=False,
    )
    try:
        result = evaluator.run([_batch_case(), _batch_case()])
    finally:
        evaluator._llm_client.close()
    assert len(requests) == 4
    assert result.passed_cases == 1
    assert result.failed_cases == 1
    assert result.accuracy_preservation_rate == 0.5
    assert result.avg_f1_score == 0.5
    assert result.results[0].error is None
    assert result.results[0].f1_score == 1.0


def _runner_case():
    return EvalCase(
        id="runner-fixture",
        context=_batch_case().request.messages[0]["content"],
        query="What is the answer?",
        ground_truth="42",
    )


def _runner(proxy=False):
    import os

    from headroom.evals.runners.before_after import BeforeAfterRunner, LLMConfig
    from headroom.transforms.content_router import ContentRouterConfig

    return BeforeAfterRunner(
        llm_config=LLMConfig(
            provider="anthropic",
            model="claude-fixture",
            headroom_proxy_url=os.environ["ANTHROPIC_BASE_URL"] if proxy else None,
        ),
        router_config=ContentRouterConfig(enable_kompress=False),
        use_semantic_similarity=False,
    )


def _close_runner(runner):
    runner._llm_client.close()
    if runner._proxy_client is not None:
        runner._proxy_client.close()


@pytest.mark.parametrize("proxy", [False, True])
@pytest.mark.parametrize(
    "outcomes,compressed_failed",
    [
        ([TimeoutError("42"), TimeoutError("42")], True),
        ([TimeoutError("42"), "42"], False),
        (["42", TimeoutError("42")], True),
    ],
)
def test_cli_runner_failures_are_not_answers(batch_endpoint, proxy, outcomes, compressed_failed):
    responses, requests = batch_endpoint
    responses.extend(outcomes)
    runner = _runner(proxy)
    try:
        result = runner.evaluate_case(_runner_case())
    finally:
        _close_runner(runner)
    assert len(requests) == 2
    assert result.compressed_tokens < result.original_tokens
    assert not result.accuracy_preserved
    assert result.exact_match is False
    assert result.f1_score == 0.0
    assert result.semantic_similarity is None
    assert result.contains_ground_truth is (not compressed_failed)
    assert result.error is not None


@pytest.mark.parametrize("proxy", [False, True])
def test_cli_runner_literal_error_success_is_preserved(batch_endpoint, proxy):
    responses, requests = batch_endpoint
    responses.extend(["ERROR: 42", "ERROR: 42"])
    runner = _runner(proxy)
    try:
        result = runner.run(EvalSuite("literal", [_runner_case()]))
    finally:
        _close_runner(runner)
    assert len(requests) == 2
    assert result.passed_cases == 1
    assert result.accuracy_preservation_rate == 1.0
    assert result.results[0].error is None


def test_cli_ground_truth_failure_skips_judge_and_bfcl_matching(batch_endpoint):
    responses, requests = batch_endpoint
    # The spare high-scoring judge reply must not be consumed after provider failure.
    responses.extend([TimeoutError("42"), "Reasoning: fixture\nScore: 5"])
    runner = _runner()
    case = _runner_case()
    case.ground_truth = '[{"answer": {"value": [42]}}]'
    case.metadata["source"] = "BFCL"
    try:
        result = runner.evaluate_case(case, mode=EvalMode.GROUND_TRUTH)
    finally:
        _close_runner(runner)
    assert len(requests) == 1
    assert runner._judge_fn is None
    assert not result.accuracy_preserved
    assert result.contains_ground_truth is False
    assert result.f1_score == 0.0
    assert result.error.startswith("compressed: ")


@pytest.mark.parametrize("judge_fails,bfcl", [(False, False), (True, False), (True, True)])
def test_cli_ground_truth_success_and_judge_fallback_unchanged(batch_endpoint, judge_fails, bfcl):
    responses, requests = batch_endpoint
    responses.extend(
        ["42", TimeoutError("judge unavailable") if judge_fails else "Reasoning: fixture\nScore: 5"]
    )
    runner = _runner()
    case = _runner_case()
    if bfcl:
        case.ground_truth = '[{"answer": {"value": [42]}}]'
        case.metadata["source"] = "BFCL"
    try:
        result = runner.evaluate_case(case, mode=EvalMode.GROUND_TRUTH)
    finally:
        _close_runner(runner)
    assert len(requests) == 2
    assert result.accuracy_preserved
    assert result.contains_ground_truth is True
    assert result.f1_score == (0.0 if judge_fails else 1.0)
    assert result.error is None
