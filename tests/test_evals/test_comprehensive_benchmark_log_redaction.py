"""run_lm_eval must not log --model_args, which can carry an API key (CodeQL #177).

The redaction is log-only: lm_eval itself still has to receive the real value,
or the benchmark silently runs against the wrong endpoint/credentials.
"""

import logging
from types import SimpleNamespace

import pytest

from headroom.evals import comprehensive_benchmark

FAKE_SECRET = "sk-test-FAKE-0123456789abcdef"


def test_model_args_redacted_in_log_but_passed_to_lm_eval(monkeypatch, caplog, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        # Non-zero exit stops run_lm_eval right after the call, before it
        # looks for result files we did not create.
        return SimpleNamespace(returncode=1, stdout="", stderr="stopped by test")

    monkeypatch.setattr(comprehensive_benchmark, "run", fake_run)

    with caplog.at_level(logging.INFO, logger=comprehensive_benchmark.__name__):
        with pytest.raises(RuntimeError, match="stopped by test"):
            comprehensive_benchmark.run_lm_eval(
                model_args=f"model=gpt-4o-mini,api_key={FAKE_SECRET}",
                base_url="http://localhost:8787/v1",
                tasks=["arc_easy"],
                output_path=str(tmp_path),
            )

    running = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Running:")]
    assert len(running) == 1
    assert "--model_args <redacted>" in running[0]
    assert FAKE_SECRET not in caplog.text

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index("--model_args") + 1] == (
        f"model=gpt-4o-mini,api_key={FAKE_SECRET},base_url=http://localhost:8787/v1"
    )


def test_no_model_args_logs_no_placeholder(monkeypatch, caplog, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=1, stdout="", stderr="stopped by test")

    monkeypatch.setattr(comprehensive_benchmark, "run", fake_run)

    with caplog.at_level(logging.INFO, logger=comprehensive_benchmark.__name__):
        with pytest.raises(RuntimeError):
            comprehensive_benchmark.run_lm_eval(tasks=["arc_easy"], output_path=str(tmp_path))

    assert "--model_args" not in caplog.text
    assert "--model_args" not in calls[0]
