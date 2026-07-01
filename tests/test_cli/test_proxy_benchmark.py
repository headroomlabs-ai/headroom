"""Tests for the proxy benchmark CLI helper."""

from __future__ import annotations

import json

from click.testing import CliRunner

from headroom.cli.main import main


def _write_stats(path, *, input_tokens: int, saved_tokens: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "input": input_tokens,
                    "saved": saved_tokens,
                    "savings_percent": round(100 * saved_tokens / (input_tokens + saved_tokens), 2),
                },
                "requests": {"total": 3},
            }
        )
    )


def test_proxy_benchmark_compare_text(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    _write_stats(baseline, input_tokens=1000)
    _write_stats(optimized, input_tokens=700, saved_tokens=300)

    result = CliRunner().invoke(
        main,
        ["proxy-benchmark", "compare", str(baseline), str(optimized)],
    )

    assert result.exit_code == 0, result.output
    assert "Local LLM prefill benchmark" in result.output
    assert "baseline input tokens" in result.output
    assert "1,000" in result.output
    assert "700" in result.output
    assert "300" in result.output
    assert "30.00%" in result.output


def test_proxy_benchmark_compare_json(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    _write_stats(baseline, input_tokens=153_600)
    _write_stats(optimized, input_tokens=106_700, saved_tokens=46_900)

    result = CliRunner().invoke(
        main,
        [
            "proxy-benchmark",
            "compare",
            str(baseline),
            str(optimized),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["baseline_input_tokens"] == 153_600
    assert data["optimized_input_tokens"] == 106_700
    assert data["tokens_saved"] == 46_900
    assert data["savings_percent"] == 30.53


def test_proxy_benchmark_compare_rejects_missing_tokens(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    baseline.write_text(json.dumps({"tokens": {}}))
    _write_stats(optimized, input_tokens=700, saved_tokens=300)

    result = CliRunner().invoke(
        main,
        ["proxy-benchmark", "compare", str(baseline), str(optimized)],
    )

    assert result.exit_code != 0
    assert "missing tokens.input" in result.output
