"""The token-counting accuracy eval must not grade an estimate against an estimate.

With the reference tokenizer unavailable (offline with a cold
``TIKTOKEN_CACHE_DIR``), the eval used ``len(text) // 4`` as the "actual" count
and reported exact matches and error rates as if they had been measured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from headroom.evals import batch_compression_eval as bce
from headroom.tokenizers.tiktoken_counter import TiktokenVocabNotCached


def _refuse(_name: str) -> object:
    raise TiktokenVocabNotCached("cl100k_base is not cached")


def test_a_cold_reference_reports_accuracy_as_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bce, "load_encoding", _refuse)
    result = bce.evaluate_token_counting_accuracy()

    assert not result.reference_available
    assert "not cached" in (result.reference_error or "")
    assert (result.exact_matches, result.avg_error_percent, result.max_error_percent) == (
        None,
        None,
        None,
    )
    assert result.test_cases == len(result.results) > 0
    assert all(set(r) == {"text_length", "estimated_tokens"} for r in result.results)
    assert "Accuracy not measured" in result.summary()


def test_a_warm_reference_is_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    words = SimpleNamespace(encode=lambda text: text.split())
    monkeypatch.setattr(bce, "load_encoding", lambda _name: words)
    result = bce.evaluate_token_counting_accuracy()

    assert result.reference_available
    assert result.exact_matches is not None and result.avg_error_percent is not None
    assert result.results[0]["actual_tokens"] == len("Hello, world!".split())
    assert "Exact matches" in result.summary()


def test_cli_offline_cold_cache_exports_no_accuracy(tmp_path) -> None:
    out = tmp_path / "results.json"
    env = {
        "HEADROOM_OFFLINE": "1",
        "TIKTOKEN_CACHE_DIR": str(tmp_path / "empty-cache"),
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "headroom.evals.batch_compression_eval"]
        + ["--token-counting", "--output", str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Accuracy not measured" in proc.stdout
    data = json.loads(out.read_text())
    assert data["reference_available"] is False
    assert data["exact_matches"] is None and data["avg_error_percent"] is None
    assert all("actual_tokens" not in r for r in data["results"])
